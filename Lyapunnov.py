# -*- coding: utf-8 -*-
"""
Espectro de exponentes de Lyapunov por ecuaciones variacionales.

Un único motor (Benettin + QR) aplicado a todos los sistemas de la tesis:

    rossler            control de validación (valores de referencia conocidos)
    roman_ciclo        Roman y Bertolotti en red, delta = 1,5 delta_c  (ciclo límite)
    roman_red          Roman y Bertolotti en red, delta = 10 delta_c   (hipercaos)
    roman_migracion    idem con migración, sigma = 0,0315              (toro)
    handy_red          HANDY igualitaria extendido a 3 sociedades
    handy_forzado      HANDY igualitaria con forzado climático de dos escalas

El método: junto con la trayectoria se integra la ecuación variacional
v' = J(x(t)) v sobre n vectores tangentes, que se reortonormalizan cada tau
años con una descomposición QR; el exponente k-ésimo es el promedio temporal
de log|R_kk| (Benettin et al. 1980). Esto evita el estimador de dos
trayectorias, que sobre un oscilador de relajación devuelve un lambda_1
positivo espurio incluso sobre un ciclo límite.

Tres detalles que importan:

  * COORDENADAS LOGARÍTMICAS. Cada sistema declara qué variables se integran
    en log. Los exponentes de Lyapunov son invariantes ante este cambio de
    coordenadas (es un difeomorfismo), pero en coordenadas lineales el
    integrador lleva a cero las variables que caen muchos órdenes de magnitud.

  * TIEMPO ABSOLUTO. En los sistemas no autónomos (HANDY forzado) el tiempo
    tiene que avanzar de un bloque de reortonormalización al siguiente; si
    cada bloque se integra sobre (0, tau) el forzado se reinicia y el
    resultado no significa nada.

  * VERIFICACIÓN DEL JACOBIANO. Antes de cada corrida se compara el jacobiano
    analítico contra diferencias finitas centradas. Si el error relativo no
    es chico, el espectro no sirve y conviene no mirarlo.

''

# %% imports y utilidades
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Optional
from scipy.integrate import solve_ivp

plt.rcParams.update({"font.size": 13})

RTOL, ATOL = 1e-9, 1e-11


@dataclass
class Sistema:
    nombre: str
    etiqueta: str                     # para las leyendas de las figuras
    f: Callable                       # f(t, u) -> du/dt   (coordenadas físicas)
    J: Callable                       # J(t, u) -> jacobiano (coordenadas físicas)
    u0: np.ndarray                    # condición inicial  (coordenadas físicas)
    log: np.ndarray                   # máscara: True = esa variable va en log
    tau: float = 100.0                # paso de reortonormalización [años]
    K: int = 600                      # número de reortonormalizaciones
    t_trans: float = 3.0e4            # transitorio descartado [años]
    diagnostico: Optional[Callable] = None  # diagnóstico opcional sobre el estado
    autonomo: bool = True             # False si el campo depende de t (forzado)


def _a_U(u, mask):
    """Coordenadas físicas -> coordenadas de integración."""
    return np.where(mask, np.log(np.maximum(u, 1e-300)), u)


def _a_u(U, mask):
    """Coordenadas de integración -> coordenadas físicas."""
    return np.where(mask, np.exp(np.clip(U, -700, 700)), U)


def campo_transformado(sis):
    """Devuelve (g, G): campo y jacobiano en las coordenadas de integración.

    Si U_i = ln u_i, entonces  g_i = f_i / u_i  y

        G_ij = (1/u_i) J_ij u_j - delta_ij f_i / u_i,

    que es la expresión que se usa para las variables marcadas en `log`;
    las variables lineales pasan sin cambio.
    """
    mask = sis.log

    def g(t, U):
        u = _a_u(U, mask)
        du = np.asarray(sis.f(t, u), float).copy()
        du[mask] /= u[mask]
        return du

    def G(t, U):
        u = _a_u(U, mask)
        Jm = np.asarray(sis.J(t, u), float)
        col = np.where(mask, u, 1.0)          # du_j / dU_j
        fil = np.ones_like(u)
        fil[mask] = 1.0 / u[mask]             # factor de fila
        M = (fil[:, None] * Jm) * col[None, :]
        diag = np.zeros_like(u)
        diag[mask] = np.asarray(sis.f(t, u), float)[mask] / u[mask]
        M -= np.diag(diag)
        return M

    return g, G


def verificar_jacobiano(sis, U, t=0.0, h=1e-6):
    """Compara el jacobiano analítico contra diferencias finitas centradas."""
    g, G = campo_transformado(sis)
    n = U.size
    Gfd = np.array([(g(t, U + h * e) - g(t, U - h * e)) / (2 * h)
                    for e in np.eye(n)]).T
    Gan = G(t, U)
    esc = max(np.abs(Gan).max(), 1e-30)
    return np.abs(Gfd - Gan).max() / esc


# %% Benettin + QR
def espectro_lyapunov(sis, verbose=True):
    """Espectro completo por ecuaciones variacionales con reortonormalización QR.

    Devuelve un diccionario con el espectro, la historia de convergencia
    (estimación acumulada en función del tiempo) y la dimensión de Kaplan-Yorke.
    """
    g, G = campo_transformado(sis)
    n = sis.u0.size
    U = _a_U(np.asarray(sis.u0, float), sis.log)
    t = 0.0

    # --- transitorio ---------------------------------------------------
    if sis.t_trans > 0:
        sol = solve_ivp(g, (t, t + sis.t_trans), U, method="LSODA",
                        rtol=RTOL, atol=ATOL)
        U = sol.y[:, -1]
        t = sol.t[-1]

    # --- control del jacobiano ------------------------------------------
    err = verificar_jacobiano(sis, U, t)
    if verbose:
        print(f"  error relativo del jacobiano: {err:.2e}"
              + ("   <-- ATENCION, deberia ser < 1e-5" if err > 1e-5 else ""))
    if sis.diagnostico is not None and verbose:
        print("  " + sis.diagnostico(_a_u(U, sis.log)))

    # --- lazo de Benettin ------------------------------------------------
    def rhs(tt, s):
        Uu = s[:n]
        Qq = s[n:].reshape(n, n)
        return np.concatenate([g(tt, Uu), (G(tt, Uu) @ Qq).ravel()])

    Q = np.eye(n)
    S = np.zeros(n)
    hist = np.empty((sis.K, n))
    tiempo = np.empty(sis.K)

    for k in range(sis.K):
        sol = solve_ivp(rhs, (t, t + sis.tau),
                        np.concatenate([U, Q.ravel()]),
                        method="LSODA", rtol=RTOL, atol=ATOL)
        U = sol.y[:n, -1]
        t = sol.t[-1]

        Qn, R = np.linalg.qr(sol.y[n:, -1].reshape(n, n))
        d = np.sign(np.diag(R))
        d[d == 0] = 1.0
        Q = Qn * d                      # frame ortonormal con diag(R) > 0
        S += np.log(np.abs(np.diag(R)))

        hist[k] = S / ((k + 1) * sis.tau)
        tiempo[k] = (k + 1) * sis.tau

        if verbose and (k + 1) % max(1, sis.K // 4) == 0:
            print(f"    {k+1:5d}/{sis.K}  lambda(x1e-3) = "
                  f"{np.round(np.sort(hist[k])[::-1][:4] * 1e3, 3)}")

    orden = np.argsort(S)[::-1]
    lam = (S / (sis.K * sis.tau))[orden]
    hist = hist[:, orden]

    # --- dimensión de Kaplan-Yorke ---------------------------------------
    # En un sistema NO AUTÓNOMO el espectro medido no contiene el exponente
    # nulo asociado a la fase del forzado, porque el tiempo entra como
    # parámetro y no como variable de estado. Para la dimensión hay que
    # agregarlo a mano: si no, D_KY sale una unidad por debajo.
    lam_dky = lam if sis.autonomo else np.sort(np.append(lam, 0.0))[::-1]
    cs = np.cumsum(lam_dky)
    if (cs >= 0).any():
        j = int(np.max(np.where(cs >= 0)[0]))
        dky = (j + 1 + cs[j] / abs(lam_dky[j + 1])
               if j + 1 < lam_dky.size else float("nan"))
    else:
        dky = float("nan")

    return dict(nombre=sis.nombre, etiqueta=sis.etiqueta, lam=lam, hist=hist,
                t=tiempo, dky=dky, err_jac=err, tau=sis.tau, K=sis.K)


# %% Rossler (control)
def sistema_rossler(a=0.2, b=0.2, c=5.7):
    """Rossler. Referencia: lambda = (0,0714 ; 0 ; -5,39)."""
    def f(t, u):
        x, y, z = u
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    def J(t, u):
        x, y, z = u
        return np.array([[0.0, -1.0, -1.0],
                         [1.0, a, 0.0],
                         [z, 0.0, x - c]])

    return Sistema(nombre="rossler", etiqueta="Rössler",
                   f=f, J=J, u0=np.array([1.0, 1.0, 0.0]),
                   log=np.array([False, False, False]),
                   tau=1.0, K=4000, t_trans=500.0)


# %% Roman y Bertolotti en red
PAR_ROMAN = dict(beta=2.0, alpha=0.05, gamma=0.05, lam=100.0, c=0.05, eps=0.01)


def delta_critico_roman(p=PAR_ROMAN):
    """delta_c de la bifurcación de Hopf supercrítica."""
    b, al, c, g, lam = p["beta"], p["alpha"], p["c"], p["gamma"], p["lam"]
    S = 2 * c * al + (c + 2 * al) ** 2
    return S / (4 * lam * b) * (1 + np.sqrt(1 + 8 * g * c * al * (c + 2 * al) / S ** 2))


def sistema_roman(mult_delta=10.0, sigma=0.0, N=3, p=PAR_ROMAN,
                  nombre=None, etiqueta=None, K=600, t_trans=5.0e4):
    """Roman y Bertolotti en red, en las 3N variables (x_i, y_i, w_i).

    A = 11^T/N + nu I  (acoplamiento extractivo),  B = sigma * L  (migración).
    Condición inicial igualitaria, como en el trabajo original: las tres
    sociedades parten del mismo estado.
    """
    beta, alpha, gamma = p["beta"], p["alpha"], p["gamma"]
    lam, c, eps = p["lam"], p["c"], p["eps"]
    nu = 1e-2 / N
    delta = mult_delta * delta_critico_roman(p)
    A = np.ones((N, N)) / N + nu * np.eye(N)
    L = N * np.eye(N) - np.ones((N, N))
    B = sigma * L
    n = 3 * N
    idx = np.arange(N)

    def f(t, u):
        x, y, w = u[:N], u[N:2 * N], u[2 * N:]
        Ax = A @ x
        h = 1.0 - np.exp(-w / eps)
        ext = delta * x * y * Ax
        dx = (beta * w / (x + eps) - alpha * x) * h - B @ x
        dy = gamma * y * (1.0 - y / lam) - ext
        dw = ext - c * w
        return np.concatenate([dx, dy, dw])

    def J(t, u):
        x, y, w = u[:N], u[N:2 * N], u[2 * N:]
        Ax = A @ x
        E = np.exp(-w / eps)
        h = 1.0 - E
        M = np.zeros((n, n))
        # bloque de la capacidad
        M[idx, idx] = (-beta * w / (x + eps) ** 2 - alpha) * h
        M[idx, 2 * N + idx] = (beta / (x + eps) * h
                               + (beta * w / (x + eps) - alpha * x) * E / eps)
        M[:N, :N] -= B
        # derivadas del término extractivo
        dext_dx = delta * (np.diag(y * Ax) + (x * y)[:, None] * A)
        dext_dy = delta * x * Ax
        M[N:2 * N, :N] = -dext_dx
        M[N + idx, N + idx] = gamma * (1.0 - 2.0 * y / lam) - dext_dy
        M[2 * N:, :N] = dext_dx
        M[2 * N + idx, N + idx] = dext_dy
        M[2 * N + idx, 2 * N + idx] = -c
        return M

    def diagnostico(u):
        """Apartamiento del subespacio homogéneo (ver nota al final)."""
        x = u[:N]
        return (f"apartamiento del subespacio homogéneo: "
                f"max|x_i - <x>|/<x> = {np.abs(x - x.mean()).max() / x.mean():.2e}")

    u0 = np.concatenate([np.ones(N), lam * np.ones(N), np.ones(N)])
    return Sistema(nombre=nombre or f"roman_d{mult_delta:g}_s{sigma:g}",
                   etiqueta=etiqueta or rf"Roman, $\delta={mult_delta:g}\delta_c$, $\sigma={sigma:g}$",
                   f=f, J=J, u0=u0, log=np.ones(n, bool),
                   tau=100.0, K=K, t_trans=t_trans, diagnostico=diagnostico)


# %%  HANDY igualitaria extendido a N sociedades
PAR_HANDY = dict(alpha_m=1e-2, alpha_M=7e-2, beta_C=3e-2,
                 s=5e-4, rho=5e-3, gamma=1e-2, lam=100.0)


def _handy_campos(p, N, A, delta, gamma_t):
    """Campo y jacobiano de HANDY igualitaria en red, variables (x_C, y, w).

    El consumo y la mortalidad son funciones a trozos: con w_th = rho*x_C,

        w >= w_th :  alpha_C = alpha_m,            C_C = s x_C
        w <  w_th :  alpha_C = alpha_m + (1-w/w_th)(alpha_M-alpha_m),
                     C_C = s w / rho

    En la rama de escasez el consumo no depende de x_C, de modo que ambas
    ramas son lineales en w y el jacobiano es exacto salvo en el pliegue
    w = w_th, que es un conjunto de medida nula.
    """
    am, aM, bC = p["alpha_m"], p["alpha_M"], p["beta_C"]
    s, rho, lam = p["s"], p["rho"], p["lam"]
    da = aM - am
    n = 3 * N
    idx = np.arange(N)

    def f(t, u):
        x, y, w = u[:N], u[N:2 * N], u[2 * N:]
        P = A @ x
        ext = delta * y * P
        escasez = w < rho * x
        aC = np.where(escasez, am + (1.0 - w / (rho * x)) * da, am)
        CC = np.where(escasez, s * w / rho, s * x)
        dx = (bC - aC) * x
        dy = gamma_t(t) * y * (lam - y) - ext
        dw = ext - CC
        return np.concatenate([dx, dy, dw])

    def J(t, u):
        x, y, w = u[:N], u[N:2 * N], u[2 * N:]
        P = A @ x
        escasez = w < rho * x
        M = np.zeros((n, n))
        # d(dx)/dx  y  d(dx)/dw
        M[idx, idx] = np.where(escasez, bC - am - da, bC - am)
        M[idx, 2 * N + idx] = np.where(escasez, da / rho, 0.0)
        # d(dy)/dx , d(dy)/dy
        M[N:2 * N, :N] = -delta * y[:, None] * A
        M[N + idx, N + idx] = gamma_t(t) * (lam - 2.0 * y) - delta * P
        # d(dw)/dx , d(dw)/dy , d(dw)/dw
        M[2 * N:, :N] = delta * y[:, None] * A
        M[2 * N + idx, 2 * N + idx] = np.where(escasez, -s / rho, 0.0)
        M[2 * N + idx, idx] += np.where(escasez, 0.0, -s)
        M[2 * N + idx, N + idx] = delta * P
        return M

    return f, J


def sistema_handy_red(delta=3.3e-5, N=3, p=PAR_HANDY, K=600, t_trans=6.0e4):
    """HANDY igualitaria extendido a N sociedades, con la matriz A de Roman."""
    nu = 1e-2 / N
    A = np.ones((N, N)) / N + nu * np.eye(N)
    f, J = _handy_campos(p, N, A, delta, gamma_t=lambda t: p["gamma"])
    n = 3 * N
    # (ln x_C, ln y, w): la riqueza se integra lineal, como en el barrido
    mask = np.array([True] * N + [True] * N + [False] * N)
    u0 = np.concatenate([100.0 * np.ones(N), p["lam"] * np.ones(N), np.zeros(N)])

    def diagnostico(u):
        x = u[:N]
        return (f"apartamiento del subespacio homogéneo: "
                f"max|x_i - <x>|/<x> = {np.abs(x - x.mean()).max() / x.mean():.2e}")

    return Sistema(nombre="handy_red",
                   etiqueta=rf"HANDY en red, $\delta={delta:.2e}$",
                   f=f, J=J, u0=u0, log=mask,
                   tau=100.0, K=K, t_trans=t_trans, diagnostico=diagnostico)


# %% HANDY igualitaria con forzado climático
def sistema_handy_forzado(delta=2.40e-5, p=PAR_HANDY,
                          gamma0=1e-2, Ae=0.3, Te=1.0, Ai=0.7, Ti=75.0,
                          K=2000, t_trans=2.0e4):
    """HANDY igualitaria (una sociedad) con gamma(t) de dos escalas.

    Sistema NO AUTÓNOMO: el motor lleva el tiempo absoluto a través de los
    bloques de reortonormalización, que es imprescindible para que el forzado
    no se reinicie en cada bloque.

    Los exponentes son los del flujo variacional de dimensión 3; la fase del
    forzado entra como parámetro y no aporta su exponente nulo.
    """
    N = 1
    A = np.ones((1, 1))

    def gamma_t(t):
        return gamma0 * (1.0 + Ae * np.sin(2 * np.pi * t / Te)
                         + Ai * np.sin(2 * np.pi * t / Ti))

    f, J = _handy_campos(p, N, A, delta, gamma_t)
    u0 = np.array([100.0, p["lam"], 1e-12])
    return Sistema(nombre="handy_forzado",
                   etiqueta=rf"HANDY forzado, $\delta={delta:.2e}$",
                   f=f, J=J, u0=u0, log=np.ones(3, bool),
                   tau=25.0, K=K, t_trans=t_trans, autonomo=False)


# %% 
def catalogo():
    dc = delta_critico_roman()
    return {
        "rossler": sistema_rossler,
        "roman_ciclo": lambda: sistema_roman(
            1.5, 0.0, nombre="roman_ciclo",
            etiqueta=r"Roman, $\delta=1{,}5\,\delta_c$ (ciclo límite)"),
        "roman_red": lambda: sistema_roman(
            10.0, 0.0, nombre="roman_red",
            etiqueta=r"Roman en red, $\delta=10\,\delta_c$"),
        "roman_migracion": lambda: sistema_roman(
            10.0, 0.0315, nombre="roman_migracion", t_trans=8.0e4,
            etiqueta=r"Roman con migración, $\sigma=0{,}0315$"),
        "handy_red": sistema_handy_red,
        "handy_forzado": sistema_handy_forzado,
    }


REFERENCIA = {"rossler": np.array([0.0714, 0.0, -5.39])}


# %% corrida
def correr(nombres, hacer_figura=True):
    cat = catalogo()
    if nombres == ["todos"]:
        nombres = list(cat)

    resultados = []
    for nom in nombres:
        if nom not in cat:
            print(f"caso desconocido: {nom}   (opciones: {', '.join(cat)})")
            continue
        sis = cat[nom]()
        print(f"\n=== {sis.nombre} : {sis.K} bloques de tau={sis.tau:g} "
              f"-> {sis.K * sis.tau:g} años ===")
        r = espectro_lyapunov(sis)
        lam = r["lam"]
        print(f"  espectro (x1e-3): {np.round(lam * 1e3, 3)}")
        print(f"  suma = {lam.sum():.3e}    D_KY = {r['dky']:.3f}")
        # Un exponente cuenta como positivo sólo si supera el ruido de
        # convergencia, que escala con el exponente más negativo del espectro.
        tol = max(1e-8, 1e-3 * np.abs(lam).max())
        pos = lam[lam > tol]
        if len(pos):
            print(f"  exponentes positivos: {len(pos)}"
                  f"   ->  1/lambda_1 = {1.0 / lam[0]:.0f} años"
                  + ("   (HIPERCAOS)" if len(pos) > 1 else ""))
        else:
            print("  ningún exponente positivo (ciclo límite o punto fijo)")
        if nom in REFERENCIA:
            print(f"  referencia:       {np.round(REFERENCIA[nom], 4)}")
            print(f"  obtenido:         {np.round(lam, 4)}")
        np.savez(f"lyap_{sis.nombre}.npz", lam=lam, hist=r["hist"],
                 t=r["t"], dky=r["dky"])
        resultados.append(r)

    if hacer_figura and resultados:
        figura_convergencia(resultados)
    return resultados


def figura_convergencia(resultados, nombre="lyapunov_convergencia"):
    """Convergencia de la estimación acumulada de lambda_1 (y lambda_2)."""
    m = len(resultados)
    fig, axes = plt.subplots(1, m, figsize=(5.0 * m, 4.2), squeeze=False)
    letras = "abcdefgh"
    for j, (ax, r) in enumerate(zip(axes[0], resultados)):
        hist, t = r["hist"], r["t"]
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax.plot(t, hist[:, 0], color="tab:blue", lw=1.5, label=r"$\lambda_1$")
        if hist.shape[1] > 1:
            ax.plot(t, hist[:, 1], color="tab:orange", lw=1.5, label=r"$\lambda_2$")
        ax.axhline(r["lam"][0], color="tab:blue", ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("Tiempo de integración [años]", fontsize=14)
        if j == 0:
            ax.set_ylabel(r"$\lambda$ acumulado [año$^{-1}$]", fontsize=14)
        ax.text(0.03, 0.94, f"({letras[j]})", transform=ax.transAxes,
                fontsize=15, va="top")
        ax.text(0.97, 0.06, r["etiqueta"], transform=ax.transAxes,
                fontsize=11, ha="right")
        ax.legend(loc="center right", fontsize=11, frameon=False)
    plt.tight_layout()
    plt.savefig(f"{nombre}.png", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(f"{nombre}.pdf", bbox_inches="tight", pad_inches=0.2)
    print(f"\nfigura guardada: {nombre}.png")
