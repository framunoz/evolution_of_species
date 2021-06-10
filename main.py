import numpy as np

from perthame_edp.model.integral_functional import FunctionalG
from perthame_edp.model.solver import DICT_SOLVERS_R, DICT_SOLVERS_U


def f1(x, y): return y ** 2


def f2(t, x): return np.ones_like(x)


N, M, T = 100, 110, 120
x_lims, y_lims, dt = (0, 1), (0, 2), 0.01
x = np.linspace(*x_lims, N + 2)
y = np.linspace(*y_lims, M + 2)
t = np.array([dt * n for n in range(T + 1)])
BOUNDARIES_CONFIG = {
    "x_lims": x_lims,
    "y_lims": y_lims,
    "dt": dt,
    "N": N,
    "M": M,
    "T": T,
}
r = np.linspace(-1, 0, N + 2)
F1 = np.array([f1(x_, y) for x_ in x])
F2 = np.array([f2(t_, x) for t_ in t])
# print(transform_to_discrete_function(F1, "K", N+2, x, M+2, y))
G = FunctionalG(r, F1, F2[0], **BOUNDARIES_CONFIG)
G.actualize_row(F2[3], 3)
print(G[3, 4])


def resolver_perthame(u_0, R_0, r, R_in, m_1, m_2, K, eps, solver_u="first schema", solver_R="method 2",
                      x_lims=(0, 1), y_lims=None, N=100, M=None, dt=0.01, T=100):
    """
    u_0(x) cond inicial de u
    R_0(y) cond inicial de R
    r(x) tasa de crecimiento dependiente del trait
    R_in(y) Trait dependent resource-supply rate
    m_1(x) tasa de mortalidad de las especies consumidoras
    m_2(y) Tasa de decaimiento de los recursos
    K(x, y) tasa de consumo del recurso y por los individuos de trait x
    eps tasa de mutación
    """
    # Armar diccionario de configuraciones de la discretización
    disc_configs = {
        "x_lims": x_lims, "y_lims": y_lims,
        "N": N, "M": M,
        "dt": dt, "T": T
    }
    # Redefinir T si es que este fuera None
    T = T or 100
    # Instanciar solvers
    R = DICT_SOLVERS_R.get(solver_R, None)(m_2, R_in, r, K, u_0, R_0, **disc_configs)
    u = DICT_SOLVERS_U.get(solver_u, None)(m_1, eps, r, K, u_0, R_0, **disc_configs)
    # Iterar sobre n
    for n in range(T):
        u.actualize_row(R[n], n)
        u.actualize_step_np1(n)
        R.actualize_row(u[n + 1], n + 1)
        R.actualize_step_np1(n)
    return u, R


x_lims = None  # (-4, 5)
y_lims = (-1, 2)
N = 100  # None  # 100
M = 110
dt = None  # 0.02
T = 120

CONFIG_LIMITES = {
    "x_lims": x_lims,
    "y_lims": y_lims,
    "N": N,
    "M": M,
    "dt": dt,
    "T": T
}

M_1_CONST, M_2_CONST = 0.5, 1.
EPS = 0.001
SIG_K, SIG_IN = 0.6, 1.
R_CONST = 1.
M_IN = 3.
U_MAX, MU_U, SIG_U = 1., 0.25, 1.
R_MAX, MU_R, SIG_R = 5., -0.25, 1.


def K(x, y, sig_K=SIG_K) -> float:
    num = np.exp(- (x - y) ** 2 / (2 * sig_K ** 2))
    den = sig_K * np.sqrt(2 * np.pi)
    return num / den


def R_in(y, M_in=M_IN, sig_in=SIG_IN) -> float:
    num = M_in * np.exp(- y ** 2 / (2 * sig_in ** 2))
    den = sig_in * np.sqrt(2 * np.pi)
    return num / den


def m_1(x) -> float:
    return M_1_CONST


def m_2(x) -> float:
    return M_2_CONST


def r(x) -> float:
    return R_CONST


# Distribuciones Gaussianas
def u_0(x, u_max=U_MAX, mu_u=MU_U, sig_u=SIG_U) -> float:
    num = u_max * np.exp(- (x - mu_u) ** 2 / (2 * sig_u ** 2))
    den = sig_u * np.sqrt(2 * np.pi)
    return num / den


def R_0(y, R_max=R_MAX, mu_R=MU_R, sig_R=SIG_R) -> float:
    num = R_max * np.exp(- (y - mu_R) ** 2 / (2 * sig_R ** 2))
    den = sig_R * np.sqrt(2 * np.pi)
    return num / den


CONFIG_LIMITES["x_lims"] = (0, 1)

u, R = resolver_perthame(u_0, R_0, r, R_in, m_1, m_2, K, EPS, **CONFIG_LIMITES)
print(u)
print(R)

a_matrix = np.zeros((2, 3), dtype=bool)
print(a_matrix)
a_matrix[0] = True
print(a_matrix)
