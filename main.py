# !/usr/bin/python3
# -*-coding utf-8 -*-
# @Time     : 2021/06/30 22:23
# @Author   : Francisco Muñoz
# @Project  : evolution_of_species
# @File     : main.py
# @Software : PyCharm
import numpy as np

from perthame_pde.model.solver import solve_perthame, Solver1R, Solver2U

print(np.min(
    [np.min(arr) for arr in [np.array([1, 2]), np.array([1, 3, 0])]]
))

x_lims = (-4, 4)
y_lims = (-4, 4)
N = 400
M = 100
dt = 0.05
T = 100

CONFIG_LIMITES = {
    "x_lims": x_lims,
    "y_lims": y_lims,
    "N": N,
    "M": M,
    "dt": dt,
    "T": T
}

M_1_CONST, M_2_CONST = 0.3, 1.
EPS = 0.001
SIG_K, SIG_IN = 0.6, 1.
R_CONST = 1.
M_IN = 3.
U_MAX, MU_U, SIG_U = 3., 1., 0.01
R_MAX, MU_R, SIG_R = 5., -1., 1.


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


def m_2(y) -> float:
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


u, R = solve_perthame(
    u_0, R_0, r, R_in, m_1, m_2, K, EPS,
    solver_R=Solver1R, solver_u=Solver2U.set_theta(1),
    verbose=True, reports_every=50,
    **CONFIG_LIMITES
)

# print(u, R)
print(u.F)
