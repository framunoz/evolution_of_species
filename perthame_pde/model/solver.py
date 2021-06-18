# !/usr/bin/python3
# -*-coding utf-8 -*-
# @Time     : 2021/06/11 10:59
# @Author   : Francisco Muñoz
# @Project  : evolution_of_species
# @File     : solver.py
# @Software : PyCharm
import sys
import time
import warnings
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp2d

from perthame_pde.model.discrete_function import AbstractDiscreteFunction
from perthame_pde.model.integral_functional import FunctionalF, FunctionalG
from perthame_pde.utils.validators import validate_nth_row, Float, \
    DiscreteFunctionValidator, InitialDiscreteFunctionValidator


def solve_perthame(u_0, R_0, r, R_in, m_1, m_2, K, eps=0, solver_u=None, solver_R=None, verbose=False, reports_every=1,
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
    # Building discrete functions as arrays
    if verbose:
        print("Creating arrays...")
    x = np.linspace(*x_lims, N + 2)
    y = np.linspace(*y_lims, M + 2)
    u_0 = np.array([u_0(x_) for x_ in x])
    R_0 = np.array([R_0(y_) for y_ in y])
    r = np.array([r(x_) for x_ in x])
    R_in = np.array([R_in(y_) for y_ in y])
    m_1 = np.array([m_1(x_) for x_ in x])
    m_2 = np.array([m_2(y_) for y_ in y])
    K = np.array([[K(x_, y_) for y_ in y] for x_ in x])

    # Building dictionary with discretization settings
    disc_configs = {
        "x_lims": x_lims, "y_lims": y_lims,
        "N": N, "M": M,
        "dt": dt, "T": T
    }

    # Define T as default, if None was given
    T = 100 if T is None else T

    # Building solvers
    if verbose:
        print("Building solvers...")
    solver_u = solver_u if solver_u is not None else Solver1U
    solver_R = solver_R if solver_R is not None else Solver1R
    R = solver_R(m_2, R_in, r, K, u_0, R_0, **disc_configs)
    u = solver_u(m_1, eps, r, K, u_0, R_0, **disc_configs)
    # Start iterations
    if verbose:
        print("Starting iterations")
    time_iter = 0
    for n in range(T):
        start_iter = time.time()

        u.actualize_row(R[n], n)
        u.actualize_step_np1(n)
        R.actualize_row(u[n + 1], n + 1)
        R.actualize_step_np1(n)

        time_iter += time.time() - start_iter

        if verbose and n % reports_every == 0:
            sys.stdout.write(f"\rIteration n = {n}, "
                             + "Total Time: {0:.3f}[seg], ".format(time_iter)
                             + "Time/Iter: {0:.3f}[seg]".format(time_iter / (n + 1)))
    if verbose:
        print()
    return u, R


class AbstractSolver(AbstractDiscreteFunction, ABC):
    """
    Abstract class for general solvers.
    """
    r = DiscreteFunctionValidator("x")
    K = DiscreteFunctionValidator("x", "y")
    u = InitialDiscreteFunctionValidator("t", "x")
    R = InitialDiscreteFunctionValidator("t", "y")

    def __init__(self, r, K, u_0, R_0, **kwargs):
        AbstractDiscreteFunction.__init__(self, **kwargs)
        # Save r and K
        self.r = r
        self.K = K
        # Create functions with initial data
        self.u = u_0
        self.R = R_0

    @abstractmethod
    def actualize_row(self, row: np.ndarray, n: int):
        """
        Actualize the needed row

        Parameters
        ----------
        row : np.ndarray
            row to actualize
        n : int
            number of the row
        """
        pass

    @abstractmethod
    def actualize_step_np1(self, n: int) -> np.ndarray:
        """
        Actualize the solver to the next step.

        Parameters
        ----------
        n : int
            current step.
        """
        pass

    def calculate_total_mass(self) -> np.ndarray:
        """
        Calculates the total mass.
        """
        self._warning_row_not_calculated()
        return self._calculate_total_mass()

    def function_interpolated(self):
        """
        Returns the current function interpolated on the mesh
        """
        self._warning_row_not_calculated()
        return self._function_interpolated()

    def _warning_row_not_calculated(self):
        for n in range(len(self._mask)):
            if not self._is_row_calculated(n):
                warnings.warn(f"There is a row that has not yet been calculated. n = {n}", Warning, 2)

    @abstractmethod
    def _calculate_total_mass(self) -> np.ndarray:
        """
        To use template pattern.
        """
        pass

    @abstractmethod
    def _function_interpolated(self):
        """
        To use template pattern.
        """
        pass


class AbstractSolverU(AbstractSolver, ABC):
    """
    Abstract class for function solvers of u.
    """
    eps = Float(lower_bound=(0, True))
    m_1 = DiscreteFunctionValidator("x")

    def __init__(self, m_1, eps, r, K, u_0, R_0, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        m_1 : function or array[N+2,]
            Mortality of consumer species.
        eps : float
            Mutation rate.
        r : function or array[N+2,]
            Trait dependent growth rate.
        K : function or array[N+2, M+2]
            Consumption rate of resource y by individuals of trait x.
        u_0 : function or array[N+2,]
            Initial data of the Consumer species distribution.
        R_0 : function or array[M+2,]
            Initial data of the Resource distribution.

        Other Parameters
        ----------------
        x_lims : tuple[int, int]
            Limits of the interval of x. Value by default is (0, 1).
        y_lims : tuple[int, int]
            Limits of the interval of y. Value by default is x_lims.
        dt : float
            Discretization of the time. Value by default is 0.01
        N : int
            Number of points of the mesh x. Value by default is 100.
        M : int
            Number of points of the mesh y. Value by default is N.
        T : int
            Number of points of the mesh t. Value by default is 100.
        """
        AbstractSolver.__init__(self, r, K, u_0, R_0, **kwargs)
        # Save m_1
        self.m_1 = m_1
        self.eps = eps
        # Generates an instance of F
        self._F = FunctionalF(K, R_0, **kwargs)
        # Matrix that represents the current function
        self._matrix = self._u
        self._mask = self._create_mask()
        self._update_mask_row(0, True)

    def actualize_row(self, row: np.ndarray, n: int):
        validate_nth_row(row, self._R[n])
        self._R[n] = row
        # Update the row of F
        self._F.actualize_row(row, n)
        # Update the mask
        self._update_mask_row(n + 1, False)
        return self

    def _calculate_total_mass(self) -> np.ndarray:
        return simpson(self._matrix, dx=self.h1, axis=1)

    def _function_interpolated(self):
        return interp2d(self.t, self.x, self.u.T)


class AbstractSolverR(AbstractSolver, ABC):
    """
    Abstract class for function solvers of R.
    """
    m_2 = DiscreteFunctionValidator("y")
    R_in = DiscreteFunctionValidator("y")

    def __init__(self, m_2, R_in, r, K, u_0, R_0, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        m_2 : function or array[M+2,]
            Decay rate of resource.
        R_in : function or array[M+2,]
            Trait dependent resource-supply rate.
        r : function or array[N+2,]
            Trait dependent growth rate.
        K : function or array[N+2, M+2]
            Consumption rate of resource y by individuals of trait x.
        u_0 : function or array[N+2,]
            Initial data of the Consumer species distribution.
        R_0 : function or array[M+2,]
            Initial data of the Resource distribution.

        Other Parameters
        ----------------
        x_lims : tuple[int, int]
            Limits of the interval of x. Value by default is (0, 1).
        y_lims : tuple[int, int]
            Limits of the interval of y. Value by default is x_lims.
        dt : float
            Discretization of the time. Value by default is 0.01
        N : int
            Number of points of the mesh x. Value by default is 100.
        M : int
            Number of points of the mesh y. Value by default is N.
        T : int
            Number of points of the mesh t. Value by default is 100.
        """
        AbstractSolver.__init__(self, r, K, u_0, R_0, **kwargs)
        # Save m_2 and R_in
        self.m_2 = m_2
        self.R_in = R_in
        self._G = FunctionalG(r, K, u_0, **kwargs)
        # Matrix that represents the current function
        self._matrix = self._R
        self._mask = self._create_mask()
        self._update_mask_row(0, True)

    def actualize_row(self, row: np.ndarray, n: int):
        validate_nth_row(row, self._u[n])
        self._u[n] = row
        # Update the row of G
        self._G.actualize_row(row, n)
        # Update the mask
        self._update_mask_row(n + 1, False)
        return self

    def _calculate_total_mass(self) -> np.ndarray:
        return simpson(self._matrix, dx=self.h2, axis=1)

    def _function_interpolated(self):
        return interp2d(self.t, self.y, self.R.T)


class Solver1U(AbstractSolverU):
    """
    Solver that uses the first method for u.
    """

    def _create_transition_matrix(self, n: int) -> np.ndarray:
        # Calculates upper and lower diagonals
        alpha = self.eps * self.dt / self.h1 ** 2
        alpha_array = np.ones((self.N + 1,)) * alpha
        upper_matrix = np.diag(alpha_array, 1)
        lower_matrix = np.diag(alpha_array, -1)
        # Replace the Neumann conditions
        upper_matrix[0, 1] = lower_matrix[-1, -2] = 2 * alpha
        # Calculates the central diagonal: beta
        A_n = -self.m_1 + self.r * self._F[n]
        beta_array = 1 - 2 * alpha + self.dt * A_n
        central_matrix = np.diag(beta_array, 0)
        return lower_matrix + central_matrix + upper_matrix

    def actualize_step_np1(self, n: int) -> np.ndarray:
        if not self._is_row_calculated(n + 1):
            self._u[n + 1] = self._create_transition_matrix(n).dot(self._u[n])
            self._update_mask_row(n + 1, True)
        return self._u[n + 1]


class Solver2U(AbstractSolverU):
    """
    Solver that uses the first method for u. theta-implicit scheme
    """
    theta = Float(lower_bound=(0, True), upper_bound=(1, True))

    def __init__(self, m_1, eps, r, K, u_0, R_0, theta=0.5, **kwargs):
        AbstractSolverU.__init__(self, m_1, eps, r, K, u_0, R_0, **kwargs)
        self.theta = theta

    def actualize_step_np1(self, n: int) -> np.ndarray:
        raise NotImplementedError("Implementar!")

    @classmethod
    def set_theta(cls, theta: float):
        """
        Factory method that sets the theta value before init the class.

        Parameters
        ----------
        theta: float
            Value of the theta-implicit scheme.
        """
        return partial(cls, theta=theta)


class Solver1R(AbstractSolverR):
    """
    Solver that uses the first equation proposed in the paper.
    """

    def actualize_step_np1(self, n: int) -> np.ndarray:
        if not self._is_row_calculated(n + 1):
            self._R[n + 1] = (1 - self.dt * (self.m_2 + self._G[n])) * self.R[n] + self.dt * self.R_in
            self._update_mask_row(n + 1, True)
        return self._R[n + 1]


class Solver2R(AbstractSolverR):
    """
    Solver that uses the second equation proposed in the paper.
    """

    def __init__(self, m_2, R_in, r, K, u_0, R_0, **kwargs):
        AbstractSolverR.__init__(self, m_2, R_in, r, K, u_0, R_0, **kwargs)
        # Update the quasi-static equation (does not matters the inicial data)
        self._R[0] = self.R_in / (self.m_2 + self._G[0])

    def actualize_step_np1(self, n: int) -> np.ndarray:
        if not self._is_row_calculated(n + 1):
            G_np1 = self._G[n + 1]
            R_np1 = self.R_in / (self.m_2 + G_np1)
            self._R[n + 1] = R_np1
            self._update_mask_row(n + 1, True)
        return self.R[n + 1]


class Solver3R(AbstractSolverR):
    """
    Solver that uses the first equation proposed in the paper with a second order method.
    """

    def actualize_step_np1(self, n: int) -> np.ndarray:
        if not self._is_row_calculated(n + 1):
            A_n = self.dt * (self.m_2 + self._G[n])
            B_n = self.dt * self.R_in
            if n == 0:
                self._R[n + 1] = (1 - A_n) * self.R[n] + B_n
            else:
                self._R[n + 1] = self.R[n - 1] - 2 * A_n * self.R[n] + 2 * B_n
            self._update_mask_row(n + 1, True)
        return self._R[n + 1]
