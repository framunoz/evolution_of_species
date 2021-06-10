from abc import ABC, abstractmethod

import numpy as np

from perthame_edp.model.discrete_function import AbstractDiscreteFunction
from perthame_edp.model.integral_functional import FunctionalF, FunctionalG
from perthame_edp.utils.validators import validate_nth_row, Float, \
    DiscreteFunctionValidator, InitialDiscreteFunctionValidator


# TODO: Hacer un método para ver el tiempo de ejecución
# TODO: Hacer un método para calcular la integral de solvers
# TODO: Observar casos críticos: ver si los animales se extinguen


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
        # A counter of the current step
        self._current_step = 0

    def _is_next_step(self, n: int) -> bool:
        """
        Returns True if n is at least the next step. Otherwise, return False.

        Parameters
        ----------
        n : int
            Step to verify
        """
        return n <= self._current_step + 1

    def _update_step(self, n: int):
        """
        Update the current step.

        Parameters
        ----------
        n : int
            Step to set as current
        """
        self._current_step = n + 1

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
    def actualize_step_np1(self, n: int):
        """
        Actualize the solver to the next step.

        Parameters
        ----------
        n : int
            current step.

        Returns
        -------

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
        self._R[n] = np.copy(row)
        # Update the row of F
        self._F.actualize_row(row, n)
        # Update the mask
        self._update_mask_row(n, False)
        return self


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
        self._u[n] = np.copy(row)
        # Update the row of G
        self._G.actualize_row(row, n)
        # Update the mask
        self._update_mask_row(n, False)
        return self


class SolverU(AbstractSolverU):
    """
    Solver that uses the first method for u.
    """

    def actualize_step_np1(self, n: int):
        if not self._mask[n + 1].all():
            # TODO: Mejorar esta parte con álgebra lineal
            alpha = self.eps * self.dt / self.h1 ** 2
            gamma = alpha
            # TODO: Ver si las cond. de borde se puede hacer mejor
            self._u[n + 1, 0], self._u[n + 1, -1] = self._u[n, 0], self._u[n, -1]
            for j in range(1, self.N + 1):
                A_n_j = -self.m_1[j] + self.r[j] * self._F[n, j]
                beta = 1 - 2 * alpha + self.dt * A_n_j
                u_np1_j = (alpha * self._u[n, j - 1]
                           + beta * self._u[n, j]
                           + gamma * self._u[n, j + 1])
                self._u[n + 1, j] = u_np1_j
            self._update_mask_row(n + 1, True)
        return self._u[n + 1]


DICT_SOLVERS_U = {
    "first schema": SolverU
}


class SolverRMethod1(AbstractSolverR):
    """
    Solver that uses the first equation proposed in the paper.
    """

    def actualize_step_np1(self, n: int):
        pass


class SolverRMethod2(AbstractSolverR):
    """
    Solver that uses the second equation proposed in the paper.
    """

    def actualize_step_np1(self, n: int):
        if not self._mask[n + 1].all():
            G_np1 = self._G[n + 1]
            R_np1 = self.R_in / (self.m_2 + G_np1)
            self._R[n + 1] = R_np1
            self._update_mask_row(n + 1, True)
        return self.R[n + 1]


DICT_SOLVERS_R = {
    "method 1": SolverRMethod1,
    "method 2": SolverRMethod2,
}
