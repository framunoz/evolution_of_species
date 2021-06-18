# !/usr/bin/python3
# -*-coding utf-8 -*-
# @Time     : 2021/06/11 10:59
# @Author   : Francisco Muñoz
# @Project  : evolution_of_species
# @File     : integral_functional.py
# @Software : PyCharm

from abc import ABC, abstractmethod
from typing import Union, Callable

import numpy as np
from scipy.integrate import simpson

from perthame_pde.model.discrete_function import AbstractDiscreteFunction
from perthame_pde.utils.validators import validate_nth_row, DiscreteFunctionValidator, \
    InitialDiscreteFunctionValidator

OneDimDiscreteFunction = Union[np.ndarray, Callable[[float], float]]
TwoDimDiscreteFunction = Union[np.ndarray, Callable[[float, float], float]]


class AbstractIntegralFunctional(AbstractDiscreteFunction, ABC):
    """
    Abstract class that represents the functions that are like an integral functional.
    """
    K = DiscreteFunctionValidator("x", "y")

    def __init__(self, K: TwoDimDiscreteFunction, **kwargs):
        AbstractDiscreteFunction.__init__(self, **kwargs)
        self.K = K

    @abstractmethod
    def actualize_row(self, row: np.ndarray, n: int) -> None:
        """
        Actualize the needed row.

        Parameters
        ----------
        row : np.ndarray
            row to actualize
        n : int
            number of the row
        """
        pass


class FunctionalF(AbstractIntegralFunctional):
    R = InitialDiscreteFunctionValidator("t", "y")

    def __init__(self, K: TwoDimDiscreteFunction, R_0: OneDimDiscreteFunction, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        K : function or array[N+2, M+2]
            Consumption rate of resource y by individuals of trait x.
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
        AbstractIntegralFunctional.__init__(self, K, **kwargs)
        # Create R function
        self.R = R_0
        # Create the internal representation
        self._matrix = np.zeros((self.T + 1, self.N + 2))
        # Create the mask
        self._mask = self._create_mask()

    def actualize_row(self, row: np.ndarray, n: int):
        validate_nth_row(row, self.R[n])
        self._R[n] = row
        self._update_mask_row(n, False)
        return self

    def _calculate_row(self, n: int):
        if not self._is_row_calculated(n):
            matrix_to_integrate = self.K * self.R[n].reshape(1, -1)
            integral_array = simpson(matrix_to_integrate, x=self.y, axis=1)
            self._matrix[n] = integral_array
            self._update_mask_row(n, True)
        return self.matrix[n]

    def _calculate_component(self, n: int, j: int):
        if not self._is_component_calculated(n, j):
            self._calculate_row(n)
        return self.matrix[n, j]


class FunctionalG(AbstractIntegralFunctional):
    r = DiscreteFunctionValidator("x")
    u = InitialDiscreteFunctionValidator("t", "x")

    def __init__(self, r: OneDimDiscreteFunction, K: TwoDimDiscreteFunction,
                 u_0: OneDimDiscreteFunction, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        r : function or array[N+2,]
            Trait dependent resource-supply rate.
        K : function or array[N+2, M+2]
            Consumption rate of resource y by individuals of trait x.
        u_0 : function or array[N+2,]
            Initial data of the consumer species distribution.

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
        AbstractIntegralFunctional.__init__(self, K, **kwargs)
        self.r = r
        # Create u function
        self.u = u_0
        # Create the internal representation
        self._matrix = np.zeros((self.T + 1, self.M + 2))
        # Create the mask
        self._mask = self._create_mask()

    def actualize_row(self, row: np.ndarray, n: int):
        validate_nth_row(row, self._u[n])
        self._u[n] = row
        self._update_mask_row(n, False)
        return self

    def _calculate_row(self, n: int):
        if not self._is_row_calculated(n):
            matrix_to_integrate = (self.r.reshape(-1, 1) * self.K) * self.u[n].reshape(-1, 1)
            integral_array = simpson(matrix_to_integrate, x=self.x, axis=0)
            self._matrix[n] = integral_array
            self._update_mask_row(n, True)
        return self.matrix[n]

    def _calculate_component(self, n: int, k: int):
        if not self._is_component_calculated(n, k):
            self._calculate_row(n)
        return self.matrix[n, k]