# !/usr/bin/python3
# -*-coding utf-8 -*-
# @Time     : 2021/06/11 10:59
# @Author   : Francisco Muñoz
# @Project  : evolution_of_species
# @File     : discrete_function.py
# @Software : PyCharm

from abc import ABC
from typing import Tuple, Union

import numpy as np

from perthame_pde.utils.validators import Boundary, Float, Integer, validate_index, DiscreteFunctionValidator

BoundaryType = Tuple[int, int]


class AbstractDiscreteFunction(ABC):
    """
    Abstract class that represents a function, but that discretizes it as if it were an array, so that accessing a
    component of that array is equivalent to evaluating that component as an argument in the function.
    """
    x_lims = Boundary()
    y_lims = Boundary()
    dt = Float(lower_bound=(0, False))
    N = Integer()
    M = Integer()
    T = Integer()
    x = DiscreteFunctionValidator("x")
    y = DiscreteFunctionValidator("y")
    h1 = Float(lower_bound=(0, False))
    h2 = Float(lower_bound=(0, False))

    def __init__(self,
                 x_lims: BoundaryType = (0, 1), y_lims: BoundaryType = None, dt: float = 0.01,
                 N: int = 100, M: int = None, T: int = 100):
        """
        Constructor.

        Parameters
        ----------
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
        self.x_lims = x_lims if x_lims is not None else (0, 1)
        self.y_lims = y_lims if y_lims is not None else self.x_lims
        self.dt = dt if dt is not None else 0.01
        self.N = N if N is not None else 100
        self.M = M if M is not None else self.N
        self.T = T if T is not None else 100
        # Construct mesh
        self.x = np.linspace(*self.x_lims, self.N + 2)
        self.y = np.linspace(*self.y_lims, self.M + 2)
        self.t = np.linspace(0, self.dt * self.T, self.T + 1)
        # Determine steps
        self.h1 = 1 / (self.N + 1)
        self.h2 = 1 / (self.M + 1)
        # Matrix unsigned
        self._matrix = None
        # Mask for cache
        self._mask = None

    def __repr__(self, space=2):
        tab = " " * space
        matrix_str = tab + str(self.matrix).replace('\n', '\n' + tab)
        return (self.__class__.__name__
                + f"(\n"
                + tab + f"N={self.N}, M={self.M}, T={self.T},\n"
                + tab + f"x_lims={self.x_lims}, y_lims={self.y_lims}, T_max={self.T * self.dt:.2f}\n"
                + tab + f"shape={self.matrix.shape}, matrix=\n{matrix_str}\n"
                + f")")

    def __getitem__(self, item: Union[int, Tuple[int, int]]):
        validate_index(item)
        if isinstance(item, int):
            # Check if the values are already calculated
            if not self._mask[item].all():
                self._calculate_row(item)
            return self.matrix[item]
        if isinstance(item, tuple):
            return self._calculate_component(*item)

    def _calculate_component(self, i: int, j: int):
        """
        Calculates and update a specific component

        Parameters
        ----------
        i: int
            i-th row
        j: int
            j-th column
        """
        return self.matrix[i, j]

    def _calculate_row(self, i: int):
        """
        Calculates and update a specific row
        Parameters
        ----------
        i: int
            i-th row
        """
        for j in range(self.matrix.shape[1]):
            self._calculate_component(i, j)

    def _update_mask_row(self, n: int, bool_to_update: bool = False) -> None:
        """
        Cleans the n-th row of the mask.

        Parameters
        ----------
        n : int
            A row of the mask.
        """
        if n < len(self._mask):
            self._mask[n] = bool_to_update

    def _create_mask(self) -> np.ndarray:
        """
        Creates a mask.

        Returns
        -------
        np.ndarray
            A mask with the same shape of the matrix representation
        """
        return np.zeros_like(self.matrix, dtype=bool)

    def _is_component_calculated(self, i: int, j: int) -> bool:
        """
        Verify if the (i, j)-th component was calculated.

        Parameters
        ----------
        i: int
            i-th row
        j: int
            j-th column
        """
        if i < len(self._mask) and j < len(self._mask[i]):
            return self._mask[i, j]
        else:
            return False

    def _is_row_calculated(self, i: int) -> bool:
        """
        Verify if the i-th row was calculated.

        Parameters
        ----------
        i: int
            i-th row
        """
        if i < len(self._mask):
            return self._mask[i].all()
        else:
            return False

    @property
    def matrix(self) -> np.ndarray:
        return np.copy(self._matrix)
