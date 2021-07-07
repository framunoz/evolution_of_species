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

from perthame_pde.model.mesh import TMesh, XMesh, YMesh, ZMesh
from perthame_pde.utils.validators import validate_index

BoundaryType = Tuple[int, int]


class AbstractDiscreteFunction(ABC):
    """
    Abstract class that represents a function, but that discretizes it as if it were an array, so that accessing a
    component of that array is equivalent to evaluating that component as an argument in the function.
    """

    def __init__(self,
                 x_lims: BoundaryType = None, y_lims: BoundaryType = None, z_lims: BoundaryType = None,
                 N: int = None, M: int = None, O: int = None,
                 dt: float = None, T: int = None):
        """
        Constructor.

        Parameters
        ----------
        x_lims : tuple[int, int]
            Limits of the interval of x. Value by default is (0, 1).
        y_lims : tuple[int, int]
            Limits of the interval of y. Value by default is (0, 1).
        z_lims : tuple[int, int]
            Limits of the interval of z. Value by default is (0, 1).
        N : int
            Number of points of the mesh x. Value by default is 100.
        M : int
            Number of points of the mesh y. Value by default is 100.
        O : int
            Number of points of the mesh z. Value by default is 100.
        dt : float
            Discretization of the time. Value by default is 0.01
        T : int
            Number of points of the mesh t. Value by default is 100.
        """
        self.t = TMesh(dt, T)
        self.x = XMesh(x_lims, N)
        self.y = YMesh(y_lims, M)
        self.z = ZMesh(z_lims, O)
        # Matrix unsigned
        self._matrix = None

    def __repr__(self, space=2):
        tab = " " * space
        matrix_str = tab + str(self.matrix).replace('\n', '\n' + tab)
        return (self.__class__.__name__
                + f"(\n"
                + tab + f"N={self.x.N}, M={self.y.M}, T={self.t.T},\n"
                + tab + f"x_lims={self.x.lims}, y_lims={self.y.lims}, T_max={self.t.T * self.t.dt:.2f}\n"
                + tab + f"shape={self.matrix.shape}, matrix=\n{matrix_str}\n"
                + f")")

    def __getitem__(self, item: Union[int, Tuple[int, int]]):
        validate_index(item)
        return self.matrix[item]

    @property
    def matrix(self) -> np.ndarray:
        return np.copy(self._matrix)
