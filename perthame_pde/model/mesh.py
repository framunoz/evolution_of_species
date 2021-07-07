# !/usr/bin/python3
# -*-coding utf-8 -*-
# @Time     : 2021/07/07 12:17
# @Author   : Francisco Muñoz
# @Project  : evolution_of_species
# @File     : mesh.py
# @Software : PyCharm
from abc import ABC

import numpy as np


class AbstractMesh(ABC):

    def __init__(self, lims=None, N=None, margen=0):
        self.lims = lims if lims is not None else (0, 1)
        self._N = N if N is not None else 100
        self.mesh = np.linspace(*self.lims, self._N + margen)
        self._h = np.diff(self.mesh)[0]


class TMesh(AbstractMesh):
    def __init__(self, dt=None, T=None):
        dt = dt if dt is not None else 0.001
        T = T if T is not None else 100
        AbstractMesh.__init__(self, (0, T * dt), T, 1)
        self.dt = self._h
        self.T = self._N


class XMesh(AbstractMesh):
    def __init__(self, x_lims=None, N=None):
        AbstractMesh.__init__(self, x_lims, N, 2)
        self.dx = self._h
        self.N = self._N


class YMesh(AbstractMesh):
    def __init__(self, y_lims=None, M=None):
        AbstractMesh.__init__(self, y_lims, M, 2)
        self.dy = self._h
        self.M = self._N


class ZMesh(AbstractMesh):
    def __init__(self, z_lims=None, O=None):
        AbstractMesh.__init__(self, z_lims, O, 2)
        self.dz = self._h
        self.O = self._N
