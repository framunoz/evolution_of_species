# !/usr/bin/python3
# -*-coding utf-8 -*-
# @Time     : 2021/06/17 11:13
# @Author   : Francisco Muñoz
# @Project  : evolution_of_species
# @File     : plotters.py
# @Software : PyCharm

from copy import deepcopy
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from matplotlib.collections import LineCollection

from perthame_pde.model.solver import AbstractSolverU, AbstractSolverR

_FIGSIZE = (10, 5)


def plot_phase_plane(u: AbstractSolverU, R: AbstractSolverR,
                     fig_config: dict = None, ax_config: dict = None, eps=0.01):
    x = R.calculate_total_mass()
    y = u.calculate_total_mass()
    t = R.t.mesh
    x_delay = eps * max(abs(x.min()), abs(x.max()))
    y_delay = eps * max(abs(y.min()), abs(y.max()))

    if ax_config is None:
        ax_config = {
            "title": "Trayectorias en el plano de fase (R, u)",
            "xlabel": "Masa total de los recursos",
            "ylabel": "Masa total de la población",
            "xlim": (x.min() - x_delay, x.max() + x_delay),
            "ylim": (y.min() - y_delay, y.max() + y_delay),
        }
    if fig_config is None:
        fig_config = {
            "figsize": _FIGSIZE
        }

    fig = plt.figure(**fig_config)
    ax = fig.add_subplot(111, **ax_config)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap="plasma", norm=norm)
    lc.set_array(t)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    cbar.ax.set_ylabel("Tiempo", rotation=270)

    plt.show()


def plot_total_mass(u: AbstractSolverU, R: AbstractSolverR,
                    fmt: list = None, label: list = None,
                    fig_config: dict = None, ax_config: dict = None):
    if ax_config is None:
        ax_config = {
            "title": "Biomasa total de los rasgos y recursos",
            "xlabel": "Tiempo",
            "ylabel": "Biomasa"
        }
    if fig_config is None:
        fig_config = {
            "figsize": _FIGSIZE
        }
    fmt = fmt if fmt is not None else ["-", "-."]
    label = label if label is not None else ["rasgo", "recurso"]
    fig = plt.figure(**fig_config)
    ax = fig.add_subplot(111, **ax_config)

    valores_t = u.t.mesh
    valores_u = u.calculate_total_mass()
    valores_R = R.calculate_total_mass()

    plt.plot(valores_t, valores_u, fmt[0], label=label[0])
    plt.plot(valores_t, valores_R, fmt[1], label=label[1])

    plt.legend(ncol=2, loc="best", prop={'size': 12})


def plot_anim(u: AbstractSolverU, R: AbstractSolverR,
              speed: float = 1.0, t_min: int = None, t_max: int = None,
              fmt: list = None, label: list = None, T0: float = None,
              fig_config: dict = None, ax_config: dict = None, fig_label: dict = None):
    if fig_label is None:
        fig_label = {
            "title": "Gráfico de la distribución de rasgos y recursos",
            "xlabel": "rasgo/recurso",
            "ylabel": "densidad"
        }
    if fig_config is None:
        fig_config = {
            "figsize": _FIGSIZE
        }
    if ax_config is None:
        ax_config = {}
    t_min = t_min if t_min is not None else 0
    t_max = t_max if t_max is not None else u.t.T
    fmt = fmt if fmt is not None else ["-", "-."]
    label = label if label is not None else ["rasgo", "recurso"]
    T0 = T0 if T0 is not None else u.t.dt * t_min

    anim = Animar(
        x=[u.x.mesh, R.y.mesh],
        NT=t_max - t_min,
        dt=u.t.dt,
        funcs=[u.matrix[t_min:t_max + 1], R.matrix[t_min:t_max + 1]],
        fmt=fmt,
        label=label,
        T0=T0
    )
    anim.crear_fig(**fig_config)
    anim.crear_ax(**ax_config)
    anim.animar(speed)
    anim.rotular(**fig_label)

    return anim.mostrar_en_notebook()


# TODO: Ver si hay validadores que se pueden pasar a utils.validators
def validar_tipo_funcs(funcs):
    if not (isinstance(funcs, dict) or isinstance(funcs, list)):
        raise TypeError("'funcs' tiene que ser lista o diccionario")


def validar_args_funcs(funcs):
    for f in funcs:
        if not (isinstance(f, FunctionType) or isinstance(f, np.ndarray)):
            raise TypeError("'funcs' tiene que tener funciones o arreglos")


def validar_funcs(funcs):
    validar_tipo_funcs(funcs)
    validar_args_funcs(funcs)
    return funcs


def validar_label(label, funcs):
    if label is None:
        return [None] * len(funcs)
    if len(label) != len(funcs):
        raise ValueError("'label' y 'funcs' deben tener el mismo largo")
    for lbl in label:
        if not isinstance(lbl, str):
            raise TypeError("'label' debe ser una secuencia de str")
    return label


def validar_fmt(fmt, funcs):
    if fmt is None:
        return [''] * len(funcs)
    for f in fmt:
        if not isinstance(f, str):
            raise TypeError("'fmt' debe ser una lista de str")
    return fmt


def parse_funcs(funcs, x, dt, NT, N):
    for i, f in enumerate(funcs):
        if isinstance(f, FunctionType):
            try:
                funcs[i] = np.array([f(dt * n, x) for n in range(NT + 1)])
            except:
                funcs[i] = np.array([[f(dt * n, x_) for x_ in x] for n in range(NT + 1)])
        if funcs[i].shape != (NT + 1, N + 2):
            raise ValueError(f"El arreglo {i} tiene dimensiones {funcs[i].shape} y debe ser {(NT + 1, N + 2)}")
    return funcs


def crear_label_funcs(label, funcs, x, dt, NT, N):
    if isinstance(funcs, dict):
        funcs_ = validar_funcs(list(funcs.values()))
        label, funcs = validar_label(list(funcs.keys()), funcs_), funcs_
    else:
        label, funcs = validar_label(label, funcs), validar_funcs(funcs)
    funcs = parse_funcs(funcs, x, dt, NT, N)
    return label, funcs


# TODO: Refactor esta clase
class Animar:
    ANIMAR = True
    TIEMPO_REAL = True
    MOSTRAR_TIEMPO = True

    def __init__(self, x: list, NT, dt, funcs: list, fmt=None, label=None, prefijo='', T0=0., fps=30):
        """
        :param x: Intervalo del dominio de las funciones. De largo N+2.
        :param NT: Número de sub-intervalos de tiempo.
        :param dt: Pasos de tiempo.
        :param funcs: Lista o diccionario de funciones o matrices de (NT+1)x(N+2).
            Si es diccionario no es necesario agregar labels.
        :param fmt: Formato de las líneas de las funciones.
        :param label: Labels de las líneas de las funciones.
        :param prefijo: Lo que irá antes del path al salvarse, por si se quiere que
            se guarden en carpetas.
        :param fps: Número de Frames por Segundo que se quiere que se guarde.
        """
        # Información básica
        self._x = x

        self._dt = dt
        self._NT = NT
        self._T = NT * dt
        self._T0 = T0

        self._label = label
        self._funcs = funcs

        self._fmt = validar_fmt(fmt, funcs)

        # Path en donde se guardará la imagen
        self._path = (prefijo + '{}.gif').format

        # Información sobre la animación
        self._anim = None
        self._fps = fps
        self._thr = 1 / fps

        # Setear figure y axes
        self._fig = None
        self._ax = None
        self._lines = []

    @property
    def funcs(self):
        if self._existe_label():
            return dict(zip(self._label, deepcopy(self._funcs)))
        else:
            return deepcopy(self._funcs)

    def _existe_label(self):
        """
        Verifica si existe algún label.
        :return:
        """
        return any(e is not None for e in self._label)

    def _hallar_xy_min_max(self):
        """
        Halla el mínimo y máximo de cada eje para que todos los datos de la figura calcen en los ejes.
        :return:
        """
        x_min, x_max = np.min([np.min(arr) for arr in self._x]), np.max([np.max(arr) for arr in self._x])

        y_min, y_max = np.min(self._funcs[0]), np.max(self._funcs[0])
        for i in range(1, len(self._funcs)):
            y_min_, y_max_ = np.min(self._funcs[i]), np.max(self._funcs[i])
            y_min, y_max = min(y_min, y_min_), max(y_max, y_max_)

        return (x_min, x_max), (y_min, y_max)

    def crear_fig(self, **kwargs):
        """
        Crea la figura base. Se le puede entregar el resto de parámetros que necesita la figura.
        :param kwargs:
        :return:
        """
        self._fig = plt.figure(**kwargs)

    def crear_ax(self, **kwargs):
        """
        Crea el eje base. Se le puede entregar el resto de parámetros que necesita la figura.
        :param kwargs:
        :return:
        """
        if self.ANIMAR:
            if self._fig is None:
                self.crear_fig()
            xlim, ylim = self._hallar_xy_min_max()
            kwargs['xlim'], kwargs['ylim'] = kwargs.get('xlim', xlim), kwargs.get('ylim', ylim)
            self._ax = self._fig.add_subplot(111, **kwargs)

    def rotular(self, title=None, xlabel=None, ylabel=None):
        """
        Rotula el eje.
        :param title:
        :param xlabel:
        :param ylabel:
        :return:
        """
        if self._ax is None:
            self.crear_ax()
        self._ax.set_title(title)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

    def _agregar_leyenda(self, **kwargs):
        kwargs['loc'] = kwargs.get('loc', 1)
        if self._existe_label():
            plt.legend(**kwargs)

    def _crear_fig_ax(self):
        """
        Crea una figura y un eje por defecto, si es que aún no se crean.
        :return:
        """
        if self._fig is None:
            self.crear_fig()
        if self._ax is None:
            self.crear_ax()

    def _crear_lines(self):
        """
        Crea tantas lineas como funciones haya y los agrega al eje.
        :return:
        """
        for i in range(len(self._funcs)):
            line, = self._ax.plot([], [], self._fmt[i],
                                  label=self._label[i], lw=2)
            self._lines.append(line)

    def _cargar_lines(self, i):
        """
        Carga la información de las funciones en el tiempo 'i' en las lineas.
        :param i:
        :return:
        """
        for x, func, line in zip(self._x, self._funcs, self._lines):
            line.set_data(x, func[i])

    def _agregar_tiempo(self, i):
        if self.MOSTRAR_TIEMPO:
            self._time.set_text(f'$t=${self._dt * i + self._T0:.2f}')

    def _init(self):
        """
        Función que inicializa la animación.
        :return:
        """
        self._ultimo_j = 0
        for line in self._lines:
            line.set_data([], [])
        if self.MOSTRAR_TIEMPO:
            self._time = self._ax.text(0.02, 0.93, '$t=$0.00', transform=self._ax.transAxes)
        return tuple(self._lines)

    def _animate(self, i):
        """
        Anima el i-ésimo frame.
        :param i:
        :return:
        """
        if self.TIEMPO_REAL:
            for j in range(self._ultimo_j, self._NT + 1):
                if self._dt * j >= self._thr * i or j == self._NT:
                    self._ultimo_j = j
                    self._cargar_lines(j)
                    self._agregar_tiempo(j)
                    break
        else:
            self._cargar_lines(i)
            self._agregar_tiempo(i)

        return tuple(self._lines)

    def _animar(self, speed=1.):
        # Setear el threshold
        self._thr *= speed
        # Crear fig y ax en caso de que aun no se haya hecho
        self._crear_fig_ax()
        self._crear_lines()
        self._agregar_leyenda()
        # Setear el número de frames total
        frames = int(self._T / self._thr) if self.TIEMPO_REAL else self._NT + 1
        self._anim = animation.FuncAnimation(self._fig, self._animate,
                                             init_func=self._init,
                                             frames=frames,
                                             blit=False)

    def animar(self, speed=1.):
        """
        Función para animar. 'speed' es para modificar la velocidad de la animación.
        :param speed:
        :return:
        """
        if self.ANIMAR:
            self._animar(speed)

    def mostrar_en_notebook(self):
        """
        Método para mostrar la animación en el notebook.
        """
        anim_html = HTML(self._anim.to_jshtml())
        plt.close(self._fig)
        return anim_html

    def save(self, name='animacion'):
        """
        Salva la animación en caso de que exista.
        :param name:
        :return:
        """
        if self._anim is None:
            pass
        else:
            self._anim.save(self._path(name), fps=self._fps,
                            extra_args=['-vcodec', 'libx264'])
