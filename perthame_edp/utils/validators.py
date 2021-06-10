from abc import ABC, abstractmethod

import numpy as np


def validate_index(item):
    """
    Validates if an index represents a row or a component.
    """
    if not isinstance(item, (int, tuple)):
        raise TypeError(f"'item' must be an 'int' or 'tuple'."
                        + f" Currently is '{type(item).__name__}'.")
    if isinstance(item, tuple) and len(item) != 2:
        raise ValueError("'item' must be length 2."
                         + f" Currently is {len(item)}.")
    if isinstance(item, tuple) and not (isinstance(item[0], int) or isinstance(item[1], int)):
        raise ValueError("The components of 'item' must be 'int'.")


def validate_nth_row(row, row_to_replace):
    """
    Validates if the new row is compatible with the row to replace.
    """
    if not isinstance(row, np.ndarray):
        raise TypeError(f"'row' is not an 'array'."
                        + f" Currently is '{type(row).__name__}'.")
    if not row.shape == row_to_replace.shape:
        raise ValueError(f"Dimensions does not fits."
                         + f" Currently is {row.shape}, it must be {row_to_replace.shape}.")


class Validator(ABC):
    """
    Abstract Class for General Validators
    """

    def __set_name__(self, owner, name):
        self.public_name = name
        self.protected_name = "_" + name
        # To make this variable final
        setattr(owner, self.protected_name + "_attr_set", False)

    def __get__(self, obj, obj_type=None):
        return getattr(obj, self.protected_name)

    def __set__(self, obj, value):
        # Ask if the variable is not set yet
        if not getattr(obj, self.protected_name + "_attr_set"):
            self._validate(obj, value)
            setattr(obj, self.protected_name, value)
            setattr(obj, self.protected_name + "_attr_set", True)
        # If it was, raise an error
        else:
            raise AttributeError(f"Attribute {self.public_name} was already set.")

    @abstractmethod
    def _validate(self, obj, value):
        """
        To use the template pattern.

        Parameters
        ----------
        obj : Object
            An instance of the current object
        value : object
            Value to _validate.
        """
        pass


class Boundary(Validator):
    """
    Validator for bounds, like (0.1, 10.5)
    """

    def _validate(self, obj, value):
        if not isinstance(value, tuple):
            raise TypeError(f"'{self.public_name}' must be a 'tuple'."
                            + f" Currently is '{type(self.public_name).__name__}'.")
        if len(value) != 2:
            raise ValueError(f"'{self.public_name}' must be a length 2."
                             + f" Currently is {len(value)}.")
        if not (isinstance(value[0], (int, float)) and isinstance(value[1], (int, float))):
            raise ValueError(f"The values of '{self.public_name}' must be 'int' or 'float'.")


class Integer(Validator):
    """
    Validator for integers, like 10
    """

    def _validate(self, obj, value):
        if not isinstance(value, int):
            raise TypeError(f"'{self.public_name}' must be an 'int'."
                            + f" Currently is '{type(self.public_name).__name__}'")
        if not value >= 0:
            raise ValueError(f"'{self.public_name}' must be greater or equals to 0.")


class Float(Validator):
    """
    Validator for floats, like 3.1415
    """

    def __init__(self, lower_bound=(None, None), upper_bound=(None, None)):
        self._lower_bound = -float("inf") if lower_bound[0] is None else lower_bound[0]
        self._lower_bound_eq = True if lower_bound[1] is None else lower_bound[1]
        self._upper_bound = float("inf") if upper_bound[0] is None else upper_bound[0]
        self._upper_bound_eq = True if upper_bound[1] is None else upper_bound[1]
        str_low_bound = "[" if self._lower_bound_eq else "("
        str_upp_bound = "]" if self._lower_bound_eq else ")"
        self._str_bounds = f"{str_low_bound}{self._lower_bound}, {self._upper_bound}{str_upp_bound}"

    def _validate(self, obj, value):
        if not isinstance(value, float):
            raise TypeError(f"'{self.public_name}' must be a 'float'."
                            + f" Currently is '{type(self.public_name).__name__}'.")
        satisfied_lower_bound = value > self._lower_bound or (self._lower_bound_eq and value == self._lower_bound)
        satisfied_upper_bound = value < self._upper_bound or (self._upper_bound_eq and value == self._upper_bound)
        if not (satisfied_lower_bound and satisfied_upper_bound):
            raise ValueError(f"'{self.public_name}' must be in {self._str_bounds}.")


class MatrixValidator(Validator, ABC):
    """
    Abstract Class for General Matrix Validators
    """

    def __get__(self, obj, obj_type=None):
        # Returns a copy of the attribute
        # return getattr(obj, self.protected_name)
        return np.copy(getattr(obj, self.protected_name))

    def __set__(self, obj, value):
        if value is None:
            setattr(obj, self.protected_name, value)
        # Ask if the variable is not set yet
        elif not getattr(obj, self.protected_name + "_attr_set"):
            self._validate(obj, value)
            value_transformed = self._transform_discrete_function(obj, value)
            setattr(obj, self.protected_name, value_transformed)
            setattr(obj, self.protected_name + "_attr_set", True)
        # If it was, raise an error
        else:
            raise AttributeError(f"Attribute {self.public_name} was already set.")

    @abstractmethod
    def _transform_discrete_function(self, obj, value):
        """
        To use template pattern.
        """
        pass


class DiscreteFunctionValidator(MatrixValidator):
    def __init__(self, x=None, y=None):
        self._x = x
        self._y = y
        self._dict_obj = {}

    def _transform_discrete_function(self, obj, value):
        if self._x is None and self._y is None:
            return value
        if isinstance(value, np.ndarray):
            return value
        x = np.linspace(*obj.x_lims, self._dict_obj[self._x])
        if self._dict_obj[self._y] is None:
            func_to_return = np.array([value(x_) for x_ in x])
        else:
            y = np.linspace(*obj.y_lims, self._dict_obj[self._y])
            func_to_return = np.array([[value(x_, y_) for y_ in y] for x_ in x])
        return func_to_return

    def _validate(self, obj, value):
        if self._x is None and self._y is None:
            return None
        # Dictionary to save the current setting
        self._dict_obj = {"x": obj.N + 2, "y": obj.M + 2, "t": obj.T + 1, None: None}
        # Make a tuple to compare
        x_size, y_size = self._dict_obj[self._x], self._dict_obj[self._y]
        tuple_to_compare = (x_size, y_size) if y_size is not None else (x_size,)
        size = len(tuple_to_compare)
        # Check if value is an array
        is_ndarray = isinstance(value, np.ndarray)
        # Case not is an array or a function
        if not (is_ndarray or callable(value)):
            raise TypeError(f"'{self.public_name}' must be an 'array' or a 'function'."
                            + f" Currently is {type(value).__name__}.")
        # If it is an array, make an instance to ensure the operations
        if is_ndarray:
            value = np.array(value)
        # Case shape does not fit
        if is_ndarray and len(value.shape) != size:
            raise ValueError(f"The dimensions of '{self.public_name}' must be {size}."
                             + f" Currently is {len(value)}.")
        # Case dimensions does not fit
        if is_ndarray and value.shape != tuple_to_compare:
            raise ValueError(f"The dimensions of '{self.public_name}' must be equals to {tuple_to_compare}."
                             + f" Currently is {value.shape}.")


class InitialDiscreteFunctionValidator(DiscreteFunctionValidator):
    def _transform_discrete_function(self, obj, value):
        if self._x is None and self._y is None:
            return value
        func_to_return = np.zeros((self._dict_obj[self._x], self._dict_obj[self._y]))
        if isinstance(value, np.ndarray):
            func_to_return[0] = value
            return func_to_return
        y = np.linspace(*obj.y_lims, self._dict_obj[self._y])
        func_to_return[0] = np.array([value(y_) for y_ in y])
        return func_to_return

    def _validate(self, obj, value):
        if self._x is None and self._y is None:
            return None
        # Dictionary to save the current setting
        self._dict_obj = {"x": obj.N + 2, "y": obj.M + 2, "t": obj.T + 1, None: None}
        # Make a tuple to compare
        y_size = self._dict_obj[self._y]
        tuple_to_compare = (y_size,)
        size = len(tuple_to_compare)
        # Check if value is an array
        is_ndarray = isinstance(value, np.ndarray)
        # Case not is an array or a function
        if not (is_ndarray or callable(value)):
            raise TypeError(f"'{self.public_name}' must be an 'array' or a 'function'."
                            + f" Currently is {type(value).__name__}.")
        # If it is an array, make an instance to ensure the operations
        if is_ndarray:
            value = np.array(value)
        # Case shape does not fit
        if is_ndarray and len(value.shape) != size:
            raise ValueError(f"The dimensions of '{self.public_name}' must be {size}."
                             + f" Currently is {len(value)}.")
        # Case dimensions does not fit
        if is_ndarray and value.shape != tuple_to_compare:
            raise ValueError(f"The dimensions of '{self.public_name}' must be equals to {tuple_to_compare}."
                             + f" Currently is {value.shape}.")
