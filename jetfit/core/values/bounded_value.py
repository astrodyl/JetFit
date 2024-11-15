import operator
from enum import Enum

from jetfit.core.utilities import maths


class BoundedValue:
    """
    A value with lower and upper bounds.

    Whether the bounds store (lower error, upper error), or (value + lower
    error, value + upper error) is up to the inheriting classes' implementation.

    Attributes
    ----------
    value : float
        The value's magnitude.

    lower : float
        The lower bound of the value.

    upper : float
        The upper bound of the value.

    units: one of `jetfit.core.defns.enums`
        The units of the value and bounds.
    """
    def __init__(self, value: float, lower: float, upper: float, units: Enum):
        self.value = value
        self.units = units
        self.lower = lower
        self.upper = upper

    def __add__(self, other):
        """
        Performs addition between the calling object and `other`.

        Parameters
        ----------
        other : BoundedValue or float
            The value to add.

        Returns
        -------
        BoundedValue
            Resultant BoundedValue with propagated error.
        """
        return self.operate(operator.__add__, other)

    def __sub__(self, other):
        """
        Performs subtraction between the calling object and `other`.

        Parameters
        ----------
        other : BoundedValue or float
            The value to subtract.

        Returns
        -------
        BoundedValue
            Resultant BoundedValue with propagated error.
        """
        return self.operate(operator.__sub__, other)

    def __truediv__(self, other):
        """
        Performs division between the calling object and `other`.

        Parameters
        ----------
        other : BoundedValue or float
            The value to divide by.

        Returns
        -------
        BoundedValue
            Resultant BoundedValue with propagated error.
        """
        return self.operate(operator.__truediv__, other)

    def __mul__(self, other):
        """
        Performs multiplication between the calling object and `other`.

        Parameters
        ----------
        other : BoundedValue or float
            The value to multiply by.

        Returns
        -------
        BoundedValue
            Resultant BoundedValue with propagated error.
        """
        return self.operate(operator.__mul__, other)

    def operate(self, op: operator, other):
        """
        Performs the `op` operation between the calling object and `other`.

        Parameters
        ----------
        op : operator
            The operation to perform.

        other : BoundedValue or float
            The value to multiply by.

        Returns
        -------
        BoundedValue
            The resultant BoundedValue with propagated error.
        """
        if isinstance(other, float):
            return self.__class__(op(self.value, other), self.lower, self.upper, self.units)

        if isinstance(other, self.__class__):
            return maths.bounded_operation(op, self, other)

        raise NotImplementedError(f"Unsupported operation between {self.__class__} "
                                  f"and {other.__class__}: {op.__name__}.")

    def encompasses(self, value: float) -> bool:
        """
        Checks if the value is contained within the bounds.

        Parameters
        ----------
        value : float
            The value to check.

        Returns
        -------
        bool
            True if value is contained within the bounds else False.
        """
        return self.lower <= value <= self.upper