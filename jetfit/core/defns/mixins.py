from enum import Enum


class BoundedMixin:
    """
    Adds tuple bounds with helper methods.

    Adds support for values that have an associated error region. Whether
    the bounds store (lower error, upper error), or (value + lower error,
    value + upper error) is up to the inheriting classes' implementation.

    Attributes
    ----------
    lower : float
        The lower bound of the value.

    upper : float
        The upper bound of the value.
    """
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def encompasses(self, value: float):
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


class UnitsMixin:
    """
    Adds support for values that have units.

    Notes
    -----
    All classes inheriting from this mixin must have a `_units_enum` class
    attribute. It is required to convert the string representation of the
    units to the enum representation.
    """
    _units_enum = None

    @property
    def units(self) -> Enum:
        """ Returns the units Enum for the instance."""
        return self._units

    @units.setter
    def units(self, units: Enum | str) -> None:
        """
        Sets the units.

        Parameters
        ----------
        units : str or Enum
            Must match the `_units_enum` attribute defined in the inheriting
            classes' attributes.

        Notes
        -----
        If a str is passed, converts it to its enum representation as defined
        in the inheriting classes' variables.

        Raises
        ------
        ValueError
            If the provided units argument does not match, or is not convertable
            to, the inheriting classes' `_units_enum` attribute.

        TypeError
            If provided `units` is not a str or Enum instance.
        """
        if isinstance(units, self._units_enum):
            self._units = units

        elif isinstance(units, Enum):
            raise ValueError(f"{units} is not supported for {self.__class__}. "
                             f"Units must be of type {self._units_enum}.")

        elif isinstance(units, str):
            try:
                self._units = self._units_enum(units)
            except ValueError:
                raise ValueError(f"{units} is not supported. Supported units include "
                                 f"{[u.value for u in self._units_enum]}")

        else:
            raise TypeError("Units must be a string or an Enum instance.")
