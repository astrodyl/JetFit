

class BoundedMixin:
    """
    Adds tuple bounds with helper methods.

    Adds support for values that have an associated error region. Whether
    the bounds store (lower error, upper error), or (value + lower error,
    value + upper error) is up to the inheriting classes' implementation.

    Attributes
    ----------
    lower : float or astropy.units.Quantity
        The lower bound of the value.

    upper : float or astropy.units.Quantity
        The upper bound of the value.
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __repr__(self) -> str:
        return f"BoundedMixin(lower={self.lower}, upper={self.upper})"

    def bounds(self) -> tuple:
        """
        Returns the lower and upper bounds as a tuple.

        Returns
        -------
        tuple with length 2
            The lower and upper bound.
        """
        return self.lower, self.upper

    def encompasses(self, value):
        """
        Checks if the value is contained within the bounds.

        Parameters
        ----------
        value : float or astropy.units.Quantity
            The value to check.

        Returns
        -------
        bool
            True if value is contained within the bounds else False.
        """
        return self.lower <= value <= self.upper
