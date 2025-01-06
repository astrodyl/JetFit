import copy

from jetfit.core.defns.enums import TimeUnits
from jetfit.core.defns.mixins import UnitsMixin
from jetfit.core.utils.physics import TimeConversions


class TimeValue(UnitsMixin):
    """
    A time measurement value with units.

    Attributes
    ----------
    value : float
        The time measurement value.

    units : `jetfit.core.enums.TimeUnits`
        The unit of the time measurement value.
    """
    _units_enum = TimeUnits

    def __init__(self, value: float, units: TimeUnits | str):
        self.value = value
        self.units = units

    @classmethod
    def from_csv_row(cls, row, desired_u: TimeUnits = None):
        """
        Returns instance parsed from a row of a CSV row.

        Parameters
        ----------
        row : NamedTuple
            CSV row with `Time` and `TimeUnits`.

        desired_u : `jetfit.core.enums.TimeUnits`, optional
            Units to store the time.

        Raises
        ------
        TypeError
            If `Time` or `TimeUnits` is `None` or not of the expected types.

        Returns
        -------
        TimeValue
            Populated from the CSV row.
        """
        for value in row:
            if value is None:
                raise TypeError(f'Missing {row.name} in row {row.Index}.')

        instance = cls(row.Time, row.TimeUnits)
        instance.convert_to(desired_u)

        return instance

    def copy(self):
        """
        Deep copies the calling object.

        Returns
        -------
        TimeValue
            The deepcopy of the object.
        """
        return copy.deepcopy(self)

    def get_value(self, units: TimeUnits | str) -> float:
        """
        Returns the value converted to `units` from `self.units`.

        Parameters
        ----------
        units : `jetfit.core.enums.TimeUnits`
            The units of the returned value.

        Returns
        -------
        float
            The value in the provided `units`.
        """
        return TimeConversions.convert_to(self.value, self.units, units)

    def to_seconds(self):
        """ Converts time value with units of ``TimeUnits.SEC``. """
        self.value = TimeConversions.to_seconds(self.value, self.units)
        self.units = TimeUnits.SEC

    def to_hours(self) :
        """ Converts the time value with units of ``TimeUnits.HRS``. """
        self.value = TimeConversions.to_hours(self.value, self.units)
        self.units = TimeUnits.HRS

    def to_days(self) -> None:
        """ Converts the time value with units of ``TimeUnits.DAY``. """
        self.value = TimeConversions.to_days(self.value, self.units)
        self.units = TimeUnits.DAY

    def convert_to(self, units: TimeUnits) -> None:
        """
        Converts the time value with units of ``units``.

        units : TimeUnits
            The time units to convert to.

        Raises
        ------
        NotImplementedError
            If ``units`` is not one of ``TimeUnits``.
        """
        if units not in TimeUnits:
            raise NotImplementedError(f"Unsupported time units: {units}")

        getattr(self, f'to_{units.value}')()
