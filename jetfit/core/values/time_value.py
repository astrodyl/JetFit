from jetfit.core.defns.enums import TimeUnits
from jetfit.core.defns.mixins import UnitsMixin
from jetfit.core.utilities import maths


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

        instance =  cls(row.Time, row.TimeUnits)

        if instance.units != desired_u:
            instance.convert_to(desired_u)

        return instance

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
        return maths.convert_time_to(self.value, self.units, units)

    def convert_to(self, units: TimeUnits) -> None:
        """
        Converts `value` to the provided `units`.

        Parameters
        ----------
        units : `jetfit.core.enums.TimeUnits`
            The units to convert to.
        """
        if units == TimeUnits.SEC:
            return self.to_secs()

        if units == TimeUnits.HRS:
            return self.to_hours()

        if units == TimeUnits.DAY:
            return self.to_days()

    def to_secs(self) -> None:
        """ Converts the value and units to seconds. """
        self.value = maths.to_seconds(self.value, self.units)
        self.units = TimeUnits.SEC

    def to_hours(self) -> None:
        """ Converts the value and units to hours. """
        self.value = maths.to_hours(self.value, self.units)
        self.units = TimeUnits.HRS

    def to_days(self) -> None:
        """ Converts the value and units to days. """
        self.value = maths.to_days(self.value, self.units)
        self.units = TimeUnits.DAY
