from jetfit.core.defns.enums import IndexType
from jetfit.core.defns.mixins import BoundedMixin
from jetfit.core.values.time import TimeValue


class SpectralIndexValue(BoundedMixin):
    """
    A spectral index.

    Attributes
    ----------
    value : float
        The spectral index value. A negative value indicates

    start : TimeValue
        The start integration time for the spectral index calculation.

    stop : TimeValue
        The stop integration time for the spectral index calculation.
    """
    def __init__(self, value: float, lower: float, upper: float,
                 start: TimeValue, stop: TimeValue, frequency_range: tuple):

        super().__init__(lower, upper)

        self.value = value
        self.start = start
        self.stop = stop
        self.frequency_range = frequency_range

    @property
    def type(self) -> IndexType:
        """ Returns the type of flux. """
        return IndexType.SPECTRAL

    @property
    def avg_error(self) -> float:
        """ Returns the average of the error bounds. """
        return (self.upper + self.lower) / 2

    @classmethod
    def from_csv_row(cls, row):
        """
        Returns instance parsed from a CSV row.

        Parameters
        ----------
        row : NamedTuple
            CSV row.

        Returns
        -------
        SpectralIndexValue
            Populated from the CSV row.
        """
        start = TimeValue(row.TimeLower, row.TimeUnits)
        stop = TimeValue(row.TimeUpper, row.TimeUnits)

        return cls(
            value=row.Value,
            lower=row.ValueLower,
            upper=row.ValueUpper,
            start=start,
            stop=stop,
            frequency_range=(
                row.FrequencyLower,
                row.FrequencyUpper
            )
        )
