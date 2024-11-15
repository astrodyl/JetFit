import operator

from jetfit.core.defns.enums import FluxUnits, FluxType
from jetfit.core.defns.mixins import UnitsMixin
from jetfit.core.utilities import maths
from jetfit.core.values.bounded_value import BoundedValue


class FluxValue(BoundedValue, UnitsMixin):
    """ A flux measurement value with units and bounds. """
    _units_enum = FluxUnits

    def __init__(self, value: float, lower: float, upper: float,
                 units: FluxUnits | str):

        BoundedValue.__init__(self, value, lower, upper, units)

    @property
    def avg_error(self) -> float:
        """ Returns the average of the error bounds. """
        return (self.upper + self.lower) / 2

    def get_value(self, units: str | FluxUnits) -> float:
        """
        Returns the value in the provided `units`.

        Parameters
        ----------
        units: str or FluxUnits
            The units of the returned value.

        Returns
        -------
        float
            The value in the provided `units`.
        """
        return  maths.convert_flux_to(self.value, self.units, units)

    def convert_to(self, units: FluxUnits) -> None:
        """
        Converts `value`, `lower`, and `upper` to the `units`.

        Parameters
        ----------
        units : FluxUnits
            The units to convert to.
        """
        if units == FluxUnits.MJY:
            return self.to_mjy()

        if units == FluxUnits.CGS:
            return self.to_cgs()

    def to_mjy(self) -> None:
        """ Converts the flux value and its bounds to milli-Jansky units. """
        self.value = maths.to_mjy(self.value, self.units)
        self.lower = maths.to_mjy(self.lower, self.units)
        self.upper = maths.to_mjy(self.upper, self.units)
        self.units = FluxUnits.MJY

    def to_cgs(self) -> None:
        """ Converts the flux value and its bounds to CGS units. """
        self.value = maths.to_cgs(self.value, self.units)
        self.lower = maths.to_cgs(self.lower, self.units),
        self.upper = maths.to_cgs(self.upper, self.units)
        self.units = FluxUnits.CGS


class IntegratedFluxValue(FluxValue):
    """
    An integrated flux measurement.

    Attributes
    ----------
    frequency_range : tuple
        The frequency range to integrate over measured in Hz.

    Notes
    -----
    Integrated flux refers to the total energy received from a source over a
    specific wavelength or frequency range, calculated by integrating the flux
    values across that range. Typical units are erg / cm^2 / s.

    Recommended for x-ray bands since their associated ranges can be large.
    For example, Swift XRT uses (0.3keV - 10keV) = (7.25e16 Hz - 2.42e18 Hz).
    """
    def __init__(self, value: float, lower: float, upper: float,
                 units: FluxUnits | str, frequency_range: tuple[float, float]):

        FluxValue.__init__(self, value, lower, upper, units)
        self.frequency_range = frequency_range

    def __add__(self, other):
        """"""
        bv = self.operate(operator.__add__, other)
        return self.__class__(bv.value, bv.lower, bv.upper, bv.units, self.frequency_range)

    def __sub__(self, other):
        """"""
        bv = self.operate(operator.__sub__, other)
        return self.__class__(bv.value, bv.lower, bv.upper, bv.units, self.frequency_range)

    def __truediv__(self, other):
        """"""
        bv = self.operate(operator.__truediv__, other)
        return self.__class__(bv.value, bv.lower, bv.upper, bv.units, self.frequency_range)

    def __mul__(self, other):
        """"""
        bv = self.operate(operator.__mul__, other)
        return self.__class__(bv.value, bv.lower, bv.upper, bv.units, self.frequency_range)

    @property
    def type(self) -> FluxType:
        """ Returns the type of flux. """
        return FluxType.INTEGRATED

    @classmethod
    def from_csv_row(cls, row, desired_u: FluxUnits = None):
        """
        Returns instance parsed from a CSV row.

        Parameters
        ----------
        row : NamedTuple
            CSV row with `Flux`, `FluxUnits`, `FluxLowerError`,
            `FluxUpperError`, `FrequencyLower`, and `FrequencyUpper`.

        desired_u : FluxUnits, optional
            Units to store the flux values.

        Raises
        ------
        TypeError
            If `Flux`, `FluxUnits`, `FluxLowerError`, `FluxUpperError`,
            `FrequencyLower`, or `FrequencyUpper` is None.

        Returns
        -------
        IntegratedFluxValue
            Populated from the CSV row.
        """
        # TODO: Write CSV validation class. For now, checking for nones is fine.

        for value in row:
            if value is None:
                raise TypeError(f'Missing {row.name} in row {row.Index}.')

        instance = cls(value=row.Flux,
                       units=row.FluxUnits,
                       lower=row.FluxLowerError,
                       upper=row.FluxUpperError,
                       frequency_range=(
                           row.FrequencyLower,
                           row.FrequencyUpper
                       ))

        if instance.units != desired_u:
            instance.convert_to(desired_u)

        return instance

class SpectralFluxValue(FluxValue):
    """
    A spectral flux (flux density) measurement.

    Attributes
    ----------
    frequency : float
        The average frequency of the spectral flux measurement.

    Notes
    -----
    Spectral flux refers to the flux per unit frequency and describes how the
    flux is distributed over a spectrum. Typical units are mJy.

    Recommended for optical, near-infrared, and radio since their associated
    ranges are small enough that a single frequency is a good approximation.
    """
    def __init__(self, value: float, lower: float, upper: float,
                 frequency: float, units: FluxUnits | str):

        FluxValue.__init__(self, value,lower, upper, units)
        self.frequency = frequency

    @property
    def type(self) -> FluxType:
        """ Returns the type of flux. """
        return FluxType.SPECTRAL

    @classmethod
    def from_csv_row(cls, row):
        """
        Returns instance parsed from a row of a CSV row.

        row : NamedTuple
            CSV row with `Flux`, `FluxUnits`, `FluxLowerError`,
            `FluxUpperError`, `FrequencyLower`, and `FrequencyUpper`.

        Raises
        ------
        TypeError
            If `Flux`, `FluxUnits`, `FluxLowerError`, `FluxUpperError`,
            `FrequencyLower`, or `FrequencyUpper` is None.

        Raises
        ------
        TypeError
            If any cell of the row is `None`.

        Returns
        -------
        SpectralFluxValue
            Populated from the CSV row.
        """
        for value in row:
            if value is None:
                raise TypeError(f'Missing {row.name} in row {row.Index}.')

        return cls(
            value=row.Flux,
            units=row.FluxUnits,
            lower=row.FluxLowerError,
            upper=row.FluxUpperError,
            frequency=row.Frequency
        )
