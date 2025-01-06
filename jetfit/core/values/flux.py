import copy
import operator
from enum import Enum

from jetfit.core.defns.enums import FluxUnits, FluxType
from jetfit.core.defns.mixins import UnitsMixin
from jetfit.core.utils.physics import FluxConversions
from jetfit.core.values.bounded import BoundedValue


class FluxValue(BoundedValue, UnitsMixin):
    """ A flux measurement value with units and bounds. """
    _units_enum = FluxUnits

    def __init__(
            self,
            value: float,
            lower: float,
            upper: float,
            units: FluxUnits | str
    ):
        BoundedValue.__init__(self, value, lower, upper, units)

    @property
    def avg_error(self) -> float:
        """ Returns the average of the error bounds. """
        return (self.upper + self.lower) / 2

    def copy(self):
        """
        Deep copies the calling object.

        Returns
        -------
        FluxValue
            The deepcopy of the object.
        """
        return copy.deepcopy(self)


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
    def __init__(
            self,
            value: float,
            lower: float,
            upper: float,
            units: FluxUnits | str,
            frequency_range: tuple[float, float]
    ):
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
    def frequency(self) -> float:
        """
        Returns the average of the frequency range.

        Returns
        -------
        float
            The average of the frequency range.
        """
        return (self.frequency_range[1] + self.frequency_range[0]) / 2

    @property
    def type(self) -> FluxType:
        """ Returns the type of flux. """
        return FluxType.INTEGRATED

    @classmethod
    def from_csv_row(cls, row):
        """
        Returns instance parsed from a CSV row.

        Parameters
        ----------
        row : NamedTuple
            ????

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
        instance = cls(
            value=row.Value,
            units=row.ValueUnits,
            lower=row.ValueLower,
            upper=row.ValueUpper,
            frequency_range=(
                row.FrequencyLower,
                row.FrequencyUpper
            )
        )

        return instance

    def get_spectral(self):
        """ Returns the flux as an equivalent spectral flux. """
        frequency_range = self.frequency_range[1] - self.frequency_range[0]

        return SpectralFluxValue(
            self.value / (1.0e-26 * frequency_range),
            self.lower / (1.0e-26 * frequency_range),
            self.upper / (1.0e-26 * frequency_range),
            self.frequency,
            FluxUnits.MJY
        )


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
    def __init__(
            self,
            value: float,
            lower: float,
            upper: float,
            frequency: float,
            units: FluxUnits | str | Enum
    ):
        FluxValue.__init__(self, value, lower, upper, units)
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
            value=row.Value,
            units=row.ValueUnits,
            lower=row.ValueLower,
            upper=row.ValueUpper,
            frequency=row.Frequency
        )

    def to_mjy(self) -> None:
        """ Converts the flux value, lower, and upper to units of ``FluxUnits.MJY``. """
        if self.units == FluxUnits.CGS:
            self.value = FluxConversions.to_mjy(self.value, FluxUnits.CGS)
            self.lower = FluxConversions.to_mjy(self.lower, FluxUnits.CGS)
            self.upper = FluxConversions.to_mjy(self.upper, FluxUnits.CGS)
            self.units = FluxUnits.MJY

    def to_cgs(self) -> None:
        """ Converts the flux value, lower, and upper to units of ``FluxUnits.CGS``. """
        if self.units == FluxUnits.MJY:
            self.value = FluxConversions.to_cgs(self.value, FluxUnits.MJY)
            self.lower = FluxConversions.to_cgs(self.lower, FluxUnits.MJY)
            self.upper = FluxConversions.to_cgs(self.upper, FluxUnits.MJY)
            self.units = FluxUnits.CGS

    def convert_to(self, units: FluxUnits) -> None:
        """
        Converts the flux value, lower, and upper to units of ``units``.

        units : FluxUnits
            The flux units to convert to.

        Raises
        ------
        NotImplementedError
            If ``units`` is not one of ``FluxUnits``.
        """
        if units not in FluxUnits:
            raise NotImplementedError(f"Unsupported flux units: {units}")

        getattr(self, f'to_{units.value}')()
