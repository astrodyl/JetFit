from jetfit.core.defns.enums import FluxUnits
from jetfit.core.values.flux import FluxValue


class FluxModel:
    """
    Flux Class

    Attributes
    ----------
    pf : float
        The peak flux.

    cf : float
        The cooling frequency.

    sf : float
        The synchrotron frequency.

    p : float
        The electron energy index.

    units : FluxUnits
        The units of the peak flux.

    _regime : str
        The cooling regime (fast or slow).

    _segment : str
        The spectral segment.

    See Also
    --------
    jetfit.models.data.flux.spectral.spsf_model.SpectralFluxModel :
        The Sari & Piran analytic spectral flux (flux density) model.

    jetfit.models.data.flux.integrated.spif_model.IntegratedFluxModel :
        The Sari & Piran analytic integrated flux model.
    """
    def __init__(
            self,
            pf: float,
            cf: float,
            sf: float,
            p: float,
            units: str | FluxUnits
    ):
        self.pf = pf
        self.cf = cf
        self.sf = sf
        self.p = p

        self._units = units
        self._regime = self.regime

    @property
    def units(self) -> FluxUnits:
        """ Returns the peak flux units. """
        return self._units

    @units.setter
    def units(self, units: str | FluxUnits) -> None:
        """
        Sets the peak flux units.

        Parameters
        ----------
        units : str or FluxUnits
            The units of the peak flux. If a `str` is given, it is first
            converted to a ``FluxUnits`` enum before setting.
        """
        self._units = units if isinstance(units, FluxUnits) else FluxUnits(units)

    @property
    def regime(self) -> str:
        """
        Defining the regime as a  property allows the user to modify the
        spectral frequencies.
        """
        return 'fast' if self.sf > self.cf else 'slow'

    def update_peak_flux(self, pf: float | FluxValue, units: str | FluxUnits = None) -> None:
        """
        Updates the peak flux and its units.

        Parameters
        ----------
        pf : float or FluxValue
            The peak flux.

        units : str or FluxUnits
            The units of the peak flux.

        Examples
        --------
        >>> model = FluxModel(1.0e-7, 1.0e15, 1.0e18, 2.5, 'mjy')

        Update the peak flux using a ``FluxValue`` object.

        >>> flux_obj = FluxValue(1.0e-7, 1.1e-8, 1.2e-8, FluxUnits.CGS)
        >>> model.update_peak_flux(flux_obj)
        >>> model.units
        FluxUnits.CGS

        Update the peak flux using a float and a `str`.

        >>> model.update_peak_flux(1.1e-7, 'mjy')
        >>> model.units
        FluxUnits.MJY

        Update the peak flux using a float and a ``FluxUnits`` enum.

        >>> model.update_peak_flux(1.1e-7, FluxUnits.CGS)
        >>> model.units
        FluxUnits.CGS
        """
        if isinstance(units, FluxUnits):
            pf = pf.value
            units = pf.units

        self.pf = pf
        self.units = units
