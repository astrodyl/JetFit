import numpy as np

from jetfit.core.defns.enums import FluxUnits
from jetfit.core.values.flux import SpectralFluxValue
from jetfit.models.data.flux.extinction.ccm_model import CCMExtinction
from jetfit.models.data.flux.flux_model import FluxModel


class SpectralFluxModel(FluxModel):
    """
    Sari Piran Spectral Flux Model (Flux Density).

    Sari & Piran model describes the synchrotron emission model from a
    decelerating relativistic shell that collides with an external medium
    [1]_. An extinction model is applied for frequencies contained within
    the Optical/NIR regime [2]_.

    Ignores the self-absorption regime (segment A for fast cooling and segment
    E for slow cooling) since it does not affect either the optical nor the
    X-ray radiation in which we are interested.

    Attributes
    ----------
    frequency : float
        The average band frequency to model.

    segment : str
        The Sari & Piran spectral segment.

    ebv_milky_way : float, default: 0
        The Milky Way E(B - V) strength. Defaults to zero which does not
        affect the flux strength.

    ebv_source_frame : float, default: 0
        The source frame E(B - V) strength. Defaults to zero which does not
        affect the flux strength.

    redshift : float
        The redshift of the event. The redshift is used to transform the
        observed frequency into the source frame for the extinction
        calculation.

    References
    ----------
    .. [1] Sari & Piran (1998)
        https://ui.adsabs.harvard.edu/abs/1998ApJ...497L..17S/abstract

    .. [2] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. (1989)
        https://articles.adsabs.harvard.edu//full/1989ApJ...345..245C/0000249.000.html

    See Also
    --------
    jetfit.models.data.flux.extinction.extinction :
        Implements the CCM Optical/NIR extinction model.

    jetfit.models.afterglow.sari_piran.sari_piran.SariPiran :
        A wrapper class that provides an easy way to evaluate Sari & Piran
        models.
    """
    def __init__(
            self,
            pf: float,
            cf: float,
            sf: float,
            p: float,
            frequency: float,
            units: str | FluxUnits,
            ebv_milky_way: float = 0.0,
            ebv_source_frame: float = 0.0,
            redshift: float = 0.0
    ):
        super().__init__(pf, cf, sf, p, units)

        self.frequency = frequency
        self.ebv_milky_way = ebv_milky_way
        self.ebv_source_frame = ebv_source_frame
        self.redshift = redshift
        self._segment = self.segment

    @property
    def segment(self) -> str:
        """
        Determines the Sari & Piran spectral segment.

        Returns
        -------
        str
            The letter representing the spectral segment.
        """
        if self.regime == 'fast':
            f1, f2 = self.cf, self.sf
            c1, c2, c3 = 'B', 'C', 'D'

        else:
            f1, f2 = self.sf, self.cf
            c1, c2, c3 = 'F', 'G', 'H'

        if self.frequency <= f1:
            return c1

        if self.frequency < f2:
            return c2

        if self.frequency >= f2:
            return c3

    @segment.setter
    def segment(self, s: str) -> None:
        """"""
        raise NotImplemented(
            'Setting the segment is not allowed since it is determined by the'
            'observed frequency, cooling frequency, and synchrotron frequency.'
        )

    def evaluate(self, obj: bool = False) -> SpectralFluxValue | float:
        """
        Evaluates the model for a given ``segment`` and ``regime``.

        Parameters
        ----------
        obj : bool, optional
            If ``True``, returns a ``SpectralFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        SpectralFluxValue or float
            The modeled spectral flux value.
        """
        return getattr(self, f"get_segment_{self.segment.lower()}_flux")(obj)

    # <editor-fold desc="Fast Cooling Regime">
    def get_segment_b_flux(self, obj: bool = False) -> SpectralFluxValue | float:
        """
        Calculates and returns a flux in the fast cooling regime (B).

        Parameters
        ----------
        obj : bool, optional
            If ``True``, returns a ``SpectralFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        SpectralFluxValue or float
            The modeled flux in the fast cooling regime (B).
        """
        if self.cf == 0.0:
            return np.nan

        return self._return(self.pf * (self.frequency / self.cf) ** (1 / 3), obj)

    def get_segment_c_flux(self, obj: bool = False) -> SpectralFluxValue | float:
        """
        Calculates and returns a flux in the fast cooling regime (C).

        Parameters
        ----------
        obj : bool, optional
            If ``True``, returns a ``SpectralFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        SpectralFluxValue or float
            The modeled flux in the fast cooling regime (C).
        """
        if self.cf == 0.0:
            return np.nan

        return self._return(self.pf * (self.frequency / self.cf) ** (-1 / 2), obj)

    def get_segment_d_flux(self, obj: bool = False) -> SpectralFluxValue | float:
        """
        Calculates and returns a flux in the fast cooling regime (D).

        Parameters
        ----------
        obj : bool, optional
            If ``True``, returns a ``SpectralFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        SpectralFluxValue or float
            The modeled flux in the fast cooling regime (D).
        """
        if self.cf == 0.0 or self.sf == 0.0:
            return np.nan

        eq1 = self.pf * ((self.sf / self.cf) ** (-1 / 2))
        eq2 = (self.frequency / self.sf) ** (-self.p / 2)

        return self._return(eq1 * eq2, obj)
    # </editor-fold>

    # <editor-fold desc="Slow Cooling Regime">
    def get_segment_f_flux(self, obj: bool = False) -> SpectralFluxValue | float:
        """
        Calculates and returns a flux in the slow cooling segment (F).

        Parameters
        ----------
        obj : bool, optional
            If ``True``, returns a ``SpectralFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        SpectralFluxValue or float
            The modeled flux in the slow cooling segment (F).
        """
        if self.sf == 0.0:
            return np.nan

        return self._return(self.pf * (self.frequency / self.sf) ** (1 / 3), obj)

    def get_segment_g_flux(self, obj: bool = False) -> SpectralFluxValue | float:
        """
        Calculates and returns a flux in the slow cooling segment (G).

        Parameters
        ----------
        obj : bool, optional
            If ``True``, returns a ``SpectralFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        SpectralFluxValue or float
            The modeled flux in the slow cooling regime (G).
        """
        if self.sf == 0.0:
            return np.nan

        return self._return(self.pf * (self.frequency / self.sf) ** -((self.p - 1) / 2), obj)

    def get_segment_h_flux(self, obj: bool = False) -> SpectralFluxValue | float:
        """
        Calculates and returns a flux in the slow cooling regime (H).

        Parameters
        ----------
        obj : bool, optional
            If ``True``, returns a ``SpectralFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        SpectralFluxValue or float
            The modeled flux in the slow cooling regime (H).
        """
        if self.sf == 0.0 or self.cf == 0.0:
            return np.nan

        eq1 = self.pf
        eq2 = (self.cf / self.sf) ** (-(self.p - 1) / 2)
        eq3 = (self.frequency / self.cf) ** (-self.p / 2)

        return self._return(eq1 * eq2 * eq3, obj)
    # </editor-fold>

    def _return(self, val: float, obj: bool) -> SpectralFluxValue | float:
        """
        Returns the modeled flux in the desired format with units of
        ``self.units``.

        Parameters
        ----------
        val : float
            The flux value.

        obj : bool
            If ``True``, returns a ``SpectralFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        SpectralFluxValue or float
            The modeled spectral flux with units of `units`.
        """
        if self.ebv_milky_way != 0.0:
            val *= CCMExtinction().evaluate(
                self.frequency, ebv=self.ebv_milky_way
            )

        if self.ebv_source_frame != 0.0:
            val *= CCMExtinction().evaluate(
                self.frequency, self.redshift, self.ebv_source_frame
            )

        if obj:
            return SpectralFluxValue(
                value=val,
                lower=0.0,
                upper=0.0,
                units=self.units,
                frequency=self.frequency
            )

        return val
