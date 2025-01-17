import numpy as np

from jetfit.core.defns.enums import FluxUnits
from jetfit.core.values.flux import IntegratedFluxValue
from jetfit.models.data.flux.flux_model import FluxModel


class IntegratedFluxModel(FluxModel):
    """
    Implements the integrated flux models from Sari & Piran (1998) [1]_.

    Ignores the self-absorption regime (segment A for fast cooling and segment
    E for slow cooling) since it does not affect either the optical nor the
    X-ray radiation in which we are interested.

    If the ``lower`` and ``upper`` span multiple spectral segments, the model
    is evaluated as a sum of the integrands for each relevant segment.

    For example, ``get_segment_cd_flux(lower, upper)`` is equivalent to:
    ``get_segment_c_flux(lower, sf)`` + ``get_segment_c_flux(sf, upper)``.

    Attributes
    ----------
    lower : float
        The lower integration limit measured in Hz.

    upper : float
        The upper integration limit measured in Hz.

    References
    ----------
    .. [1] Sari & Piran (1998)
        https://ui.adsabs.harvard.edu/abs/1998ApJ...497L..17S/abstract

    Examples
    --------
    Create an instance:

    >>> ranges = (7.25e16, 2.42e18)  # Swift XRT integration limits
    >>> pf, cf, sf, p = 0, 0, 0, 0  # Placeholder values for example
    >>> int_model = IntegratedFluxModel(pf, cf, sf, p, ranges[0], ranges[1], FluxUnits.CGS)

    Get the regime and segment:

    >>> int_model.regime
    'fast'

    >>> int_model.segment
    'C'

    Model the integrated flux returning a float:

    >>> int_model.evaluate()
    1.0E-19

    Model the integrated flux returning an IntegratedFluxValue:

    >>> obj = int_model.evaluate(obj=True)
    >>> print(obj.value, obj.units)
    1.0E-19, FluxUnits.CGS

    See Also
    --------
    jetfit.models.afterglow.sari_piran.sari_piran.SariPiran :
        A wrapper class that provides an easy way to evaluate Sari & Piran
        models.

    `test.sari_piran.test_if_model.py` :
        Unit tests for this class.
    """
    def __init__(
            self,
            pf: float,
            cf: float,
            sf: float,
            p: float,
            lower: float,
            upper: float,
            units: str | FluxUnits = None,
    ):
        super().__init__(pf, cf, sf, p, units)
        self.lower = lower
        self.upper = upper
        self._segment = self.segment

    @property
    def segment(self) -> str:
        """
        Returns the Sari Piran spectral segment.

        If ``lower`` and ``upper`` span multiple segments, they are
        concatenated together. For example, 'CD' spans segments 'C',
        and 'D'.

        Returns
        -------
        str
            The Sari Piran spectral segment.
        """
        if self.regime == 'fast':
            f1, f2 = self.cf, self.sf
            c1, c2, c3 = 'B', 'C', 'D'

        else:
            f1, f2 = self.sf, self.cf
            c1, c2, c3 = 'F', 'G', 'H'

        # Entire range is below cf(sf) for fast(slow).
        if self.upper <= f1:
            return c1

        # Upper is between cf(sf) and sf(cf) for fast(slow).
        # Check where the lower frequency lies.
        if self.upper <= f2:
            if self.lower < f1:
                return c1 + c2
            return c2

        # Upper is above sf(cf) for fast(slow).
        # Check where the lower frequency lies.
        if self.lower < f1:
            return c1 + c2 + c3

        if self.lower < f2:
            return c2 + c3

        # Entire range is above sf(cf) for fast(slow).
        return c3

    def evaluate(self, obj: bool = False):
        """
        Evaluates the integrated flux model using the class attributes.

        Parameters
        ----------
        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        IntegratedFluxValue or float
            The modeled integrated flux with units of erg / s / cm2.
        """
        return getattr(self, f"get_segment_{self.segment.lower()}_flux")(obj)

    # <editor-fold desc="Fast Cooling">
    def get_segment_b_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Models a fast cooling flux for the Sari & Piran segment (B).

        Parameters
        ----------
        lower : float, optional
            Lower frequency to evaluate the integrand.

        upper : float, optional
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
            The integrated flux value in the fast cooling regime (B) with
            units of erg / s / cm2.
        """
        if self.cf == 0.0:
            return np.nan

        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        eq1 = (3 / 4) * self.pf
        eq2 = (1 / self.cf) ** (1 / 3)
        eq3 = (upper ** (4 / 3)) - (lower ** (4 / 3))

        return self._return(eq1 * eq2 * eq3, self.units, obj)

    def get_segment_c_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Models a fast cooling flux for the Sari & Piran segment (C).

        Parameters
        ----------
        lower : float, optional
            Lower frequency to evaluate the integrand.

        upper : float, optional
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
            The integrated flux value in the fast cooling regime (C) with
            units of erg / s / cm2.
        """
        if self.cf == 0.0:
            return np.nan

        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        eq1 = 2 * self.pf * (1 / self.cf) ** -0.5
        eq2 = (upper ** 0.5) - (lower ** 0.5)

        return self._return(eq1 * eq2, self.units, obj)

    def get_segment_d_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Models a fast cooling flux for the Sari & Piran segment (D).

        Parameters
        ----------
        lower : float, optional
            Lower frequency to evaluate the integrand.

        upper : float, optional
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
            The integrated flux value in the fast cooling regime (D) with
            units of erg / s / cm2.
        """
        if self.cf == 0.0 or self.sf == 0.0:
            return np.nan

        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        eq1 = self.pf * (2 / (2 - self.p))
        eq2 = (self.sf / self.cf) ** -0.5
        eq3 = (1 / self.sf) ** (-self.p / 2)
        eq4 = (upper ** ((2 - self.p) / 2)) - (lower ** ((2 - self.p) / 2))

        return self._return(eq1 * eq2 * eq3 * eq4, self.units, obj)

    def get_segment_bc_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Calculates a flux for the Sari & Piran a fast cooling segments BC.

        Parameters
        ----------
        lower : float, optional
            Lower frequency to evaluate the integrand.

        upper : float, optional
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        float
            The integrated flux value for the Sari & Piran a fast cooling
            segments BC with units of erg / s / cm2.
        """
        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        return self._return(
            self.get_segment_b_flux(lower, self.cf) +
            self.get_segment_c_flux(self.cf, upper),
            FluxUnits.CGS,
            obj
        )

    def get_segment_bcd_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Calculates a flux for the Sari & Piran a fast cooling segments BCD.

        Parameters
        ----------
        lower : float, optional
            Lower frequency to evaluate the integrand.

        upper : float, optional
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        float
            The integrated flux value for the Sari & Piran a fast cooling
            segments BCD with units of erg / s / cm2.
        """
        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        return self._return(
            self.get_segment_bc_flux(lower, self.sf) +
            self.get_segment_d_flux(self.sf, upper),
            FluxUnits.CGS,
            obj
        )

    def get_segment_cd_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Calculates a flux for the Sari & Piran a fast cooling segments CD.

        Parameters
        ----------
        lower : float, optional
            Lower frequency to evaluate the integrand.

        upper : float, optional
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        float
            The integrated flux value for the Sari & Piran a fast cooling
            segments CD with units of erg / s / cm2.
        """
        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        return self._return(
            self.get_segment_c_flux(lower, self.sf) +
            self.get_segment_d_flux(self.sf, upper),
            FluxUnits.CGS,
            obj
        )
    # </editor-fold>

    # <editor-fold desc="Slow Cooling">
    def get_segment_f_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Calculates a flux for the Sari & Piran a slow cooling segment F.

        Parameters
        ----------
        lower : float, optional, default: self.lower
            Lower frequency to evaluate the integrand.

        upper : float, optional, default: self.upper
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        float
            The integrated flux value for the Sari & Piran a slow cooling
            segment F with units of erg / s / cm2.
        """
        if self.sf == 0.0:
            return np.nan

        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        eq1 = (3 / 4) * self.pf
        eq2 = (1 / self.sf) ** (1 / 3)
        eq3 = (upper ** (4 / 3)) - (lower ** (4 / 3))

        return self._return(eq1 * eq2 * eq3, self.units, obj)

    def get_segment_g_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Calculates a flux for the Sari & Piran a slow cooling segment G.

        Parameters
        ----------
        lower : float, optional, default: self.lower
            Lower frequency to evaluate the integrand.

        upper : float, optional, default: self.upper
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        float
            The integrated flux value for the Sari & Piran a slow cooling
            segment G with units of erg / s / cm2.
        """
        if self.sf == 0.0:
            return np.nan

        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        eq1 = (2 / (3 - self.p)) * self.pf
        eq2 = (1 / self.sf) ** (-(self.p - 1) / 2)
        eq3 = (upper ** ((3 - self.p) / 2)) - (lower ** ((3 - self.p) / 2))

        return self._return(eq1 * eq2 * eq3, self.units, obj)

    def get_segment_h_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Calculates a flux for the Sari & Piran a slow cooling segment H.

        Parameters
        ----------
        lower : float, optional, default: self.lower
            Lower frequency to evaluate the integrand.

        upper : float, optional, default: self.upper
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        float
            The integrated flux value for the Sari & Piran a slow cooling
            segment H with units of erg / s / cm2.
        """
        if self.cf == 0.0 or self.sf == 0.0:
            return np.nan

        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        eq1 = self.pf * (2 / (2 - self.p))
        eq2 = (self.cf / self.sf) ** (-(self.p - 1) / 2)
        eq3 = (1 / self.cf) ** (-self.p / 2)
        eq4 = (upper ** ((2 - self.p) / 2)) - (lower ** ((2 - self.p) / 2))

        return self._return(eq1 * eq2 * eq3 * eq4, self.units, obj)

    def get_segment_fg_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Calculates a flux for the Sari & Piran a slow cooling segments FG.

        Parameters
        ----------
        lower : float, optional, default: self.lower
            Lower frequency to evaluate the integrand.

        upper : float, optional, default: self.upper
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        float
            The integrated flux value for the Sari & Piran a slow cooling
            segments FG with units of erg / s / cm2.
        """
        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        return self._return(
            self.get_segment_f_flux(lower, self.sf) +
            self.get_segment_g_flux(self.sf, upper),
            FluxUnits.CGS,
            obj
        )

    def get_segment_fgh_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Calculates a flux for the Sari & Piran a slow cooling segments FGH.

        Parameters
        ----------
        lower : float, optional, default: self.lower
            Lower frequency to evaluate the integrand.

        upper : float, optional, default: self.upper
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        float
            The integrated flux value for the Sari & Piran a slow cooling
            segments FGH with units of erg / s / cm2.
        """
        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        return self._return(
            self.get_segment_fg_flux(lower, self.cf) +
            self.get_segment_h_flux(self.cf, upper),
            FluxUnits.CGS,
            obj
        )

    def get_segment_gh_flux(self, lower: float = None, upper: float = None, obj: bool = False):
        """
        Calculates a flux for the Sari & Piran a slow cooling segments GH.

        Parameters
        ----------
        lower : float, optional, default: self.lower
            Lower frequency to evaluate the integrand.

        upper : float, optional, default: self.upper
            Upper frequency to evaluate the integrand.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        float
            The integrated flux value for the Sari & Piran a slow cooling
            segments GH with units of erg / s / cm2.
        """
        lower = lower if lower else self.lower
        upper = upper if upper else self.upper

        return self._return(
            self.get_segment_g_flux(lower, self.cf) +
            self.get_segment_h_flux(self.cf, upper),
            FluxUnits.CGS,
            obj
        )
    # </editor-fold>

    def _return(self, val: float, units: FluxUnits, obj: bool) -> IntegratedFluxValue | float:
        """
        Returns the modeled flux in the desired format.

        Parameters
        ----------
        val : float
            The flux value.

        units : FluxUnits
            The units of `val`. We don't want to use `self.units` because
            the integrate-by-parts methods will double convert the results.

        obj : bool
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        IntegratedFluxValue or float
            The modeled flux with units of erg / s / cm2.
        """
        if units == FluxUnits.MJY:
            val *= 1.0e-26

        if obj:
            return IntegratedFluxValue(
                value=val,
                lower=0.0,
                upper=0.0,
                frequency_range=(
                    self.lower,
                    self.upper
                ),
                units=FluxUnits.CGS
            )

        return val
