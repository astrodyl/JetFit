import numpy as np

from jetfit.core.defns.enums import FluxType, IndexType, FluxUnits
from jetfit.core.values.flux import IntegratedFluxValue, SpectralFluxValue
from jetfit.models.data.flux.integrated.spif_model import IntegratedFluxModel
from jetfit.models.data.flux.spectral.spsf_model import SpectralFluxModel


class SariPiran:
    """
    Wrapper class that provides an easy way to evaluate Sari & Piran models.

    Sari & Piran model describes the synchrotron emission model from a
    decelerating relativistic shell that collides with an external medium
    [1]_.

    Attributes
    ----------
    pfs : np.ndarray of float
        The peak fluxes.

    cfs : np.ndarray of float
        The cooling frequencies.

    sfs : np.ndarray of float
        The synchrotron frequencies.

    p : float
        The electron energy index.

    units : FluxUnits
        The units of ``pfs``. Assumes that all elements in ``pfs`` have the
        same units.

    ebv_milky_way : float
        The milky way E(B - V) strength.

    ebv_source_frame : float
        The source frame E(B - V) strength.

    redshift : float
        The redshift of the event. The redshift is used to transform the
        observed frequency into the source frame for the extinction
        calculation.

    References
    ----------
    .. [1] Sari & Piran (1998)
        https://ui.adsabs.harvard.edu/abs/1998ApJ...497L..17S/abstract

    See Also
    --------
    jetfit.models.data.flux.spectral.spsf_model.SpectralFluxModel :
        The Sari & Piran analytic spectral flux (flux density) model.

    jetfit.models.data.flux.integrated.spif_model.IntegratedFluxModel :
        The Sari & Piran analytic integrated flux model.
    """
    def __init__(
            self,
            pfs: np.ndarray[float],
            cfs: np.ndarray[float],
            sfs: np.ndarray[float],
            p: float,
            units: FluxUnits,
            ebv_milky_way: float = 0.0,
            ebv_source_frame: float = 0.0,
            redshift: float = 0.0,
    ):
        if isinstance(pfs, np.ndarray):
            if len(pfs) != len(cfs) != len(sfs):
                raise ValueError('Mismatch in spectral function shapes.')

        self.pfs = pfs
        self.cfs = cfs
        self.sfs = sfs
        self.p = p

        self.units = units
        self.ebv_milky_way = ebv_milky_way
        self.ebv_source_frame = ebv_source_frame
        self.redshift = redshift

    def evaluate(self, data: list) -> np.ndarray:
        """
        Evaluates the Sari & Piran models for the corresponding data types.

        Initializes a numpy array with `np.nan` values. If a nan is modeled,
        the method will not evaluate the remaining the points. Instead, it
        will return the array in its current state with nans. This is done
        to speed up calculations when used with MCMC sampling since a single
        nan value will cause the log likelihood to be negative infinity.

        Parameters
        ----------
        data : list of Measurement

        Returns
        -------
        np.ndarray of float
            The modeled flux values.
        """
        res, o = np.full(len(data), np.nan), 0

        for i, ev in enumerate(data):

            if ev.y.type == FluxType.SPECTRAL:
                res[i] = self.spectral_flux(ev.y.frequency, i + o)

            elif ev.y.type == FluxType.INTEGRATED:
                res[i] = self.integrated_flux(ev.y.frequency_range, i + o)

            elif ev.y.type == IndexType.SPECTRAL:
                res[i] = self.spectral_index(ev.y.frequency_range, i + o)
                o += 1

            if np.isnan(res[i]):
                return res

        return res

    def spectral_flux(
            self,
            frequency: float,
            i: int,
            obj: bool = False
    ) -> SpectralFluxValue | float:
        """
        Models the spectral flux using the spectral function object attributes
        at the given index.

        Parameters
        ----------
        frequency : float
            The average band frequency.

        i : int
            The index to use for the spectral functions.

        obj : bool, optional
            If ``True``, returns a ``SpectralFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        SpectralFluxValue or float
            The modeled spectral flux.
        """
        return SpectralFluxModel(
            self.pfs[i],
            self.cfs[i],
            self.sfs[i],
            self.p,
            frequency,
            self.units,
            self.ebv_milky_way,
            self.ebv_source_frame,
            self.redshift
        ).evaluate(obj)

    def integrated_flux(
            self,
            int_range: tuple,
            i: int,
            obj: bool = False
    ) -> IntegratedFluxValue | float:
        """
        Models the spectral flux using the spectral function object attributes
        at the given index.

        Parameters
        ----------
        int_range : tuple of float
            The frequency range to integrate.

        i : int
            The index to use for the spectral functions.

        obj : bool, optional
            If ``True``, returns a ``IntegratedFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        IntegratedFluxValue or float
            The modeled spectral flux.
        """
        return IntegratedFluxModel(
            self.pfs[i], self.cfs[i], self.sfs[i], self.p, int_range[0], int_range[1], self.units
        ).evaluate(obj)

    def spectral_index(self, int_range: tuple, i: int) -> float:
        """
        Models the spectral index using a two point approximation.

        Parameters
        ----------
        int_range : tuple of float
            The frequency range to integrate.

        i : int
            The index to use for the spectral functions.

        Returns
        -------
        float
            The modeled spectral index.
        """
        y2 = IntegratedFluxModel(
            self.pfs[i + 1], self.cfs[i + 1], self.sfs[i + 1], self.p, int_range[0], int_range[1], self.units
        ).evaluate()

        y1 = IntegratedFluxModel(
            self.pfs[i], self.cfs[i], self.sfs[i], self.p, int_range[0], int_range[1], self.units
        ).evaluate()

        return np.log(y2 / y1) / np.log(int_range[1] / int_range[0])
