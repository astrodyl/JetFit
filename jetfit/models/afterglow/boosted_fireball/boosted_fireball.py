import numpy as np

from jetfit.core.defns.enums import FluxUnits
from jetfit.core.defns.evidence import Evidence
from jetfit.models.afterglow.boosted_fireball.parameters.parameters import BFModelParams
from jetfit.models.afterglow.boosted_fireball.hydro_sim.scaler import Scaler
from jetfit.models.afterglow.sari_piran.sari_piran import SariPiran


class BoostedFireball:
    """

    """
    def __init__(self):
        self.model = SariPiran
        self.scaler = Scaler()
        self.parameters = BFModelParams()

    def evaluate(
            self,
            evidence: Evidence,
            params: BFModelParams,
            obj : bool = False
    ) -> np.ndarray:
        """
        Evaluates the Boosted Fireball model for a set of parameters, theta.

        Parameters
        ----------
        evidence : Evidence
            The evidence object. Although the measured values are not needed
            to evaluate the model, this method needs to know which type of
            data that it needs to model (e.g., spectral vs. integrated flux).
            To prevent unnecessary computations that could slow down the MCMC
            sampling routine, it's quickest to simply pass the evidence object.

        params : BFModelParams
            The model parameter values.

        obj : bool, optional, default=False
            If ``True``, returns a ``SpectralFluxValue`` object. If
            ``False``, returns a float.

        Returns
        -------
        np.ndarray of float
            The modeled afterglow lightcurve.

        References
        ----------
        Duffell, MacFadyen (2013)
            https://ui.adsabs.harvard.edu/abs/2013ApJ...776L...9D/abstract
        """
        pfs, cfs, sfs = (
            self.scaler.scaled_characteristics(
                evidence.optimized_x,
                params
            )
        )

        if np.isnan(pfs[0]):
            return pfs

        return self.model(
            pfs, cfs, sfs,
            params.p,
            FluxUnits.MJY,
            params.ebv_milky_way,
            params.ebv_source_frame,
            params.redshift,
        ).evaluate(evidence.values, obj)
