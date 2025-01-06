import numpy as np
from numpy import ndarray

from jetfit.core.utils import paths
from jetfit.models.afterglow.boosted_fireball.parameters.parameters import BFModelParams
from jetfit.models.afterglow.boosted_fireball.hydro_sim.hydro_sim import HydroSimTable


class Scaler:
    """
    Applies the scaling relations [1]_ for the observing time, peak flux,
    synchrotron frequency, and cooling frequency.

    Given the tabulated characteristic spectral functions that are in the
    `HydroSimTable` class, the scaling relations are able to calculate the
    spectral parameters, peak flux, synchrotron frequency, and cooling
    frequency [2]_ which fully describe the spectral model of synchrotron
    emission.

    This reduces the original high-dimensional problem to a four-dimensional
    problem of observing time, peak flux, synchrotron frequency, and cooling
    frequency since the spectrum of synchrotron emission is a series of
    connected power laws [2]_, [3]_.

    Attributes
    ----------
    hydro_sim_table : HydroSimTable
        If the characteristic spectral functions have not already been
        loaded, __init__ calls the load() method before storing
        as a value.

    References
    ----------
    .. [1] Ryan et al. (2015)

        https://ui.adsabs.harvard.edu/abs/2015ApJ...799....3R/abstract

    .. [2] Sari & Piran (1998)

        https://ui.adsabs.harvard.edu/abs/1998ApJ...497L..17S/abstract

    .. [3] Wu et al. (2018)

        https://ui.adsabs.harvard.edu/abs/2018ApJ...869...55W/abstract
    """
    def __init__(self, hydro_sim_table: HydroSimTable = None):

        if hydro_sim_table is None:
            hydro_sim_table = HydroSimTable(
                paths.get_hydro_sim_table_path()
            )

        self.hydro_sim_table = hydro_sim_table

    def scaled_characteristics(self, times: np.ndarray, p: BFModelParams) -> tuple:
        """
        Applies the scaling relation to the times array as defined
        in equation (4) of `GAMMA RAY BURSTS ARE OBSERVED OFF-AXIS`
        (Ryan et al., 2015).

        Parameters
        ----------
        times : np.ndarray
            Observation times measured in time since the GRB trigger.

        p : BFModelParams
            The model parameter values.

        Returns
        -------
        tuple of np.ndarray, with shapes ??
            Spectral function values corresponding to sampled params.
        """
        # HydroSimTable stores time in natural log scale.
        position = np.array([
            [np.log(tau), p.eta, p.gamma_b, p.theta]
            for tau in self.scale_times(times, p)
        ])

        spectral_functions = (
            self.hydro_sim_table.get_combined_characteristics(position)
        )

        try:
            peak_fluxes, cooling_frequencies, synchrotron_frequencies = (
                spectral_functions[:, 0], spectral_functions[:, 1], spectral_functions[:, 2]
            )
        except IndexError:
            nans = np.full(len(position), np.nan)
            return nans, nans, nans

        if np.isnan(peak_fluxes[0]):
            return peak_fluxes, cooling_frequencies, synchrotron_frequencies

        return (peak_fluxes * self.get_peak_scale(p),
                cooling_frequencies * self.get_cooling_scale(p),
                synchrotron_frequencies * self.get_synchrotron_scale(p))

    @staticmethod
    def scale_times(times: np.ndarray, p: BFModelParams) -> ndarray:
        """
        Applies the scaling relation to the times array as defined
        in equation (3) of `GAMMA RAY BURSTS ARE OBSERVED OFF-AXIS`
        (Ryan et al., 2015).

        Parameters
        ----------
        times : np.ndarray
            Observation times measured in time since the GRB trigger.

        p : BFModelParams

        Returns
        -------
        np.ndarray
            The scaled times.
        """
        return times * ((p.n / p.E) ** (1 / 3)) / (1 + p.z)

    @staticmethod
    def get_peak_scale(p: BFModelParams) -> float:
        """
        Calculates and returns the scaling factor for the peak fluxes.

        Returns
        -------
        float
            Scaling factor for the peak flux.
        """
        eq1 = (1 + p.z) / (p.d ** 2)
        eq2 = (p.p - 1) / (3 * p.p - 1)
        eq3 = p.E * (p.n ** 0.5) * (p.eps_b ** 0.5)
        eq4 = p.zeta

        return eq1 * eq2 * eq3 * eq4

    @staticmethod
    def get_cooling_scale(p: BFModelParams) -> float:
        """
        Calculates and returns the scaling factor for the cooling
        frequencies.

        Returns
        -------
        float
            Scaling factor for the cooling frequencies.
        """
        eq1 = 1 / (1 + p.z)
        eq2 = (p.E ** (-2 / 3)) * (p.n ** (-5 / 6))
        eq3 = p.eps_b ** (-3 / 2)

        return eq1 * eq2 * eq3

    @staticmethod
    def get_synchrotron_scale(p: BFModelParams) -> float:
        """
        Calculates and returns the scaling factor for the synchrotron
        frequencies.

        Returns
        -------
        float
            Scaling factor for the synchrotron frequencies.
        """
        eq1 = 1 / (1 + p.z)
        eq2 = ((p.p - 2) / (p.p - 1)) ** 2
        eq3 = (p.n ** 0.5) * (p.eps_e ** 2)
        eq4 = (p.eps_b ** 0.5) * (p.zeta ** -2)

        return eq1 * eq2 * eq3 * eq4
