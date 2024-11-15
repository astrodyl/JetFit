import numpy as np
from numpy import ndarray

from jetfit.core.values.time_value import TimeValue
from jetfit.model.parameters import ModelParameters
from jetfit.scalers.hydro_sim import HydroSimTable


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

    times : np.ndarray
        Unscaled observation times measured in time since the GRB trigger.

    Notes
    -----
    To accommodate parallelization, the `Scaler` class does not store the
    model parameters in the class. The parameters required to scale the light
    curves are sampled via the MCMC routine. Unfortunately, this makes the
    method arguments a bit cumbersome.

    Requires a list of TimeValues even though only an array of the floats is
    stored in the class. I do this for two reasons: (1) The time needs to be
    stored in seconds since the trigger. I don't trust the user to actually
    ensure that the units are correct. (2) Even if the user knows that they
    need to have the time measured in seconds, it's more convenient to just
    pass in the TimeValues and for get about it.

    References
    ----------
    (Ryan et al., 2015)
        https://ui.adsabs.harvard.edu/abs/2015ApJ...799....3R/abstract

    (Sari & Piran, 1998)
        https://ui.adsabs.harvard.edu/abs/1998ApJ...497L..17S/abstract

    (Wu et al., 2018)
        https://ui.adsabs.harvard.edu/abs/2018ApJ...869...55W/abstract
    """
    def __init__(self, hydro_sim_table: HydroSimTable, times: list[TimeValue]):

        if hydro_sim_table.peak_fluxes is None:
            hydro_sim_table.load()

        self.hydro_sim_table = hydro_sim_table
        self.times = np.array([t.get_value('seconds') for t in times])

    def scaled_characteristics(self, p: ModelParameters) -> tuple:
        """
        Applies the scaling relation to the times array as defined
        in equation (4) of `GAMMA RAY BURSTS ARE OBSERVED OFF-AXIS`
        (Ryan et al., 2015).

        Returns
        -------
        tuple of np.ndarray, with shapes ??
            Spectral function values corresponding to sampled params.
        """
        # HydroSimTable stores time in natural log scale.
        position = np.array([
            [
                np.log(tau),
                p.asymptotic_lorentz,
                p.boost,
                p.obs_angle
            ]
            for tau in self.scale_times(p)
        ])

        peak_fluxes, cooling_frequencies, synchrotron_frequencies = (
            self.hydro_sim_table.get_characteristics_at(position)
        )

        if np.isnan(peak_fluxes[0]):
            return peak_fluxes, cooling_frequencies, synchrotron_frequencies

        return (peak_fluxes * self.get_peak_scale(p),
                cooling_frequencies * self.get_cooling_scale(p),
                synchrotron_frequencies * self.get_synchrotron_scale(p))

    def scale_times(self, p: ModelParameters) -> ndarray:
        """
        Applies the scaling relation to the times array as defined
        in equation (3) of `GAMMA RAY BURSTS ARE OBSERVED OFF-AXIS`
        (Ryan et al., 2015).

        Returns
        -------
        np.ndarray
            Scaled times
        """
        return self.times * ((p.density / p.energy) ** (1 / 3)) / (1 + p.z)

    @staticmethod
    def get_peak_scale(p: ModelParameters) -> float:
        """
        Calculates and returns the scaling factor for the peak fluxes.

        Returns
        -------
        float
            Scaling factor for the peak flux.
        """
        eq1 = (1 + p.z) / (p.distance ** 2)
        eq2 = (p.electron_index - 1) / (3 * p.electron_index - 1)
        eq3 = p.energy * (p.density ** 0.5) * (p.magnetic_energy_fraction ** 0.5)
        eq4 = p.accel_electron_fraction

        return eq1 * eq2 * eq3 * eq4

    @staticmethod
    def get_cooling_scale(p: ModelParameters) -> float:
        """
        Calculates and returns the scaling factor for the cooling
        frequencies.

        Returns
        -------
        float
            Scaling factor for the cooling frequencies.
        """
        eq1 = 1 / (1 + p.z)
        eq2 = (p.energy ** (-2 / 3)) * (p.density ** (-5 / 6))
        eq3 = p.magnetic_energy_fraction ** (-3 / 2)

        return eq1 * eq2 * eq3

    @staticmethod
    def get_synchrotron_scale(p: ModelParameters) -> float:
        """
        Calculates and returns the scaling factor for the synchrotron
        frequencies.

        Returns
        -------
        float
            Scaling factor for the synchrotron frequencies.
        """
        eq1 = 1 / (1 + p.z)
        eq2 = ((p.electron_index - 2) / (p.electron_index - 1)) ** 2
        eq3 = (p.density ** 0.5) * (p.electron_energy_fraction ** 2)
        eq4 = (p.magnetic_energy_fraction ** 0.5) * (p.accel_electron_fraction ** -2)

        return eq1 * eq2 * eq3 * eq4
