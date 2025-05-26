import astropy.units as u
import numpy as np
from dust_extinction.parameter_averages import CCM89

from jetfit.core.defns.enums import DataType
from jetfit.core.input import Observation
from jetfit.models2.basemodels import SpectralFluxModel, IntegratedFluxModel, SpectralIndexModel


class BoostedFireballModel:
    """

    """

    # noinspection PyPep8Naming
    def __init__(
            self, E, n, boost, eta, p, zeta, eps_e, eps_b, z, obs_angle, dL,
            ebv_mw=None, ebv_sf=None, hydro_sim_table=None
    ):
        # Hydrodynamic Parameters
        self.E50 = E
        self.n = n
        self.boost = boost
        self.eta = eta

        # Radiation Parameters
        self.p = p
        self.zeta = zeta
        self.eps_e = eps_e
        self.eps_b = eps_b

        # Observational Parameters
        self.z = z
        self.obs_angle = obs_angle
        self.dL28 = dL

        self.hydro_sim_table = hydro_sim_table

        # temp
        self.ebv_mw = ebv_mw
        self.ebv_sf = ebv_sf
        self.ext_model = CCM89(Rv=3.1)

    # noinspection PyPep8Naming
    @property
    def E50(self) -> float:
        """ Returns the explosion energy normalized to 10e50 ergs. """
        return self._E50

    # noinspection PyPep8Naming
    @E50.setter
    def E50(self, e: float | u.Quantity) -> None:
        """
        Sets the explosion energy normalized to 10e50 ergs.

        Parameters
        ----------
        e : float or astropy.units.Quantity
            The explosion energy. If a float is provided, assumes
            that the value is already normalized to 1e50 ergs.
        """
        if isinstance(e, u.Quantity):
            e = e.to_value('erg') / 1e50

        self._E50 = e

    # noinspection PyPep8Naming
    @property
    def dL28(self) -> float:
        """ Returns the luminosity distance normalized to 1e28 cm. """
        return self._dL28

    # noinspection PyPep8Naming
    @dL28.setter
    def dL28(self, d: float | u.Quantity) -> None:
        """
        Sets the luminosity distance normalized to 1e28 cm.

        Parameters
        ----------
        d : float or astropy.units.Quantity
            The luminosity distance. If a float is provided,
            assumes the value is already normalized to 1e28 cm.
        """
        if isinstance(d, u.Quantity):
            d = d.to_value('cm') / 1e28

        self._dL28 = d

    @property
    def peak_scale(self) -> float:
        """
        Calculates and returns the scaling factor for the peak fluxes.

        Returns
        -------
        float
            Scaling factor for the peak flux.
        """
        eq1 = (1 + self.z) / (self.dL28 ** 2)
        eq2 = (self.p - 1) / (3 * self.p - 1)
        eq3 = self.E50 * (self.n ** 0.5) * (self.eps_b ** 0.5)
        eq4 = self.zeta

        # return self.zeta * (
        #     (1 + self.z) / (self.dL28 ** 2) *
        #     (self.p - 1) / (3 * self.p - 1) *
        #     self.E50 * (self.n ** 0.5) * (self.eps_b ** 0.5)
        # )

        return eq1 * eq2 * eq3 * eq4

    @property
    def cooling_scale(self) -> float:
        """
        Calculates and returns the scaling factor for the cooling
        frequencies.

        Returns
        -------
        float
            Scaling factor for the cooling frequencies.
        """
        eq1 = 1 / (1 + self.z)
        eq2 = (self.E50 ** (-2 / 3)) * (self.n ** -(5 / 6))
        eq3 = self.eps_b ** (-3 / 2)

        return eq1 * eq2 * eq3

    @property
    def synchrotron_scale(self) -> float:
        """
        Calculates and returns the scaling factor for the synchrotron
        frequencies.

        Returns
        -------
        float
            Scaling factor for the synchrotron frequencies.
        """
        eq1 = 1 / (1 + self.z)
        eq2 = ((self.p - 2) / (self.p - 1)) ** 2
        eq3 = (self.n ** 0.5) * (self.eps_e ** 2)
        eq4 = (self.eps_b ** 0.5) * (self.zeta ** -2)

        return eq1 * eq2 * eq3 * eq4

    def model(self, obs: Observation):
        """
        Models the observational data.

        Parameters
        ----------
        obs : Observation
            The `Observation` object to model.

        Returns
        -------
        np.ndarray
            The modeled observational data.
        """
        res = np.full(len(obs.data), np.nan)

        f_peaks, nu_cs, nu_ms = self.scaled_characteristics(obs.time_array[obs.flux_loc])

        if np.isnan(f_peaks.min()):
            return f_peaks

        # Model flux values
        for i in obs.flux_loc:
            nu_m, nu_c, f_peak = nu_ms[i], nu_cs[i], f_peaks[i]

            if obs.data[i].type == DataType.SPECTRAL_FLUX:
                res[i] = SpectralFluxModel(nu_m, nu_c, f_peak, self.p).model(obs.data[i])

            elif obs.data[i].type == DataType.INTEGRATED_FLUX:
                res[i] = IntegratedFluxModel(nu_m, nu_c, f_peak, self.p).model(obs.data[i])

            if np.isnan(res[i]):
                return res

        # Model spectral indices
        for i in obs.spectral_index_loc:
            res[i] = SpectralIndexModel(
                f1=res[np.argwhere(obs.time_array == obs.data[i].time_range.upper.value)],
                f2=res[np.argwhere(obs.time_array == obs.data[i].time_range.lower.value)]
            ).model(obs.data[i])

        # Apply extinction to spectral flux values
        if self.ebv_mw or self.ebv_sf:
            mask = obs.flux_types == DataType.SPECTRAL_FLUX
            wn = obs.wave_number_array[mask]

            if self.ebv_mw:  # milky way
                res[mask] *= self.ext_model.extinguish(wn, Ebv=self.ebv_mw)

            if self.ebv_sf:  # source frame
                res[mask] *= self.ext_model.extinguish((1 + self.z) * wn, Ebv=self.ebv_sf)

        return res

    def scale_times(self, times: np.ndarray) -> np.ndarray:
        """
        Applies the scaling relation to the times array as defined
        in equation (3) of `GAMMA RAY BURSTS ARE OBSERVED OFF-AXIS`
        (Ryan et al., 2015).

        Parameters
        ----------
        times : np.ndarray of float
            The observer times [d].

        Returns
        -------
        np.ndarray of float
            The scaled times.
        """
        return times * ((self.n / self.E50) ** (1 / 3)) / (1 + self.z)

    def scaled_characteristics(self, times: np.ndarray) -> tuple:
        """
        Applies the scaling relation to the times array as defined
        in equation (4) of `GAMMA RAY BURSTS ARE OBSERVED OFF-AXIS`
        (Ryan et al., 2015).

        Parameters
        ----------
        times : np.ndarray of float
            The observer times [d].

        Returns
        -------
        tuple of np.ndarray of float
            Spectral function values corresponding to sampled params.
        """
        # HydroSimTable stores time in natural log scale.
        position = np.array([
            [np.log(tau), self.eta, self.boost, self.obs_angle]
            for tau in self.scale_times(times)
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

        if np.isnan(peak_fluxes.min()):
            return peak_fluxes, cooling_frequencies, synchrotron_frequencies

        return (
            peak_fluxes * self.peak_scale,
            cooling_frequencies * self.cooling_scale,
            synchrotron_frequencies * self.synchrotron_scale
        )
