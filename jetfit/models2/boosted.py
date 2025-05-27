import numpy as np

from jetfit.core.input import Observation
from jetfit.models2.basemodels import ObservedSpectrumModel, JetBreakModel, SpectralFluxModel, IntegratedFluxModel, \
    SpectralIndexModel


class BoostedFireballModel:
    """
    """
    k = 0.0

    # noinspection PyPep8Naming
    def __init__(
            self, E, n0, gamma_b, eta, p, zeta, eps_e, eps_b, z, theta_obs, dL28,
            tj=None, sj=None, sji=None, hydro_sim_table=None
    ):

        # Hydrodynamic Parameters
        self.E50 = E
        self.n0 = n0
        self.gamma_b = gamma_b
        self.eta = eta

        # Radiation Parameters
        self.p = p
        self.zeta = zeta
        self.eps_e = eps_e
        self.eps_b = eps_b

        # Observational Parameters
        self.theta_obs = theta_obs
        self.dL28 = dL28
        self.z = z

        # Jet Break Parameters
        self.tj = tj
        self.sj = (sj or 1 / sji) if (sj or sji) else None

        self.hydro_sim_table = hydro_sim_table

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
        eq3 = self.E50 * (self.n0 ** 0.5) * (self.eps_b ** 0.5)
        eq4 = self.zeta

        # x =  self.zeta * (
        #     (1 + self.z) / (self.dL28 ** 2) *
        #     (self.p - 1) / (3 * self.p - 1) *
        #     self.E50 * (self.n0 ** 0.5) * (self.eps_b ** 0.5)
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
        eq2 = (self.E50 ** (-2 / 3)) * (self.n0 ** -(5 / 6))
        eq3 = self.eps_b ** (-3 / 2)

        # y = (
        #     self.eps_b ** (-3 / 2) * self.E50 ** (-2 / 3) * self.n0 ** -(5 / 6)
        # ) / (1 + self.z)

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
        eq3 = (self.n0 ** 0.5) * (self.eps_e ** 2)
        eq4 = (self.eps_b ** 0.5) * (self.zeta ** -2)

        # z = (
        #     ((self.p - 2) / (self.p - 1)) ** 2 *
        #     self.n0 ** 0.5 * self.eps_e ** 2 *
        #     self.eps_b ** 0.5 * self.zeta ** -2
        # ) / (1 + self.z)

        return eq1 * eq2 * eq3 * eq4

    @property
    def is_valid(self) -> bool:
        """ Whether the model is parameters are valid. """
        # Smoothing parameters are unstable around 0
        if self.sj is not None and abs(self.sj) <= 0.1:
            return False
        return (self.eps_b + self.eps_e) < 1.0

    def model(self, obs: Observation, subset = None):
        """
        Models an ``Observation`` object.

        Parameters
        ----------
        obs : Observation
            The observation object to model.

        subset : np.ndarray of bool, optional
            The truth array of which values to model.

        Returns
        -------
        np.ndarray of float
            The unextinguished modeled flux.
        """
        if not self.is_valid:
            return np.array([np.nan])

        jet_break = self.jet_break(obs.as_arrays.times)

        return ObservedSpectrumModel(**self.spectrum(obs.as_arrays.times),
            arrays=obs.as_arrays, jet=jet_break, sharp=True
        ).model(subset)

    def spectrum(self, t):
        """
        Returns the characteristics that define a GRB spectrum.

        Parameters
        ----------
        t : np.ndarray of float or float
            The observer time [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        f_pk, nu_c, nu_m = self.scaled_characteristics(t)

        return {
            'p': self.p, 'k': 0.0,
            'f_peak': f_pk, 'nu_m': nu_m, 'nu_c': nu_c,
        }

    def jet_break(self, t):
        """
        Jet break model.

        Parameters
        ----------
        t : np.ndarray of float
            The observer times [d] used to smooth the break.

        Returns
        -------
        JetBreakModel
        """
        if self.tj is not None and self.sj is not None:
            return JetBreakModel(SpectralFluxModel(
                **self.spectrum(self.tj)), self.tj, t, self.p, self.sj)

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
        return (86_400 * times) * ((self.n0 / self.E50) ** (1 / 3)) / (1 + self.z)

    def f_peak(self, t):
        """ Return the scaled peak fluxes. """
        pos = np.array([
            [np.log(tau), self.eta, self.gamma_b, self.theta_obs]
            for tau in self.scale_times(t)
        ])
        return self.hydro_sim_table.get_peak_fluxes_at(pos)

    def nu_m(self, t):
        """ Return the scaled synchrotron frequencies. """
        pos = np.array([
            [np.log(tau), self.eta, self.gamma_b, self.theta_obs]
            for tau in self.scale_times(t)
        ])
        return self.hydro_sim_table.get_synchrotron_frequencies_at(pos)

    def nu_c(self, t):
        """ Return the scaled cooling frequencies. """
        pos = np.array([
            [np.log(tau), self.eta, self.gamma_b, self.theta_obs]
            for tau in self.scale_times(t)
        ])
        return self.hydro_sim_table.get_cooling_frequencies_at(pos)

    def nu_a(self, *args, **kwargs):
        """"""
        return None

    def spectral_flux(self, t, f, fts=False):
        """
        """
        return SpectralFluxModel(**self.spectrum(t)).evaluate(
            f, False, self.jet_break(t), sharp=True
        )

    def integrated_flux(self, t, lower, upper, fts=False):
        """
        """
        return IntegratedFluxModel(**self.spectrum(t)).evaluate(
            lower, upper, False, self.jet_break(t), sharp=True
        )

    def spectral_index(self, t, lower, upper, fts=False):
        """
        """
        return SpectralIndexModel(**self.spectrum(t)).evaluate(
            lower, upper, False, self.jet_break(t), sharp=True
        )

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
            [np.log(tau), self.eta, self.gamma_b, self.theta_obs]
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
