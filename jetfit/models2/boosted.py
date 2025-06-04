import numpy as np

from jetfit.core.input import Observation
from jetfit.models2.basemodels import ObservedSpectrumModel, SpectralFluxModel
from jetfit.models2.basemodels import IntegratedFluxModel, SpectralIndexModel


class BoostedFireballModel:
    """
    A two parameter, physically motivated off-axis afterglow model.

    The Boosted Fireball model [1]_ assumes that the blast-wave
    is self-similar and propagating through a constant density
    medium.

    Attributes
    ----------
    E50 : float
        The explosion energy. Normalized to 1e50 ergs.

    n0 : float
        The circumburst number density [cm-3].

    gamma_b : float
        The Lorentz boost factor.

    eta : float
        The specific internal energy.

    p : float
        The electron energy index.

    zeta : float
        The fraction of electrons accelerated by the shock.
        Must be in the range [0, 1].

    eps_e : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    eps_b : float
        The fraction of thermal energy carried by relativistic
        electrons. Must be in the range [0, 1].

    theta_obs : float
        The viewing angle [rad]. Must be in the range [0, 1].

    dL28 : float
        The luminosity distance corresponding the redshift, ``z``.
        Normalized to 1e28 cm.

    z : float
        The redshift.

    hydro_sim_table : HydroSimTable
        The tabulated results from the numerical simulations.

    References
    ----------
    .. [1] A "Boosted Fireball" Model for Structured Relativistic Jets.
        https://ui.adsabs.harvard.edu/abs/2013ApJ...776L...9D/abstract
    """
    k = 0.0
    sharp = True

    # noinspection PyPep8Naming
    def __init__(
        self, E, n0, gamma_b, eta, p, zeta, eps_e, eps_b, z, theta_obs, dL28, hydro_sim_table
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

        self.hydro_sim_table = hydro_sim_table

    @property
    def is_valid(self) -> bool:
        """ Are the model parameters physically valid? """
        return (self.eps_b + self.eps_e) < 1.0 and self.p >= 2

    @property
    def peak_scale(self) -> float:
        """ The scaling factor for the peak flux. """
        return self.zeta * (
            (1 + self.z) / (self.dL28 ** 2) *
            (self.p - 1) / (3 * self.p - 1) *
            self.E50 * (self.n0 ** 0.5) * (self.eps_b ** 0.5)
        )

    @property
    def cooling_scale(self) -> float:
        """ The scaling factor for the cooling frequencies. """
        return (
            self.eps_b ** (-3 / 2) *
            self.E50 ** (-2 / 3) *
            self.n0 ** -(5 / 6)
        ) / (1 + self.z)

    @property
    def synchrotron_scale(self) -> float:
        """ The scaling factor for the synchrotron frequencies. """
        return (
            ((self.p - 2) / (self.p - 1)) ** 2 *
            self.n0 ** 0.5 * self.eps_e ** 2 *
            self.eps_b ** 0.5 * self.zeta ** -2
        ) / (1 + self.z)

    def scale_times(self, t):
        """
        Scales the observer time(s).

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer times.

        Returns
        -------
        np.ndarray of float
            The scaled source-frame times with units as ``t``.

        References
        ----------
        .. [1] GAMMA RAY BURSTS ARE OBSERVED OFF-AXIS (Ryan et al. 2015)
            https://ui.adsabs.harvard.edu/abs/2015ApJ...799....3R/abstract
        """
        return t * ((self.n0 / self.E50) ** (1 / 3)) / (1 + self.z)

    def model(self, obs: Observation, subset = None):
        """
        Models an ``Observation`` object.

        Parameters
        ----------
        obs : Observation
            The ``Observation`` to model.

        subset : np.ndarray of bool, optional
            The truth array of which values to model.

        Returns
        -------
        np.ndarray of float
            The modeled observational data.
        """
        if not self.is_valid:
            return np.array([np.nan])

        spectrum = self.spectrum(obs.as_arrays.times)

        if np.isnan(spectrum.get('f_peak').min()):
            return np.array([np.nan])

        return ObservedSpectrumModel(
            **spectrum, arrays=obs.as_arrays, sharp=self.sharp
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
        f_pk, nu_c, nu_m = self.scaled_csf(t)

        return {
            'p': self.p, 'k': 0.0,
            'f_peak': f_pk, 'nu_m': nu_m, 'nu_c': nu_c,
        }

    def f_peak(self, t, scale=True):
        """
        Retrieves the peak flux value(s) from the numerical
        simulation table.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        scale : bool, optional, default=True
            Should the value(s) be scaled?

        Returns
        -------
        float or np.ndarray of float
            The peak flux value(s) [mJy] at time(s) ``t``.
        """
        f_pk = self.hydro_sim_table.get_peak_fluxes_at(
            np.array(
                [[np.log(tau), self.eta, self.gamma_b, self.theta_obs]
                for tau in self.scale_times(t * 86_400)]
            )
        )
        return f_pk * self.peak_scale if scale else f_pk

    def nu_m(self, t, scale=True):
        """
        Retrieves the synchrotron frequency value(s) from
        the numerical simulation table.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        scale : bool, optional, default=True
            Should the value(s) be scaled?

        Returns
        -------
        float or np.ndarray of float
            The synchrotron frequency value(s) [Hz] at time(s) ``t``.
        """
        nu_m = self.hydro_sim_table.get_synchrotron_frequencies_at(
            np.array(
                [[np.log(tau), self.eta, self.gamma_b, self.theta_obs]
                 for tau in self.scale_times(t * 86_400)]
            )
        )
        return nu_m * self.synchrotron_scale if scale else nu_m

    def nu_c(self, t, scale=True):
        """
        Retrieves the cooling frequency value(s) from
        the numerical simulation table.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        scale : bool, optional, default=True
            Should the value(s) be scaled?

        Returns
        -------
        float or np.ndarray of float
            The cooling frequency value(s) [Hz] at time(s) ``t``.
        """
        nu_c = self.hydro_sim_table.get_cooling_frequencies_at(
            np.array(
                [[np.log(tau), self.eta, self.gamma_b, self.theta_obs]
                 for tau in self.scale_times(t * 86_400)]
            )
        )
        return nu_c * self.cooling_scale if scale else nu_c

    @staticmethod
    def nu_a(*args, **kwargs):
        """
        Dummy method. The numerical simulation table does not
        store the self-absorption frequency.
        """
        return None

    def spectral_flux(self, t, f, fts=False):
        """
        Calculates the spectral fluxes at times ``t`` for the
        frequencies ``f``.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer times [d].

        f : float or np.ndarray of float
            The average band frequencies [Hz].

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        Returns
        -------
        float np.ndarray of float
            The modeled spectral flux [mJy].

        See Also
        --------
        `models2.basemodels.SpectralFluxModel.evaluate`
            See for information on how various shapes
            of t and f are handled.
        """
        return SpectralFluxModel(**self.spectrum(t)).evaluate(
            f, fts=fts, sharp=self.sharp
        )

    def integrated_flux(self, t, lower, upper, fts=False):
        """
        Calculates the integrated fluxes at times ``t`` for the
        lower and upper integration bounds, ``lower`` and ``upper``.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer times [d].

        lower, upper : float or np.ndarray of float
            The integration bounds [Hz].

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        Returns
        -------
        float np.ndarray of float
            The modeled spectral flux [erg cm-2 s-1].

        See Also
        --------
        `models2.basemodels.SpectralFluxModel.evaluate`
            See for information on how various shapes
            of t, lower, upper are handled.
        """
        return IntegratedFluxModel(**self.spectrum(t)).evaluate(
            lower, upper, fts=fts, sharp=self.sharp
        )

    def spectral_index(self, t, lower, upper, fts=False):
        """
        Calculates the spectral index at times ``t`` for the
        lower and upper integration bounds, ``lower`` and ``upper``.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer times [d].

        lower, upper : float or np.ndarray of float
            The integration bounds [Hz].

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        Returns
        -------
        float np.ndarray of float
            The modeled spectral index.

        See Also
        --------
        `models2.basemodels.SpectralFluxModel.evaluate`
            See for information on how various shapes
            of t, lower, upper are handled.
        """
        return SpectralIndexModel(**self.spectrum(t)).evaluate(
            lower, upper, fts=fts, sharp=self.sharp
        )

    def scaled_csf(self, t, scale=True) -> tuple:
        """
        Returns the scaled characteristic spectral functions.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer times [d].

        scale : bool, optional, default=True
            Should the value(s) be scaled?

        Returns
        -------
        tuple of np.ndarray of float
            The characteristic spectral functions.
        """
        position = np.array([
            [np.log(tau), self.eta, self.gamma_b, self.theta_obs]
            for tau in self.scale_times(t * 86_400)
        ])

        spectral_functions = (
            self.hydro_sim_table.get_combined_characteristics(position)
        )

        try:
            f_pk, nu_c, nu_m = (
                spectral_functions[:, 0],
                spectral_functions[:, 1],
                spectral_functions[:, 2]
            )
        except IndexError:
            nans = np.full(len(position), np.nan)
            return nans, nans, nans

        if np.isnan(f_pk.min()):
            return f_pk, nu_c, nu_m

        if scale:
            return (
                f_pk * self.peak_scale,
                nu_c * self.cooling_scale,
                nu_m * self.synchrotron_scale
            )
        else:
            return f_pk, nu_c, nu_m
