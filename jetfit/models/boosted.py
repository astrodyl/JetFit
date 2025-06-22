import h5py
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator as RGInterpolator

from jetfit.core.structs import ScaleType
from jetfit.core.input import Observation
from jetfit.models.basemodels import ObservedSpectrumModel, SpectralFluxModel
from jetfit.models.basemodels import IntegratedFluxModel, SpectralIndexModel


def nans(n: int):
    return np.array([[np.nan, np.nan, np.nan] for _ in range(n)])


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
        The asymptotic Lorentz factor.

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
        t : np.ndarray of float or float
            The observer time [d].

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
            The observational data.

        subset : np.ndarray of bool, optional
            The subset of data to model.

        Returns
        -------
        np.ndarray of float
            The modeled observational data.
        """
        if not self.is_valid:
            return np.array([np.nan])

        spectrum = self.spectrum(obs.times())

        if np.isnan(spectrum.get('f_peak').min()):
            return np.array([np.nan])

        return ObservedSpectrumModel(
            **spectrum, arrays=obs.as_arrays, sharp=self.sharp
        ).model(subset)

    def spectrum(self, t):
        """
        Returns the characteristics that define the GRB spectrum.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

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
        return self.scaled_csf(t)[0]

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
        return self.scaled_csf(t)[2]

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
        return self.scaled_csf(t)[1]

    @staticmethod
    def nu_a(*args, **kwargs):
        """
        Dummy method. The numerical simulation table does not
        store the self-absorption frequency.
        """
        return None

    def spectral_flux(self, t, nu, fts=False):
        """
        Calculates the spectral fluxes at times ``t`` for
        the frequencies ``nu``.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer times [d].

        nu : float or np.ndarray of float
            The average band frequencies [Hz].

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        Returns
        -------
        float or np.ndarray of float
            The modeled spectral flux [mJy].

        See Also
        --------
        `models.basemodels.SpectralFluxModel.evaluate`
            See for information on how various shapes
            of t and f are handled.
        """
        return SpectralFluxModel(**self.spectrum(t)).evaluate(
            nu, fts=fts, sharp=self.sharp
        )

    def integrated_flux(self, t, lower, upper, fts=False):
        """
        Calculates the integrated fluxes at times ``t`` for
        the integration bounds, ``lower`` and ``upper``.

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
        float or np.ndarray of float
            The modeled spectral flux [erg cm-2 s-1].

        See Also
        --------
        `models.basemodels.SpectralFluxModel.evaluate`
            See for information on how various shapes
            of t, lower, upper are handled.
        """
        return IntegratedFluxModel(**self.spectrum(t)).evaluate(
            lower, upper, fts=fts, sharp=self.sharp
        )

    def spectral_index(self, t, lower, upper, fts=False):
        """
        Calculates the spectral index at times ``t`` for
        the integration bounds, ``lower`` and ``upper``.

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
        float or np.ndarray of float
            The modeled spectral index.

        See Also
        --------
        `models.basemodels.SpectralFluxModel.evaluate`
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
            self.hydro_sim_table.get_characteristics(position)
        )

        f_pk, nu_c, nu_m = (
            spectral_functions[:, 0],
            spectral_functions[:, 1],
            spectral_functions[:, 2],
        )

        if np.isnan(f_pk.min()):
            return f_pk, nu_c, nu_m

        if scale:
            f_pk = f_pk * self.peak_scale
            nu_c = nu_c * self.cooling_scale
            nu_m = nu_m * self.synchrotron_scale

        return f_pk, nu_c, nu_m


class HydroSimTable:
    """
    A Hydrodynamic Simulations Table.

    Attributes
    ----------
    path : str or Path
        Path to the Boosted Fireball hydrodynamic simulation table
        that is described in [1]_.

    peak_fluxes : `scipy.interpolate.RegularGridInterpolator`
        Rectangular grid of peak flux values.

    cooling_frequencies : `scipy.interpolate.RegularGridInterpolator`
        Rectangular grid of cooling frequency values.

    synchrotron_frequencies : `scipy.interpolate.RegularGridInterpolator`
        Rectangular grid of synchrotron frequency values.

    char_spec_params : dict

    References
    ----------
    .. [1] Y. Wu & A. MacFadyen, "Constraining the Outflow Structure
        of the Binary Neutron Star Merger Event GW170817/GRB170817A
        with a Markov Chain Monte Carlo Analysis," The Astrophysical
        Journal, vol. 869, pp. 55-65, 2018.
    """
    spectral_scale = ScaleType.LN
    spectral_axes = ('f_peak', 'f_nu_c', 'f_nu_m')

    spectrum_axes = ('tau', 'Eta0', 'GammaB', 'theta_obs')
    spectrum_scales = {}

    def __init__(self, path: str | Path, load: bool = True):
        """

        Parameters
        ----------
        load : bool, optional
            If `True`, loads the table upon instantiation. If `False`,
            the table can be loaded at a later time by calling the
            load() method.
        """
        self.path = path
        self.peak_fluxes = None
        self.cooling_frequencies = None
        self.synchrotron_frequencies = None
        self.combined_functions = None

        self.char_spec_params = {}

        if load:
            self.load()

    def load(self) -> None:
        """ Loads the hydrodynamic simulation table. """
        char_spec_funcs = {}

        with h5py.File(self.path, 'r') as table:

            for key in table.keys():

                if key in self.spectral_axes:
                    char_spec_funcs[key] = np.ma.log(table[key][...]).filled(-np.inf)

                elif key == 'tau':
                    self.char_spec_params[key] = np.log(table[key][...])
                    self.spectrum_scales[key] = ScaleType.LN

                elif key in self.spectrum_axes:
                    self.char_spec_params[key] = table[key][...]
                    self.spectrum_scales[key] = ScaleType.LINEAR

        char_spec_params_list = [
            self.char_spec_params[axis] for axis in self.spectrum_axes
        ]

        combined = np.stack((
            char_spec_funcs['f_peak'],
            char_spec_funcs['f_nu_c'],
            char_spec_funcs['f_nu_m']),
            axis=-1
        )
        self.combined_functions = RGInterpolator(char_spec_params_list, combined)
        self.peak_fluxes = RGInterpolator(char_spec_params_list, char_spec_funcs['f_peak'])
        self.cooling_frequencies = RGInterpolator(char_spec_params_list, char_spec_funcs['f_nu_c'])
        self.synchrotron_frequencies = RGInterpolator(char_spec_params_list, char_spec_funcs['f_nu_m'])

    def get_characteristics(self, position):
        """"""
        return self.get_characteristic_at(position, self.combined_functions)

    def get_characteristic_at(self, pos: np.ndarray, func, ex=True):
        """
        Calls the provided ``interpolate.RegularGridInterpolator``
        ``func`` and returns the characteristic spectral function
        at the provided ``position``.

        Parameters
        ----------
        pos : np.ndarray of float

        func : `scipy.interpolate.RegularGridInterpolator`

        ex : bool, optional, default=True
            Should values be extrapolated?

        Returns
        -------
        np.ndarray
        """
        try:
            return np.exp(func(pos))

        except ValueError:
            if not ex: return nans(len(pos))

            # Attempted to interpolate outside the grid
            taus = self.char_spec_params['tau']

            # Where do I need to extrapolate?
            early = pos[:, 0] < taus.min()
            late  = pos[:, 0] > taus.max()
            valid = ~early & ~late

            # Can't extrapolate with 1 point
            if len(pos[valid]) < 2:
                return nans(len(pos))

            csf = np.empty((len(pos), 3))

            # Check for -inf after each to prevent accessing
            # the interpolator which is slow.
            if valid.any():
                csf[valid] = func(pos[valid])

                if -np.inf in csf[valid]:
                    return nans(len(pos))

            if early.any():
                csf[early] = self.extrapolate(
                    pos[early], taus[0], taus[1], func, regime='early')

                if -np.inf in csf[early]:
                    return nans(len(pos))

            if late.any():
                csf[late] = self.extrapolate(
                    pos[late], taus[-2], taus[-1], func, regime='late')

                if -np.inf in csf[late]:
                    return nans(len(pos))

            return np.exp(csf)

    @staticmethod
    def extrapolate(pos, t_lower, t_upper, func, regime):
        """"""
        pos_lower = np.array(pos, copy=True)
        pos_upper = np.array(pos, copy=True)

        # Overwrite with outermost valid times
        pos_lower[:, 0] = t_lower
        pos_upper[:, 0] = t_upper

        # Interpolate
        y_lower = func(pos_lower)
        y_upper = func(pos_upper)

        # Calculate the slope
        m = (y_upper - y_lower) / (t_upper - t_lower)

        # Which direction am I extrapolating?
        y0 = y_lower if regime == 'early' else y_upper
        t0 = t_lower if regime == 'early' else t_upper

        # Extrapolate and return in ln space
        return np.array([
            y0[:, 0] + m[:, 0] * (pos[:, 0] - t0),  # f_pk
            y0[:, 1] + m[:, 1] * (pos[:, 0] - t0),  # nu_c
            y0[:, 2] + m[:, 2] * (pos[:, 0] - t0),  # nu_m
        ]).T
