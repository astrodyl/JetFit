import numpy as np

from jetfit.core.utils import days_to_sec

try:
    from jetsimpy import Jet, Gaussian
except ImportError:
    pass


"""
Adapter for integrating JetSimPy [1]_ with AMPy's infrastructure.

This module wraps the ``jetsimpy`` package and adapts it for use
with AMPy's MCMC framework.

Notes
-----
The implementation supports ISM, wind, and ISM + wind density
environments.

Although ``jetsimpy`` is blazing fast considering it is a
numerical simulation codebase, it is somewhat slow for MCMC
fitting. It is recommended to use multiprocessing when using
this model. To make use of multithreading instead, the C++
source code will need to be modified and then recompiled.

References
----------
.. [1] jetsimpy: A Highly Efficient Hydrodynamic Code for
    Gamma-ray Burst Afterglow. https://arxiv.org/abs/2402.19359.
"""


class JetSimpy:
    """
    Adapter for the numerical afterglow model ``jetsimpy`` [1]_.

    Parameters
    ----------
    Eiso : float
        The isotropic equivalent energy [erg].

    n0 : float
        The ISM number density [cm-3].

    A : float
        The wind number density amplitude [cm-3].

    eps_e : float
        The fraction of thermal energy in the electric field.
        Must be in the range [0, 1].

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    p : float
        The electron energy power-law index.

    theta_c : float
        The half-opening angle [rad].

    theta_v : float
        The viewing angle [rad].

    z : float
        The redshift.

    d : float
        The luminosity distance [Mpc].

    lf : float
        The initial Lorentz factor.

    References
    ----------
    .. [1] jetsimpy: A Highly Efficient Hydrodynamic Code for Gamma-ray
        Burst Afterglow : https://arxiv.org/abs/2402.19359
    """
    # noinspection PyPep8Naming
    def __init__(self, Eiso, n0, A, eps_e, eps_b, p, theta_c, theta_v, z, d, lf):
        self.Eiso = Eiso
        self.n0 = n0
        self.A = A
        self.eps_e = eps_e
        self.eps_b = eps_b
        self.p = p
        self.theta_c = theta_c
        self.theta_v = theta_v
        self.z = z
        self.d = d
        self.lf = lf

        self.jet = Jet(Gaussian(theta_c, Eiso, lf), A, n0, tmax=1e12)

    @property
    def is_valid(self) -> bool:
        """ Is the model physically valid? """
        return ((self.eps_b + self.eps_e) < 1.0) and (self.p >= 2.0)

    def to_dict(self, exclude=None) -> dict:
        """
        Returns the attributes as a dictionary.

        Parameters
        ----------
        exclude : array_like, optional
            Which attributes should be excluded?

        Returns
        -------
        dict
        """
        return {
            k: v for k, v in vars(self).items()
            if k not in set(exclude or [])
        }

    def model(self, obs, subset=None):
        """
        Models an ``observation`` object.

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
        res = np.full(obs.as_arrays.times.size, np.nan)

        try:
            # Model the spectral flux
            if (sfm := obs.as_arrays.sflux_loc).any():
                if subset is not None:
                    sfm = np.logical_and(sfm, subset)

                res[sfm] = self.spectral_flux(
                    obs.times()[sfm], obs.freqs()[sfm]
                )

            # Model the integrated flux
            if (ifm := obs.as_arrays.iflux_loc).any():
                if subset is not None:
                    ifm = np.logical_and(ifm, subset)

                res[ifm] = self.integrated_flux(
                    obs.times()[ifm], obs.int_lowers()[ifm], obs.int_uppers()[ifm]
                )

            # Model the spectral indices
            if (sim := obs.as_arrays.sindex_loc).any():
                if subset is not None:
                    sim = np.logical_and(sim, subset)

                res[sim] = self.spectral_index(
                    obs.times()[sim], obs.int_lowers()[sim], obs.int_uppers()[sim]
                )
        except Exception as e:
            print(e)
            return np.array([np.nan])

        return res

    def spectral_flux(self, t, nu, model='sync', **kwargs):
        """
        Calculates the spectral flux at times ``t`` for the
        frequencies ``nu``.

        Parameters
        ----------
        t : np.ndarray of float
            The observer times [d].

        nu : float or np.ndarray of float
            The average band frequencies [Hz].

        model : str, optional, default="sync"
            The radiation model.

        Returns
        -------
        np.ndarray of float
            The modeled spectral flux [mJy].
        """
        return self.jet.FluxDensity(days_to_sec(t), nu, self.to_dict(('jet',)), model=model)

    def integrated_flux(self, t, lower, upper, model='sync', **kwargs):
        """
        Calculates the integrated flux at times ``t`` for
        the integration bounds, ``lower`` and ``upper``.

        Parameters
        ----------
        t : np.ndarray of float
            The observer times [d].

        lower, upper : float or np.ndarray of float
            The integration bounds [Hz].

        model : str, optional, default="sync"
            The radiation model.

        Returns
        -------
        np.ndarray of float
            The modeled integrated flux [erg cm-2 s-1].
        """
        sflux = self.spectral_flux(t, lower, model)
        b = self.spectral_index(t, lower, upper, model)

        return 1e-26 * (
            (sflux * lower / (b + 1)) *
            (((upper / lower) ** (b + 1)) - 1)
        )

    def integrated_flux2(self, t, lower, upper, **kwargs):
        """
        Calculates the integrated flux at times ``t`` for
        the integration bounds, ``lower`` and ``upper``
        using the ``jetsimpy`` implementation.

        Warning: Super slow. Not viable for MCMC.

        Parameters
        ----------
        t : np.ndarray of float
            The observer times [d].

        lower, upper : float or np.ndarray of float
            The integration bounds [Hz].

        Returns
        -------
        np.ndarray of float
            The modeled integrated flux [erg cm-2 s-1].
        """
        return self.jet.Flux(t, lower, upper, self.to_dict(('jet',)))

    def spectral_index(self, t, lower, upper, model='sync', **kwargs):
        """
        Calculates the spectral index at times ``t`` for
        the integration bounds, ``lower`` and ``upper``.

        Parameters
        ----------
        t : np.ndarray of float
            The observer times [d].

        lower, upper : float or np.ndarray of float
            The integration bounds [Hz].

        model : str, optional, default="sync"
            The radiation model.

        Returns
        -------
        np.ndarray of float
            The modeled spectral index.
        """
        return (
            np.log10(
                self.spectral_flux(t, upper, model) /
                self.spectral_flux(t, lower, model)
            ) /
            np.log10(upper / lower)
        )

    def spectrum(self, t):
        """
        Returns the characteristics that define the GRB spectrum.

        Parameters
        ----------
        t : np.ndarray of float or float
            The observer time [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        # What about when A and n0 both != 0?
        k = 2.0 if self.A != 0.0 else 0.0

        return {
            'p': self.p, 'k': k, 'f_peak': self.f_peak(t),
            'nu_m': self.nu_m(t), 'nu_c': self.nu_c(t), 'nu_a': self.nu_a(t),
        }

    def f_peak(self, t, **kwargs):
        """
        Calculates the peak flux at times ``t``
        as a weighted average.

        Parameters
        ----------
        t : np.ndarray of float
            The observer times [d].

        Returns
        -------
        np.ndarray of float
            The peak flux [mJy].
        """
        t, p, nu = days_to_sec(t), self.to_dict(('jet',)), 1e16

        return self.jet.FluxDensity(t, nu, p, model='intensity')

    def nu_m(self, t, model='sync', **kwargs):
        """
        Calculates the synchrotron frequency at times ``t``
        as a weighted average.

        Parameters
        ----------
        t : np.ndarray of float
            The observer times [d].

        model : str, optional, default="sync"
            The radiation model.

        Returns
        -------
        np.ndarray of float
            The source-frame synchrotron frequencies [Hz].
        """
        t, p, nu = days_to_sec(t), self.to_dict(('jet',)), 1e16

        return self.jet.WeightedAverage(
            t, nu, p, model, average_model='nu_m'
        )

    def nu_c(self, t, model='sync', **kwargs):
        """
        Calculates the synchrotron frequency at times ``t``
        as a weighted average.

        Parameters
        ----------
        t : np.ndarray of float
            The observer times [d].

        model : str, optional, default="sync"
            The radiation model.

        Returns
        -------
        np.ndarray of float
            The source-frame synchrotron frequencies [Hz].
        """
        t, p, nu = days_to_sec(t), self.to_dict(('jet',)), 1e16

        return self.jet.WeightedAverage(
            t, nu, p, model, average_model='nu_c'
        )

    def nu_a(self, t, model='sync', **kwargs):
        """
        Calculates the sel-absorption frequency at times ``t``
        as a weighted average.

        Parameters
        ----------
        t : np.ndarray of float
            The observer times [d].

        model : str, optional, default="sync"
            The radiation model.

        Returns
        -------
        np.ndarray of float
            The source-frame self-absorption frequencies [Hz].
        """
        t, p, nu = days_to_sec(t), self.to_dict(('jet',)), 1e16

        return self.jet.WeightedAverage(
            t, nu, p, model, average_model='nu_a'
        )
