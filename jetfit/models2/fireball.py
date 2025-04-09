import astropy.units as u
import numpy as np

from jetfit.core.input import Observation
from jetfit.models2.basemodels import IntegratedFluxModel, SpectralIndexModel
from jetfit.models2.basemodels import SynchrotronFrequencyModel, SpectralFluxModel
from jetfit.models2.basemodels import CoolingFrequencyModel, PeakFluxModel

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class FireballModel:
    """
    Implements the ultra-relativistic shock moving into an
    external medium with density rho = rho_0 * R^-k.

    Parameters can be passed either as an Astropy Quantity with
    an associated unit, or as a simple float. However, they will
    only be stored as floats. Quantities will be converted to the
    appropriate units before storing the float value. This is done
    because operating with Astropy Quantities is very slow and makes
    MCMC fitting extremely difficult. If a float is passed, it is
    assumed that the value is already in the expected units.

    Attributes
    ----------
    E : float or astropy.units.Quantity
        The explosion energy normalized to 1e52 ergs.

    p : float
        The electron energy index (dimensionless).

    eps_b : float
        The fraction of thermal energy in the magnetic field.

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons.

    z : float
        The redshift to the event.

    dL : float or astropy.units.Quantity
        The luminosity distance to the event.

    rho0 : float or astropy.units.Quantity
        The density normalization.

    k : float
        The density power-law index.

    X : float
        The hydrogen mass fraction.

    tj : float, optional
        The jet break time in days.

    sj : float, optional
        The jet break smoothing factor.

    References
    ----------
    [1] Broadband view of blast wave physics: A study
        of gamma-ray burst afterglows
    """

    # noinspection PyPep8Naming
    def __init__(self, E, p, eps_b, eps_e, z, dL, rho0, k, X, tj=None, sj=None):
        # intrinsic properties
        self.E = E
        self.p = p
        self.eps_b = eps_b
        self.eps_e = eps_e
        self.k = k
        self.rho0 = rho0
        self.X = X

        # extrinsic properties
        self.dL = dL
        self.z = z

        # Jet break properties
        self.tj = tj
        self.sj = sj

    def __repr__(self):
        """ Human-readable representation. """
        return f'FireballModel(E={self.E}, n={self.rho0}, .., p={self.p}, k={self.k})'

    # noinspection PyPep8Naming
    @property
    def E(self) -> float:
        """ Returns the explosion energy normalized to 10e52 ergs. """
        return self._E

    # noinspection PyPep8Naming
    @E.setter
    def E(self, e: float | u.Quantity) -> None:
        """
        Sets the explosion energy normalized to 10e52 ergs.

        Parameters
        ----------
        e : float or astropy.units.Quantity
            The explosion energy. If a float is provided, assumes
            that the value is already normalized to 1e52 ergs.
        """
        if isinstance(e, u.Quantity):
            e = e.to_value('erg') / 1e52

        self._E = e

    @property
    def rho0(self) -> float:
        """ Returns the density normalization, normalized to the proton mass. """
        return self._rho0

    @rho0.setter
    def rho0(self, rho0) -> None:
        """
        Sets the density normalization as a simple float.

        Define rho as:

        rho = rho_x * R^-k = rho_0 * (R/R_0)^-k

        such that:

        rho_x = rho_0 * R_0^k

        where R_0 is the characteristic radius which we take to
        be 1e17 cm. Then `rho0` is the reference density at 1e17
        cm with units of g/cm^3. However, I normalize to the proton
        mass such that rho0 is a number density with units of cm^-3.

        Parameters
        ----------
        rho0 : float or astropy.units.Quantity
            The density normalization. If a float or unit-less
            quantity is provided, assumes that the value is
            ??.
        """
        if isinstance(rho0, u.Quantity):
            rho0 = rho0.cgs.value

        self._rho0 = rho0

    # noinspection PyPep8Naming
    @property
    def dL(self) -> float:
        """ Returns the luminosity distance normalized to 1e28 cm. """
        return self._dL

    # noinspection PyPep8Naming
    @dL.setter
    def dL(self, d: float | u.Quantity) -> None:
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

        self._dL = d

    def model(
            self,
            observation: Observation,
            group: str = None
    ) -> np.ndarray:
        """
        Models an observation object.

        Parameters
        ----------
        observation : Observation
            The observation object to model.

        group : str, optional
            The data group to model.

        Returns
        -------
        np.ndarray of float
            The modeled flux.
        """
        res = np.full(len(observation.data), np.nan)

        # For speed, get observation as arrays
        arrays = observation.as_arrays

        # Save locations for readability
        sf_mask = arrays.sflux_loc
        if_mask = arrays.iflux_loc
        si_mask = arrays.sindex_loc

        if group is not None:
            g_mask = observation.data_groups[group]
            sf_mask = np.logical_and(sf_mask, g_mask)
            if_mask = np.logical_and(if_mask, g_mask)
            si_mask = np.logical_and(si_mask, g_mask)

        # Calculate the spectral functions
        f_peaks = self.f_peak(arrays.times)
        nu_ms = self.nu_m(arrays.times)
        nu_cs = self.nu_c(arrays.times)

        # Model spectral fluxes
        res[sf_mask] = SpectralFluxModel(
            nu_ms[sf_mask], nu_cs[sf_mask], f_peaks[sf_mask], self.p, self.k
        )(arrays.frequencies[sf_mask])

        # Model Integrated fluxes
        res[if_mask] = IntegratedFluxModel(
            nu_ms[if_mask], nu_cs[if_mask], f_peaks[if_mask], self.p, self.k
        )(arrays.if_lower_freqs[if_mask], arrays.if_upper_freqs[if_mask])

        # Model Spectral indices
        res[si_mask] = SpectralIndexModel(
            nu_ms[si_mask], nu_cs[si_mask], f_peaks[si_mask], self.p, self.k
        )(arrays.si_lower_freqs[si_mask], arrays.si_upper_freqs[si_mask])

        # Smooth the flux values if there is a jet break
        if self.tj:
            res[sf_mask] = self.smooth_jet_break(
                f=res[sf_mask],
                t=arrays.times[sf_mask],
                nu=arrays.frequencies[sf_mask]
            )
            res[if_mask] = self.smooth_jet_break(
                f=res[if_mask],
                t=arrays.times[if_mask],
                lower=arrays.if_lower_freqs[if_mask],
                upper=arrays.if_upper_freqs[if_mask]
            )

        # return modeled observational data
        return res[observation.data_groups[group]] if group else res

    def evaluate_spectral_flux(self, t, f):
        """ Model spectral fluxes. """
        res = SpectralFluxModel(
            self.nu_m(t), self.nu_c(t), self.f_peak(t), self.p, self.k
        ).evaluate(f)

        if self.tj:
            return self.smooth_jet_break(res, t, nu=f)

        return res

    def evaluate_integrated_flux(self, t, lower, upper):
        """ Model Integrated fluxes. """
        res = IntegratedFluxModel(
            self.nu_m(t), self.nu_c(t), self.f_peak(t), self.p, self.k
        ).evaluate(lower, upper)

        if self.tj:
            return self.smooth_jet_break(res, t, lower=lower, upper=upper)

        return res

    def smooth_jet_break(self, f, t, **kwargs):
        """
        Smooths the flux across a jet break.

        Parameters
        ----------
        f : np.ndarray or float
            The flux to smooth.

        t : np.ndarray or float
            The time of the flux measurements.

        kwargs :
            Do not pass both `nu` and `lower`/`upper`. OR ELSE.

            nu : np.ndarray or float
                The frequency to evaluate the jet break flux.
                Required if passing spectral fluxes.

            lower : np.ndarray or float
                The lower frequency to evaluate the jet break flux.
                Required if passing integrated fluxes.

            upper : np.ndarray or float
                The upper frequency to evaluate the jet break flux.
                Required if passing integrated fluxes.

        Returns
        -------
        np.ndarray or float
            The jet-break smoothed flux.

        Raises
        ------
        ValueError
            If `nu` and `lower` and `upper` are not provided.
        """
        f_peak_jet = self.f_peak(self.tj)
        nu_m_jet = self.nu_m(self.tj)
        nu_c_jet = self.nu_c(self.tj)

        if 'nu' in kwargs:
            model = SpectralFluxModel
        elif 'lower' in kwargs and 'upper' in kwargs:
            model = IntegratedFluxModel
        else:
            raise ValueError(
                'Must provide either `nu` or `lower` and `upper`'
            )

        jet_model = model(nu_m_jet, nu_c_jet, f_peak_jet, self.p, self.k)
        jet_flux = jet_model(**kwargs)

        return (
            f ** -self.sj + (jet_flux * (t / self.tj) ** -self.p) ** -self.sj
        ) ** -(1 / self.sj)

    def f_peak(self, t):
        """
        Calculates the peak flux in the case of an ultra-
        relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float or u.Quantity['time']
            The peak flux in mJy at time `t`.
        """
        return PeakFluxModel(
            self.E, self.rho0, self.eps_b, self.dL, self.z, self.k, self.X)(t)

    def nu_c(self, t):
        """
        Calculates the cooling frequency in the case of an ultra-
        relativistic shock moving into an external medium with
        density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float or u.Quantity['time']
            The cooling frequency in Hz at time `t`.
        """
        return CoolingFrequencyModel(
            self.E, self.rho0, self.eps_b, self.k, self.z)(t)

    def nu_m(self, t):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float or u.Quantity['time']
            The synchrotron frequency in Hz at time `t`.
        """
        return SynchrotronFrequencyModel(
            self.E, self.eps_e, self.eps_b, self.k, self.z, self.X, self.p)(t)
