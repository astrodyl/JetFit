import math

import astropy.units as u
import numpy as np

from jetfit.core.defns.enums import DataType
from jetfit.core.values import SpectralFlux, IntegratedFlux, SpectralIndex
from jetfit.core.core import ipl, pl


class FluxSegment:
    """
    A power-law segment for the GRB fireball spectrum.

    Attributes
    ----------
    beta : float
        The spectral index of the segment.

    amplitude : float or u.Quantity['spectral flux density']
        The amplitude at the reference frequency.

    nu_0 : float or u.Quantity['frequency']
        The reference frequency.

    name: str, optional, default=None
        The name of the segment.
    """
    def __init__(self, beta, amplitude, nu_0, name: str = None):
        self.name = name
        self.beta = beta
        self.nu_0 = nu_0
        self.amplitude = amplitude

    @property
    def nu_0(self) -> float:
        """ Returns the reference frequency in Hz. """
        return self._nu_0

    @nu_0.setter
    def nu_0(self, val):
        """ Sets the reference frequency in Hz. """
        if isinstance(val, u.Quantity):
            val = val.to_value('Hz')

        self._nu_0 = val

    @property
    def amplitude(self) -> float:
        """ Returns the amplitude in mJy. """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, val):
        """ Sets the amplitude in mJy. """
        if isinstance(val, u.Quantity):
            val = val.to_value('mJy')

        self._amplitude = val

    def model(self, flux):
        """
        Given a `Flux` value, return an equivalent flux
        measurement given the objects attributes.

        Parameters
        ----------
        flux : `SpectralFlux` or `IntegratedFlux`
            The flux value to model.

        Returns
        -------
        float or array_like
            The modeled flux.
        """
        if flux.type == DataType.SPECTRAL_FLUX:
            return self.spectral_flux(flux.frequency)

        elif flux.type == DataType.INTEGRATED_FLUX:
            limits = flux.int_range
            return self.integrated_flux(limits.lower, limits.upper)

        raise TypeError(f'Unsupported DataType: {flux.type}')

    def spectral_flux(self, nu):
        """
        Calculates the spectral flux of the segment.

        Parameters
        ----------
        nu : float or u.Quantity or array_like
            The frequency(ies) to evaluate.

        Returns
        -------
        float or u.Quantity or array_like
            The spectral flux(es) measured in `mJy`.
        """
        return pl(self.amplitude, nu, self.nu_0, self.beta)

    def integrated_flux(self, lower, upper):
        """
        Calculates the integrated flux of the segment.

        Parameters
        ----------
        lower : float or u.Quantity or array_like
            The lower integration limit(s).

        upper : float or u.Quantity or array_like
            The upper integration limit(s).

        Returns
        -------
        float or u.Quantity or array_like
            The integrated flux(es) in erg / cm(2) / s
        """
        return 1e-26 * ipl(self.amplitude, lower, upper, self.nu_0, self.beta)


class BaseFluxModel:
    """

    Attributes
    ----------
    f_peak : float or u.Quantity['spectral flux density']
        The peak flux.

    nu_m : float or u.Quantity['frequency']
        The synchrotron frequency.

    nu_c : float or u.Quantity['frequency']
        The cooling frequency.

    p : float
        The electron energy power-law index.

    k : float
        The circumburst density power-law index.
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k):
        if isinstance(nu_m, np.ndarray) or isinstance(nu_c, np.ndarray):
            if nu_m.size != nu_c.size:
                raise ValueError('nu_m, nu_c must have the same size.')

        self.f_peak = f_peak
        self.nu_m = nu_m
        self.nu_c = nu_c
        self.p = p
        self.k = k

    @property
    def f_peak(self):
        """ Returns the peak flux in mJy. """
        return self._f_peak

    @f_peak.setter
    def f_peak(self, val):
        """ Sets the peak flux in mJy. """
        if isinstance(val, u.Quantity):
            val = val.to_value('mJy')

        self._f_peak = val

    @property
    def fast_regime(self) -> np.ndarray:
        """ """
        nu_m = np.atleast_1d(self.nu_m)
        nu_c = np.atleast_1d(self.nu_c)
        return nu_m > nu_c

    @property
    def slow_regime(self) -> np.ndarray:
        """ """
        return ~self.fast_regime


class SpectralFluxModel(BaseFluxModel):
    """
    Spectral Fireball Flux Model

    Provides methods for calculating spectral fluxes
    using the spectrum for the GRB fireball model.
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k):
        super().__init__(nu_m, nu_c, f_peak, p, k)

    def __call__(self, nu):
        """ Calls the `evaluate_smooth` method. """
        return self.evaluate(nu)

    def model(self, val: SpectralFlux):
        """
        Models a `SpectralFlux` value using its frequency.

        Parameters
        ----------
        val : SpectralFlux
            The spectral flux value to model.

        Returns
        -------
        float
            The modeled spectral flux value.
        """
        return self.evaluate(val.frequency.value)

    def evaluate(self, nu):
        """
        Calculates the smoothed flux at a given frequency, `nu`.

        Supports four cases:
            (1) One `nu` and many spectral functions:
                Returns an array of flux with length of the
                spectral functions (i.e., nu_m.size).

            (2) Many `nu` and one spectral function:
                Returns an array of flux with length of `nu`.

            (3) Many `nu` and many spectral functions:
                All arrays must be of the same size and the
                returned array will have the same size.

            (4) One `nu` and one spectral function:
                Returns a single flux value.

        Parameters
        ----------
        nu : float or np.ndarray
            The frequency to evaluate.

        Returns
        -------
        float or np.ndarray of float
            The modeled flux with units of `f_peak`.
        """
        nu = np.atleast_1d(nu)
        nu_m = np.atleast_1d(self.nu_m)
        nu_c = np.atleast_1d(self.nu_c)

        # Handles the cases for varying sizes of inputs
        size = nu_m.size if nu.size == 1 else nu_m.size

        # Initialize spectral indices with slow-cooling params
        b1 = np.full(size, 1 / 3)
        b2 = np.full(size, (1 - self.p) / 2)
        b3 = np.full(size, -self.p / 2)

        # Initialize smoothing factors with slow-cooling params
        s12 = np.full(size, 1.84 - (0.040 * self.k) - (0.40 - 0.010 * self.k) * self.p)
        s23 = np.full(size, 1.15 - (0.125 * self.k) - (0.06 - 0.015 * self.k) * self.p)

        # Initialize critical frequencies in slow-cooling order
        nu12 = np.array(nu_m, copy=True)
        nu23 = np.array(nu_c, copy=True)

        # Overwrite with any fast-cooling parameters
        fast_regime = self.fast_regime

        if fast_regime.any():
            b2[fast_regime] = -0.5
            s12[fast_regime] = 0.597
            nu12[fast_regime] = nu_c[fast_regime]
            nu23[fast_regime] = nu_m[fast_regime]
            s23[fast_regime] = 3.34 + 0.17 * self.k - (0.82 + 0.035 * self.k) * self.p

        # return spectral flux smoothed across segments [mJy]
        return self.f_peak * (
            (((nu / nu12) ** -(s12 * (b1 - b2)) + 1) ** (s23 / s12)) *
            ((nu / nu12) ** -(s23 * b2)) +
            (((nu23 / nu12) ** -(s23 * b2)) * ((nu/nu23) ** -(s23 * b3)))
        ) ** -(1 / s23)


class IntegratedFluxModel(BaseFluxModel):
    """
    Integrated Fireball Flux Model

    Provides methods for calculating integrated fluxes
    using the spectrum for the GRB fireball model.
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k):
        super().__init__(nu_m, nu_c, f_peak, p, k)

    def __call__(self, lower, upper):
        """ Calls the `evaluate` method. """
        return self.evaluate(lower, upper)

    def model(self, val: IntegratedFlux):
        """
        Models an `IntegratedFlux` value using its integration
        range.

        Parameters
        ----------
        val : IntegratedFlux
            The Integrated flux value to model.

        Returns
        -------
        float
            The modeled integrated flux value.
        """
        return self.evaluate(
            lower=val.int_range.lower.value,
            upper=val.int_range.upper.value
        )

    def evaluate(self, lower: float, upper: float):
        """
        Evaluates the integrated flux model using the
        `lower` and `upper` integration limits.

        Parameters
        ----------
        lower : float
            The lower integration limit measured in Hz.

        upper : float
            The upper integration limit measured in Hz.

        Returns
        -------
        float
            The integrated flux with units of erg cm-2 s-1.
        """
        beta = SpectralIndexModel(
            self.nu_m, self.nu_c, self.f_peak, self.p, self.k
        ).evaluate(lower, upper)

        flux = SpectralFluxModel(
            self.nu_m, self.nu_c, self.f_peak, self.p, self.k
        ).evaluate(lower)

        # return smoothed integrated flux [erg cm-2 s-1]
        return 1e-26 * (
            (flux * lower / (beta + 1)) *
            (((upper / lower) ** (beta + 1)) - 1)
        )


class SpectralIndexModel(BaseFluxModel):
    """
    Spectral Index Model
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k):
        super().__init__(nu_m, nu_c, f_peak, p, k)

    def __call__(self, lower, upper):
        """ Calls the `evaluate` method. """
        return self.evaluate(lower, upper)

    def model(self, val: SpectralIndex):
        """
        Models a `SpectralIndex` value using its integration
        limits.

        Parameters
        ----------
        val : SpectralIndex
            The spectral index value to model.

        Returns
        -------
        float
            The modeled spectral index value.
        """
        return self.evaluate(
            lower=val.int_range.lower.value,
            upper=val.int_range.upper.value,
        )

    def evaluate(self, lower, upper):
        """
        Approximates the spectral index using a two
        point approximation.

        Parameters
        ----------
        lower : float or np.ndarray of float
            The lower integration limit.

        upper : float or np.ndarray of float
            The upper integration limit.

        Returns
        -------
        float or np.ndarray of float
            The modeled spectral index.
        """
        model = SpectralFluxModel(
            self.nu_m, self.nu_c, self.f_peak, self.p, self.k)

        # return spectral index [dimension less]
        return (
            np.log(model(upper) / model(lower)) /
            np.log(upper / lower)
        )


class BaseSpectralModel:
    """
    Base Spectral Model. Not intended for direct use.

    Parameters
    ----------
    E : float or u.Quantity['energy']
        The explosion energy. If a float is provided, assumes
        that the value is already normalized to 1e52 ergs. If
        passing a `Quantity`, it will be normalized before
        storing it as a float.

    eps_b : float
        The fraction of thermal energy in the magnetic field.

    k : float
        The density power-law index.

    z : float
        The redshift.

    Attributes
    ----------
    E : float
        The explosion energy normalized to 1e52 ergs.

    eps_b : float
        The fraction of thermal energy in the magnetic field.

    k : float
        The density power-law index.

    z : float
        The redshift.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, eps_b, k, z):
        self.E = E
        self.eps_b = eps_b
        self.k = k
        self.z = z

    def __repr__(self) -> str:
        """ Returns readable string for printing. """
        name = self.__class__.__name__
        return f"{name}(E={self.E}, z={self.z}, k={self.k})"

    def __call__(self, t):
        """ Wrapper for the evaluate method. """
        return self.evaluate(t)

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
    def alpha(self) -> float:
        """ Returns the temporal coefficient. """
        return 16 / (17 - 4 * self.k)

    @property
    def beta(self) -> float:
        """ Returns the spectral coefficient. """
        return 4 - self.k

    def evaluate(self, t):
        """ Placeholder evaluate method. """
        raise NotImplementedError(f'evaluate not implemented.')


class PeakFluxModel(BaseSpectralModel):
    """
    Peak flux model. Assumes an ultra-relativistic shock moving
    through an external medium with rho = rho0 * R^-k density.

    Both radiative and adiabatic models are supported.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, rho0, eps_b, dL, z, k, X):
        super().__init__(E, eps_b, k, z)
        self.rho0 = rho0
        self.dL = dL
        self.X = X

    @property
    def n_p(self):
        """ Returns the inverse particle density. """
        return 0.5 * (1 + self.X)

    def evaluate(self, t: u.Quantity | float) -> float:
        """
        Calculates the peak flux at time `t` for a shock's
        movement that is described by `evo`.

        Parameters
        ----------
        t : u.Quantity['time'] or float
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float
            The peak flux at time `t` measured in mJy.
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # convenience variables
        k, x = self.k, 4 - self.k

        # evaluate exponents once
        exp_z   = (0.5 * (8 - self.k) / x)
        exp_c   = -0.5 * (24 - 7 * k) / x
        exp_en  = 0.5 * (8 - 3 * k) / x
        exp_t   = -0.5 * k / x
        exp_rho = 2 / x

        # exponents in log-space to prevent overflow
        log_pot = (
            (10 * exp_c) +              # speed of light [cm]
            (52 * exp_en) +             # 1e52 erg normalization
            ((17 * k - 24) * exp_rho) + # proton mass [g] and radius normalization
            (4 * exp_t) -               # time conversion (d -> s)
            8.0                         # e(q_e)^3 * e(m_e)^-1 * e(m_p)^-1 - e(dL)^2 + e(cgs->mJy)
                                        # = -30 + 28 + 24 -56 + 26 = -8
        )

        # return peak flux [mJy]
        return (
            # k-independent mantissas
            13.71383 *  # = 4/3 * sqrt(2) * m(q_e)^3 * m(m_e)^-1 * m(m_p)^-1

            # k-dependent mantissas
            (2.9979 ** exp_c) *     # speed of light [cm]
            (1.67262 ** exp_rho) *  # density normalization [g]
            (8.64 ** exp_t) *       # time conversion (d -> s)

            # k-dependent terms
            (math.pi ** -((2 - k) / x)) *
            (self.alpha ** -(0.5 * (8 - 3 * k) / x)) *
            (self.beta ** -(0.5 * k / x)) *

            # model parameters
            (self.eps_b ** 0.5) *       # magnetic field fraction
            ((1 + self.z) ** exp_z) *   # redshift
            (self.E ** exp_en) *        # explosion energy / 1e52 erg
            self.n_p *                  # particle density
            (self.rho0 ** exp_rho) *    # number density / m_p / R_*
            (self.dL ** -2) *           # luminosity distance / 1e28 cm
            (t ** exp_t) *              # time in days

            # exponents in linear-space
            (10 ** log_pot)
        )


class CoolingFrequencyModel(BaseSpectralModel):
    """
    Cooling frequency model. Assumes an ultra-relativistic
    shock moving through an external medium with rho = rho0
    * R^-k density.

    Both radiative and adiabatic models are supported.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, rho0, eps_b, k, z):
        super().__init__(E, eps_b, k, z)
        self.rho0 = rho0

    def evaluate(self, t: u.Quantity | float) -> float:
        """
        Calculates the cooling frequency at time `t`
        for a shock's movement that is described by `evo`.

        Parameters
        ----------
        t : u.Quantity['time'] or float
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float
            The cooling frequency at time `t` measured in Hz.
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # convenience variables
        k, x = self.k, 4 - self.k

        # evaluate exponents once
        exp_c   = 0.5 * (68 - 19 * k) / x
        exp_en  = -0.5 * (4 - 3 * k) / x
        exp_t   = -0.5 * (4 - 3 * k) / x
        exp_z   = -0.5 * (4 + k) / x
        exp_rho = -4 / x

        # exponents in log-space to prevent overflow
        log_pot = (
            (10 * exp_c) + (52 * exp_en) - 70 +
            (4 * exp_t) + ((17 * k - 24) * exp_rho)
        )

        # return cooling frequency [Hz]
        return (
            # k-independent mantissas
            0.014871 *  # 81/8192 * sqrt(2) * m(q_e)^-7 * m(m_e)^5

            # k-dependent mantissas
            (2.9979 ** exp_c) *     # speed of light
            (1.67262 ** exp_rho) *  # density normalization
            (8.64 ** exp_t) *       # time conversion (d -> s)

            # k-dependent terms
            (math.pi ** -((8 - k) / x)) *
            (self.alpha ** (0.5 * (4 - 3 * k) / x)) *
            (self.beta ** (0.5 * (12 - k) / x)) *

            # model parameters
            ((1 + self.z) ** exp_z) *   # redshift
            (self.eps_b ** -1.5) *      # magnetic field fraction
            (self.E ** exp_en) *        # explosion energy
            (self.rho0 ** exp_rho) *    # density normalization
            (t ** exp_t) *              # time in days

            # exponents in linear-space
            (10 ** log_pot)
        )


class SynchrotronFrequencyModel(BaseSpectralModel):
    """
    Synchrotron frequency model. Assumes an ultra-relativistic
    shock moving through an external medium with rho = rho0
    * R^-k density.

    Both radiative and adiabatic models are supported.

    Attributes
    ----------
    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons, unit=None.

    X : float
        The hydrogen mass fraction, unit=None.

    p : float
        The electron energy power-law index, unit=None.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, eps_e, eps_b, k, z, X, p):
        super().__init__(E, eps_b, k, z)
        self.eps_e = eps_e
        self.X = X
        self.p = p

    @property
    def n_p(self):
        """ Returns the particle density. """
        return 0.5 * (1 + self.X)

    def evaluate(self, t: u.Quantity | float) -> float:
        """
        Calculates the synchrotron frequency at time `t`
        for a shock's movement that is described by `evo`.

        To prevent overflow exceptions and generally slow
        calculations, I take the sum of the log of all powers
        of ten rather than evaluating each separately. The
        model parameters span many, many orders of magnitude.

        To improve efficiency, constant factors are evaluated
        and combined beforehand and the result is used.

        Parameters
        ----------
        t : u.Quantity['time'] or float
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float
            The cooling frequency at time `t` measured in Hz.
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # return synchrotron frequency [Hz]
        return (
            # all constants evaluated
            4.049782158231e+16 *

            # k-dependent factors
            (self.alpha ** -0.5) *
            (self.beta ** -1.5) *

            # model parameters
            (self.n_p ** -2) *      # particle density
            (self.eps_e ** 2) *     # electric field fraction
            (self.eps_b ** 0.5) *   # magnetic field fraction
            ((1 + self.z) ** 0.5) * # redshift
            (self.E ** 0.5) *       # explosion energy
            ((self.p - 2) ** 2) *   # electron energy index
            ((self.p - 1) ** -2) *  # electron energy index
            (t ** -1.5)             # time in days
        )


# class SynchrotronFrequencyExponents:
#     """
#     The exponents for SynchrotronFrequencyModel parameters.
#
#     Exponents for both adiabatic and radiative evolution
#     are implemented.
#
#     Attributes
#     ----------
#     k : float
#         The density power-law index.
#
#     evo : str, {'adiabatic', 'radiative'}
#             The evolution type.
#     """
#     def __init__(self, k: float, evo: str):
#         self.k = k
#         self.evo = evo
#
#     @property
#     def q_e(self) -> float:
#         """ Returns the electron charge exponent. """
#         return 1.0
#
#     @property
#     def m_e(self) -> float:
#         """ Returns the electron mass exponent. """
#         return -3.0
#
#     @property
#     def m_p(self) -> float:
#         """ Returns the proton mass exponent. """
#         return 2.0
#
#     @property
#     def n_p(self) -> float:
#         """ Returns the particle density exponent. """
#         return -2.0
#
#     @property
#     def eps_b(self) -> float:
#         """ Returns the magnetic field fraction exponent. """
#         return 0.5
#
#     @property
#     def eps_e(self) -> float:
#         """ Returns the electric field fraction exponent. """
#         return 2.0
#
#     @property
#     def pi(self) -> float:
#         """ Returns the pi exponent. """
#         return {
#             'adiabatic': -1,
#             'radiative': -0.5 * (15 - 4 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def c(self) -> float:
#         """ Returns the speed of light exponent. """
#         return {
#             'adiabatic': -2.5,
#             'radiative': -0.5 * (40 - 11 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def alpha(self) -> float:
#         """ Returns the temporal coefficient exponent. """
#         return {
#             'adiabatic': -0.5,
#             'radiative': -(4 - self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def beta(self) -> float:
#         """ Returns the spectral coefficient exponent. """
#         return {
#             'adiabatic': -1.5,
#             'radiative': -0.5 * (24 - 7 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def t(self) -> float:
#         """ Returns the time exponent. """
#         return {
#             'adiabatic': -1.5,
#             'radiative': -0.5 * (24 - 7 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     # noinspection PyPep8Naming
#     @property
#     def E(self) -> float:
#         """ Returns the energy exponent. """
#         return {
#             'adiabatic': 0.5,
#             'radiative': (4 - self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def rho0(self) -> float:
#         """ Returns the density normalization exponent. """
#         return {
#             'adiabatic': 0.0,
#             'radiative': -0.5 / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def z(self) -> float:
#         """ Returns the redshift term exponent. """
#         return {
#             'adiabatic': 0.5,
#             'radiative': 0.5 * (10 - 3 * self.k) / (7 - 2 * self.k)
#         }[self.evo]


# class PeakFluxExponents:
#     """
#     The exponents for the PeakFluxModel's parameters.
#
#     Exponents for both adiabatic and radiative evolution
#     are available.
#
#     Attributes
#     ----------
#     k : float
#         The density power-law index.
#
#     evo : str, {'adiabatic', 'radiative'}
#             The evolution type.
#     """
#     def __init__(self, k: float, evo: str):
#         self.k = k
#         self.evo = evo
#
#     @property
#     def q_e(self) -> float:
#         """ Returns the electron charge exponent. """
#         return -3.0
#
#     @property
#     def m_e(self) -> float:
#         """ Returns the electron mass exponent. """
#         return -1.0
#
#     @property
#     def m_p(self) -> float:
#         """ Returns the proton mass exponent. """
#         return -1.0
#
#     # noinspection PyPep8Naming
#     @property
#     def dL(self) -> float:
#         """ Returns the luminosity distance exponent. """
#         return -2.0
#
#     @property
#     def eps_b(self) -> float:
#         """ Returns the magnetic field fraction exponent. """
#         return 0.5
#
#     @property
#     def pi(self) -> float:
#         """ Returns the pi exponent. """
#         return {
#             'adiabatic': -(2 - self.k) / (4 - self.k),
#             'radiative': -(9 - 4 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def c(self) -> float:
#         """ Returns the speed of light exponent. """
#         return {
#             'adiabatic': -0.5 * (24 - 7 * self.k) / (4 - self.k),
#             'radiative': -0.5 * (52 - 17 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def alpha(self) -> float:
#         """ Returns the temporal coefficient exponent. """
#         return {
#             'adiabatic': -0.5 * (8 - 3 * self.k) / (4 - self.k),
#             'radiative': -(8 - 3 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def beta(self) -> float:
#         """ Returns the spectral coefficient exponent. """
#         return {
#             'adiabatic': -0.5 * self.k / (4 - self.k),
#             'radiative': -0.5 * (6 - self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def t(self) -> float:
#         """ Returns the time contribution exponent. """
#         return {
#             'adiabatic': -0.5 * self.k / (4 - self.k),
#             'radiative': -0.5 * (6 - self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     # noinspection PyPep8Naming
#     @property
#     def E(self) -> float:
#         """ Returns the energy exponent. """
#         return {
#             'adiabatic': 0.5 * (8 - 3 * self.k) / (4 - self.k),
#             'radiative': (8 - 3 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def rho0(self) -> float:
#         """ Returns the density normalization exponent. """
#         return {
#             'adiabatic': 2 / (4 - self.k),
#             'radiative': 2.5 / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def z(self) -> float:
#         """ Returns the redshift term exponent. """
#         return {
#             'adiabatic': 0.5 * (8 - self.k) / (4 - self.k),
#             'radiative': 2.5 * (4 - self.k) / (7 - 2 * self.k)
#         }[self.evo]


# class CoolingFrequencyExponents:
#     """
#     The exponents for CoolingFrequencyModel parameters.
#
#     Exponents for both adiabatic and radiative evolution
#     are implemented.
#
#     Attributes
#     ----------
#     k : float
#         The density power-law index.
#
#     evo : str, {'adiabatic', 'radiative'}
#             The evolution type.
#     """
#     def __init__(self, k: float, evo: str):
#         self.k = k
#         self.evo = evo
#
#     @property
#     def q_e(self) -> float:
#         """ Returns the electron charge exponent. """
#         return -7.0
#
#     @property
#     def m_e(self) -> float:
#         """ Returns the electron mass exponent. """
#         return 5.0
#
#     @property
#     def eps_b(self) -> float:
#         """ Returns the magnetic field fraction exponent. """
#         return -1.5
#
#     @property
#     def pi(self) -> float:
#         """ Returns the pi contribution [unit-less]. """
#         return {
#             'adiabatic': -(8 - self.k) / (4 - self.k),
#             'radiative': -(27 - 4 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def c(self) -> float:
#         """ Returns the speed of light exponent. """
#         return {
#             'adiabatic': 0.5 * (68 - 19 * self.k) / (4 - self.k),
#             'radiative': 0.5 * (124 - 41 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def alpha(self) -> float:
#         """ Returns the temporal coefficient exponent. """
#         return {
#             'adiabatic': 0.5 * (4 - 3 * self.k) / (4 - self.k),
#             'radiative': (4 - 3 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def beta(self) -> float:
#         """ Returns the spectral coefficient exponent. """
#         return {
#             'adiabatic': 0.5 * (12 - self.k) / (4 - self.k),
#             'radiative': 0.5 * (24 - 5 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def t(self) -> float:
#         """ Returns the time exponent. """
#         return {
#             'adiabatic': -0.5 * (4 - 3 * self.k) / (4 - self.k),
#             'radiative': -0.5 * (4 - 3 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     # noinspection PyPep8Naming
#     @property
#     def E(self) -> float:
#         """ Returns the energy exponent. """
#         return {
#             'adiabatic': -0.5 * (4 - 3 * self.k) / (4 - self.k),
#             'radiative': -(4 - 3 * self.k) / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def rho0(self) -> float:
#         """ Returns the density normalization exponent. """
#         return {
#             'adiabatic': -4 / (4 - self.k),
#             'radiative': -6.5 / (7 - 2 * self.k)
#         }[self.evo]
#
#     @property
#     def z(self) -> float:
#         """ Returns the redshift term exponent. """
#         return {
#             'adiabatic': -0.5 * (4 + self.k) / (4 - self.k),
#             'radiative': -0.5 * (10 - 3 * self.k) / (7 - 2 * self.k)
#         }[self.evo]