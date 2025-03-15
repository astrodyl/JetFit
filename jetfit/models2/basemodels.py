import math

import astropy.units as u
import numpy as np

from jetfit.core.defns.enums import DataType
from jetfit.core.values import SpectralFlux, IntegratedFlux, SpectralIndex
from jetfit.core.core import ipl, pl, two_point_approx


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
    def regime(self) -> str:
        """ Returns the regime of the power-law spectrum. """
        return 'fast' if self.nu_m > self.nu_c else 'slow'

    @property
    def seg_b(self):
        """ Returns the object for fast cooling segment B. """
        return FluxSegment(1 / 3, self.f_peak, self.nu_c, name='B')

    @property
    def seg_c(self):
        """ Returns the object for fast cooling segment C. """
        return FluxSegment(-0.5, self.f_peak, self.nu_c, name='C')

    @property
    def seg_d(self) -> FluxSegment:
        """ Returns the object for fast cooling segment D. """
        amp = self.f_peak * (self.nu_m / self.nu_c) ** -0.5
        return FluxSegment(-0.5 * self.p, amp, self.nu_m, name='D')

    @property
    def seg_f(self) -> FluxSegment:
        """ Returns the object for slow cooling segment G. """
        return FluxSegment(1 / 3, self.f_peak, self.nu_m, name='F')

    @property
    def seg_g(self) -> FluxSegment:
        """ Returns the object for slow cooling segment G. """
        return FluxSegment(-0.5 * (self.p - 1), self.f_peak, self.nu_m, name='G')

    @property
    def seg_h(self) -> FluxSegment:
        """ Returns the object for slow cooling segment H. """
        amp = self.f_peak * (self.nu_c / self.nu_m) ** -(0.5 * (self.p - 1))
        return FluxSegment(-0.5 * self.p, amp, self.nu_c, name='H')


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
        return self.evaluate_smooth(nu)

    def segment(self, f) -> FluxSegment:
        """
        Determines the segment containing the frequency
        provided `f` in `regime`.

        Parameters
        ----------
        f : u.Quantity['frequency']
            The frequency to evaluate.

        Returns
        -------
        FluxSegment
            The segment containing the frequency `f`.
        """
        r = self.regime
        f1, f2 = self.nu_m, self.nu_c

        if r == 'fast':
            f1, f2 = f2, f1

        if f <= f1:  # f is below the critical frequencies
            return self.seg_b if r == 'fast' else self.seg_f

        if f < f2:  # f is between the critical frequencies
            return self.seg_c if r == 'fast' else self.seg_g

        # f is above the largest critical frequency
        return self.seg_d if r == 'fast' else self.seg_h

    def model_smooth(self, val: SpectralFlux):
        """"""
        return self.evaluate_smooth(val.frequency.value)

    def evaluate_smooth(self, nu: float) -> float:
        """
        Calculates the smoothed flux at a given frequency, `nu`.

        Parameters
        ----------
        nu : float
            The frequency to evaluate.

        Returns
        -------
        float
            The modeled flux with units of `f_peak`.
        """
        # Critical frequencies
        nu12, nu23 = self.nu_m, self.nu_c

        # Segment spectral indices
        b1, b2, b3 = 1 / 3, (1 - self.p) / 2, -self.p / 2

        # Smoothing factors
        s12 = 1.84 - (0.040 * self.k) - (0.40 - 0.010 * self.k) * self.p
        s23 = 1.15 - (0.125 * self.k) - (0.06 - 0.015 * self.k) * self.p

        if self.regime == 'fast':
            nu12, nu23 = self.nu_c, self.nu_m
            b2  = -0.50
            s12 = 0.597
            s23 = 3.34 + 0.17 * self.k - (0.82 + 0.035 * self.k) * self.p

        # return smoothed flux density
        return self.f_peak * (
            (((nu / nu12) ** -(s12 * (b1 - b2)) + 1) ** (s23 / s12)) *
            ((nu / nu12) ** -(s23 * b2)) +
            (((nu23 / nu12) ** -(s23 * b2)) * ((nu/nu23) ** -(s23 * b3)))
        ) ** -(1 / s23)

    def model(self, val: SpectralFlux):
        """
        Models the flux at `val`'s frequency `f`.

        Parameters
        ----------
        val : SpectralFlux
            The spectral flux value to model.

        Returns
        -------
        u.Quantity['spectral flux density']
            The modeled flux at frequency `f`.
        """
        return self.evaluate(val.frequency.value)

    def evaluate(self, f):
        """
        Evaluates the flux at the frequency `f`.

        Parameters
        ----------
        f : u.Quantity['frequency'] or array_like
            The frequency to evaluate.

        Returns
        -------
        u.Quantity['spectral flux density']
            The modeled flux at frequency `f`.
        """
        return self.segment(f).spectral_flux(f)
    # </editor-fold>


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
        return self.evaluate_smooth(lower, upper)

    def model_smooth(self, val: IntegratedFlux):
        """"""
        return self.evaluate_smooth(
            lower=val.int_range.lower.value,
            upper=val.int_range.upper.value
        )

    def evaluate_smooth(self, lower: float, upper: float):
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
        ).evaluate_smooth(lower)

        return 1e-26 * (
            (flux * lower / (beta + 1)) *
            (((upper / lower) ** (beta + 1)) - 1)
        )

    def model(self, val: IntegratedFlux):
        """
        Models the flux `val` using its integration range.

        Parameters
        ----------
        val : IntegratedFlux
            The Integrate Flux value.

        Returns
        -------
        ??
            The modeled flux for `val`'s integration range.
        """
        return self.evaluate(val.int_range.lower.value, val.int_range.upper.value)

    def evaluate(self, lower, upper):
        """
        Calculates the flux for the frequency range `lower`, `upper`.

        Parameters
        ----------
        lower : u.Quantity['frequency']
            The lower frequency.

        upper : u.Quantity['frequency']
            The upper frequency.

        Returns
        -------
        u.Quantity['energy flux']
            The modeled flux for the frequency range `lower`, `upper`.
        """
        seg_name = self.segment(lower, upper)

        # Evaluate ranges that span a single segment
        if seg_name in ('B', 'C', 'D', 'F', 'G', 'H'):
            seg = getattr(self, f"seg_{seg_name.lower()}")
            return seg.integrated_flux(lower, upper)

        # Evaluate ranges that span two segments
        if seg_name == 'BC' or seg_name == 'GH':
            seg1 = getattr(self, f"seg_{seg_name[0].lower()}")
            seg2 = getattr(self, f"seg_{seg_name[1].lower()}")

            return (
                seg1.integrated_flux(lower, self.nu_c) +
                seg2.integrated_flux(self.nu_c, upper)
            )

        if seg_name == 'CD' or seg_name == 'FG':
            seg1 = getattr(self, f"seg_{seg_name[0].lower()}")
            seg2 = getattr(self, f"seg_{seg_name[1].lower()}")

            return (
                seg1.integrated_flux(lower, self.nu_m) +
                seg2.integrated_flux(self.nu_m, upper)
            )

        # Evaluate ranges that span the entire regime
        if seg_name == 'BCD':
            seg_b = self.seg_b.integrated_flux(lower, self.nu_c)
            seg_c = self.seg_c.integrated_flux(self.nu_c, self.nu_m)
            seg_d = self.seg_d.integrated_flux(self.nu_m, upper)
            return seg_b + seg_c + seg_d

        if seg_name == 'FGH':
            seg_f = self.seg_f.integrated_flux(lower, self.nu_m)
            seg_g = self.seg_g.integrated_flux(self.nu_m, self.nu_c)
            seg_h = self.seg_h.integrated_flux(self.nu_c, upper)
            return seg_f + seg_g + seg_h

    def segment(self, lower, upper) -> str:
        """
        Determines the segment(s) that the lower, upper range
        spans.

        Parameters
        ----------
        lower : u.Quantity['frequency']
            The lower frequency.

        upper : u.Quantity['frequency']
            The upper frequency.

        Returns
        -------
        str
            The segment name(s) that contain `lower` and `upper`.
        """
        if self.regime == 'fast':
            f1, f2 = self.nu_c, self.nu_m
            c1, c2, c3 = 'B', 'C', 'D'

        else:
            f1, f2 = self.nu_m, self.nu_c
            c1, c2, c3 = 'F', 'G', 'H'

        # Entire range is below nu_c(nu_m) for fast(slow).
        if upper <= f1:
            return c1

        # Upper is between nu_c(nu_m) and nu_m(nu_c) for fast(slow).
        # Check where the lower frequency lies.
        if upper <= f2:
            if lower < f1:
                return c1 + c2
            return c2

        # Upper is above nu_m(nu_c) for fast(slow).
        # Check where the lower frequency lies.
        if lower < f1:
            return c1 + c2 + c3

        if lower < f2:
            return c2 + c3

        # Entire range is above nu_m(nu_c) for fast(slow).
        return c3


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
        """"""
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
        lower : ??
            The lower integration limit.

        upper : ??
            The upper integration limit.

        Returns
        -------
        float
            The modeled spectral index.
        """
        model = SpectralFluxModel(
            self.nu_m, self.nu_c, self.f_peak, self.p, self.k)

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