import math

import astropy.units as u
import astropy.constants as const

from jetfit.core.defns.enums import DataType
from jetfit.core.values import SpectralFlux, IntegratedFlux, SpectralIndex
from jetfit.core.core import ipl, pl, two_point_approx

# Define constants in useful units
m_e = const.m_e.cgs # Mass of electron [g]
m_p = const.m_p.cgs # Mass of proton [g]
c   = const.c.cgs   # Speed of light [cm/s]


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

    """
    def __init__(self, nu_m, nu_c, f_peak, p):
        self.f_peak = f_peak
        self.nu_m = nu_m
        self.nu_c = nu_c
        self.p = p

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
    def __init__(self, nu_m, nu_c, f_peak, p):
        super().__init__(nu_m, nu_c, f_peak, p)

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
            return self.seg_b if r == 'fast' else self.seg_d

        if f < f2:  # f is between the critical frequencies
            return self.seg_c if r == 'fast' else self.seg_g

        # f is above the largest critical frequency
        return self.seg_d if r == 'fast' else self.seg_h

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
    def __init__(self, nu_m, nu_c, f_peak, p):
        super().__init__(nu_m, nu_c, f_peak, p)

    def __call__(self, lower, upper) -> u.Quantity:
        """ Calls the `evaluate` method. """
        return self.evaluate(lower, upper)

    def model(self, val: IntegratedFlux) -> u.Quantity:
        """
        Models the flux `val` using its integration range.

        Parameters
        ----------
        val : IntegratedFlux
            The Integrate Flux value.

        Returns
        -------
        u.Quantity['energy flux']
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


class SpectralIndexModel:
    """
    Spectral Index Model
    """
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

    def model(self, index: SpectralIndex):
        """
        Models the spectral index `index` using a two
        point approximation.

        Parameters
        ----------
        index : SpectralIndex
            The spectral index to model.

        Returns
        -------
        float
            The approximated spectral index.
        """
        return self.evaluate(
            lower=index.int_range.lower.value,
            upper=index.int_range.upper.value
        )

    def evaluate(self, lower, upper):
        """
        Approximates the spectral index using a two
        point approximation.

        Parameters
        ----------
        lower : ??
            The start integration time.

        upper : ??
            The stop integration time.

        Returns
        -------

        """
        return two_point_approx(self.f2, self.f1, lower, upper, log=True)


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
    _exponents = None

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

    def __call__(self, t, evo: str = 'adiabatic'):
        """ Wrapper for the evaluate method. """
        return self.evaluate(t, evo)

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

    def evaluate(self, t: u.Quantity, evo: str) -> float:
        """
        Evaluates the shared model parameters between all
        spectral models at time `t` for a shock's movement
        that is described by `evo`.

        As this is a base class and a base method, the
        returned value is not a complete calculation. All
        child classes must implement an evaluate method and
        either call this method, or reimplement the terms
        that are evaluated within.

        Parameters
        ----------
        t : u.Quantity['time']
            The time to evaluate.

        evo : str, {'adiabatic', 'radiative'}, default='adiabatic'
            The evolution of the external shock.

        Returns
        -------
        float
            The evaluated terms that are shared amongst all
            spectral models.
        """
        e = self._exponents(self.k, evo)

        result = 1.0

        # k-dependent unit-less factors
        result *= math.pi ** e.pi
        result *= self.alpha ** e.alpha
        result *= self.beta ** e.beta

        # shared model parameters
        result *= self.eps_b ** e.eps_b
        result *= (1 + self.z) ** e.z
        result *= t.to_value('day') ** e.t
        result *= self.E ** e.E

        return result

    def exponents(self, evo: str):
        """
        Returns the exponents for each model parameter.

        Parameters
        ----------
        evo : str, {'adiabatic', 'radiative'}
            The evolution type.

        Returns
        -------
            The object containing the exponents for the calling
            model's parameters.
        """
        return self._exponents(self.k, evo)


class PeakFluxExponents:
    """
    The exponents for the PeakFluxModel's parameters.

    Exponents for both adiabatic and radiative evolution
    are available.

    Attributes
    ----------
    k : float
        The density power-law index.

    evo : str, {'adiabatic', 'radiative'}
            The evolution type.
    """
    def __init__(self, k: float, evo: str):
        self.k = k
        self.evo = evo

    @property
    def q_e(self) -> float:
        """ Returns the electron charge exponent. """
        return -3.0

    @property
    def m_e(self) -> float:
        """ Returns the electron mass exponent. """
        return -1.0

    @property
    def m_p(self) -> float:
        """ Returns the proton mass exponent. """
        return -1.0

    # noinspection PyPep8Naming
    @property
    def dL(self) -> float:
        """ Returns the luminosity distance exponent. """
        return -2.0

    @property
    def eps_b(self) -> float:
        """ Returns the magnetic field fraction exponent. """
        return 0.5

    @property
    def pi(self) -> float:
        """ Returns the pi exponent. """
        return {
            'adiabatic': -(2 - self.k) / (4 - self.k),
            'radiative': -(9 - 4 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def c(self) -> float:
        """ Returns the speed of light exponent. """
        return {
            'adiabatic': -0.5 * (24 - 7 * self.k) / (4 - self.k),
            'radiative': -0.5 * (52 - 17 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def alpha(self) -> float:
        """ Returns the temporal coefficient exponent. """
        return {
            'adiabatic': -0.5 * (8 - 3 * self.k) / (4 - self.k),
            'radiative': -(8 - 3 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def beta(self) -> float:
        """ Returns the spectral coefficient exponent. """
        return {
            'adiabatic': -0.5 * self.k / (4 - self.k),
            'radiative': -0.5 * (6 - self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def t(self) -> float:
        """ Returns the time contribution exponent. """
        return {
            'adiabatic': -0.5 * self.k / (4 - self.k),
            'radiative': -0.5 * (6 - self.k) / (7 - 2 * self.k)
        }[self.evo]

    # noinspection PyPep8Naming
    @property
    def E(self) -> float:
        """ Returns the energy exponent. """
        return {
            'adiabatic': 0.5 * (8 - 3 * self.k) / (4 - self.k),
            'radiative': (8 - 3 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def rho0(self) -> float:
        """ Returns the density normalization exponent. """
        return {
            'adiabatic': 2 / (4 - self.k),
            'radiative': 2.5 / (7 - 2 * self.k)
        }[self.evo]

    @property
    def z(self) -> float:
        """ Returns the redshift term exponent. """
        return {
            'adiabatic': 0.5 * (8 - self.k) / (4 - self.k),
            'radiative': 2.5 * (4 - self.k) / (7 - 2 * self.k)
        }[self.evo]


class PeakFluxModel(BaseSpectralModel):
    """
    Peak flux model. Assumes an ultra-relativistic shock moving
    through an external medium with rho = rho0 * R^-k density.

    Both radiative and adiabatic models are supported.
    """
    _exponents = PeakFluxExponents

    # noinspection PyPep8Naming
    def __init__(self, E, rho0, eps_b, dL, z, k, X):
        super().__init__(E, eps_b, k, z)
        self.rho0 = rho0
        self.dL = dL
        self.X = X

    @property
    def n_p(self):
        """ Returns the particle density. """
        return 0.5 * (1 + self.X)

    def evaluate(self, t, evo: str = 'adiabatic') -> float:
        """
        Calculates the peak flux at time `t` for a shock's
        movement that is described by `evo`.

        Parameters
        ----------
        t : u.Quantity['time']
            The time to evaluate.

        evo : str, {'adiabatic', 'radiative'}, default='adiabatic'
            The evolution of the external shock.

        Returns
        -------
        float
            The peak flux at time `t` measured in mJy.
        """
        e = self.exponents(evo)

        result = super().evaluate(t, evo)

        # k-independent unit-less factors
        # 4/3 * sqrt(2) * q_em^-3 * m_em^-1 * m_pm^-1
        result *= 13.71383

        # k-dependent unit factors
        result *= 2.9979 ** e.c
        result *= 1.67262 ** e.rho0
        result *= 8.64 ** e.t

        # k-dependent powers-of-ten
        kd_pot = 10 * e.c + 52 * e.E - 24 * e.rho0 + 4 * e.t

        # k-independent powers-of-ten
        # log(q_e * m_e * m_p * 1e28 * 1e26)
        ki_pot = -8.0

        # powers-of-ten
        result *= pow(10, ki_pot + kd_pot)

        # parameters
        result *= self.n_p
        result *= self.rho0 ** e.rho0
        result *= self.dL ** e.dL

        # return peak flux [mJy]
        return result


class CoolingFrequencyExponents:
    """
    The exponents for CoolingFrequencyModel parameters.

    Exponents for both adiabatic and radiative evolution
    are implemented.

    Attributes
    ----------
    k : float
        The density power-law index.

    evo : str, {'adiabatic', 'radiative'}
            The evolution type.
    """
    def __init__(self, k: float, evo: str):
        self.k = k
        self.evo = evo

    @property
    def q_e(self) -> float:
        """ Returns the electron charge exponent. """
        return -7.0

    @property
    def m_e(self) -> float:
        """ Returns the electron mass exponent. """
        return 5.0

    @property
    def eps_b(self) -> float:
        """ Returns the magnetic field fraction exponent. """
        return -1.5

    @property
    def pi(self) -> float:
        """ Returns the pi contribution [unit-less]. """
        return {
            'adiabatic': -(8 - self.k) / (4 - self.k),
            'radiative': -(27 - 4 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def c(self) -> float:
        """ Returns the speed of light exponent. """
        return {
            'adiabatic': 0.5 * (68 - 19 * self.k) / (4 - self.k),
            'radiative': 0.5 * (124 - 41 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def alpha(self) -> float:
        """ Returns the temporal coefficient exponent. """
        return {
            'adiabatic': 0.5 * (4 - 3 * self.k) / (4 - self.k),
            'radiative': (4 - 3 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def beta(self) -> float:
        """ Returns the spectral coefficient exponent. """
        return {
            'adiabatic': 0.5 * (12 - self.k) / (4 - self.k),
            'radiative': 0.5 * (24 - 5 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def t(self) -> float:
        """ Returns the time exponent. """
        return {
            'adiabatic': -0.5 * (4 - 3 * self.k) / (4 - self.k),
            'radiative': -0.5 * (4 - 3 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    # noinspection PyPep8Naming
    @property
    def E(self) -> float:
        """ Returns the energy exponent. """
        return {
            'adiabatic': -0.5 * (4 - 3 * self.k) / (4 - self.k),
            'radiative': -(4 - 3 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def rho0(self) -> float:
        """ Returns the density normalization exponent. """
        return {
            'adiabatic': -4 / (4 - self.k),
            'radiative': -6.5 / (7 - 2 * self.k)
        }[self.evo]

    @property
    def z(self) -> float:
        """ Returns the redshift term exponent. """
        return {
            'adiabatic': -0.5 * (4 + self.k) / (4 - self.k),
            'radiative': -0.5 * (10 - 3 * self.k) / (7 - 2 * self.k)
        }[self.evo]


class CoolingFrequencyModel(BaseSpectralModel):
    """
    Cooling frequency model. Assumes an ultra-relativistic
    shock moving through an external medium with rho = rho0
    * R^-k density.

    Both radiative and adiabatic models are supported.
    """
    _exponents = CoolingFrequencyExponents

    # noinspection PyPep8Naming
    def __init__(self, E, rho0, eps_b, k, z):
        super().__init__(E, eps_b, k, z)
        self.rho0 = rho0

    def evaluate(self, t, evo: str = 'adiabatic') -> float:
        """
        Calculates the cooling frequency at time `t`
        for a shock's movement that is described by `evo`.

        Parameters
        ----------
        t : u.Quantity['time']
            The time to evaluate.

        evo : str, {'adiabatic', 'radiative'}, default='adiabatic'
            The evolution of the external shock.

        Returns
        -------
        float
            The cooling frequency at time `t` measured in Hz.
        """
        e = self.exponents(evo)

        result = super().evaluate(t, evo)

        # k-independent unit-less factors
        # 81/8192 * sqrt(2) * q_em^-7 * m_em^5
        result *= 0.014871

        # k-dependent unit factors
        result *= 2.9979 ** e.c
        result *= 1.67262 ** e.rho0
        result *= 8.64 ** e.t

        # k-dependent powers-of-ten
        kd_pot = 10 * e.c + 52 * e.E - 24 * e.rho0 + 4 * e.t

        # log(q_e^-7 * m_e^5) = 70 - 140 = -70
        ki_pot = -70.0

        # powers-of-ten
        result *= pow(10, ki_pot + kd_pot)

        # unshared parameters
        result *= self.rho0 ** e.rho0

        # return cooling frequency [Hz]
        return result


class SynchrotronFrequencyExponents:
    """
    The exponents for SynchrotronFrequencyModel parameters.

    Exponents for both adiabatic and radiative evolution
    are implemented.

    Attributes
    ----------
    k : float
        The density power-law index.

    evo : str, {'adiabatic', 'radiative'}
            The evolution type.
    """
    def __init__(self, k: float, evo: str):
        self.k = k
        self.evo = evo

    @property
    def q_e(self) -> float:
        """ Returns the electron charge exponent. """
        return 1.0

    @property
    def m_e(self) -> float:
        """ Returns the electron mass exponent. """
        return -3.0

    @property
    def m_p(self) -> float:
        """ Returns the proton mass exponent. """
        return 2.0

    @property
    def n_p(self) -> float:
        """ Returns the particle density exponent. """
        return -2.0

    @property
    def eps_b(self) -> float:
        """ Returns the magnetic field fraction exponent. """
        return 0.5

    @property
    def eps_e(self) -> float:
        """ Returns the electric field fraction exponent. """
        return 2.0

    @property
    def pi(self) -> float:
        """ Returns the pi exponent. """
        return {
            'adiabatic': -1,
            'radiative': -0.5 * (15 - 4 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def c(self) -> float:
        """ Returns the speed of light exponent. """
        return {
            'adiabatic': -2.5,
            'radiative': -0.5 * (40 - 11 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def alpha(self) -> float:
        """ Returns the temporal coefficient exponent. """
        return {
            'adiabatic': -0.5,
            'radiative': -(4 - self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def beta(self) -> float:
        """ Returns the spectral coefficient exponent. """
        return {
            'adiabatic': -1.5,
            'radiative': -0.5 * (24 - 7 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def t(self) -> float:
        """ Returns the time exponent. """
        return {
            'adiabatic': -1.5,
            'radiative': -0.5 * (24 - 7 * self.k) / (7 - 2 * self.k)
        }[self.evo]

    # noinspection PyPep8Naming
    @property
    def E(self) -> float:
        """ Returns the energy exponent. """
        return {
            'adiabatic': 0.5,
            'radiative': (4 - self.k) / (7 - 2 * self.k)
        }[self.evo]

    @property
    def rho0(self) -> float:
        """ Returns the density normalization exponent. """
        return {
            'adiabatic': 0.0,
            'radiative': -0.5 / (7 - 2 * self.k)
        }[self.evo]

    @property
    def z(self) -> float:
        """ Returns the redshift term exponent. """
        return {
            'adiabatic': 0.5,
            'radiative': 0.5 * (10 - 3 * self.k) / (7 - 2 * self.k)
        }[self.evo]


class SynchrotronFrequencyModel(BaseSpectralModel):
    """
    Synchrotron frequency model. Assumes an ultra-relativistic
    shock moving through an external medium with rho = rho0
    * R^-k density.

    Both radiative and adiabatic models are supported.

    Attributes
    ----------
    rho0 : float
        The density normalization, normalized to the proton mass,
        unit=cm(k-3).

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons, unit=None.

    X : float
        The hydrogen mass fraction, unit=None.

    p : float
        The electron energy power-law index, unit=None.
    """
    _exponents = SynchrotronFrequencyExponents

    # noinspection PyPep8Naming
    def __init__(self, E, rho0, eps_e, eps_b, k, z, X, p):
        super().__init__(E, eps_b, k, z)
        self.rho0 = rho0
        self.eps_e = eps_e
        self.X = X
        self.p = p

    @property
    def n_p(self):
        """ Returns the particle density. """
        return 0.5 * (1 + self.X)

    def evaluate(self, t, evo: str = 'adiabatic') -> float:
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
        t : u.Quantity['time']
            The time to evaluate.

        evo : str, {'adiabatic', 'radiative'}, default='adiabatic'
            The evolution of the external shock.

        Returns
        -------
        float
            The cooling frequency at time `t` measured in Hz.
        """
        e = self.exponents(evo)

        result = super().evaluate(t, evo)

        # k-independent mantissas
        # = 2 * sqrt(2) * mantissa(q_e * m_e^-3 * m_p^2)
        result *= 0.050281

        # k-dependent mantissas
        result *= 2.9979 ** e.c
        result *= 1.67262 ** e.rho0
        result *= 8.64 ** e.t

        # k-dependent powers-of-ten [cgs]
        # = log(c^f(k) * E^g(k) * rho^h(k))
        log_kd_exp = 10 * e.c + 52 * e.E - 24 * e.rho0 + 4 * e.t

        # k-independent powers-of-ten [cgs]
        # = log(q_e * m_e^-3 * m_p^2) = -10 + 84 - 48 = 26
        log_ki_exp = 26.0

        # undo the log of the exponents
        result *= pow(10, log_ki_exp + log_kd_exp)

        # unshared parameters
        result *= self.n_p ** e.n_p
        result *= self.rho0 ** e.rho0
        result *= self.eps_e ** e.eps_e
        result *= ((self.p - 2) ** 2) * ((self.p - 1) ** -2)

        # return synchrotron frequency [Hz]
        return result
