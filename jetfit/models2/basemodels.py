import math

import astropy.units as u
import astropy.constants as const
import numpy as np

from jetfit.core.values import SpectralFlux, IntegratedFlux, SpectralIndex
from jetfit.mcmc.parameters.parameters import Parameters


# noinspection PyPep8Naming
class BaseBlastWaveModel:
    """
    Base BlastWaveModel. Not intended for direct use.

    Parameters
    ----------
    E : float or u.Quantity['energy']
        If a float is provided, assumes the energy is
        normalized to 1e52 ergs.

    n17 : float or u.Quantity['number density']
        The number density at `ref` cm.

    k : float
        The density power-law index.
    """
    c = const.c.cgs.value  # type: ignore
    m_p = const.m_p.cgs.value  # type: ignore

    def __init__(self, E, n17, k, ref):
        self._E = E
        self._n17 = n17
        self.k = k
        self.r_ref = ref

    @property
    def E(self) -> float:
        """ Returns the energy normalized to 1e52 ergs. """
        return self._E

    @E.setter
    def E(self, e: float | u.Quantity):
        """
        Sets the energy normalized to 1e52 ergs.

        Parameters
        ----------
        e : float or astropy.units.Quantity['energy']
            The explosion energy. If a float is provided,
            assumes the energy is normalized to 1e52 ergs.
        """
        if isinstance(e, u.Quantity):
            e = e.to_value('erg') / 1e52
        self._E = e

    @property
    def n17(self) -> float:
        """ Returns the density normalization. """
        return self._n17

    @n17.setter
    def n17(self, n17) -> None:
        """
        Sets the density normalization as a simple float.

        Define rho as:

        rho = rho_x * R^-k = rho_0 * (R/R_0)^-k

        such that:

        rho_x = rho_0 * R_0^k = n0 * m_p * R_0^k

        where R_0 is the characteristic radius which is
        taken to be 1e17 cm. Then, `n17` is defined as
        the number density with respect to 1e17 cm.

        Parameters
        ----------
        n17 : float or u.Quantity['number density', 'mass density']
            The number density at 1e17 cm.
        """
        if isinstance(n17, u.Quantity):
            if n17.physical_type == 'number density':
                n17 = n17.cgs.value

            elif n17.physical_type == 'mass density':
                n17 = n17.cgs.value / self.m_p

        self._n17 = n17

    @property
    def alpha(self) -> float:
        """ Returns the hydrodynamic coefficient. """
        return 16 / (17 - 4 * self.k)

    @property
    def beta(self) -> float:
        """ Returns the hydrodynamic coefficient. """
        return 4 - self.k

    @property
    def rho17(self):
        """ Returns the density normalization. """
        return self.n17 * self.m_p * (self.r_ref ** self.k)


# noinspection PyPep8Naming
class BlastWaveModel(BaseBlastWaveModel):
    """
    Models a self-similar, ultra-relativistic blast wave.
    """
    def __init__(self, E, n17, k, ref=1e17):
        super().__init__(E, n17, k, ref)

    def lorentz_factor(self, z, t):
        """
        The Lorentz factor of the shocked fluid, gamma.

        Parameters
        ----------
        z : float
            The redshift.

        t : float or u.Quantity['time'] or np.ndarray
            The observer time in days.

        Returns
        -------
        float or u.Quantity['dimensionless'] or np.ndarray
            The Lorentz factor of the shocked fluid.
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        k = self.k

        # Convert to source frame time [s]
        t = t * 86_400 / (1 + z)

        return (
            self.alpha * self.beta ** (3 - k) *
            np.pi * self.c ** (5 - k) * self.rho17 *
            (1e52 * self.E) ** -1 * t ** (3 - k)
        ) ** -(0.5 / (4 - k))

    def shock_radius(self, z, t, t_decel=0.0):
        """
        The shock radius, R(t).

        Parameters
        ----------
        z : float
            The redshift.

        t : float or u.Quantity['time'] or np.ndarray
            The observer time [days since trigger].

        t_decel : float or u.Quantity['time'] or np.ndarray
            The burst frame (z=0) deceleration time
            of the blast wave [days since trigger].

        Returns
        -------
        float or u.Quantity['length'] or np.ndarray
            The shock radius evaluated at `t` [cm].
        """
        if isinstance(t, u.Quantity):
            if t.unit.physical_type != 'time':
                raise TypeError(
                    f'Expected a time Quantity. Received '
                    f'{t.unit.physical_type} instead.'
                )
            t = t.to_value('d')

        # Add the deceleration time [s]
        t = 86_400 * (t_decel + (t / (1 + z)))

        return (
            self.beta * 1e52 * self.E * t /
            (self.alpha * np.pi * self.rho17 * self.c)
        ) ** (1 / (4 - self.k))

    def decel_radius(self, gamma=300):
        """
        The burst-frame deceleration radius measured in cm.

        Parameters
        ----------
        gamma : float or np.ndarray, default=300
            The initial Lorentz factor.

        Returns
        -------
        float or np.ndarray
            The deceleration radius [cm].
        """
        return (
            ((3 - self.k) * 1e52 * self.E) /
            (4 * np.pi * self.rho17 * self.c ** 2 * gamma ** 2)
        ) ** (1 / (3 - self.k))

    def decel_time(self, gamma=300, z=0.0):
        """
        Calculates the deceleration time of the shock
        measured in seconds. If the redshift, `z`, is
        provided, returns the observer-frame time. Else,
        returns the burst frame time (i.e., z=0.0).

        Parameters
        ----------
        gamma : float or np.ndarray, default=300
            The initial Lorentz factor.

        z : float, optional, default=0.0
            The redshift.

        Returns
        -------
        float or np.ndarray
            The deceleration time [seconds since trigger].
        """
        return (1 + z) * (
            self.decel_radius(gamma) /
            ((4 - self.k) * gamma ** 2 * self.c)
        )


# noinspection PyPep8Naming
class OpeningAngleModel:
    """
    Jet opening angle model.

    Parameters
    ----------
    E : float
        The isotropic energy normalized to 1e52 erg.

    rho0 : float
        The number density. Normalized to the proton
        mass and (1e17cm)^k such that the units are
        1 / cm^3.

    k : float
        The density power-law index.

    z : float
        The redshift.
    """
    def __init__(self, E, rho0, k, z):
        self.rho0 = rho0
        self.E = E
        self.k = k
        self.z = z

    def __repr__(self):
        """ Human-readable string """
        return f'OpeningAngle(E={self.E}, rho0={self.rho0}, k={self.k})'

    def __call__(self, *args, **kwargs):
        """ Calls the evaluate method. """
        return self.evaluate(*args, **kwargs)

    @property
    def alpha(self) -> float:
        """ Returns the hydrodynamic coefficient. """
        return 16 / (17 - 4 * self.k)

    @property
    def beta(self) -> float:
        """ Returns the hydrodynamic coefficient. """
        return 4 - self.k

    def evaluate(self, t):
        """
        Evaluates the jet opening angle at the jet break
        time `t`.

        Parameters
        ----------
        t : float or np.ndarray of float
            The jet break time in days since trigger.

        Returns
        -------
        float or np.ndarray of float
            The jet opening angle.
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        rho_norm = 1.67e-24 * (1e17 ** self.k)

        # return the jet opening angle
        return (
            np.pi * self.alpha *
            (self.beta ** (3 - self.k)) *
            ((1 + self.z) ** -(3 - self.k)) *
            (2.99e10 ** (5 - self.k)) *     # [cm s-1] ^ (5-k)
            (rho_norm * self.rho0) *        # [g cm(k-3)]
            ((1e52 * self.E) **-1) *        # [g cm2 s-2] ^ -1
            ((86_400 * t) ** (3 - self.k))  # [s] ^ (3 - k)
        ) ** (0.5 / (4 - self.k))


class BaseFireballModel:
    """
    Base model. Not intended for direct use.

    Implements the ultra-relativistic shock moving into an
    external medium with density rho = rho_0 * R^-k.

    Attributes
    ----------
    E : float or astropy.units.Quantity['energy']
        The explosion energy [1e52 ergs].

    rho0 : float or astropy.units.Quantity['number density']
        The density normalization [cm-3].

    dL : float or astropy.units.Quantity['length']
        The luminosity distance to the event [1e28 cm].

    p : float
        The electron energy index.

    k : float or np.ndarray of float
        The density power-law index. The model requires that
        `k` < 4.

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons. Must be in the range [0, 1].

    z : float
        The redshift to the event.

    X : float
        The hydrogen mass fraction. Must be in the range [0, 1].
        0 indicates hydrogen depleted. 1 indicates hydrogen rich.

    References
    ----------
    [1] Broadband view of blast wave physics: A study
        of gamma-ray burst afterglows
    """
    m_p = const.m_p.cgs.value  # type: ignore

    # noinspection PyPep8Naming
    def __init__(self, E, p, eps_b, eps_e, z, dL, rho0, k, X):
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

    def __repr__(self):
        """ Human-readable representation. """
        name = self.__class__.__name__
        return f'{name}(E={self.E}, n={self.rho0}, .., p={self.p}, k={self.k})'

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

        rho_x = rho_0 * R_0^k = n0 * m_p * R_0^k

        where R_0 is the characteristic radius which is
        taken to be 1e17 cm. Then, `n17` is defined as
        the number density with respect to 1e17 cm.

        Parameters
        ----------
        rho0 : float or u.Quantity['number density', 'mass density']
            The number density at 1e17 cm.
        """
        if isinstance(rho0, u.Quantity):
            if rho0.unit.physical_type == 'number density':
                rho0 = rho0.cgs.value

            elif rho0.unit.physical_type == 'mass density':
                rho0 = rho0.cgs.value / self.m_p

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
            assumes the value is normalized to 1e28 cm.
        """
        if isinstance(d, u.Quantity):
            d = d.to_value('cm') / 1e28
        self._dL = d

    @property
    def is_physical(self):
        """ Whether the model is parameters are physically valid. """
        return self.eps_b + self.eps_e < 1.0


class BaseFluxModel:
    """
    Base flux model. Not intended for direct use.

    Attributes
    ----------
    f_peak : float or np.ndarray or u.Quantity['spectral flux density']
        The peak flux. If a simple float is provided,
        assumes it is measured in mJy.

    nu_m : float or np.ndarray or u.Quantity['frequency']
        The synchrotron frequency. If a simple float
        is provided, assumes it is measured in Hz.

    nu_c : float or np.ndarray or u.Quantity['frequency']
        The cooling frequency. If a simple float
        is provided, assumes it is measured in Hz.

    nu_a : float or np.ndarray or u.Quantity['frequency']
        The self-absorption frequency. If a simple float
        is provided, assumes it is measured in Hz.

    p : float
        The electron energy power-law index.

    k : float
        The circumburst density power-law index.
    """
    def __init__(self, nu_m, nu_c, nu_a, f_peak, p, k):
        self.f_peak = f_peak
        self.nu_m = np.atleast_1d(nu_m)
        self.nu_c = np.atleast_1d(nu_c)
        self.nu_a = np.atleast_1d(nu_a)

        if self.nu_m.size != self.nu_c.size != self.nu_a.size:
            raise ValueError(
                f'nu_m, nu_c, nu_a must have the same size: '
                f'{self.nu_m.size, self.nu_c.size, self.nu_a.size}'
            )

        self.p = p
        self.k = k

        # Determine regimes (slow, slow with self-absorption, fast)
        # TODO: Not updated when frequencies are updated. Not good
        # TODO: practice, but its faster. Revisit this before release.
        self.slow = self.nu_m < self.nu_c
        self.sabs = np.logical_and(self.slow, nu_m < nu_a)
        self.fast = ~self.slow

    @property
    def f_peak(self) -> float | np.ndarray:
        """ Returns the peak flux [mJy]. """
        return self._f_peak

    @f_peak.setter
    def f_peak(self, val):
        """
        Sets the peak flux in mJy.

        Parameters
        ----------
        val : float or np.ndarray or u.Quantity['spectral flux density']
            The peak flux. If a simple float is provided,
            assumes it is measured in mJy.
        """
        if isinstance(val, u.Quantity):
            val = val.to_value('mJy')
        self._f_peak = val

    @property
    def nu_m(self) -> float | np.ndarray:
        """ Returns the synchrotron frequency [Hz]. """
        return self._nu_m

    @nu_m.setter
    def nu_m(self, val):
        """
        Sets synchrotron frequency in Hz.

        Parameters
        ----------
        val : float or np.ndarray or u.Quantity['frequency']
            The synchrotron frequency. If a simple float
            is provided, assumes it is measured in Hz.
        """
        if isinstance(val, u.Quantity):
            val = val.to_value('Hz')
        self._nu_m = val

    @property
    def nu_c(self) -> float | np.ndarray:
        """ Returns the synchrotron frequency [Hz]. """
        return self._nu_c

    @nu_c.setter
    def nu_c(self, val):
        """
        Sets cooling frequency in Hz.

        Parameters
        ----------
        val : float or np.ndarray or u.Quantity['frequency']
            The cooling frequency. If a simple float is
            provided, assumes it is measured in Hz.
        """
        if isinstance(val, u.Quantity):
            val = val.to_value('Hz')
        self._nu_c = val

    @property
    def nu_a(self) -> float | np.ndarray:
        """ Returns the self-absorption frequency [Hz]. """
        return self._nu_a

    @nu_a.setter
    def nu_a(self, val) :
        """
        Sets self-absorption frequency in Hz.

        Parameters
        ----------
        val : float or np.ndarray or u.Quantity['frequency']
            The self-absorption frequency. If a simple float is
            provided, assumes it is measured in Hz.
        """
        if isinstance(val, u.Quantity):
            val = val.to_value('Hz')
        self._nu_a = val


class SpectralFluxModel(BaseFluxModel):
    """
    Spectral Fireball Flux Model

    The notation in this class is as follows:
        - b1 = spectral index of segment 1
        - s12 = smoothing between segments 1 and 2
        - nu12 = characteristic frequency at `v_12`

    F_ν
    │                     _
    │              _⎽⎽⎼⎼⎻⎻⎺⎺ ‾│‾---__
    │        _⎽⎽⎼⎼⎻⎻⎺⎺        │      ‾‾---__
    │      ╱ │            │            │\
    │     ╱  │            │            │ \
    │    ╱   │    seg 1   │    seg 2   │  \
    │   ╱    │            │            │   \
    │  ╱     │            │            │    \
    │ ╱      │            │            │
    ├───────────────────────────────────────────▶ ν
            v_0          v_1          v_2
    """
    def __init__(self, nu_m, nu_c, nu_a, f_peak, p, k):
        super().__init__(nu_m, nu_c, nu_a, f_peak, p, k)

    def __call__(self, nu):
        """ Calls the `evaluate` method. """
        return self.evaluate(nu)

    def model(self, val: SpectralFlux) -> float:
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
        Calculates the smoothed flux for frequency, `nu`.

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
            The observed frequency [Hz].

        Returns
        -------
        float or np.ndarray of float
            The modeled smoothed spectral flux [mJy].
        """
        nu = np.atleast_1d(nu)

        # Get stuff done
        nu12, nu23 = self.spectral_breaks()
        b1, b2, b3 = self.spectral_indices()
        s12, s23 = self.smoothing()

        # Transform for readability
        x12, x23 = nu / nu12, nu / nu23

        # Smooth the spectrum across spectral breaks
        flux = self.f_peak * (
            (x12 ** -(s12 * (b1 - b2)) + 1) ** (s23 / s12) * x12 ** -(s23 * b2) +
            ((nu23 / nu12) ** -(s23 * b2)) * (x23 ** -(s23 * b3))
        ) ** -(1 / s23)

        # return the smoothed spectral flux [mJy]
        return flux[0] if flux.size == 1 else flux

    def spectral_breaks(self):
        """
        Creates arrays critical frequencies that define
        the GRB spectrum.

        Returns
        -------
        tuple of np.ndarray of float
            The critical frequencies [Hz].
        """
        # Default: nu_a < nu_m < nu_c
        nu12 = np.array(self.nu_m, copy=True)
        nu23 = np.array(self.nu_c, copy=True)

        if self.fast.any():
            # Overwrite: nu_m < nu_a < nu_c
            nu12[self.fast] = self.nu_c[self.fast]
            nu23[self.fast] = self.nu_m[self.fast]

        if self.sabs.any():
            # Overwrite: nu_a < nu_c < nu_m
            nu12[self.sabs] = self.nu_a[self.sabs]

        return nu12, nu23

    def spectral_indices(self):
        """
        Calculates the spectral indices using Sari, Piran,
        & Narayan 1998 [1]_.

        Returns
        -------
        tuple of np.ndarray of float
            The spectral indices for each segment.

        References
        ----------
        .. [1] Sari, Piran, & Narayan (1998)
            https://iopscience.iop.org/article/10.1086/311269/pdf
        """
        # Default: nu_a < nu_m < nu_c
        b1 = np.full(self.fast.size, 1 / 3)
        b2 = np.full(self.fast.size, (1 - self.p) / 2)
        b3 = np.full(self.fast.size, -self.p / 2)

        if self.sabs.any():
            # Overwrite: nu_m < nu_a < nu_c
            b1[self.sabs] = 5/2

        if self.fast.any():
            # Overwrite: nu_a < nu_c < nu_m
            b2[self.fast] = -0.5

        return b1, b2, b3

    def smoothing(self):
        """
        Determines the smoothing factors between breaks.

        Supports smoothing between three segments / two breaks:
            - (nu_m, nu_c) for nu_a < nu_m < nu_c
            - (nu_a, nu_c) for nu_m < nu_a < nu_c
            - (nu_c, nu_m) for nu_a < nu_c < nu_m

        Smoothing factors are derived from Table 2, column s(p) in
        Granot & Sari 2002 [1]_. GS02 present smoothing factors
        for `k=0` and `k=2`. The smoothing factors used here are
        generalized for any value of `k`.

        Returns
        -------
        tuple of np.ndarray of float
            The smoothing factors.

        References
        ----------
        .. [1] Granot & Sari (2002)
            https://iopscience.iop.org/article/10.1086/338966
        """
        k, p = self.k, self.p

        # Generalized s(p) from GS02 for break 2 (s12) and break 3 (s23)
        if isinstance(k, np.ndarray):
            # Default: nu_a < nu_m < nu_c
            s12 = 1.84 - (0.040 * k) - (0.40 - 0.010 * k) * p
            s23 = 1.15 - (0.125 * k) - (0.06 - 0.015 * k) * p
        else:
            # Default: nu_a < nu_m < nu_c
            s12 = np.full(self.fast.size, 1.84 - (0.040 * k) - (0.40 - 0.010 * k) * p)
            s23 = np.full(self.fast.size, 1.15 - (0.125 * k) - (0.06 - 0.015 * k) * p)

        # Generalized s(p) from GS02 for break 9 (s23) and break 11 (s12)
        if self.fast.any():
            # Overwrite: nu_a < nu_c < nu_m
            fast_k = k[self.fast] if isinstance(k, np.ndarray) else k
            s23[self.fast] = 3.34 + 0.17 * fast_k - (0.82 + 0.035 * fast_k) * p
            s12[self.fast] = 0.597

        # Generalized s(p) from GS02 for break 5 (s12)
        if self.sabs.any():
            # Overwrite: nu_m < nu_a < nu_c
            sabs_k = k[self.sabs] if isinstance(k, np.ndarray) else k
            s12[self.sabs] = 1.47 - 0.11 * sabs_k - (0.21 - 0.015 * sabs_k) * p

        return s12, s23


class IntegratedFluxModel(BaseFluxModel):
    """
    Integrated Fireball Flux Model

    Provides methods for calculating integrated fluxes
    using the spectrum for the GRB fireball model.
    """
    def __init__(self, nu_m, nu_c, nu_a, f_peak, p, k):
        super().__init__(nu_m, nu_c, nu_a, f_peak, p, k)

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

    def evaluate(self, lower, upper):
        """
        Evaluates the integrated flux model using the
        `lower` and `upper` integration limits.

        Parameters
        ----------
        lower : float or np.ndarray of float
            The lower integration limit measured in Hz.

        upper : float or np.ndarray of float
            The upper integration limit measured in Hz.

        Returns
        -------
        float or np.ndarray of float
            The integrated flux with units of erg cm-2 s-1.
        """
        beta = SpectralIndexModel(
            self.nu_m, self.nu_c, self.nu_a, self.f_peak, self.p, self.k
        ).evaluate(lower, upper)

        flux = SpectralFluxModel(
            self.nu_m, self.nu_c, self.nu_a, self.f_peak, self.p, self.k
        ).evaluate(lower)

        # return the smoothed integrated flux [erg cm-2 s-1]
        return 1e-26 * (
            (flux * lower / (beta + 1)) *
            (((upper / lower) ** (beta + 1)) - 1)
        )


class SpectralIndexModel(BaseFluxModel):
    """
    Spectral Index Model
    """
    def __init__(self, nu_m, nu_c, nu_a, f_peak, p, k):
        super().__init__(nu_m, nu_c, nu_a, f_peak, p, k)

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
            self.nu_m, self.nu_c, self.nu_a, self.f_peak, self.p, self.k)

        # return the spectral index [dimension less]
        return (
            np.log10(model(upper) / model(lower)) /
            np.log10(upper / lower)
        )


class BaseSpectralModel:
    """
    Base Spectral Model. Not intended for direct use.

    Parameters
    ----------
    E : float or u.Quantity['energy']
        The explosion energy. If a float is provided, assumes
        that the value is already normalized to 1e52 erg. If
        passing a `Quantity`, it will be normalized before
        storing it as a float.

    eps_b : float
        The fraction of thermal energy in the magnetic field.

    k : float or np.ndarray of float
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

    def __call__(self, *args, **kwargs):
        """ Wrapper for the evaluate method. """
        return self.evaluate(*args, **kwargs)

    # noinspection PyPep8Naming
    @property
    def E(self) -> float:
        """ Returns the explosion energy normalized to 10e52 erg. """
        return self._E

    # noinspection PyPep8Naming
    @E.setter
    def E(self, e: float | u.Quantity) -> None:
        """
        Sets the explosion energy normalized to 10e52 erg.

        Parameters
        ----------
        e : float or astropy.units.Quantity
            The explosion energy. If a float is provided, assumes
            that the value is already normalized to 1e52 erg.
        """
        if isinstance(e, u.Quantity):
            e = e.to_value('erg') / 1e52
        self._E = e

    @property
    def alpha(self) -> float:
        """ Returns the hydrodynamic coefficient. """
        return 16 / (17 - 4 * self.k)

    @property
    def beta(self) -> float:
        """ Returns the hydrodynamic coefficient. """
        return 4 - self.k

    def evaluate(self, *args, **kwargs):
        """ Placeholder evaluate method. """
        raise NotImplementedError(f'evaluate not implemented.')


class PeakFluxModel(BaseSpectralModel):
    """
    Peak flux model. Assumes an ultra-relativistic shock moving
    through an external medium with rho = rho0 * R^-k density.
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

    def evaluate(self, t, ref=17):
        """
        Calculates the peak flux at time `t` for a shock's
        movement that is described by `evo`.

        Parameters
        ----------
        t : float or np.array of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        ref : float
            The radius normalization [cm] in log space.

        Returns
        -------
        float or np.array of u.Quantity['time']
            The peak flux at time `t` measured in mJy.
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # Convenience variables
        k, x = self.k, 4 - self.k

        # Evaluate exponents once
        exp_c   = -0.5 * (24 - 7 * k) / x
        exp_en  = 0.5 * (8 - 3 * k) / x
        exp_z   = 0.5 * (8 - k) / x
        exp_t   = -0.5 * k / x
        exp_rho = 2 / x

        # Exponents in log-space to prevent overflow
        log_pot = (
            (10 * exp_c) +                # speed of light [cm]
            (52 * exp_en) +               # 1e52 erg normalization
            ((ref * k - 24) * exp_rho) +  # proton mass [g] and radius normalization
            (4 * exp_t) -                 # time conversion (d -> s)
            8.0                           # e(q_e)^3 * e(m_e)^-1 * e(m_p)^-1 - e(dL)^2 + e(cgs->mJy)
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
            (self.rho0 ** exp_rho) *    # number density
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
    """

    # noinspection PyPep8Naming
    def __init__(self, E, rho0, eps_b, k, z):
        super().__init__(E, eps_b, k, z)
        self.rho0 = rho0

    def evaluate(self, t, ref=17):
        """
        Calculates the cooling frequency at time `t`
        for a shock's movement that is described by `evo`.

        Parameters
        ----------
        t : float or np.array of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        ref : float
            The radius normalization [cm] in log space.

        Returns
        -------
        float or np.array of float
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
            (4 * exp_t) + ((ref * k - 24) * exp_rho)
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

    def evaluate(self, t):
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
        t : float or np.array of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.array of float
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


# noinspection PyPep8Naming
class AbsorptionFrequencyModel(BaseSpectralModel):
    """
    Absorption frequency model. Assumes an ultra-relativistic
    shock moving through an external medium with rho = rho0
    * R^-k density.

    I implement only the slow-cooling scenarios:
        - nu_a < nu_m < nu_c
        - nu_m < nu_a < nu_c

    Although nu_a < nu_c < nu_m is perfectly valid, only the
    upper two of three breaks are considered when smoothing
    the flux. Thus, this scenario will never be applied.

    All other combinations, though physically plausible in
    extreme scenarios, are not implemented because they are
    considered physically unrealistic [1]_.

    Attributes
    ----------
    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons, unit=None.

    X : float
        The hydrogen mass fraction, unit=None.

    p : float
        The electron energy power-law index, unit=None.

    References
    ----------
    .. [1] Gao et al. (2013)
        https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2520G/abstract
    """
    c = const.c.cgs.value      # noqa
    m_p = const.m_p.cgs.value  # noqa
    m_e = const.m_e.cgs.value  # noqa
    q_e = 4.8032e-10           # [g1/2 cm3/2 s-1]

    def __init__(self, E, rho0, eps_e, eps_b, k, z, X, p):
        super().__init__(E, eps_b, k, z)
        self.eps_e = eps_e
        self.rho0 = rho0
        self.X = X
        self.p = p

    def evaluate(self, t, order, ref=17):
        """
        ??

        Parameters
        ----------
        t : float or np.array of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        order : str, {'amc', 'mac'}
            The order of the spectral breaks.

        ref : float, optional, default=17
            The reference radius [cm] in log space.

        Returns
        -------
        float or np.array of float
            The self-absorption frequency at time `t` measured in Hz.
        """
        return getattr(self, f'evaluate_{order}')(t, ref)

    def evaluate_amc(self, t, ref=17):
        """"""
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # convenience variables
        k, x = self.k, 4 - self.k

        # exponents for readability
        e_z     = -(0.8 * (5 - 2*k) / x)
        e_c     = -0.8 * ((5 - 2*k) / x)
        e_alpha = -(0.8 * (1 - k) / x)
        e_pi    = (0.2 * (4 + 2*k) / x)
        e_en    = (0.8 * (1 - k) / x)
        e_beta  = -(0.6*k / x)
        e_rho   = (2.4 / x)

        # return self-absorption frequency [Hz]
        return 10 ** (

            # Dimension-less quantities
            np.log10(2 * 3 ** 0.8) +
            e_alpha * np.log10(self.alpha) +        # hydrodynamics coefficient
            e_beta * np.log10(self.beta) +          # hydrodynamics coefficient
            e_pi * np.log10(np.pi) +                # pi
            (1.6 * np.log10(0.5 * (1 + self.X))) +  # hydrogen mass fraction
            1.6 * np.log10(self.p - 1) +            # electron energy index
            -0.6 * np.log10(self.p + 2/3) +         # electron energy index
            0.6 * np.log10(self.p + 2) +            # electron energy index

            # Dimensional quantities
            1.6 * np.log10(self.q_e) +   # electron charge [g1/2 cm3/2 s-1]
            -1.6 * np.log10(self.m_p) +  # proton mass [g]
            e_c * np.log10(self.c) +     # speed of light [cm s-1]

            # Model parameters
            -np.log10(self.eps_e) +                       # electron field energy fraction
            0.2 * np.log10(self.eps_b) +                  # magnetic field electron fraction
            e_rho * (ref*k + np.log10(self.m_p * self.rho0)) +  # density [g cm-3]
            e_en * (52 + np.log10(self.E)) +              # energy [erg]
            e_z * np.log10(1 + self.z) +                  # redshift

            # Evaluated at time, `t`
            -0.6 * (k / x) * np.log10(86_400 * t)  # time [s]
        ) * ((self.p - 2) ** -1)  # electron energy index (avoids inf for p < 2)

    def evaluate_mac(self, t, ref=17):
        """"""
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # Convenience variables
        p, k = self.p, self.k
        x, y, z = p + 2, p + 4, 4 - k

        # Transformations
        t = t * 86_400
        E = 1e52 * self.E
        rho0 = self.rho0 * self.m_p * (10 ** ref) ** k

        # Shared exponents
        exp_bt = -0.5 * (4 * (3 * p + 2) - k * (3 * p - 2)) / (y * z)
        exp_ae = 0.5 * (4 * x - k * (p + 6)) / (y * z)

        # Linear terms
        pre_factor = (
            (p - 2) ** (2 * (p - 1) / y) * (p - 1) ** -(2 * (p - 2) / y) * x ** (2 / y) *
            2 ** ((9 * p - 22) / (6 * y)) * 3 ** (8 / (3 * y)) *
            np.pi ** -(0.5 * (8 * x - 2 * k * y) / (4 - k) / y) *
            self.alpha ** -exp_ae * self.beta ** exp_bt *
            math.gamma(p / 2 + 1 / 3) ** (2 / y)
        )

        # return self-absorption frequency [Hz]
        return pre_factor * 10 ** (
            # Pre-factors
            np.log10(self.q_e) * ((p + 6) / y) +
            np.log10(self.m_e) * -((3 * p + 2) / y) +
            np.log10(self.m_p) * (2 * (p - 2) / y) +
            np.log10(self.c) * -((4 * (5 * p + 10) - k * (5 * p + 14)) / (2 * y * z)) +

            # Model parameters
            np.log10(1 + self.z) * (0.5 * (4 * (p - 6)  - k * (p - 10)) / (y * z)) +
            np.log10(0.5 * (1 + self.X)) * -(2 * x / y) +
            np.log10(self.eps_e) * (2 * (p - 1) / y) +
            np.log10(self.eps_b) * (0.5 * x / y) +
            np.log10(rho0) * (8 / z / y) +
            np.log10(E) * exp_ae +
            np.log10(t) * exp_bt
        )


class ObservedFluxModel:
    """
    Container for computing the observed afterglow flux.

    Parameters
    ----------
    afterglow_model :
        The afterglow flux model to use. Can be any custom
        defined model as long as it has a `model` method
        that takes an `Observation` and returns an array.

    extinction_model :
        The dust extinction model to use. Models from
        `dust_extinction` package or any custom object
        that has an `extinguish` method.

    ext_sf : np.array, optional
        The pre-computed source frame extinction values.

    ext_mw : np.array, optional
        The pre-computed milky way extinction values.
    """
    def __init__(
            self,
            afterglow_model,
            extinction_model,
            ext_sf=None,
            ext_mw=None,
    ):
        self.afterglow_model = afterglow_model
        self.extinction_model = extinction_model
        self.ext_sf = ext_sf
        self.ext_mw = ext_mw

    def __repr__(self):
        """ Human-readable representation. """
        return (
            f'ObservedFluxModel('
            f'ag={self.afterglow_model}, '
            f'ext={self.extinction_model})'
        )

    def __call__(self, *args, **kwargs):
        """ Calls the `model` method. """
        return self.model(*args, **kwargs)

    def model(self, obs, params, **kwargs):
        """
        Models the observed GRB afterglow flux.

        Parameters
        ----------
        obs : Observation
            The `Observation` object to model.

        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        kwargs : dict, optional
            Any additional arguments needed to instantiate the
            flux model.

        Returns
        -------
        np.ndarray of float
            The modeled observed GRB afterglow flux.
        """

        # Model the GRB afterglow flux
        modeled = self.model_afterglow(obs, params, **kwargs)

        # Apply dust extinction and host galaxy corrections
        modeled[obs.sflux_loc] = self.model_extinction(
            modeled[obs.sflux_loc], **Parameters.extinction(obs, params)
        )

        return modeled

    def model_afterglow(self, obs, params, **kwargs):
        """
        Models the unextinguished GRB afterglow flux.

        Parameters
        ----------
        obs : Observation
            The `Observation` object to model.

        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        kwargs : optional
            Any additional arguments needed to instantiate the
            flux model.

        Returns
        -------
        np.ndarray of float
            The modeled GRB afterglow flux.
        """
        if params.get('shared') is not None:
            # Model the afterglow flux with sets of parameters
            # applied to different subsets of the data.
            modeled = self.model_segmented_afterglow(
                obs, params, **kwargs
            )

        else:
            # Model the afterglow flux all together. Nice and simple.
            model = self.afterglow_model(**params.get('model'), **kwargs)
            modeled = model.model(obs)

        # return the unextinguished GRB afterglow flux
        return modeled

    def model_segmented_afterglow(self, obs, params, **kwargs):
        """
        Models the unextinguished GRB afterglow flux divided
        into an arbitrarily defined number of subsets.

        This method is useful for light curves that display
        different behaviors in different temporal regimes.
        Note that it is up to the user to determine whether
        fitting multiple sets of parameters is physically
        meaningful or not. This method simply provides the
        ability to do so.

        Parameters
        ----------
        obs : Observation
            The `Observation` object to model.

        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        kwargs : optional
            Any additional arguments needed to instantiate
            the flux model.

        Returns
        -------
        np.ndarray of float
            The modeled unextinguished GRB afterglow flux.
        """
        modeled = np.full(obs.length, np.nan, dtype=float)

        for group, mask in obs.data_groups.items():
            model_params = params.get(group).get('model')

            modeled[mask] = self.afterglow_model(
                **model_params, **kwargs).model(obs, mask)

        return modeled

    def model_extinction(
        self, modeled, wn, z=None, ebv_sf=None,
        ebv_mw=None, host_pos=None, host_vals=None,
        rv_sf=None, rv_mw=None
    ):
        """
        Corrects the afterglow flux, `modeled`, for
        dust extinction and host galaxy contributions.

        Parameters
        ----------
        modeled : np.array
            The modeled flux.

        wn : np.array
            The observed wave numbers measured in inverse
            microns.

        z : float, optional
            The redshift. If provided, transforms `wn` to
            the source frame when extinguishing for source
            frame dust.

        ebv_sf : float, optional
            The E(B - V) value for the source frame. The
            precomputed source frame extinction values are
            given priority over `ebv_sf` (if defined).

        ebv_mw : float, optional
            The E(B - V) value for the Milky Way. The
            precomputed source frame extinction values are
            given priority over `ebv_mw` (if defined).

        rv_sf, rv_mw : float, optional
            R(V) = A(V)/E(B-V) = total-to-selective extinction
            for source frame/Milky Way.

        host_pos, host_vals : dict, optional
            The positions and values of the host galaxy corrections.
            Both must be provided to apply host galaxy corrections.
            Assumes that the values are measured in the same space
            as the intrinsic flux.

        Returns
        -------
        np.ndarray of float
            The extinguished and host galaxy corrected flux.
        """

        # Apply source frame extinction
        if self.ext_sf is not None:
            modeled *= self.ext_sf  # pre-computed

        elif ebv_sf is not None:
            # Reuse model object if not fitting for Rv
            model = self.extinction_model if rv_sf is None \
                else self.extinction_model.__class__(Rv=rv_sf)
            modeled *= model.extinguish((1 + z) * wn, Ebv=ebv_sf)

        # Apply host galaxy correction
        if host_vals is not None and host_pos is not None:
            for name, corr in host_vals.items():
                modeled[np.where(host_pos[name])] += corr

        # Apply Milky Way extinction
        if self.ext_mw is not None:
            modeled *= self.ext_mw  # pre-computed

        elif ebv_mw is not None:
            # Reuse model object if not fitting for Rv
            model = self.extinction_model if rv_mw is None \
                else self.extinction_model.__class__(Rv=rv_mw)
            modeled *= model.extinguish(wn, Ebv=ebv_mw)

        # return (afterglow_flux * ext_sf + host_correction) * ext_mw
        return modeled


# noinspection PyPep8Naming
class StratifiedMediumModel:
    """

    Parameters
    ----------
    E : float
        The energy normalized to 1e52 ergs.

    n17_1 : float
        The density before the transition.

    n17_2 : float
        The density after the transition.

    k1 : float
        The density power-law index for `n1`.

    k2 : float
        The density power-law index for `n2`.
    """
    m_p = const.m_p.cgs.value  # type: ignore
    c = const.c.cgs.value  # type: ignore

    def __init__(self, E, n17_1, n17_2, k1, k2, ref=1e17):
        self.E = E
        self.n17_1 = n17_1
        self.n17_2 = n17_2
        self.k1 = k1
        self.k2 = k2
        self.r_ref = ref

    @staticmethod
    def alpha(k) -> float:
        """ Returns the hydrodynamic coefficient. """
        return 16 / (17 - 4 * k)

    @staticmethod
    def beta(k) -> float:
        """ Returns the hydrodynamic coefficient. """
        return 4 - k

    def rho17(self, n17, k) -> float:
        """ Returns the density at 1e17cm. """
        return n17 * self.m_p * (self.r_ref ** k)

    def transition_radius(self):
        """
        Calculates the transition radius in a stratified
        density.

        Returns
        -------
        float
            The transition radius [cm].
        """
        # Evaluate in log space to prevent overflow when k1 ~= k2
        return 10 ** (np.log10(self.r_ref) + np.log10(self.n17_2 / self.n17_1) / (self.k2 - self.k1))

    def transition_time(self, z):
        """
        Calculates the observer-frame transition time
        in a stratified density.

        Uses the relation R = beta * gamma ** 2 * c * t and
        the definition of gamma to solve for t.

        Parameters
        ----------
        z : float
            The redshift.

        Returns
        -------
        float
            The observer-frame transition time [s].

        See Also
        --------
        `models2.basemodels.BlastWaveModel.lorentz_factor`
            See for the definition of gamma.
        """
        r = self.transition_radius()
        a, b = self.alpha(self.k1), self.beta(self.k1)
        rho17 = self.rho17(self.n17_1, self.k1)

        return (1 + z) * (
            (a * b ** (3 - self.k1) * np.pi *
            self.c * rho17 / (1e52 * self.E))
        ) * (r / b) ** (4 - self.k1)
