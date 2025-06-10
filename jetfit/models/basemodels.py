import math
import numpy as np
import astropy.units as u
import astropy.constants as const

from jetfit.core.values import SpectralFlux, IntegratedFlux, SpectralIndex
from jetfit.mcmc.parameters.parameters import Parameters


def has_fts_transition(nu_m, nu_c) -> bool:
    """
    Is there a fast-to-slow cooling transition?

    Parameters
    ----------
    nu_m, nu_c : np.ndarray or list
        Synchrotron (nu_m) and cooling (nu_c) frequencies.

    Returns
    -------
    bool
        True if there is a transition from fast to slow.
    """
    return np.sign(nu_c[0] - nu_m[0]) < np.sign(nu_c[-1] - nu_m[-1])


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

        t : float or np.ndarray
            The observer time [d].

        Returns
        -------
        float or np.ndarray
            The Lorentz factor of the shocked fluid.
        """
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

        t : float or np.ndarray
            The observer time [d].

        t_decel : float or np.ndarray
            The burst frame (z=0) deceleration time
            of the blast wave [d].

        Returns
        -------
        float or np.ndarray
            The shock radius evaluated at `t` [cm].
        """
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
            The deceleration time [s].
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
        time ``t``.

        Parameters
        ----------
        t : float or np.ndarray of float
            The jet break time [d].

        Returns
        -------
        float or np.ndarray of float
            The jet opening angle [rad].
        """
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


class ObservedSpectrumModel:
    """
    GRB spectrum for an observational data set.

    The fireball model classes are parameterized by the
    intrinsic properties of the afterglow. However, the
    critical frequencies and peak flux are time-dependent
    and typically calculated using observation times.

    Because of this, it doesn't really make sense to store
    the characteristics in the Fireball classes. However,
    they are used all over the place which means that I
    was constantly passing them to methods which was very
    cumbersome.

    This class is a way to keep the calculations fast but
    in an organized way. It can be used directly, but it
    probably isn't very useful outside its original
    purpose.

    Parameters
    ----------
    nu_m, nu_c : np.ndarray of float
        The characteristic frequencies [Hz].

    nu_a : np.ndarray of float, optional
        The self-absorption frequency.

    f_peak : np.ndarray of float
        The peak fluxes [mJy].

    p : float
        The electron energy index.

    k : np.ndarray of float or float
        The density power-law index.

    arrays : ObsArray
        Array representation of the `Observation` object.

    fts : bool, optional, default=`has_fts_transition()`
        Model a fast-to-slow transition?

    jet : JetBreakModel, optional
        The jet break spectrum and smoothing parameters.

    sharp : bool, optional, default=True
        Model the flux with a smoothly broken power law?
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k, arrays, nu_a=None, fts=None, jet=None, sharp=False):
        self.nu_a = nu_a
        self.nu_m = nu_m
        self.nu_c = nu_c
        self.f_peak = f_peak
        self.p = p
        self.k = k

        self.arrays = arrays
        self.sharp = sharp
        self.has_fts = has_fts_transition(
            self.nu_m, self.nu_c) if fts is None else fts
        self.jet = jet

    @property
    def is_valid(self) -> bool:
        """ Not valid when nu_a > nu_m and nu_c. """
        if self.nu_a is not None:
            return not np.logical_and(
                self.nu_a > self.nu_m, self.nu_a > self.nu_c
            ).any()
        return True

    def model(self, subset=None) -> np.ndarray:
        """
        Model the observed spectrum using the observational
        properties in `arrays`.

        Parameters
        ----------
        subset : np.ndarray
            The subset of data to model.

        Returns
        -------
        np.ndarray of float
            The unextinguished modeled GRB flux.
        """
        if not self.is_valid:
            return np.array([np.nan])

        res = np.full(self.arrays.times.size, np.nan)

        if (sfm := self.arrays.sflux_loc).any():
            if subset: sfm = np.logical_and(sfm, subset)
            res[sfm] = self.spectral_flux(sfm)

        if (ifm := self.arrays.iflux_loc).any():
            if subset: ifm = np.logical_and(ifm, subset)
            res[ifm] = self.integrated_flux(ifm)

        if (sim := self.arrays.sindex_loc).any():
            if subset: sim = np.logical_and(sim, subset)
            res[sim] = self.spectral_index(sim)

        return res[subset] if subset is not None else res

    def spectral_flux(self, mask):
        """
        Model the unextinguished spectral flux.

        Parameters
        ----------
        mask : np.ndarray of bool
            The spectral flux locations.

        Returns
        -------
        np.ndarray of float
            The unextinguished spectral flux.
        """
        jet = self.jet.subset(mask) if self.jet else None

        return SpectralFluxModel(**self.spectrum(mask))(
            self.arrays.frequencies[mask], self.has_fts, jet, self.sharp
        )

    def integrated_flux(self, mask):
        """
        Model the unextinguished spectral flux.

        Parameters
        ----------
        mask : np.ndarray of bool
            The integrated flux locations.

        Returns
        -------
        np.ndarray of float
            The unextinguished integrated flux.
        """
        jet = self.jet.subset(mask) if self.jet else None

        return IntegratedFluxModel(**self.spectrum(mask))(
            self.arrays.if_lower_freqs[mask],
            self.arrays.if_upper_freqs[mask],
            self.has_fts, jet, self.sharp
        )

    def spectral_index(self, mask):
        """
        Model the spectral indices.

        Parameters
        ----------
        mask : np.ndarray of bool
            The spectral index locations.

        Returns
        -------
        np.ndarray of float
            The spectral indices.
        """
        jet = self.jet.subset(mask) if self.jet else None

        return SpectralIndexModel(**self.spectrum(mask))(
            self.arrays.si_lower_freqs[mask],
            self.arrays.si_upper_freqs[mask],
            self.has_fts, jet, self.sharp
        )

    def spectrum(self, mask=None):
        """
        Returns the spectrum properties as a dict and
        filters based on `mask`.

        Parameters
        ----------
        mask : np.ndarray of bool
            The spectral index locations.

        Returns
        -------
        dict
            The spectrum properties.
        """
        k = nu_a = nu_m = nu_c = f_pk = None

        if mask is not None:
            # Can be an array for stratified mediums
            if isinstance(self.k, np.ndarray):
                k = self.k[mask]

            # All or none are arrays
            if isinstance(self.nu_m, np.ndarray):
                if self.nu_a is not None:
                    nu_a = self.nu_a[mask]
                nu_m = self.nu_m[mask]
                nu_c = self.nu_c[mask]
                f_pk = self.f_peak[mask]

        # return a masked dict representation
        return {
            'nu_a': nu_a if nu_a is not None else self.nu_a,
            'nu_m': nu_m if nu_m is not None else self.nu_m,
            'nu_c': nu_c if nu_c is not None else self.nu_c,
            'f_peak': f_pk if f_pk is not None else self.f_peak,
            'k': k if k is not None else self.k, 'p': self.p
        }


class BaseFireballModel:
    """
    Base model. Not intended for direct use.

    Implements the ultra-relativistic shock moving into an
    external medium with density rho = rho_0 * R^-k.

    Parameters
    ----------
    E : float or astropy.units.Quantity['energy']
        The explosion energy [1e52 ergs].

    dL : float or astropy.units.Quantity['length']
        The luminosity distance to the event [1e28 cm]. Requiring
        the distance to be provided in addition to the redshift
        prevents the need to assume a cosmology here.

    p : float
        The electron energy index. Must be > 2.

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons. Must be in the range [0, 1].

    z : float
        The redshift of the event.

    X : float
        The hydrogen mass fraction. Must be in the range [0, 1].
        0 indicates hydrogen depleted. 1 indicates hydrogen rich.

    tj : float, optional
        The jet break time in days.

    sj, sji : float, optional
        The jet break smoothing factor. `sji` is the inverse
        smoothing factor. Useful for changing MCMC basis.

    use_sa : bool, optional, default=True
        Should self-absorption be modeled?

    References
    ----------
    [1] Broadband view of blast wave physics: A study
        of gamma-ray burst afterglows
    """
    m_p = const.m_p.cgs.value  # type: ignore

    # noinspection PyPep8Naming
    def __init__(self, E, p, eps_b, eps_e, z, dL, X, tj=None, sj=None, sji=None, use_sa=True):
        # Intrinsic properties
        self.E = E
        self.p = p
        self.eps_b = eps_b
        self.eps_e = eps_e
        self.X = X

        # Extrinsic properties
        self.dL = dL
        self.z = z

        # Jet properties
        self.tj = tj
        self.sj = (sj or 1 / sji) if (sj or sji) else None

        self.use_sa = use_sa

    def __repr__(self):
        """ Human-readable representation. """
        return f'{self.__class__.__name__}(E={self.E}, p={self.p}, .., z={self.z})'

    # noinspection PyPep8Naming
    @property
    def E(self) -> float:
        """ Returns the explosion energy normalized to 1e52 ergs. """
        return self._E

    # noinspection PyPep8Naming
    @E.setter
    def E(self, e: float | u.Quantity) -> None:
        """
        Sets the explosion energy normalized to 1e52 ergs.

        Parameters
        ----------
        e : float or astropy.units.Quantity
            The explosion energy. If a float is provided, assumes
            that the value is already normalized to 1e52 ergs.
        """
        if isinstance(e, u.Quantity):
            e = e.to_value('erg') / 1e52
        self._E = e

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
    def is_valid(self) -> bool:
        """ Whether the model is parameters are valid. """
        # Smoothing parameters are unstable around 0
        if self.sj is not None and abs(self.sj) <= 0.1:
            return False
        return ((self.eps_b + self.eps_e) < 1.0) and (self.p >= 2.0)

    def spectrum(self, *args, **kwargs):
        """ Placeholder. """
        raise NotImplementedError('spectrum is not implemented.')

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

    def spectral_flux(self, t, f, fts=False):
        """
        Calculates the spectral fluxes at times `t` for the
        frequencies `f`.

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
        `models.basemodels.SpectralFluxModel.evaluate`
            See for information on how various shapes
            of t and f are handled.
        """
        return SpectralFluxModel(**self.spectrum(t)).evaluate(
            f, fts, self.jet_break(t)
        )

    def integrated_flux(self, t, lower, upper, fts=False):
        """
        Calculates the integrated fluxes at times `t` for the
        lower and upper integration bounds, `lower` and `upper`.

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
            lower, upper, fts, self.jet_break(t)
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
        `models.basemodels.SpectralFluxModel.evaluate`
            See for information on how various shapes
            of t, lower, upper are handled.
        """
        return SpectralIndexModel(**self.spectrum(t)).evaluate(
            lower, upper, fts, self.jet_break(t)
        )


class BaseFluxModel:
    """
    Base flux model. Not intended for direct use.

    Parameters
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

    Attributes
    ----------
    slow, fast, mac, cam : np.ndarray of bool
        Bools indicating the regimes.
            - slow: nu_m < nu_c
            - fast: nu_c < nu_m
            - mac: nu_m < nu_a < nu_c
            - cam: nu_c < nu_a < nu_m
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k, nu_a=None):
        self.f_peak = f_peak
        self.nu_m = np.atleast_1d(nu_m)
        self.nu_c = np.atleast_1d(nu_c)
        self.nu_a = np.atleast_1d(nu_a) if nu_a is not None else None

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
        self.fast = ~self.slow

        if self.nu_a is not None:
            self.mac = np.logical_and(self.slow, nu_m < nu_a)
            self.cam = np.logical_and(self.fast, nu_c < nu_a)
        else:
            self.mac = self.cam = None

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

    def spectral_breaks(self) -> tuple:
        """
        Creates arrays of critical frequencies that define
        the GRB spectrum.

        Returns
        -------
        tuple of np.ndarray of float
            The critical frequencies [Hz].
        """
        # Default: nu_a < nu_m < nu_c
        nu12 = np.array(self.nu_m, copy=True)
        nu23 = np.array(self.nu_c, copy=True)

        if self.mac is not None and self.mac.any():
            # Overwrite: nu_m < nu_a < nu_c
            nu12[self.mac] = self.nu_a[self.mac]

        if self.fast.any():
            # Overwrite: nu_a < nu_c < nu_m
            nu12[self.fast] = self.nu_c[self.fast]
            nu23[self.fast] = self.nu_m[self.fast]

            if self.cam is not None and self.cam.any():
                # Overwrite: nu_c < nu_a < nu_m
                nu12[self.cam] = self.nu_a[self.cam]

        return nu12, nu23

    def spectral_indices(self, fts=False) -> tuple:
        """
        Calculates the spectral indices using Sari, Piran,
        & Narayan 1998 [1]_.

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

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

        if self.mac is not None and self.mac.any():
            # Overwrite: nu_m < nu_a < nu_c
            b1[self.mac] = 2.5

        if self.fast.any():
            # Overwrite: nu_a < nu_c < nu_m
            b2[self.fast] = -0.5

            if self.cam is not None and self.cam.any():
                # Overwrite: nu_c < nu_a < nu_m
                b1[self.cam] = 2

        # b2 smoothing
        if fts:
            s12, s23 = self.smoothing()
            nu_ratio = self.nu_m / self.nu_c

            # Transition smoother
            q12 = -s12 * (b3 - b1)
            q23 = -s23 * (b3 - b1)

            b2a = -0.5 + ((1 - self.p) / 2 - -0.5) / (1 + nu_ratio ** q12)
            b2b = -0.5 + ((1 - self.p) / 2 - -0.5) / (1 + nu_ratio ** q23)

            return b1, b2a, b2b, b3

        return b1, b2, b3

    def smoothing(self, fts=False):
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

        Parameters
        ----------
        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

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
        # Default: nu_a < nu_m < nu_c
        s12 = np.full(self.fast.size, 1.84 - (0.040 * k) - (0.40 - 0.010 * k) * p)
        s23 = np.full(self.fast.size, 1.15 - (0.125 * k) - (0.06 - 0.015 * k) * p)

        # Generalized s(p) from GS02 for break 5 (s12)
        if self.mac is not None and self.mac.any():
            # Overwrite: nu_m < nu_a < nu_c
            sabs_k = k[self.mac] if isinstance(k, np.ndarray) else k
            s12[self.mac] = 1.47 - 0.11 * sabs_k - (0.21 - 0.015 * sabs_k) * p

        # Generalized s(p) from GS02 for break 9 (s23) and break 11 (s12)
        if self.fast.any():
            # Overwrite: nu_a < nu_c < nu_m
            fast_k = k[self.fast] if isinstance(k, np.ndarray) else k
            s23[self.fast] = 3.34 + 0.17 * fast_k - (0.82 + 0.035 * fast_k) * p
            s12[self.fast] = 0.597

            # Generalized s(p) from GS02 for break 8 (s12)
            if self.cam is not None and self.cam.any():
                # Overwrite: nu_c < nu_a < nu_m
                s12[self.cam] = 0.9

        # Fast-to-slow cooling smoothing
        if fts:
            b1, _, b3 = self.spectral_indices()
            nu_ratio =  self.nu_m / self.nu_c

            # Transition smoother
            q12 = -s12 * (b3 - b1)
            q23 = -s23 * (b3 - b1)

            # S12 smoothing
            s12_slow = 1.84 - (0.040 * k) - (0.40 - 0.010 * k) * p
            s12_fast = 0.597
            s12 = s12_fast + (s12_slow - s12_fast) / (1 + nu_ratio ** q12)

            # s23 smoothing
            s23_fast = 3.34 + 0.17 * k - (0.82 + 0.035 * k) * p
            s23_slow = 1.15 - (0.125 * k) - (0.06 - 0.015 * k) * p
            s23 = s23_fast + (s23_slow - s23_fast) / (1 + nu_ratio ** q23)

        return s12, s23


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
    def __init__(self, nu_m, nu_c, f_peak, p, k, nu_a=None):
        super().__init__(nu_m, nu_c, f_peak, p, k, nu_a)

    def __call__(self, *args, **kwargs):
        """ Calls the `evaluate` method. """
        return self.evaluate(*args, **kwargs)

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

    def evaluate_sharp(self, nu, jet=None):
        """ WIP. Ignores self-absorption. """
        nu = np.atleast_1d(nu)
        nu12, nu23 = self.spectral_breaks()
        b1, b2, b3 = self.spectral_indices()

        res = np.full(nu12.size, self.f_peak)

        if nu.size == 1 and nu12.size != 1:
            nu = np.full(nu12.size, nu[0])

        # Segment boundaries
        seg0 = nu <= nu12
        seg1 = (nu > nu12) & (nu < nu23)
        seg2 = nu >= nu23

        # Power-law segments
        res[seg0] *= (nu[seg0] / nu12[seg0]) ** b1[seg0]
        res[seg1] *= (nu[seg1] / nu12[seg1]) ** b2[seg1]
        res[seg2] *= (nu23[seg2] / nu12[seg2]) ** b2[seg2] * (nu[seg2] / nu23[seg2]) ** b3[seg2]

        # Smooth across the jet break
        if jet is not None:
            res = jet.smooth(res, nu)  # type: ignore

        return res[0] if res.size == 1 else res

    def evaluate(self, nu, fts=False, jet=None, sharp=False):
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
        nu : float or np.ndarray of float
            The observed frequency [Hz].

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        jet : JetBreakModel, optional
            Smooths the flux across the jet break.

        sharp : bool, optional, default=False
            Model the spectrum using sharply broken power laws?

        Returns
        -------
        float or np.ndarray of float
            The modeled smoothed spectral flux [mJy].
        """
        nu = np.atleast_1d(nu)

        if sharp:
            return self.evaluate_sharp(nu, jet)

        # Get stuff done
        nu12, nu23 = self.spectral_breaks()
        s12, s23 = self.smoothing(fts)

        # Transform for readability
        x12, x23 = nu / nu12, nu / nu23

        if fts:
            b1, b2a, b2b, b3 = self.spectral_indices(fts)
        else:
            b1, b2, b3 = self.spectral_indices()
            b2a = b2b = b2

        # Smooth the spectrum across spectral breaks
        flux = self.f_peak * (
            (x12 ** -(s12 * (b1 - b2a)) + 1) ** (s23 / s12) * x12 ** -(s23 * b2a) +
            ((nu23 / nu12) ** -(s23 * b2b)) * (x23 ** -(s23 * b3))
        ) ** -(1 / s23)

        # Apply flux normalization corrections
        if self.mac is not None and self.mac.any():
            flux = self.correct_mac_flux(flux, b2a)

        if self.cam is not None and self.cam.any():
            flux = self.correct_cam_flux(flux, nu)  # type: ignore

        # Smooth across the jet break
        if jet is not None:
            flux = jet.smooth(flux, nu, fts)  # type: ignore

        # return the smoothed spectral flux [mJy]
        return flux[0] if flux.size == 1 else flux

    def correct_mac_flux(self, flux, b2):
        """
        Applies the peak flux adjustment in the m < a < c regime.

        Parameters
        ----------
        flux : np.ndarray of float
            The flux to adjust [mJy].

        b2 : np.ndarray of float
            The spectral index of the second segment.

        Returns
        -------
        np.ndarray of float
            The corrected flux [mJy].
        """
        corr = (self.nu_a[self.mac] / self.nu_m[self.mac]) ** b2[self.mac]

        if flux.size == self.mac.size:
            flux[self.mac] *= corr
        else:
            flux *= corr

        return flux

    def correct_cam_flux(self, flux, nu):
        """
        Adjusts the flux at ``nu`` > ``nu_a`` that accounts
        for the electron pile at low frequencies.

        Parameters
        ----------
        flux : np.ndarray of float
            The flux to adjust [mJy].

        nu : np.ndarray of float
            The observed frequencies [Hz].

        Returns
        -------
        np.ndarray of float
            The corrected flux [mJy].
        """
        mask = np.logical_and(nu > self.nu_a, self.cam)

        if flux.size == self.cam.size:
            flux[mask] *= (1 / 3) * np.sqrt(self.nu_c[mask] / self.nu_a[mask])
        else:
            flux[mask] *= (1 / 3) * np.sqrt(self.nu_c / self.nu_a)

        return flux


class IntegratedFluxModel(BaseFluxModel):
    """
    Integrated Fireball Flux Model

    Provides methods for calculating integrated fluxes
    using the spectrum for the GRB fireball model.
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k, nu_a=None):
        super().__init__(nu_m, nu_c, f_peak, p, k, nu_a)

    def __call__(self, *args, **kwargs):
        """ Calls the `evaluate` method. """
        return self.evaluate(*args, **kwargs)

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

    def evaluate(self, lower, upper, fts=False, jet=None, sharp=False):
        """
        Evaluates the integrated flux model using the
        `lower` and `upper` integration limits.

        Parameters
        ----------
        lower : float or np.ndarray of float
            The lower integration limit [Hz].

        upper : float or np.ndarray of float
            The upper integration limit [Hz].

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        jet : JetBreakModel, optional, default=None
            Smooths the flux across the jet break.

        sharp : bool, optional, default=False
            Model the spectrum using sharply broken power laws?

        Returns
        -------
        float or np.ndarray of float
            The integrated flux with units of erg cm-2 s-1.
        """
        beta = SpectralIndexModel(
            self.nu_m, self.nu_c, self.f_peak, self.p, self.k, self.nu_a
        ).evaluate(lower, upper, fts, jet, sharp)

        flux = SpectralFluxModel(
            self.nu_m, self.nu_c, self.f_peak, self.p, self.k, self.nu_a
        ).evaluate(lower, fts, jet, sharp)

        # return the smoothed integrated flux [erg cm-2 s-1]
        return 1e-26 * (
            (flux * lower / (beta + 1)) *
            (((upper / lower) ** (beta + 1)) - 1)
        )


class SpectralIndexModel(BaseFluxModel):
    """
    Spectral Index Model
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k, nu_a=None):
        super().__init__(nu_m, nu_c, f_peak, p, k, nu_a)

    def __call__(self, *args, **kwargs):
        """ Calls the `evaluate` method. """
        return self.evaluate(*args, **kwargs)

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

    def evaluate(self, lower, upper, fts=False, jet=None, sharp=False):
        """
        Approximates the spectral index using a two
        point approximation.

        Parameters
        ----------
        lower : float or np.ndarray of float
            The lower integration limit.

        upper : float or np.ndarray of float
            The upper integration limit.

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        jet : JetBreakModel, optional, default=None
            Smooths the flux across the jet break.

        sharp : bool, optional, default=False
            Model the spectrum using sharply broken power laws?

        Returns
        -------
        float or np.ndarray of float
            The modeled spectral index.
        """
        model = SpectralFluxModel(
            self.nu_m, self.nu_c, self.f_peak, self.p, self.k, self.nu_a)

        # return the spectral index [dimension less]
        return (
            np.log10(
                model(upper, fts, jet, sharp) /
                model(lower, fts, jet, sharp)
            ) /
            np.log10(upper / lower)
        )


class JetBreakModel:
    """
    Models a jet break in the afterglow light curve.

    Parameters
    ----------
    t_jet : float
        The jet-break time [d].

    t_obs : np.ndarray of float
        The times to smooth over [d].

    p : float
        The electron energy index.

    s : float, optional, default=3
        The smoothing parameter.
    """
    def __init__(self, f_jet, t_jet, t_obs, p, s=3):
        self.f_jet = f_jet
        self.t_jet = t_jet
        self.t_obs = t_obs
        self.p = p
        self.s = s

    def __repr__(self):
        """ Human-readable representation """
        return f'JetBreakModel(t_jet={self.t_jet}, .., s={self.s})'

    def __call__(self, *args, **kwargs):
        """ Calls the smooth method. """
        return self.smooth(*args, **kwargs)

    def subset(self, mask):
        """ Returns a `JetBreakModel` with a subset of times. """
        return self.__class__(
            self.f_jet, self.t_jet, self.t_obs[mask], self.p, self.s
        )

    def smooth(self, f_obs, nu, fts=False):
        """
        Smooths the flux `f_obs` with the jet flux via
        a smoothly broken power law.

        Parameters
        ----------
        f_obs : np.ndarray of float
            The modeled jet-break flux [mJy or erg cm-2 s-1].

        nu : np.ndarray of float

        fts : bool, optional, default=False
            Is there a fast-to-slow cooling transition?

        Returns
        -------
        np.ndarray of float
            The smoothed flux [mJy or erg cm-2 s-1].
        """
        f_jet = self.f_jet(nu, fts)

        return (
            f_obs ** -self.s +
            (f_jet * (self.t_obs / self.t_jet) ** -self.p) ** -self.s
        ) ** -(1 / self.s)


class BaseSpectralModel:
    """
    Base Spectral Model. Not intended for direct use.

    Parameters
    ----------
    E : float
        The explosion energy normalized to 1e52 erg.

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

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
        """
        Makes the class instance callable. This behaves like
        self.evaluate(*args, **kwargs).
        """
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

    Parameters
    ----------
    rho0 : float or u.Quantity
        The number density normalization [cm-3].
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
        Calculates the peak flux at time `t`.

        Parameters
        ----------
        t : float or np.array of float
            The observer time(s) [d].

        ref : float, default=17
            The log of the characteristic radius [cm].

        Returns
        -------
        float or np.array of float
            The peak flux [mJy] at time `t`.
        """
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
        Calculates the cooling frequencies at times `t`.

        Parameters
        ----------
        t : float or np.array of float
            The observer time(s) [d].

        ref : float, default=17
            The log of the characteristic radius [cm].

        Returns
        -------
        float or np.array of float
            The cooling frequency [Hz] at time(s) `t`.
        """
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
        The fraction of thermal energy in the electric field.
        Must be in the range [0, 1].

    X : float
        The hydrogen mass fraction. Must be in the range [0, 1].
        0 indicates hydrogen depleted. 1 indicates hydrogen rich.

    p : float
        The electron energy power-law index.
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
        t : float or np.array of float
            The observer times [d].

        Returns
        -------
        float or np.array of float
            The cooling frequency at time `t` measured in Hz.
        """
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

    Attributes
    ----------
    eps_e : float
        The fraction of thermal energy in the electric field.
        Must be in the range [0, 1].

    X : float
        The hydrogen mass fraction. Must be in the range [0, 1].
        0 indicates hydrogen depleted. 1 indicates hydrogen rich.

    p : float
        The electron energy power-law index.
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
        t : float or np.array of float
            The observer times [d].

        order : str, {'amc', 'mac', 'cam', 'acm'}
            The order of the spectral breaks.

        ref : float, optional, default=17
            The log of the characteristic radius [cm].

        Returns
        -------
        float or np.array of float
            The self-absorption frequency [Hz] at time(s) `t`.
        """
        return getattr(self, f'evaluate_{order}')(t, ref)

    def evaluate_amc(self, t, ref=17):
        """
        Calculates the self-absorption frequency in the weak
        self-absorption regime (nu_c < nu_a < nu_m).

        Parameters
        ----------
        t : np.ndarray of float or float
            The observer times [d].

        ref : float
            The log of the characteristic radius [cm].

        Returns
        -------
        np.ndarray of float or float
            The self-absorption frequencies [Hz] at time(s) t.
        """
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
        """
        Calculates the self-absorption frequency in the weak
        self-absorption regime (nu_m < nu_a < nu_c).

        Parameters
        ----------
        t : np.ndarray of float or float
            The observer times [d].

        ref : float
            The log of the characteristic radius [cm].

        Returns
        -------
        np.ndarray of float or float
            The self-absorption frequencies [Hz] at time(s) t.
        """
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

    def evaluate_cam(self, t, ref=17):
        """
        Calculates the self-absorption frequency in the strong
        self-absorption regime (nu_c < nu_a < nu_m).

        Parameters
        ----------
        t : np.ndarray of float or float
            The observer times [d].

        ref : float
            The log of the characteristic radius [cm].

        Returns
        -------
        np.ndarray of float or float
            The self-absorption frequencies [Hz] at time(s) t.
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # Convenience variables
        p, k = self.p, self.k
        x = 3 * (4 - k)

        # Transformations
        t = t * 86_400
        E = 1e52 * self.E
        rho0 = self.rho0 * self.m_p * (10 ** ref) ** k

        pre_factor = (
            0.95188438 * self.m_p ** -(1 / 3) * (
                self.alpha ** (k - 2) * self.beta ** -2 * np.pi ** (2 * (k - 3))
            ) ** (1 / x)
        ) * (0.5 * (1 + self.X)) ** (1 / 3)

        # return self-absorption frequency [Hz]
        return pre_factor * 10 ** (
            (   # Model parameters
                np.log10(1 + self.z) * 2 * (k - 3) +
                np.log10(self.c) * 2 +
                np.log10(rho0) * 2 +
                np.log10(E) * (2 - k) +
                np.log10(t) * (k - 6)
            ) / x
        )

    def evaluate_acm(self, t, ref=17):
        """
        Calculates the self-absorption frequency in the weak
        self-absorption regime (nu_a < nu_c < nu_m).

        Evaluated in log-space to prevent overflow.

        Parameters
        ----------
        t : np.ndarray of float or float
            The observer-frame times [days].

        ref : float
            The log of the reference radius measured in cm.

        Returns
        -------
        np.ndarray of float or float
            The self-absorption frequency [Hz]
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # Convenience variables
        p, k = self.p, self.k
        x = 5 * (4 - k)

        # Transformations
        t = t * 86_400
        E = 1e52 * self.E
        rho0 = self.rho0 * self.m_p * (10 ** ref) ** k

        # Shared exponents
        exp_ae = (14 - 9 * k) / x

        # Linear terms
        pre_factor = 25.08968 * (
            self.alpha ** -exp_ae *
            self.beta ** -(2 * (15 - k) / x) *
            (0.5 * (1 + self.X)) ** 0.6 *
            np.pi ** ((2 + 5 * k) / x)
        )

        # return self-absorption frequency [Hz]
        return pre_factor * 10 ** (
            # Pre-factors
            np.log10(self.q_e) * (28 / 5) +
            np.log10(self.m_e) * -4 +
            np.log10(self.m_p) * -0.6 +
            np.log10(self.c) * (-2 * (65 - 19 * k) / x) +

            # Model parameters
            np.log10(1 + self.z) * -(2 * (5 - 4 * k) / x) +
            np.log10(self.eps_b) * (6 / 5) +
            np.log10(rho0) * (22 / x) +
            np.log10(E) * exp_ae +
            np.log10(t) * -(10 + 3 * k) / x
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

        if np.isnan(modeled.min()):
            return np.array([np.nan])

        # Apply dust extinction and host galaxy corrections
        modeled = self.model_extinction(
            modeled, **Parameters.extinction(obs, params)
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
            modeled = self.model_grouped_afterglow(
                obs, params, **kwargs
            )

        else:
            # Model the afterglow flux all together. Nice and simple.
            model = self.afterglow_model(**params.get('model'), **kwargs)
            modeled = model.model(obs)

        # return the unextinguished GRB afterglow flux
        return modeled

    def model_grouped_afterglow(self, obs, params, **kwargs):
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

        for group, mask in obs.groups.items():
            model_params = params.get(group).get('model')

            modeled[mask] = self.afterglow_model(
                **model_params, **kwargs).model(obs, mask)

        return modeled

    def model_extinction(
        self, modeled, wn, z=None, ebv_sf=None,
        ebv_mw=None, host_pos=None, host_vals=None,
        rv_sf=None, rv_mw=None, ext_pos=None
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
            The redshift. If provided, transforms ``wn`` to
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

        ext_pos : np.ndarray of bool, optional
            The positions to apply the extinction correction.

        Returns
        -------
        np.ndarray of float
            The extinguished and host galaxy corrected flux.
        """

        # Apply source frame extinction
        if self.ext_sf is not None:
            modeled[ext_pos] *= self.ext_sf  # pre-computed

        elif ebv_sf is not None:
            # Reuse model object if not fitting for Rv
            model = self.extinction_model if rv_sf is None \
                else self.extinction_model.__class__(Rv=rv_sf)
            modeled[ext_pos] *= model.extinguish((1 + z) * wn, Ebv=ebv_sf)

        # Apply host galaxy correction
        if host_vals is not None and host_pos is not None:
            for name, corr in host_vals.items():
                modeled[host_pos[name]] += corr

        # Apply Milky Way extinction
        if self.ext_mw is not None:
            modeled[ext_pos] *= self.ext_mw  # pre-computed

        elif ebv_mw is not None:
            # Reuse model object if not fitting for Rv
            model = self.extinction_model if rv_mw is None \
                else self.extinction_model.__class__(Rv=rv_mw)
            modeled[ext_pos] *= model.extinguish(wn, Ebv=ebv_mw)

        # return (afterglow_flux * ext_sf + host_correction) * ext_mw
        return modeled
