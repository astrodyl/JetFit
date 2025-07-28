import numpy as np

from jetfit.core.input import Observation
from jetfit.models.base import BlastWaveModel, ObservedSpectrumModel, \
    DAY2SEC, MassP, RadiationModel, BlastWaveModel2
from jetfit.models.base import BaseFireballModel


# ignore `dust_extinction` user warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class StratifiedFireballModel(BaseFireballModel):
    """
    A fully analytic description of an ultra-relativistic
    shock moving into an external stratified medium with
    density rho = rho_0 * R^-k. Where k transitions from
    one asymptotic value to another.

    Parameters
    ----------
    E52 : float
        The explosion energy normalized to 1e52 ergs.

    p : float
        The electron energy index (dimensionless).

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons. Must be in the range [0, 1].

    z : float
        The redshift to the event.

    dL28 : float
        The luminosity distance to the event [1e28 cm]. Requiring
        the distance to be provided in addition to the redshift
        prevents the need to assume a cosmology here.

    sn, sni : float, optional
        The density smoothing factor. ``sni`` is the inverse of the
        smoothing factor. Useful for changing fitting basis in MCMC.

    k1 : float
        The density power-law index before the transition.

    k2 : float
        The density power-law index after the transition.

    hmf : float
        The hydrogen mass fraction.

    tj : float, optional, default=None
        The jet break observer-frame time [d].

    sj, sji : float, optional, default=None
        The jet break smoothing factor. `sji` is the inverse of the
        smoothing factor. Useful for changing fitting basis in MCMC.
        Required if ``tj != None``.

    use_sa : bool, optional, default=True
        Should self-absorption be modeled?
    """
    # noinspection PyPep8Naming
    def __init__(
        self, E52, p, eps_b, eps_e, z, dL28, n0t, rt, hmf,
        k1=None, k2=None, lf0=None, tj=None, sj=None, sji=None,
        sn=None, sni=None, k1i=None, k2i=None, use_sa=True
    ):
        super().__init__(E52, p, eps_b, eps_e, z, dL28, hmf, lf0, tj, sj, sji, use_sa)

        if sn is None and sni is None:
            raise ValueError("Must specify either sn or sni.")

        # Medium
        self.k1 = k1 if k1 is not None else 1 / k1i
        self.k2 = k2 if k2 is not None else 1 / k2i

        self.sn = (sn or 1 / sni) if (sn or sni) else None
        self.rt = rt
        self.n0t = n0t

    @property
    def ref_radius(self):
        """ Returns the reference radius [cm]."""
        return self.rt

    @property
    def is_valid(self) -> bool:
        """ Whether the model is parameters are valid. """
        if self.sn < 0 and self.k1 < self.k2:
            return False

        if self.sn > 0 and self.k1 > self.k2:
            return False

        return super().is_valid and abs(self.sn) > 0.1

    def smooth(self, t):
        """
        Empirically smooths the number density normalizations
        and the power-law indices over the observer times ``t``.

        Parameters
        ----------
        t : np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        tuple of np.ndarray of float
            The smoothed number density normalizations [cm-3] and
            the smoothed density power-law indices.
        """
        # Rename for convenience
        s, k1, k2 = self.sn, self.k1, self.k2

        r = self.radii(t)
        x = r / self.rt

        # Calculate the effective number densities
        n = self.n0t * (2 ** (1 / s)) * (
            x ** (k1 * s) + x ** (k2 * s)
        ) ** -(1 / s)

        # Calculate the effective density power-law indices
        k_eff_num = k1 * x ** (k1 * s) + k2 * x ** (k2 * s)
        k_eff_den = x ** (k1 * s) + x ** (k2 * s)
        k_eff = k_eff_num / k_eff_den

        # Number density normalized to `rt`
        n0 = n * (r / self.rt) ** k_eff

        return n0, k_eff

    def radii(self, t):
        """
        Calculates the radius traversed by the blast wave
        during time ``t`` in a stratified medium defined by
        the power-law indices ``k1`` and ``k2``, and the radius
        and density at the transition, ``n0t`` and ``rt``.

        Parameters
        ----------
        t : float or np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        float or np.ndarray
            The radii traversed by the blast wave [cm].
        """
        bwm1 = BlastWaveModel(self.E52, self.n0t, self.k1, ref=self.rt)
        bwm2 = BlastWaveModel(self.E52, self.n0t, self.k2, ref=self.rt)
        t_decel = bwm1.decel_time(self.lf0 or 300.0) / DAY2SEC

        r1 = bwm1.shock_radius(self.z, t, t_decel)
        r2 = bwm2.shock_radius(self.z, t, t_decel)

        # Rename for convenience
        s, k1, k2 = self.sn, self.k1, self.k2
        x1, x2 = r1 / self.rt, r2 / self.rt

        return (2 ** (1 / s)) * self.rt * (x1 ** -s + x2 ** -s) ** -(1 / s)

    def model(self, obs: Observation):
        """
        Models the observational data, ``obs``.

        Parameters
        ----------
        obs : Observation
            The observational data.

        Returns
        -------
        np.ndarray of float
            The modeled observational data.
        """
        if not self.is_valid:
            return np.array([np.nan])

        # return the modeled smoothed, unextinguished flux
        return ObservedSpectrumModel(**self.spectrum(obs.times()),
            arrays=obs.as_arrays, jet=self.jet_break(obs.times()),
        ).model()

    def spectrum(self, t, n=None, k=None):
        """
        Returns the characteristics that define a GRB spectrum.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer-frame time(s) [d].

        n : float or np.ndarray of float, optional
            The effective density normalization [cm-3].

        k : float or np.ndarray of float, optional
            The effective power law indices.

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        t = np.atleast_1d(t)

        if n is None or k is None:
            n, k = self.smooth(t)

        # Number density normalization [cm(k-3)]
        n = n * self.ref_radius ** k

        radiation = RadiationModel(
            n, k, self.p, self.eps_b, self.eps_e, self.dL, self.z, self.hmf
        )

        # Default to the adiabatic spectrum
        spec = self.spectrum_adiabatic(radiation, self.E, t)

        # Should we consider radiative evolution?
        if self.radiative:

            # OK, but is there actually a radiative solution?
            if not (spec['nu_m'] > spec['nu_c']).any():
                return spec

            spec_rad = self.spectrum_radiative(radiation, t)

            # Is there still a radiative solution?
            if not (spec_rad['nu_m'] > spec_rad['nu_c']).any():
                return spec

            # When does radiative end and adiabatic begin?
            t_trans = radiation.rad_to_ad_time(
                self.E / self.lf0, t_obs=t, nu_m=spec_rad['nu_m'], nu_c=spec_rad['nu_c']
            )

            if t_trans is None:
                return spec_rad

            # What is the energy after radiative loss?
            k_t = np.interp(t_trans / DAY2SEC, t, k)
            n_t = np.interp(t_trans / DAY2SEC, t, n)

            nrg = self.E * BlastWaveModel2(self.E, self.lf0, n_t, k_t).energy_loss(
                t_trans / (1 + self.z)
            )

            # Recalculate adiabatic functions using diminished energy
            spec_ad = self.spectrum_adiabatic(radiation, nrg, t)

            # Smooth the spectral functions
            return radiation.rad_to_ad_smooth(t, t_trans / DAY2SEC, spec_rad, spec_ad)

        return spec

    def spectrum_adiabatic(self, radiation, E, t):
        """
        Returns the characteristics that define the GRB spectrum
        assuming adiabatic evolution.

        Parameters
        ----------
        radiation : RadiationModel
            The radiation model to use.

        E : float
            The energy [erg] at the start of the adiabatic evolution.

        t : float or np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        return {
            'p': self.p, 'k': radiation.k,
            'f_peak': radiation.peak_flux(E, t),
            'nu_c': radiation.cooling_frequency(E, t),
            'nu_a': radiation.absorption_frequency(E, t),
            'nu_m': radiation.synchrotron_frequency(E, t)
        }

    def spectrum_radiative(self, radiation, t):
        """
        Returns the characteristics that define the GRB spectrum
        assuming radiative evolution.

        Parameters
        ----------
        radiation : RadiationModel
            The radiation model to use.

        t : float or np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        nrg = self.E / self.lf0

        return {
            'p': radiation.p, 'k': radiation.k,
            'f_peak': radiation.peak_flux(nrg, t, adiabatic=False),
            'nu_c': radiation.cooling_frequency(nrg, t, adiabatic=False),
            'nu_a': radiation.absorption_frequency(nrg, t, adiabatic=False),
            'nu_m': radiation.synchrotron_frequency(nrg, t, adiabatic=False)
        }

    def f_peak(self, t, n=None, k=None):
        """
        Calculates the peak flux in the case of an ultra-
        relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        n : float or np.ndarray of float, optional
            The smoothed density normalization [cm-3].

        k : float or np.ndarray of float, optional
            The density power-law indices.

        Returns
        -------
        float or np.ndarray of float
            The peak flux [mJy] at time `t` [d].
        """
        if k is None or n is None:
            n, k = self.smooth(t)

        return self.spectrum(t, n, k)['f_peak']

    def nu_c(self, t, n=None, k=None):
        """
        Calculates the cooling frequency in the case of an ultra-
        relativistic shock moving into an external medium with
        density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        n : np.ndarray, optional
            The smoothed density normalization [cm-3].

        k : np.ndarray, optional
            The density power-law indices.

        Returns
        -------
        float or np.ndarray of float
            The cooling frequency in Hz at time `t`.
        """
        if n is None or k is None:
            n, k = self.smooth(t)

        return self.spectrum(t, n, k)['nu_c']

    def nu_m(self, t, n=None, k=None):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        n : np.ndarray, optional
            The smoothed density normalization [cm-3].

        k : np.ndarray, optional
            The density power-law indices.

        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The synchrotron frequency [Hz] at time(s) ``t``.
        """
        if k is None:
            n, k = self.smooth(t)

        return self.spectrum(t, n, k)['nu_m']

    def nu_a(self, t, n=None, k=None, nu_m=None, nu_c=None):
        """
        Calculates the self-absorption frequency.

        The self-absorption frequency has a circular definition.
        For example, to calculate nu_a, you must first know how
        nu_a relates to the nu_m and nu_c, but nu_a isn't known
        because it needs to be known before it can be known >:)

        This solution is weak, but the self-absorption frequency
        is calculated for every case (except nu_a > both nu_c and
        nu_m, not supported). The result is a combined array where
        the self-absorptions are compared to the synchrotron and
        cooling frequencies.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        n : np.ndarray of float, optional
            The effective density normalization [cm-3].

        k : np.ndarray of float, optional
            The effective density power-law indices.

        nu_m : float or np.ndarray of float, optional
            The synchrotron frequencies [Hz] at time ``t``.

        nu_c : float or np.ndarray of float, optional
            The cooling frequencies [Hz] at time ``t``.

        Returns
        -------
        float or np.ndarray of float
            The self-absorption frequency [Hz] at time(s) ``t``.
        """
        if n is None or k is None:
            n, k = self.smooth(t)

        return self.spectrum(t, n, k)['nu_a']


class FireballModel(BaseFireballModel):
    """
    A fully analytic description of an ultra-relativistic
    shock moving into an external medium with density
    rho = rho_0 * R^-k.

    Parameters
    ----------
    E52 : float
        The explosion energy normalized to 1e52 ergs.

    p : float
        The electron energy index.

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons. Must be in the range [0, 1].

    z : float
        The redshift of the event.

    dL28 : float
        The luminosity distance to the event [1e28 cm]. Requiring
        the distance to be provided in addition to the redshift
        prevents the need to assume a cosmology here.

    n017 : float
        The density normalization normalized to 1e17 cm.

    k : float
        The density power-law index.

    hmf : float
        The hydrogen mass fraction. Must be in the range [0, 1].
        0 indicates hydrogen depleted. 1 indicates hydrogen rich.

    lf0 : float, optional, default=None
        The initial Lorentz factor.

    tj : float, optional, default=None
        The jet break observer-frame time [d].

    sj : float, optional, default=None
        The jet break smoothing factor. Required if ``tj != None``.

    use_sa : bool, optional, default=True
        Should self-absorption be modeled?
    """
    ref_radius = 1.0e17

    # noinspection PyPep8Naming
    def __init__(self, E52, p, eps_b, eps_e, z, dL28, n017, k, hmf, lf0=None, tj=None, sj=None, sji=None, use_sa=True):
        super().__init__(E52, p, eps_b, eps_e, z, dL28, hmf, lf0, tj, sj, sji, use_sa)

        self.n017 = n017
        self.k = k

        # Blast wave model
        if lf0 is not None:
            self.blast = BlastWaveModel2(
                self.E, self.lf0, self.n0, self.k
            )

        # Radiation model
        self.radiation = RadiationModel(
            self.n0, self.k, self.p, self.eps_b,
            self.eps_e, self.dL, self.z, self.hmf
        )

    @property
    def n0(self):
        """ Returns the number density normalization. """
        return self.n017 * self.ref_radius ** self.k

    @property
    def rho0(self):
        """ Returns the mass density normalization normalized to 1e17 cm. """
        return MassP * self.n017

    @property
    def rho(self):
        """ Returns the mass density [g cm-3]. """
        return MassP * self.n017 * self.ref_radius ** self.k

    def model(self, obs: Observation, subset: np.ndarray = None):
        """
        Models the observational data, ``obs``.

        Parameters
        ----------
        obs : Observation
            The observational data.

        subset : np.ndarray of bool, optional
            Models data where ``subset==True``.

        Returns
        -------
        np.ndarray of float
            The modeled observational data.
        """
        if not self.is_valid:
            return np.array([np.nan])

        # return the modeled observational data
        return ObservedSpectrumModel(**self.spectrum(obs.times()),
            jet=self.jet_break(obs.times()), arrays=obs.as_arrays
        ).model(subset)

    def smooth(self, t):
        """
        Temporary method to match stratified class. Makes
        plotting more uniform. Will do better later.

        Parameters
        ----------
        t : np.ndarray
            The observer times [d].

        Returns
        -------
        tuple of np.ndarray of float
            The number density normalizations [cm-3] and
            the density power-law indices.
        """
        return np.full(t.size, self.n017), np.full(t.size, self.k)

    def radii(self, t):
        """
        Calculates the radius traversed by the blast wave
        during time ``t`` in a stratified medium defined by
        the power-law index ``k`` density at the reference
        radius defines at 1e17cm.

        Parameters
        ----------
        t : float or np.ndarray
            The observer times [d].

        Returns
        -------
        float or np.ndarray
            The radii traversed by the blast wave [cm].
        """
        bwm = BlastWaveModel(self.E52, self.n017, self.k, ref=self.ref_radius)
        return bwm.shock_radius(self.z, t, bwm.decel_time(self.lf0 or 300.0) / DAY2SEC)

    def spectrum(self, t):
        """
        Returns the characteristics that define the GRB spectrum.

        Parameters
        ----------
        t : float or np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        t = np.atleast_1d(t)

        # Default to adiabatic evolution
        spec = self.spectrum_adiabatic(t)

        # Radiative evolution
        if self.radiative:
            rad = np.logical_and(spec['nu_m'] > spec['nu_c'], spec['nu_a'] < spec['nu_m'])

            # Is there a radiative solution?
            if not rad.any():
                return spec

            # Radiative evolution spectrum
            spec_rad = self.spectrum_radiative(t)

            # Is there still a radiative solution?
            if not (spec_rad['nu_m'] > spec_rad['nu_c']).any():
                return spec

            # When does radiative end and adiabatic begin?
            t_trans = self.radiation.rad_to_ad_time(self.E / self.lf0) / DAY2SEC

            # Will the transition affect the light curve?
            if not ((t.min() / 100.0) < t_trans < (t.max() * 100.0)):
                return spec_rad

            # Recalculate adiabatic functions using diminished energy
            spec_ad = self.spectrum_adiabatic(t, post_rad=True)

            return self.radiation.rad_to_ad_smooth(t, t_trans, spec_rad, spec_ad)

        return spec

    def spectrum_adiabatic(self, t, post_rad=False):
        """
        Returns the characteristics that define the GRB spectrum
        assuming adiabatic evolution.

        Parameters
        ----------
        t : float or np.ndarray
            The observer-frame time(s) [d].

        post_rad : bool, optional, default=False
            Is the adiabatic evolution following a radiative
            evolution? If so, accounts for the energy loss.

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        nrg = self.E

        if post_rad:
            # When does radiative end and adiabatic begin?
            t_trans = self.radiation.rad_to_ad_time(self.E / self.lf0)

            # What is the energy after radiative loss?
            nrg = self.E * self.blast.energy_loss(t_trans / (1 + self.z))

        return {
            'p': self.p, 'k': self.k,
            'f_peak': self.radiation.peak_flux(nrg, t),
            'nu_c': self.radiation.cooling_frequency(nrg, t),
            'nu_a': self.radiation.absorption_frequency(nrg, t),
            'nu_m': self.radiation.synchrotron_frequency(nrg, t)
        }

    def spectrum_radiative(self, t):
        """
        Returns the characteristics that define the GRB spectrum
        assuming radiative evolution.

        Parameters
        ----------
        t : float or np.ndarray
            The observer-frame time(s) [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        nrg = self.E / self.lf0

        return {
            'p': self.p, 'k': self.k,
            'f_peak': self.radiation.peak_flux(nrg, t, adiabatic=False),
            'nu_c': self.radiation.cooling_frequency(nrg, t, adiabatic=False),
            'nu_a': self.radiation.absorption_frequency(nrg, t, adiabatic=False),
            'nu_m': self.radiation.synchrotron_frequency(nrg, t, adiabatic=False)
        }

    def f_peak(self, t):
        """
        Calculates the peak flux in the case of an ultra-
        relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The peak flux value(s) [mJy] at time(s) ``t``.
        """
        return self.spectrum(t)['f_peak']

    def nu_c(self, t):
        """
        Calculates the cooling frequency in the case of an ultra-
        relativistic shock moving into an external medium with
        density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The cooling frequency value(s) [Hz] at time(s) ``t``.
        """
        return self.spectrum(t)['nu_c']

    def nu_m(self, t):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The synchrotron frequency value(s) [Hz] at time(s) ``t``.
        """
        return self.spectrum(t)['nu_m']

    def nu_a(self, t, nu_m=None, nu_c=None):
        """
        Calculates the self-absorption frequency.

        The self-absorption frequency has a circular definition.
        For example, to calculate nu_a, you must first know how
        nu_a relates to the nu_m and nu_c, but nu_a isn't known
        because it needs to be known before it can be known >:)

        This solution is weak, but the self-absorption frequency
        is calculated for every case (except nu_a > both nu_c and
        nu_m, not supported). The result is a combined array where
        the self-absorptions are compared to the synchrotron and
        cooling frequencies.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        nu_m : float or np.ndarray of float, optional
            The synchrotron frequencies [Hz] at time `t`.

        nu_c : float or np.ndarray of float, optional
            The cooling frequencies [Hz] at time `t`.

        Returns
        -------
        float or np.ndarray of float
            The self-absorption frequency value(s) [Hz] at time(s) ``t``.
        """
        return self.spectrum(t)['nu_a']
