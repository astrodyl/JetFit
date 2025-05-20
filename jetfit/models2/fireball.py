import numpy as np
import astropy.units as u

from jetfit.core.input import Observation
from jetfit.models2.basemodels import IntegratedFluxModel, SpectralIndexModel, BlastWaveModel, \
    ObservedSpectrumModel, JetBreakModel
from jetfit.models2.basemodels import AbsorptionFrequencyModel, BaseFireballModel
from jetfit.models2.basemodels import SynchrotronFrequencyModel, SpectralFluxModel
from jetfit.models2.basemodels import CoolingFrequencyModel, PeakFluxModel

# ignore `dust_extinction` user warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class StratifiedFireballModel(BaseFireballModel):
    """
    Implements the ultra-relativistic shock moving into an
    external medium with density rho = rho_0 * R^-k.

    Parameters
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
        The luminosity distance to the event [1e28 cm].

    sn, sni : float, optional
        The density smoothing factor. `sni` is the inverse of the
        smoothing factor. Useful for changing fitting basis in MCMC.

    k1 : float
        The density power-law index before the transition.

    k2 : float
        The density power-law index after the transition.

    X : float
        The hydrogen mass fraction.

    tj : float, optional
        The jet break time in days.

    sj, sji : float, optional
        The jet break smoothing factor. `sji` is the inverse of the
        smoothing factor. Useful for changing fitting basis in MCMC.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, p, eps_b, eps_e, z, dL, nt, rt, k1, k2, X, tj=None, sj=None, sji=None, sn=None, sni=None):
        super().__init__(E, p, eps_b, eps_e, z, dL, X, tj, sj, sji)

        if sn is None and sni is None:
            raise ValueError("Must specify either sn or sni.")

        # Medium
        self.k1 = k1
        self.k2 = k2

        self.rt = rt
        self.nt = nt
        self.sn = (sn or 1 / sni) if (sn or sni) else None

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
        and the power-law indices over the observer times `t`.

        Parameters
        ----------
        t : np.ndarray
            The observer times [days since trigger].

        Returns
        -------
        tuple of np.ndarray of float
            The smoothed number density normalizations [cm-3] and
            the smoothed density power-law indices.
        """
        k1, k2 = self.k1, self.k2

        bwm1 = BlastWaveModel(self.E, self.nt, k1, ref=self.rt)
        bwm2 = BlastWaveModel(self.E, self.nt, k2, ref=self.rt)
        t_decel = bwm1.decel_time() / 86_400

        r1 = bwm1.shock_radius(self.z, t, t_decel)
        r2 = bwm2.shock_radius(self.z, t, t_decel)

        # Rename for convenience
        sn = self.sn
        x1, x2 = r1 / self.rt, r2 / self.rt

        # Calculate the effective number densities
        n_eff = self.nt * (2 ** (1 / sn)) * (
            x1 ** (k1 * sn) + x2 ** (k2 * sn)
        ) ** -(1 / sn)

        # Calculate the effective density power-law indices
        k_eff_num = k1 * x1 ** (k1 * sn) + k2 * x2 ** (k2 * sn)
        k_eff_den = x1 ** (k1 * sn) + x2 ** (k2 * sn)

        return n_eff, k_eff_num / k_eff_den

    def radii(self, t):
        """
        Calculates the radius traversed by the blast wave
        during time `t` in a stratified medium defined by
        the power-law indices `k1` and `k2`, and the radius
        and density at the transition, `nt` and `rt`.

        Parameters
        ----------
        t : float or np.ndarray
            The observer times [days since trigger].

        Returns
        -------
        float or np.ndarray
            The radii traversed by the blast wave [cm].
        """
        bwm1 = BlastWaveModel(self.E, self.nt, self.k1, ref=self.rt)
        bwm2 = BlastWaveModel(self.E, self.nt, self.k2, ref=self.rt)
        t_decel = bwm1.decel_time() / 86_400

        r1 = bwm1.shock_radius(self.z, t, t_decel)
        r2 = bwm2.shock_radius(self.z, t, t_decel)

        # Rename for convenience
        s = self.sn
        x1, x2 = r1 / self.rt, r2 / self.rt

        return self.rt * (2 ** (1 / s)) * (x1 ** -s + x2 ** -s) ** -(1 / s)

    def model(self, obs: Observation):
        """
        Models an `observation` object.

        Parameters
        ----------
        obs : Observation
            The observation object to model.

        Returns
        -------
        np.ndarray of float
            The unextinguished modeled flux.
        """
        if not self.is_valid:
            return np.array([np.nan])

        # return the modeled smoothed, unextinguished flux
        return ObservedSpectrumModel(**self.spectrum(obs.as_arrays.times),
            arrays=obs.as_arrays, jet=self.jet_break(obs.as_arrays.times),
        ).model()

    def spectrum(self, t, n=None, k=None):
        """
        Returns the characteristics that define a GRB spectrum.

        Parameters
        ----------
        t : np.ndarray of float or float
            The observer time [d].

        n : np.ndarray of float or float, optional
            The effective density normalization [cm-3].

        k : np.ndarray of float or float, optional
            The effective power law indices.

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        if n is None or k is None:
            n, k = self.smooth(t)

        return {
            'f_peak': self.f_peak(t, n, k),
            'nu_m': (nu_m := self.nu_m(t, k)),
            'nu_c': (nu_c := self.nu_c(t, n, k)),
            'nu_a': self.nu_a(t, n, k, nu_m, nu_c),
            'p': self.p, 'k': k
        }

    def f_peak(self, t, n=None, k=None):
        """
        Calculates the peak flux in the case of an ultra-
        relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        n : float or np.ndarray of float, optional
            The smoothed density normalization [cm-3].

        k : float or np.ndarray of float, optional
            The density power-law indices.

        Returns
        -------
        float or np.ndarray of float or u.Quantity['time']
            The peak flux [mJy] at time `t` [d].
        """
        if k is None or n is None:
            n, k = self.smooth(t)

        return PeakFluxModel(
            self.E, n, self.eps_b, self.dL, self.z, k, self.X)(t, np.log10(self.rt))

    def nu_c(self, t, n=None, k=None):
        """
        Calculates the cooling frequency in the case of an ultra-
        relativistic shock moving into an external medium with
        density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, assumed
            to be measured in days since trigger.

        n : np.ndarray of float, optional
            The smoothed density normalization [cm-3].

        k : np.ndarray of float, optional
            The density power-law indices.

        Returns
        -------
        float or np.ndarray of float
            The cooling frequency in Hz at time `t`.
        """
        if n is None or k is None:
            n, k = self.smooth(t)

        return CoolingFrequencyModel(
            self.E, n, self.eps_b, k, self.z)(t, np.log10(self.rt))

    def nu_m(self, t, k=None):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        k : np.ndarray of float, optional
            The density power-law indices.

        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, assumed
            to be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float
            The synchrotron frequency [Hz] at time `t`.
        """
        if k is None:
            _, k = self.smooth(t)

        return SynchrotronFrequencyModel(
            self.E, self.eps_e, self.eps_b, k, self.z, self.X, self.p)(t)

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
        t : float or np.ndarray of float or u.Quantity['time']
            The time since trigger. If `t` is a float,
            assumed to be measured in days.

        n : np.ndarray of float, optional
            The effective density normalization [cm-3].

        k : np.ndarray of float, optional
            The effective density power-law indices.

        nu_m : float or np.ndarray of float, optional
            The synchrotron frequencies [Hz] at time `t`.

        nu_c : float or np.ndarray of float, optional
            The cooling frequencies [Hz] at time `t`.

        Returns
        -------
        float or np.ndarray of float
            The self-absorption frequency [Hz] at time `t`.
        """
        if n is None or k is None:
            n, k = self.smooth(t)

        model = AbsorptionFrequencyModel(
            self.E, n, self.eps_e, self.eps_b, k, self.z, self.X, self.p
        )

        nu_m = self.nu_m(t, k) if nu_m is None else nu_m
        nu_c = self.nu_c(t, n, k) if nu_c is None else nu_c
        fast = nu_c < nu_m

        # Evaluate nu_a for all orderings
        ref = np.log10(self.rt)
        nu_amc = model.evaluate_amc(t, ref)
        nu_mac = model.evaluate_mac(t, ref)
        nu_cam = model.evaluate_cam(t, ref)
        nu_acm = model.evaluate_acm(t, ref)

        # Initialize with slow cooling values
        res = np.where(nu_amc < nu_m, nu_amc, nu_mac)

        if fast.any():  # Overwrite with fast cooling values
            res[fast] = np.where(nu_acm < nu_c, nu_acm, nu_cam)[fast]

        return res


class FireballModel(BaseFireballModel):
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

    Parameters
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
    def __init__(self, E, p, eps_b, eps_e, z, dL, rho0, k, X, tj=None, sj=None, sji=None):
        super().__init__(E, p, eps_b, eps_e, z, dL, X, tj, sj, sji)

        self.rho0 = rho0
        self.k = k

    @property
    def rho0(self) -> float:
        """ Returns the density normalization, normalized to the proton mass. """
        return self._rho0

    @rho0.setter
    def rho0(self, rho0) -> None:
        """
        TODO: rho0 -> n0
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

        Raises
        ------
        astropy.units.UnitTypeError
            If `rho0` is a `u.Quantity` but not a mass or number density.
        """
        if isinstance(rho0, u.Quantity):
            if rho0.unit.physical_type == 'number density':
                rho0 = rho0.cgs.value
            elif rho0.unit.physical_type == 'mass density':
                rho0 = rho0.cgs.value / self.m_p
            else:
                raise u.UnitTypeError(
                    f'rho0 must be a number/mass density'
                )
        self._rho0 = rho0

    def model(self, obs: Observation, subset: np.ndarray = None):
        """
        Models an `observation` object.

        Parameters
        ----------
        obs : Observation
            The observation object to model.

        subset : np.ndarray of bool, optional
            The truth array of which values to model.

        Returns
        -------
        np.ndarray of float
            The unextinguished modeled flux.
        """
        if not self.is_valid:
            return np.array([np.nan])

        # Model the smoothed, unextinguished flux
        return ObservedSpectrumModel(**self.spectrum(obs.as_arrays.times),
            jet=self.jet_break(obs.as_arrays.times), arrays=obs.as_arrays
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
        return {
            'f_peak': self.f_peak(t),
            'nu_m': (nu_m := self.nu_m(t)),
            'nu_a': self.nu_a(t, nu_m),
            'nu_c': self.nu_c(t),
            'p': self.p, 'k': self.k
        }

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
            The time to evaluate. If `t` is a float, assumed
            to be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float
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
            The time to evaluate. If `t` is a float, assumed
            to be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float
            The synchrotron frequency in Hz at time `t`.
        """
        return SynchrotronFrequencyModel(
            self.E, self.eps_e, self.eps_b, self.k, self.z, self.X, self.p)(t)

    def nu_a(self, t, nu_m=None, nu_c=None):
        """
        Calculates the self-absorption frequency.

        Since the self-absorption frequency has different relations
        depending on its relative position to the other critical
        frequencies, I calculate the self-absorption frequency
        for both slow-cooling cases (nu_m < nu_a and nu_a < nu_m).

        The result is a combined array where the self-absorption
        is compared to the synchrotron frequency.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, assumed
            to be measured in days since trigger.

        nu_m : float or np.ndarray of float, optional
            The synchrotron frequencies [Hz] at time `t`.

        nu_c : float or np.ndarray of float, optional
            The cooling frequencies [Hz] at time `t`.

        Returns
        -------
        float or np.ndarray of float
            The self-absorption frequency in Hz at time `t`.
        """
        model = AbsorptionFrequencyModel(
            self.E, self.rho0, self.eps_e, self.eps_b,
            self.k, self.z, self.X, self.p
        )

        nu_m = self.nu_m(t) if nu_m is None else nu_m
        nu_c = self.nu_c(t) if nu_c is None else nu_c
        fast = nu_c < nu_m

        # Evaluate nu_a for all orderings
        nu_amc = model.evaluate_amc(t)
        nu_mac = model.evaluate_mac(t)
        nu_cam = model.evaluate_cam(t)
        nu_acm = model.evaluate_acm(t)

        # Initialize with slow cooling values
        res = np.where(nu_amc < nu_m, nu_amc, nu_mac)

        # Overwrite with fast cooling values
        res[fast] = np.where(nu_acm < nu_c, nu_acm, nu_cam)[fast]

        return res
