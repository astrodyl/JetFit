import numpy as np

from jetfit.core.input import Observation
from jetfit.models2.basemodels import IntegratedFluxModel, SpectralIndexModel, BlastWaveModel, StratifiedMediumModel
from jetfit.models2.basemodels import AbsorptionFrequencyModel, BaseFireballModel
from jetfit.models2.basemodels import SynchrotronFrequencyModel, SpectralFluxModel
from jetfit.models2.basemodels import CoolingFrequencyModel, PeakFluxModel

# ignore `dust_extinction` user warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class StratifiedFireballModel:
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

    sn : float
        The density smoothing factor.

    k1 : float
        The density power-law index before the transition.

    k2 : float
        The density power-law index after the transition.

    X : float
        The hydrogen mass fraction.

    tj : float, optional
        The jet break time in days.

    sj : float, optional
        The jet break smoothing factor.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, p, eps_b, eps_e, z, dL, nt, rt, k1, k2, sn, sr, X, tj=None, sj=None, sji=None):
        # Afterglow
        self.E = E
        self.p = p
        self.eps_b = eps_b
        self.eps_e = eps_e
        self.X = X

        # Observer
        self.dL = dL
        self.z = z

        # Medium
        self.k1 = k1
        self.k2 = k2

        self.rt = rt
        self.sr = sr
        self.nt = nt
        self.sn = sn

        # Jet
        self.tj = tj
        self.sj = sj
        self.sji = sji

    # def smooth(self, t, radii=None):
    #     """
    #     Empirically smooths the number density normalizations
    #     and the power-law indices over the observer times `t`.
    #
    #     Parameters
    #     ----------
    #     t : np.ndarray
    #         The observer times [days since trigger].
    #
    #     radii : np.ndarray of float, optional
    #         The pre-computed blast wave radii [cm].
    #
    #     Returns
    #     -------
    #     tuple of np.ndarray of float
    #         The smoothed number density normalizations [cm-3] and
    #         the smoothed density power-law indices [dimension less].
    #     """
    #     if radii is None:
    #         radii = self.radii(t)
    #
    #     # Smooth the density normalizations
    #     x = radii / self.transition_radius()
    #     n_eff = self.n1 + (self.n2 - self.n1) / (1 + x ** -self.sn)
    #     k_eff = self.k1 + (self.k2 - self.k1) / (1 + x ** -self.sk)
    #
    #     return n_eff, k_eff

    def smooth(self, t, radii=None):
        """
        Empirically smooths the number density normalizations
        and the power-law indices over the observer times `t`.

        Parameters
        ----------
        t : np.ndarray
            The observer times [days since trigger].

        radii : np.ndarray of float, optional
            The pre-computed blast wave radii [cm].

        Returns
        -------
        tuple of np.ndarray of float
            The smoothed number density normalizations [cm-3] and
            the smoothed density power-law indices.
        """
        bwm1 = BlastWaveModel(self.E, self.nt, self.k1)
        bwm2 = BlastWaveModel(self.E, self.nt, self.k2)
        t_decel = bwm1.decel_time() / 86_400

        r1 = bwm1.shock_radius(self.z, t, t_decel)
        r2 = bwm2.shock_radius(self.z, t, t_decel)

        # Rename for convenience
        sn, sr = self.sn, self.sr
        k1, k2 = self.k1, self.k2
        x1, x2 = r1 / self.rt, r2 / self.rt

        r_eff = self.rt * (x1 ** -sr + x2 ** -sr) ** -(1 / sr)

        # Calculate the effective number density normalizations
        x = r_eff / self.rt
        n_eff = self.nt * (x ** (k1 * sn) + x ** (k2 * sn)) ** -(1 / sn)

        # Calculate the effective density power-law indices
        k_eff_num = k1 * x ** (k1 * sn) + k2 * x ** (k2 * sn)
        k_eff_den = x ** (k1 * sn) + x ** (k2 * sn)

        return n_eff, k_eff_num / k_eff_den

    def radii(self, t):
        """
        Calculates the radius traversed by the blast wave
        during time `t`.

        Parameters
        ----------
        t : float or np.ndarray
            The observer times [days since trigger].

        Returns
        -------
        float or np.ndarray
            The radii traversed by the blast wave [cm].
        """
        bwm = BlastWaveModel(self.E, self.nt, self.k1)
        t_decel = bwm.decel_time() / 86_400  # [d]
        return bwm.shock_radius(self.z, t, t_decel)
        # r_trans = self.transition_radius()
        # t_trans = self.transition_time() / 86_400
        #
        # t_pre = t[t < t_trans]  # time since trigger [d]
        # t_post = t[t >= t_trans] - t_trans  # time since transition [d]
        #
        # # Models pre- and post-transition
        # bwm_pre = BlastWaveModel(self.E, self.n1, self.k1)
        # bwm_post = BlastWaveModel(self.E, self.n2, self.k2)
        #
        # # Calculate the blast-wave radii
        # t_decel = bwm_pre.decel_time() / 86_400  # [d]
        # r_pre = bwm_pre.shock_radius(self.z, t_pre, t_decel)  # [cm]
        # r_post = bwm_post.shock_radius(self.z, t_post, t_decel) + r_trans  # [cm]
        #
        # if r_pre.size != 0 and r_post.size != 0:
        #     return np.concatenate((r_pre, r_post))
        # return r_pre if r_pre.size != 0 else r_post

    # def transition_density(self, r_trans=None, ref=1e17):
    #     """ Returns the transition density normalization. """
    #     if r_trans is None:
    #         r_trans = self.transition_radius()
    #
    #     if r_trans < ref:
    #         return self.n1 * (r_trans / ref) ** -self.k1
    #     return self.n2 * (r_trans / ref) ** -self.k2

    # def transition_radius(self) -> float:
    #     """ Returns the transition radius [cm]. """
    #     return StratifiedMediumModel(
    #         self.E, self.n1, self.n2, self.k1, self.k2
    #     ).transition_radius()

    # def transition_time(self):
    #     """"""
    #     return StratifiedMediumModel(
    #         self.E, self.n1, self.n2, self.k1, self.k2
    #     ).transition_time(self.z)

    def model(self, observation: Observation) -> np.ndarray:
        """"""
        res = np.full(len(observation.data), np.nan)

        # Do not model physically impossible solutions
        if self.eps_b + self.eps_e >= 1.0:
            return res

        # For speed, get observation as arrays
        arrays = observation.as_arrays

        # Smooth the density profile
        n_eff, k_eff = self.smooth(arrays.times)

        # Calculate the spectral functions
        f_peaks = self.f_peak(n_eff, k_eff, arrays.times)
        nu_cs = self.nu_c(n_eff, k_eff, arrays.times)
        nu_ms = self.nu_m(k_eff, arrays.times)

        # Model spectral fluxes
        if (sf_mask := arrays.sflux_loc).any():
            res[sf_mask] = SpectralFluxModel(
                nu_ms[sf_mask], nu_cs[sf_mask], f_peaks[sf_mask], self.p, k_eff[sf_mask]
            )(arrays.frequencies[sf_mask])

        # Model integrated fluxes
        if (if_mask := arrays.iflux_loc).any():
            res[if_mask] = IntegratedFluxModel(
                nu_ms[if_mask], nu_cs[if_mask], f_peaks[if_mask], self.p, k_eff[if_mask]
            )(arrays.if_lower_freqs[if_mask], arrays.if_upper_freqs[if_mask])

        # Model spectral indices
        if (si_mask := arrays.sindex_loc).any():
            res[si_mask] = SpectralIndexModel(
                nu_ms[si_mask], nu_cs[si_mask], f_peaks[si_mask], self.p, k_eff[si_mask]
            )(arrays.si_lower_freqs[si_mask], arrays.si_upper_freqs[si_mask])

        # Smooth the flux values if there is a jet break
        if self.tj and sf_mask.any():
            res[sf_mask] = self.smooth_jet_break(
                f=res[sf_mask], t=arrays.times[sf_mask],
                nu=arrays.frequencies[sf_mask],
                n=n_eff[sf_mask], k=k_eff[sf_mask]
            )

        if self.tj and if_mask.any():
            res[if_mask] = self.smooth_jet_break(
                f=res[if_mask], t=arrays.times[if_mask],
                lower=arrays.if_lower_freqs[if_mask],
                upper=arrays.if_upper_freqs[if_mask],
                n=n_eff[if_mask], k=k_eff[if_mask]
            )

        return res

    def f_peak(self, n, k, t):
        """
        Calculates the peak flux in the case of an ultra-
        relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        n : float or np.ndarray of float
            The smoothed density normalization [cm-3].

        k : float or np.ndarray of float
            The density power-law indices.

        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float or u.Quantity['time']
            The peak flux in mJy at time `t`.
        """
        return PeakFluxModel(
            self.E, n, self.eps_b, self.dL, self.z, k, self.X)(t)

    def nu_c(self, n, k, t):
        """
        Calculates the cooling frequency in the case of an ultra-
        relativistic shock moving into an external medium with
        density rho = rho_0 * R^-k.

        Parameters
        ----------
        n : np.ndarray of float
            The smoothed density normalization [cm-3].

        k : np.ndarray of float
            The density power-law indices.

        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, assumed
            to be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float
            The cooling frequency in Hz at time `t`.
        """
        return CoolingFrequencyModel(
            self.E, n, self.eps_b, k, self.z)(t)

    def nu_m(self, k, t):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        k : np.ndarray of float
            The density power-law indices.

        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, assumed
            to be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float
            The synchrotron frequency in Hz at time `t`.
        """
        return SynchrotronFrequencyModel(
            self.E, self.eps_e, self.eps_b, k, self.z, self.X, self.p)(t)

    def nu_a(self, n, k, t, regime: str):
        """
        Calculates the self-absorption frequency in the case of
        an ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, assumed
            to be measured in days since trigger.

        regime : str, {'slow', 'fast'}
            The regime to evaluate.

        Returns
        -------
        float or np.ndarray of float
            The self-absorption frequency in Hz at time `t`.
        """
        return AbsorptionFrequencyModel(
            self.E, n, self.eps_e, self.eps_b, k, self.z, self.X, self.p
        )(t, regime)

    def smooth_jet_break(self, f, t, n, k, **kwargs):
        """ Will be moved in base or mixin. """
        if 'nu' in kwargs:
            model = SpectralFluxModel
        elif 'lower' in kwargs and 'upper' in kwargs:
            model = IntegratedFluxModel
        else:
            raise ValueError(
                'Must provide either `nu` or `lower` and `upper`.'
            )

        # Evaluate the flux at the jet break time
        f_jet = model(
            nu_m=self.nu_m(k, self.tj), nu_c=self.nu_c(n, k, self.tj),
            f_peak=self.f_peak(n, k, self.tj), p=self.p, k=k
        )(**kwargs)

        # return flux smoothed over the jet break
        s = self.sj or self.sji

        return (
            f ** -s + (f_jet * (t / self.tj) ** -self.p) ** -s
        ) ** -(1 / s)

    def spectral_flux(self, t, f):
        """
        Calculates the spectral fluxes at times `t` for the
        frequencies `f`.

        Parameters
        ----------
        t : float or np.ndarray of float u.Quantity['time']
            The observer times measured in days since trigger.

        f : float or np.ndarray of float
            The average band frequencies.

        Returns
        -------
        float np.ndarray of float
            The modeled spectral flux.
        """
        n_eff, k_eff = self.smooth(t)

        res = SpectralFluxModel(
            self.nu_m(k_eff, t), self.nu_c(n_eff, k_eff, t),
            self.f_peak(n_eff, k_eff, t), self.p, k_eff
        ).evaluate(f)

        if self.tj:
            return self.smooth_jet_break(res, t, n_eff, k_eff, nu=f)

        return res

    def integrated_flux(self, t, lower, upper):
        """
        Calculates the integrated fluxes at times `t` for the
        lower and upper integration bounds, `lower` and `upper`.

        Parameters
        ----------
        t : float or np.ndarray of float u.Quantity['time']
            The observer times measured in days since trigger.

        lower, upper : float or np.ndarray of float
            The integration bounds measured in Hz.

        Returns
        -------
        float np.ndarray of float
            The modeled spectral flux.
        """
        n_eff, k_eff = self.smooth(t)

        res = IntegratedFluxModel(
            self.nu_m(k_eff, t), self.nu_c(n_eff, k_eff, t),
            self.f_peak(n_eff, k_eff, t), self.p, k_eff
        ).evaluate(lower, upper)

        if self.tj:
            return self.smooth_jet_break(
                res, t, n_eff, k_eff, lower=lower, upper=upper)

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
        super().__init__(E, p, eps_b, eps_e, z, dL, rho0, k, X)

        # Jet break
        self.tj = tj
        self.sj = sj
        self.sji = sji

    def model(
            self,
            observation: Observation,
            subset_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Models an `observation` object.

        Parameters
        ----------
        observation : Observation
            The observation object to model.

        subset_mask : np.ndarray of bool, optional
            The truth array of which values to model.

        Returns
        -------
        np.ndarray of float
            The unextinguished modeled flux.
        """
        res = np.full(len(observation.data), np.nan)

        # Physically impossible solution
        if self.eps_b + self.eps_e >= 1.0:
            return res[subset_mask] if subset_mask is not None else res

        # For speed, get observation as arrays
        arrays = observation.as_arrays

        # Save locations for readability
        sf_mask = arrays.sflux_loc
        if_mask = arrays.iflux_loc
        si_mask = arrays.sindex_loc

        if subset_mask is not None:
            sf_mask = np.logical_and(sf_mask, subset_mask)
            if_mask = np.logical_and(if_mask, subset_mask)
            si_mask = np.logical_and(si_mask, subset_mask)

        # Calculate the spectral functions
        f_peaks = self.f_peak(arrays.times)
        nu_ms = self.nu_m(arrays.times)
        nu_cs = self.nu_c(arrays.times)

        # Model spectral fluxes
        if sf_mask.any():
            res[sf_mask] = SpectralFluxModel(
                nu_ms[sf_mask], nu_cs[sf_mask], f_peaks[sf_mask], self.p, self.k
            )(arrays.frequencies[sf_mask])

        # Model integrated fluxes
        if if_mask.any():
            res[if_mask] = IntegratedFluxModel(
                nu_ms[if_mask], nu_cs[if_mask], f_peaks[if_mask], self.p, self.k
            )(arrays.if_lower_freqs[if_mask], arrays.if_upper_freqs[if_mask])

        # Model spectral indices
        if si_mask.any():
            res[si_mask] = SpectralIndexModel(
                nu_ms[si_mask], nu_cs[si_mask], f_peaks[si_mask], self.p, self.k
            )(arrays.si_lower_freqs[si_mask], arrays.si_upper_freqs[si_mask])

        # Smooth the flux values if there is a jet break
        if self.tj and sf_mask.any():
            res[sf_mask] = self.smooth_jet_break(
                f=res[sf_mask], t=arrays.times[sf_mask],
                nu=arrays.frequencies[sf_mask]
            )

        if self.tj and if_mask.any():
            res[if_mask] = self.smooth_jet_break(
                f=res[if_mask], t=arrays.times[if_mask],
                lower=arrays.if_lower_freqs[if_mask],
                upper=arrays.if_upper_freqs[if_mask]
            )

        return res[subset_mask] if subset_mask is not None else res

    def spectral_flux(self, t, f):
        """
        Calculates the spectral fluxes at times `t` for the
        frequencies `f`.

        Parameters
        ----------
        t : float or np.ndarray of float u.Quantity['time']
            The observer times measured in days since trigger.

        f : float or np.ndarray of float
            The average band frequencies.

        Returns
        -------
        float np.ndarray of float
            The modeled spectral flux.

        See Also
        --------
        `models2.basemodels.SpectralFluxModel.evaluate`
            See for information on how various shapes
            of t and f are handled.
        """
        res = SpectralFluxModel(
            self.nu_m(t), self.nu_c(t), self.f_peak(t), self.p, self.k
        ).evaluate(f)

        if self.tj:
            return self.smooth_jet_break(res, t, nu=f)

        return res

    def integrated_flux(self, t, lower, upper):
        """
        Calculates the integrated fluxes at times `t` for the
        lower and upper integration bounds, `lower` and `upper`.

        Parameters
        ----------
        t : float or np.ndarray of float u.Quantity['time']
            The observer times measured in days since trigger.

        lower, upper : float or np.ndarray of float
            The integration bounds measured in Hz.

        Returns
        -------
        float np.ndarray of float
            The modeled spectral flux.

        See Also
        --------
        `models2.basemodels.SpectralFluxModel.evaluate`
            See for information on how various shapes
            of t, lower, upper are handled.
        """
        res = IntegratedFluxModel(
            self.nu_m(t), self.nu_c(t), self.f_peak(t), self.p, self.k
        ).evaluate(lower, upper)

        if self.tj:
            return self.smooth_jet_break(res, t, lower=lower, upper=upper)

        return res

    def spectral_index(self, t, lower, upper):
        """
        Calculates the spectral index at times `t` for the
        lower and upper integration bounds, `lower` and `upper`.

        Parameters
        ----------
        t : float or np.ndarray of float u.Quantity['time']
            The observer times measured in days since trigger.

        lower, upper : float or np.ndarray of float
            The integration bounds measured in Hz.

        Returns
        -------
        float np.ndarray of float
            The modeled spectral flux.

        See Also
        --------
        `models2.basemodels.SpectralFluxModel.evaluate`
            See for information on how various shapes
            of t, lower, upper are handled.
        """
        return SpectralIndexModel(
            self.nu_m(t), self.nu_c(t), self.f_peak(t), self.p, self.k
        ).evaluate(lower, upper)

    def smooth_jet_break(self, f, t, **kwargs):
        """
        Smooths the flux across a jet break.

        Parameters
        ----------
        f : np.ndarray or float
            The flux to smooth.

        t : np.ndarray or float
            The times corresponding to `f` measured in days
            since trigger.

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
        float or np.ndarray of float
            The jet-break smoothed flux with units equal to those
            of `f`.

        Raises
        ------
        ValueError
            If `nu` and `lower` and `upper` are not provided.
        """
        if 'nu' in kwargs:
            model = SpectralFluxModel
        elif 'lower' in kwargs and 'upper' in kwargs:
            model = IntegratedFluxModel
        else:
            raise ValueError(
                'Must provide either `nu` or `lower` and `upper`.'
            )

        # Evaluate the flux at the jet break time
        f_jet = model(
            nu_m=self.nu_m(self.tj), nu_c=self.nu_c(self.tj),
            f_peak=self.f_peak(self.tj), p=self.p, k=self.k
        )(**kwargs)

        # return flux smoothed over the jet break
        if self.sj:
            return (
                f ** -self.sj + (f_jet * (t / self.tj) ** -self.p) ** -self.sj
            ) ** -(1 / self.sj)

        if self.sji:
            return (
                f ** -(1/self.sji) + (f_jet * (t / self.tj) ** -self.p) ** -(1/self.sji)
            ) ** -self.sji

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

    def nu_a(self, t, regime: str):
        """
        Calculates the self-absorption frequency in the case of
        an ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, assumed
            to be measured in days since trigger.

        regime : str, {'slow', 'fast'}
            The regime to evaluate.

        Returns
        -------
        float or np.ndarray of float
            The self-absorption frequency in Hz at time `t`.
        """
        return AbsorptionFrequencyModel(
            self.E, self.rho0, self.eps_e, self.eps_b, self.k, self.z, self.X, self.p
        )(t, regime)
