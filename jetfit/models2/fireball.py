import numpy as np

from jetfit.core.input import Observation
from jetfit.models2.basemodels import IntegratedFluxModel, SpectralIndexModel, BlastWaveModel, has_fts_transition
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
        if sn is None and sni is None:
            raise ValueError("Must specify either sn or sni.")

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
        self.nt = nt
        self.sn = sn
        self.sni = sni

        # Jet
        self.tj = tj
        self.sj = sj
        self.sji = sji

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
        bwm1 = BlastWaveModel(self.E, self.nt, self.k1, ref=self.rt)
        bwm2 = BlastWaveModel(self.E, self.nt, self.k2, ref=self.rt)
        t_decel = bwm1.decel_time() / 86_400

        r1 = bwm1.shock_radius(self.z, t, t_decel)
        r2 = bwm2.shock_radius(self.z, t, t_decel)

        # Rename for convenience
        sn = self.sn or 1 / self.sni
        k1, k2 = self.k1, self.k2
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
        s = self.sn or 1 / self.sni
        x1, x2 = r1 / self.rt, r2 / self.rt

        return self.rt * (2 ** (1 / s)) * (x1 ** -s + x2 ** -s) ** -(1 / s)

    def model(self, observation: Observation) -> np.ndarray:
        """"""
        res = np.full(len(observation.data), np.nan)

        # Do not model physically impossible solutions
        if self.eps_b + self.eps_e >= 1.0:
            return res

        # Unstable near 0
        if abs(self.sn or 1 / self.sni) < 0.1:
            return res

        # For speed, get observation as arrays
        arrays = observation.as_arrays

        # Smooth the density profile
        n_eff, k_eff = self.smooth(arrays.times)

        # Calculate the spectral functions
        f_peak = self.f_peak(arrays.times, n_eff, k_eff)
        nu_c = self.nu_c(arrays.times, n_eff, k_eff)
        nu_m = self.nu_m(arrays.times, k_eff)
        nu_a = self.nu_a(arrays.times, n_eff, k_eff, nu_m)

        fts = has_fts_transition(nu_m, nu_c)

        # Model the spectral fluxes
        if (sfm := arrays.sflux_loc).any():
            res[sfm] = SpectralFluxModel(
                nu_m[sfm], nu_c[sfm], nu_a[sfm], f_peak[sfm], self.p, k_eff[sfm]
            ).evaluate(arrays.frequencies[sfm], fts)

        # Model the integrated fluxes
        if (ifm := arrays.iflux_loc).any():
            res[ifm] = IntegratedFluxModel(
                nu_m[ifm], nu_c[ifm], nu_a[ifm], f_peak[ifm], self.p, k_eff[ifm]
            ).evaluate(arrays.if_lower_freqs[ifm], arrays.if_upper_freqs[ifm], fts)

        # Model the spectral indices
        if (sim := arrays.sindex_loc).any():
            res[sim] = SpectralIndexModel(
                nu_m[sim], nu_c[sim], nu_a[sim], f_peak[sim], self.p, k_eff[sim]
            ).evaluate(arrays.si_lower_freqs[sim], arrays.si_upper_freqs[sim], fts)

        # Smooth the flux values if there is a jet break
        if self.tj and sfm.any():
            res[sfm] = self.smooth_jet_break(
                f=res[sfm], t=arrays.times[sfm],
                nu=arrays.frequencies[sfm],
                n=n_eff[sfm], k=k_eff[sfm], fts=fts
            )

        if self.tj and ifm.any():
            res[ifm] = self.smooth_jet_break(
                f=res[ifm], t=arrays.times[ifm],
                lower=arrays.if_lower_freqs[ifm],
                upper=arrays.if_upper_freqs[ifm],
                n=n_eff[ifm], k=k_eff[ifm], fts=fts
            )

        return res

    def model2(self, observation: Observation) -> np.ndarray:
        """"""
        res = np.full(len(observation.data), np.nan)

        # Do not model physically impossible solutions
        if self.eps_b + self.eps_e >= 1.0:
            return res

        # Unstable near 0
        if abs(self.sn or 1 / self.sni) < 0.1:
            return res

        # For speed, get observation as arrays
        arrays = observation.as_arrays

        # Model the spectral fluxes
        if (sf_mask := arrays.sflux_loc).any():
            res[sf_mask] = self.spectral_flux(
                arrays.times[sf_mask],
                arrays.frequencies[sf_mask]
            )

        # Model the integrated fluxes
        if (if_mask := arrays.iflux_loc).any():
            res[if_mask] = self.integrated_flux(
                arrays.times[if_mask],
                arrays.if_lower_freqs[if_mask],
                arrays.if_upper_freqs[if_mask]
            )

        # Model the spectral indices
        if (si_mask := arrays.sindex_loc).any():
            res[si_mask] = self.spectral_index(
                arrays.times[si_mask],
                arrays.si_lower_freqs[si_mask],
                arrays.si_upper_freqs[si_mask]
            )

        return res

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

    def nu_a(self, t, n=None, k=None, nu_m=None):
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
            The time since trigger. If `t` is a float,
            assumed to be measured in days.

        n : np.ndarray of float, optional
            The effective density normalization [cm-3].

        k : np.ndarray of float, optional
            The effective density power-law indices.

        nu_m : float or np.ndarray of float, optional
            The synchrotron frequencies [Hz] at time `t`.

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

        nu_m = self.nu_m(k, t) if nu_m is None else nu_m
        nu_amc = model.evaluate_amc(t, np.log10(self.rt))
        nu_mac = model.evaluate_mac(t, np.log10(self.rt))

        return np.where(nu_amc < nu_m, nu_amc, nu_mac)

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

        # Transform the smoothing factor
        s = self.sj or 1 / self.sji

        # Characteristics at the jet break time
        nu_m = self.nu_m(self.tj, k)
        nu_c = self.nu_c(self.tj, n, k)
        nu_a = self.nu_a(self.tj, n, k, nu_m)
        f_peak = self.f_peak(self.tj, n, k)

        # Evaluate the flux at the jet break time
        f_jet = model(nu_m, nu_c, nu_a, f_peak, self.p, k)(**kwargs)

        # return flux smoothed over the jet break
        return (f ** -s + (f_jet * (t / self.tj) ** -self.p) ** -s) ** -(1 / s)

    def spectral_flux(self, t, f, fts=None):
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

        # Characteristics
        nu_m = self.nu_m(t, k_eff)
        nu_c = self.nu_c(t, n_eff, k_eff)
        nu_a = self.nu_a(t, n_eff, k_eff, nu_m)
        f_peak = self.f_peak(t, n_eff, k_eff)

        res = SpectralFluxModel(
            nu_m, nu_c, nu_a, f_peak, self.p, k_eff)(f, fts)

        if self.tj:
            return self.smooth_jet_break(res, t, n_eff, k_eff, nu=f, fts=fts)

        # return the spectral flux [mJy]
        return res

    def integrated_flux(self, t, lower, upper, fts=None):
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
        n, k = self.smooth(t)

        # Characteristics
        nu_m = self.nu_m(t, k)
        nu_c = self.nu_c(t, n, k)
        nu_a = self.nu_a(t, n, k, nu_m)
        f_peak = self.f_peak(t, n, k)

        res = IntegratedFluxModel(
            nu_m, nu_c, nu_a, f_peak, self.p, k)(lower, upper, fts)

        if self.tj:
            return self.smooth_jet_break(res, t, n, k, lower=lower, upper=upper, fts=fts)

        # return the integrated flux [erg s-1 cm-2]
        return res

    def spectral_index(self, t, lower, upper, fts=None):
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
        n_eff, k_eff = self.smooth(t)

        # Characteristics
        nu_m = self.nu_m(t, k_eff)
        nu_c = self.nu_c(t, n_eff, k_eff)
        nu_a = self.nu_a(t, n_eff, k_eff, nu_m)
        f_peak = self.f_peak(t, n_eff, k_eff)

        return SpectralIndexModel(
            nu_m, nu_c, nu_a, f_peak, self.p, k_eff
        ).evaluate(lower, upper, fts)


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
        nu_as = self.nu_a(arrays.times, nu_ms)

        fts = has_fts_transition(nu_ms, nu_cs)

        if sf_mask.any():  # Model spectral fluxes
            res[sf_mask] = SpectralFluxModel(
                nu_ms[sf_mask], nu_cs[sf_mask], nu_as[sf_mask],
                f_peaks[sf_mask], self.p, self.k
            )(arrays.frequencies[sf_mask], fts)

        if if_mask.any():  # Model integrated fluxes
            res[if_mask] = IntegratedFluxModel(
                nu_ms[if_mask], nu_cs[if_mask], nu_as[if_mask],
                f_peaks[if_mask], self.p, self.k
            )(arrays.if_lower_freqs[if_mask], arrays.if_upper_freqs[if_mask], fts)

        if si_mask.any():  # Model spectral indices
            res[si_mask] = SpectralIndexModel(
                nu_ms[si_mask], nu_cs[si_mask], nu_as[si_mask],
                f_peaks[si_mask], self.p, self.k
            )(arrays.si_lower_freqs[si_mask], arrays.si_upper_freqs[si_mask], fts)

        # Smooth the flux values if there is a jet break
        if self.tj and sf_mask.any():
            res[sf_mask] = self.smooth_jet_break(
                f=res[sf_mask], t=arrays.times[sf_mask],
                nu=arrays.frequencies[sf_mask], fts=fts
            )

        if self.tj and if_mask.any():
            res[if_mask] = self.smooth_jet_break(
                f=res[if_mask], t=arrays.times[if_mask],
                lower=arrays.if_lower_freqs[if_mask],
                upper=arrays.if_upper_freqs[if_mask],
                fts=fts
            )

        return res[subset_mask] if subset_mask is not None else res

    def spectral_flux(self, t, f, fts=False):
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
        # Characteristics
        nu_m = self.nu_m(t)
        nu_c = self.nu_c(t)
        nu_a = self.nu_a(t, nu_m)
        f_peak = self.f_peak(t)

        res = SpectralFluxModel(
            nu_m, nu_c, nu_a, f_peak, self.p, self.k)(f, fts)

        if self.tj:
            return self.smooth_jet_break(res, t, nu=f, fts=None)

        # return the spectral flux [mJy]
        return res

    def integrated_flux(self, t, lower, upper, fts=False):
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
        # Characteristics
        nu_m = self.nu_m(t)
        nu_c = self.nu_c(t)
        nu_a = self.nu_a(t, nu_m)
        f_peak = self.f_peak(t)

        res = IntegratedFluxModel(
            nu_m, nu_c, nu_a, f_peak, self.p, self.k)(lower, upper, fts)

        if self.tj:
            return self.smooth_jet_break(res, t, lower=lower, upper=upper, fts=fts)

        # return the integrated flux [erg s-1 cm-2]
        return res

    def spectral_index(self, t, lower, upper, fts=False):
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
        # Characteristics
        nu_m = self.nu_m(t)
        nu_c = self.nu_c(t)
        nu_a = self.nu_a(t, nu_m)
        f_peak = self.f_peak(t)

        # return the spectral indices [dimension less]
        return SpectralIndexModel(
            nu_m, nu_c, nu_a, f_peak, self.p, self.k)(lower, upper, fts)

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
        # Transform the smoothing factor
        s = self.sj or 1 / self.sji

        # Characteristics
        nu_m = self.nu_m(self.tj)
        nu_c = self.nu_c(self.tj)
        nu_a = self.nu_a(self.tj, nu_m)
        f_peak = self.f_peak(self.tj)

        # Evaluate the flux at the jet break time
        f_jet = model(
            nu_m, nu_c, nu_a, f_peak, p=self.p, k=self.k
        )(**kwargs)

        # return the flux smoothed over the jet break
        return (
            f ** -s + (f_jet * (t / self.tj) ** -self.p) ** -s
        ) ** -(1 / s)

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

    def nu_a(self, t, nu_m=None):
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
        nu_amc = model.evaluate_amc(t)
        nu_mac = model.evaluate_mac(t)

        return np.where(nu_amc < nu_m, nu_amc, nu_mac)
