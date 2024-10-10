from typing import List

import numpy as np

from jetfit.fit.flux import FluxFastCoolingSpectral, FluxSlowCoolingSpectral, FluxFastCoolingIntegrated, \
    FluxSlowCoolingIntegrated
from jetfit.fit.interpolator import Interpolator


"""
    flux_generator.py

    This module contains the FluxGenerator, which generates synthetic
    light curves from the characteristic spectral functions. See: Ryan+ 2014
"""


class FluxGenerator:
    table_info = None

    def __init__(self, table: str, log_table: bool = True, log_axis: List[str] = None):
        """ Initialize FluxGenerator.

        :param table: directory to boosted fireball table
        :param log_table: whether Table is measured in log scale
        :param log_axis: whether certain axis is measured in log scale
        """
        if log_axis is None:
            log_axis = ['tau']

        self._Interpolator = Interpolator(table, log_table=log_table, log_axis=log_axis)
        self.table_info = self._Interpolator.get_table_info()

    @staticmethod
    def get_taus(times: np.ndarray, params: dict) -> np.ndarray:
        """ Calculates and returns the scaled times

        :param times: observational time in second.
        :param params: contains all parameters {z, dL, E, n, p, epse, epsb, xiN, Eta0, GammaB,theta_obs}
        :return: float: scaled time
        """
        f0 = (params['circumburst_density'] / params['explosion_energy'])**(1./3.) / (1.0 + params['redshift'])
        return f0 * times

    def get_transformed_value(self, taus: np.ndarray, params: dict) -> (float, float, float):
        """ Applies the scale relations. sees Ryan+ 2014.

        :param taus: scaled time
        :param params: contains all parameters {z, dL, E, n, p, epse, epsb, xiN, Eta0, GammaB, theta_obs}
        :return: spectral function values corresponding to params.
        """
        position = np.array([[tau,
                              params['asymptotic_lorentz_factor'],
                              params['boost_lorentz_factor'],
                              params['observation_angle']] for tau in taus])

        f_peak, f_nu_c, f_nu_m = self._Interpolator.get_value(position)

        if np.isnan(f_peak[0]):
            return f_peak, f_nu_c, f_nu_m

        f1 = ((1 + params['redshift']) / (params['luminosity_distance'] * params['luminosity_distance']) *
              (params['spectral_index'] - 1) / (3. * params['spectral_index'] - 1.) *
              params['explosion_energy'] * params['circumburst_density']**0.5 *
              params['magnetic_energy_fraction']**0.5 * params['accelerated_electron_fraction'])

        f2 = (1.0 / (1 + params['redshift']) * params['explosion_energy']**(-2./3.) *
              params['circumburst_density']**(-5./6.) * params['magnetic_energy_fraction']**(-3./2.))

        f3 = (1.0 / (1 + params['redshift']) * (params['spectral_index'] - 2.)**2 /
              (params['spectral_index'] - 1.)**2 * params['circumburst_density']**(1./2.) *
              params['electron_energy_fraction']**2 * params['magnetic_energy_fraction']**(1./2.) *
              params['accelerated_electron_fraction']**(-2))

        return f1 * f_peak, f2 * f_nu_c, f3 * f_nu_m

    # <editor-fold desc="Single Flux Calculations">
    @staticmethod
    def get_spectral_flux(frequency: float, spectral_index: float, peak_flux: float,
                          cooling_frequency: float, synchrotron_frequency: float) -> float:
        """ Returns the spectral flux for a given segment as defined in
        Sari et al. (1998). Implements equation 7 in section 2.

        :return: spectral flux in units of erg / s / cm^2 / Hz
        """
        if synchrotron_frequency > cooling_frequency:
            generator = FluxFastCoolingSpectral
        else:
            generator = FluxSlowCoolingSpectral

        return generator(frequency,
                         peak_flux=peak_flux,
                         cooling_frequency=cooling_frequency,
                         synchrotron_frequency=synchrotron_frequency,
                         spectral_index=spectral_index).flux()

    @staticmethod
    def get_integrated(bounds: tuple, spectral_index: float, peak_flux: float,
                       cooling_frequency: float, synchrotron_frequency: float) -> float:
        """ Calculates the integrated flux for a range of frequencies by
        integrating the fast cooling set of equations (section 2, equation
        7) in Sari et al. (1998). Frequency ranges that span more than one
        segment are broken into multiple integrals.

        :param bounds: tuple of (lower bound, upper bound) measured in Hz
        :param spectral_index: spectral index
        :param peak_flux: peak flux measured in mJy
        :param cooling_frequency: measured in Hz
        :param synchrotron_frequency: measured in Hz
        :return: integrated flux in units of erg / s / cm^2
        """
        if synchrotron_frequency > cooling_frequency:
            generator = FluxFastCoolingIntegrated
        else:
            generator = FluxSlowCoolingIntegrated

        return generator(bounds,
                         peak_flux=peak_flux,
                         cooling_frequency=cooling_frequency,
                         synchrotron_frequency=synchrotron_frequency,
                         spectral_index=spectral_index).integrate() * 1.0e-26
    # </editor-fold>

    def get_spectral(self, times: np.ndarray, frequencies: np.ndarray, params: dict) -> np.ndarray:
        """ Constructs the synthetic light curve as a series of connected power
        laws. See: Spectra and Light Curves of Gamma-Ray Burst Afterglows (Sari
        et al. 1998) Section 2, https://arxiv.org/abs/astro-ph/9712005.

        :param times: observational time (seconds since trigger)
        :param frequencies: frequencies corresponding to observational time
        :param params: contains all parameters {z, dL, E, n, p, epse, epsb, xiN, Eta0, GammaB, theta_obs}
        :return: synthetic light curve (flux density)
        """
        f_peak, nu_c, nu_m = self.get_transformed_value(
            self.get_taus(times, params),
            params
        )

        if np.isnan(f_peak[0]):
            return f_peak

        # Slow cooling regime
        idx_slow = (nu_m < nu_c)
        idx_slow1 = idx_slow * (frequencies < nu_m)
        idx_slow2 = idx_slow * (frequencies >= nu_m) * (frequencies < nu_c)
        idx_slow3 = idx_slow * (frequencies >= nu_c)

        # Fast cooling regime
        idx_fast1 = (~idx_slow) * (frequencies < nu_c)
        idx_fast2 = (~idx_slow) * (frequencies >= nu_c) * (frequencies < nu_m)
        idx_fast3 = (~idx_slow) * (frequencies >= nu_m)

        inu_m = 1.0 / nu_m
        inu_c = 1.0 / nu_c

        p = params['spectral_index']

        output = np.zeros(len(times))
        # Construct the flux density light curve in the slow regime
        output[idx_slow1] = np.power(frequencies[idx_slow1] * inu_m[idx_slow1], 1.0 / 3.0)
        output[idx_slow2] = np.power(frequencies[idx_slow2] * inu_m[idx_slow2], 0.5 - 0.5 * p)
        output[idx_slow3] = (np.power(nu_c[idx_slow3] * inu_m[idx_slow3], 0.5 - 0.5 * p) *
                             np.power(frequencies[idx_slow3] * inu_c[idx_slow3], -0.5 * p))

        # Construct the flux density light curve in the fast regime
        output[idx_fast1] = np.power(frequencies[idx_fast1] * inu_c[idx_fast1], 1.0 / 3.0)
        output[idx_fast2] = np.power(frequencies[idx_fast2] * inu_c[idx_fast2], -0.5)
        output[idx_fast3] = (np.power(nu_m[idx_fast3] * inu_c[idx_fast3], -0.5) *
                             np.power(frequencies[idx_fast3] * inu_m[idx_fast3], -0.5 * p))

        return f_peak * output

    # noinspection DuplicatedCode
    def get_integrated_flux(self, times: np.ndarray, frequencies, params):
        """ Constructs the synthetic light curve as a series of connected power
        laws. See: Spectra and Light Curves of Gamma-Ray Burst Afterglows (Sari
        et al. 1998) Section 2, https://arxiv.org/abs/astro-ph/9712005.

        :param times: observational time in second
        :param frequencies: frequencies. The length should be the same as times
        :param params: contains all parameters {z, dL, E, n, p, epse, epsb, xiN, Eta0, GammaB, theta_obs}
        :return: synthetic light curve (erg cm^-2 s^-1)
        """
        f_peak, nu_c, nu_m = self.get_transformed_value(
            self.get_taus(times, params),
            params
        )

        if np.isnan(f_peak[0]):
            return f_peak

        l_freqs, h_freqs = [], []
        for _ in range(len(frequencies)):
            l_freqs.append(7.25e16)
            h_freqs.append(2.42e18)

        l_freqs = np.asarray(l_freqs)
        h_freqs = np.asarray(h_freqs)

        # Slow cooling regime
        idx_slow = (nu_m < nu_c)
        idx_slow1 = idx_slow * (l_freqs < nu_m)
        idx_slow2 = idx_slow * (h_freqs >= nu_m) * (l_freqs < nu_c)
        idx_slow3 = idx_slow * (h_freqs >= nu_c)

        # Fast cooling regime
        idx_fast1 = (~idx_slow) * (l_freqs < nu_c)
        idx_fast2 = (~idx_slow) * (h_freqs >= nu_c) * (l_freqs < nu_m)
        idx_fast3 = (~idx_slow) * (h_freqs >= nu_m)

        inu_m = 1.0 / nu_m
        inu_c = 1.0 / nu_c

        p = params['spectral_index']

        # Construct the total flux light curve
        output = np.zeros(len(times))

        output[idx_slow1] = 0.75 * nu_m[idx_slow1] * (
                np.power(np.minimum(h_freqs[idx_slow1], nu_m[idx_slow1]) * inu_m[idx_slow1], 4.0 / 3.0) -
                np.power(l_freqs[idx_slow1] * inu_m[idx_slow1], 4.0 / 3.0)
        )

        output[idx_slow2] = 2.0 / (3.0 - p) * nu_m[idx_slow2] * (
                np.power(np.minimum(h_freqs[idx_slow2], nu_c[idx_slow2]) * inu_m[idx_slow2], 1.5 - 0.5 * p) -
                np.power(np.maximum(l_freqs[idx_slow2], nu_m[idx_slow2]) * inu_m[idx_slow2], 1.5 - 0.5 * p)
        )

        output[idx_slow3] = (2.0 / (2.0 - p) * nu_c[idx_slow3] *
                             np.power(nu_c[idx_slow3] * inu_m[idx_slow3], 0.5 - 0.5 * p) * (
                                     np.power(h_freqs[idx_slow3] * inu_c[idx_slow3], 1.0 - 0.5 * p) -
                                     np.power(np.maximum(l_freqs[idx_slow3], nu_c[idx_slow3]) *
                                              inu_c[idx_slow3], 1.0 - 0.5 * p)
                             ))

        output[idx_fast1] = 0.75 * nu_c[idx_fast1] * (np.power(
            np.minimum(h_freqs[idx_fast1], nu_c[idx_fast1]) *
            inu_c[idx_fast1], 4.0 / 3.0) - np.power(l_freqs[idx_fast1] * inu_c[idx_fast1], 4.0 / 3.0))

        output[idx_fast2] = 2.0 * nu_c[idx_fast2] * (np.sqrt(
            np.minimum(h_freqs[idx_fast2], nu_m[idx_fast2]) *
            inu_c[idx_fast2]) - np.sqrt(np.maximum(l_freqs[idx_fast2], nu_c[idx_fast2]) * inu_c[idx_fast2]))

        output[idx_fast3] = (2.0 / (2.0 - p) * nu_m[idx_fast3] / np.sqrt(nu_m[idx_fast3]*inu_c[idx_fast3]) *
                             (np.power(h_freqs[idx_fast3] * inu_m[idx_fast3], 1.0 - 0.5 * p) -
                              np.power(np.maximum(l_freqs[idx_fast3], nu_m[idx_fast3]) *
                                       inu_m[idx_fast3], 1.0 - 0.5 * p)))

        return 1.0e-26 * f_peak * output
