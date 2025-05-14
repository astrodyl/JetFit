import json
import unittest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from jetfit.models2.basemodels import SpectralFluxModel, fast_to_slow_time


class TestSpectralFlux(unittest.TestCase):
    def test_MAC(self):
        """"""
        # Define the frequencies
        nu_m, nu_a, nu_c = 1e9, 1e11, 1e13
        nu = np.geomspace(1e8, 1e18, 500)

        # Define other
        f_peak, p, k = 1, 2.5, 0.0

        # Get smoothed flux
        model = SpectralFluxModel(nu_m, nu_c, nu_a, f_peak, p, k)

        # Get SPN98 flux
        b1, b2, b3 = model.spectral_indices()

        smoothed_flux = model(nu) #* (nu_a / nu_m) ** b2

        # Determine the segments
        seg0 = nu <= nu_m
        seg1 = np.logical_and(nu > nu_m, nu <= nu_c)
        seg2 = np.logical_and(nu_a < nu, nu_c >= nu_m)
        seg3 = nu > nu_c

        # Calculate the sharply-broken flux
        flux = np.empty(nu.size)
        flux[seg0] = f_peak * (nu_m / nu_a) ** ((p + 4) / 2) * (nu[seg0] / nu_m) ** 2
        flux[seg1] = f_peak * (nu_a / nu_m) ** ((1 - p) / 2) * (nu[seg1] / nu_a) ** 2.5
        flux[seg2] = f_peak * (nu[seg2] / nu_m) ** ((1 - p) / 2)
        flux[seg3] = f_peak * (nu_c / nu_m) ** ((1 - p) / 2) * (nu[seg3] / nu_c) ** (-p / 2)

        # Annotation for each break
        plt.annotate(r'$\nu_m$', xy=(1.2 * nu_m, 1e-9), xytext=(1.2 * nu_m, 1e-9), fontsize=12)
        plt.annotate(r'$\nu_a$', xy=(1.2 * nu_a, 1e-9), xytext=(1.2 * nu_a, 1e-9), fontsize=12)
        plt.annotate(r'$\nu_c$', xy=(1.2 * nu_c, 1e-9), xytext=(1.2 * nu_c, 1e-9), fontsize=12)

        # Plot the two
        plt.vlines(nu_m, ymin=0.0, ymax=f_peak * (nu_m / nu_a) ** ((p + 4) / 2), color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_a, ymin=0.0, ymax=f_peak * (nu_a / nu_m) ** b2, color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_c, ymin=0.0, ymax=f_peak * (nu_c / nu_m) ** b2, color='black', linestyle='--', alpha=0.6)
        plt.title(r'$\nu_m < \nu_a < \nu_c$', fontsize=18)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Flux [mJy]')
        plt.grid(alpha=0.5)
        plt.loglog(nu, smoothed_flux, label='Smoothly Broken')
        plt.loglog(nu, flux, label='Sharply Broken')
        plt.legend()
        plt.show()

    def test_slow_cooling_spectrum(self):
        """
        Visual inspection of slow-cooling smoothing approximation.
        """
        # Define the frequencies
        nu_a, nu_m, nu_c = 6e9, 2e11, 5e12
        nu = np.geomspace(1e8, 1e18, 500)

        # Define other
        f_peak, p, k = 2e4, 2.5, 0.0

        # Get smoothed flux
        model = SpectralFluxModel(nu_m, nu_c, nu_a, f_peak, p, k)
        smoothed_flux = model(nu)

        # Get SPN98 flux
        b1, b2, b3 = model.spectral_indices()

        # Determine the segments
        seg_f = nu < nu_m
        seg_g = np.logical_and(nu_c > nu, nu > nu_m)
        seg_h = nu > nu_c

        # Calculate the sharply-broken flux
        flux = np.empty(nu.size)
        flux[seg_f] = f_peak * (nu[seg_f] / nu_m) ** b1
        flux[seg_g] = f_peak * (nu[seg_g] / nu_m) ** b2
        flux[seg_h] = f_peak * (nu_c / nu_m) ** b2 * (nu[seg_h] / nu_c) ** b3

        # Plot the two
        plt.vlines(nu_m, ymin=0.0, ymax=f_peak, color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_c, ymin=0.0, ymax=f_peak * (nu_c / nu_m) ** b2, color='black', linestyle='--', alpha=0.6)
        plt.title('Slow Cooling Spectral Flux')
        plt.loglog(nu, smoothed_flux, label='Smoothly Broken')
        plt.loglog(nu, flux, label='Sharply Broken')
        plt.legend()
        plt.show()

    def test_slow_cooling_spectrum(self):
        """
        Visual inspection of slow-cooling smoothing approximation.
        """
        # Define the frequencies
        nu_a, nu_m, nu_c = 1e12, 1e10, 1e14
        nu = np.geomspace(1e8, 1e18, 500)

        # Define other
        f_peak, p, k = 2e4, 2.5, 0.0

        # Get smoothed flux
        model = SpectralFluxModel(nu_m, nu_c, nu_a, f_peak, p, k)
        smoothed_flux = model(nu)

        # Get SPN98 flux
        b1, b2, b3 = model.spectral_indices()

        # Determine the segments
        seg_a = nu < nu_m
        # seg_b = np.logical_and(nu > nu_m, nu < nu_a)
        seg_b = nu < nu_a
        seg_c = np.logical_and(nu_c > nu, nu > nu_a)
        seg_d = nu > nu_c

        # Calculate the sharply-broken flux
        flux = np.empty(nu.size)
        # flux[seg_a] = f_peak * (nu[seg_a] / nu_m) ** b1
        flux[seg_b] = f_peak * (nu[seg_b] / nu_c) ** b1
        flux[seg_c] = f_peak * (nu[seg_c] / nu_c) ** b2
        flux[seg_d] = f_peak * (nu_m / nu_c) ** b2 * (nu[seg_d] / nu_m) ** b3

        # Plot the two
        plt.vlines(nu_a, ymin=0.0, ymax=f_peak, color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_c, ymin=0.0, ymax=f_peak * (nu_c / nu_a) ** b2, color='black', linestyle='--', alpha=0.6)
        plt.title(r'Fast Cooling Spectral Flux $(\nu_m < \nu_a < \nu_c)$')
        plt.xlabel(r'$\nu$ [Hz]')
        plt.loglog(nu, smoothed_flux, label='Smoothly Broken')
        plt.loglog(nu, flux, label='Sharply Broken')
        plt.legend()
        plt.show()

    def test_equivalence(self):
        """"""
        # Define the frequencies
        nu_a, nu_m, nu_c = 6e9, 2e11, 5e12
        nu_m = np.geomspace(1e16, 1e10, 500)
        nu_c = np.geomspace(1e10, 1e16, 500)
        nu = np.geomspace(1e8, 1e18, 500)

        # Define other
        f_peak, p, k = 2e4, 2.5, 0.0

        # Get smoothed flux
        model = SpectralFluxModel(nu_m, nu_c, nu_a, f_peak, p, k)

        b1, b2, b3 = model.spectral_indices()
        s12, s23 = model.smoothing()

        sb = s23
        nu_b = 1e13
        b2 = b1

        s_slow = 1.15 - (0.125 * k) - (0.06 - 0.015 * k) * p
        s_fast = 3.34 + 0.17 * k - (0.82 + 0.035 * k) * p
        s_t = np.where(nu_m < nu_c, s_slow, s_fast)
        q = s_t * (b3 - b1)
        s_eff = s_slow + (s_fast - s_slow) / (1 + (nu / nu_b) ** q)

        x = f_peak * (((nu / nu_b) ** -(sb * (b1 - b2)) + 1) * (nu / nu_b) ** -(sb * (b2 - b3)) + 1) ** -(1 / sb) * (nu / nu_b) ** b3
        y = f_peak * (((nu / nu_b) ** -(s_eff * (b1 - b2)) + 1) * (nu / nu_b) ** -(s_eff * (b2 - b3)) + 1) ** -(1 / s_eff) * (nu / nu_b) ** b3

        plt.axvline(nu_b, color='black', linestyle='--', alpha=0.4)
        plt.loglog(nu, x, label=r'$s_{23}$')
        plt.loglog(nu, y, label=r'$s_{eff}$')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()


# fig, ax = plt.subplots()
        # line, = ax.loglog([], [], lw=2)
        # ax.set_xlim(nu[0], nu[-1])
        # ax.set_ylim(1e-3, 1e4)
        # title = ax.set_title("")
        #
        # def update(frame):
        #     """"""
        #     if frame == 0:
        #         frame =1
        #     nu_m1 = nu_m * frame
        #     model1 = SpectralFluxModel(nu_m1, nu_c, nu_a, f_peak, p, k)
        #     line.set_data(nu, model1(nu))
        #     # title.set_text(f"nu_m = {nu_m}")
        #     return line, title
        #
        # ani = FuncAnimation(fig, update, frames=500)
        # plt.show()