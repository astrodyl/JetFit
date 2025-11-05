import unittest

import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.utils import save_plot_unique
from jetfit.models.base import SpectralFluxModel


class TestPaperPlotting(unittest.TestCase):
    """
    Tests for models!
    """
    def setUp(self):
        """
        """
        self.t_obs = np.geomspace(0.1, 10.0, 100)

    def test_plot_sharply_broken_spectrum(self):
        """"""
        fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

        nu = np.geomspace(1e8, 1e18, 500)
        f_peak, p, k = 1, 2.25, 0.0

        # ============== ACM ==============
        # ============== ACM ==============
        # ============== ACM ==============
        nu_a, nu_c, nu_m = 1e10, 1e14, 1e16
        model = SpectralFluxModel(nu_m, nu_c, f_peak, p, k, nu_a=nu_a)
        b1, b2, b3 = model.spectral_indices()

        seg0 = nu < nu_a
        seg1 = np.logical_and(nu_c > nu, nu >= nu_a)
        seg2 = np.logical_and(nu_c < nu, nu <= nu_m)
        seg3 = nu > nu_m

        flux = np.empty(nu.size)
        flux[seg0] = f_peak * (nu_a / nu_c) ** b1 * (nu[seg0] / nu_a) ** 2
        flux[seg1] = f_peak * (nu[seg1] / nu_c) ** b1
        flux[seg2] = f_peak * (nu[seg2] / nu_c) ** b2
        flux[seg3] = f_peak * (nu_m / nu_c) ** b2 * (nu[seg3] / nu_m) ** b3

        axs[0].loglog(nu, flux, color='black', label='AMC')
        axs[0].vlines(nu_a, ymin=0.0, ymax=f_peak * (nu_a / nu_c) ** b1, color='black', linestyle='--', alpha=0.6)
        axs[0].vlines(nu_c, ymin=0.0, ymax=f_peak, color='black', linestyle='--', alpha=0.6)
        axs[0].vlines(nu_m, ymin=0.0, ymax=f_peak * (nu_m / nu_c) ** b2, color='black', linestyle='--', alpha=0.6)

        # Annotate frequencies
        axs[0].annotate(r'$\nu_a$', xy=(1.3 * nu_a, 1e-5), xytext=(1.3 * nu_a, 1e-5), fontsize=16)
        axs[0].annotate(r'$\nu_c$', xy=(1.3 * nu_c, 1e-5), xytext=(1.3 * nu_c, 1e-5), fontsize=16)
        axs[0].annotate(r'$\nu_m$', xy=(1.3 * nu_m, 1e-5), xytext=(1.3 * nu_m, 1e-5), fontsize=16)

        # Annotate spectral indices
        axs[0].annotate('2',    xy=(1e9,  3e-3), xytext=(1e9,  3e-3), fontsize=14)
        axs[0].annotate('1/3',  xy=(1e12, .7e0),  xytext=(1e12, .7e0),  fontsize=14)
        axs[0].annotate('-1/2', xy=(1e15, 7e-1), xytext=(1e15, 7e-1), fontsize=14)
        axs[0].annotate('-p/2', xy=(1e17, 2.7e-2), xytext=(1e17, 2.7e-2), fontsize=14)

        axs[0].text(0.98, 0.95, r"$\nu_a < \nu_c < \nu_m$", transform=axs[0].transAxes, ha="right", va="top", fontsize=12)
        axs[0].set_ylim(1e-6, 1e2)

        # ============== AMC ==============
        # ============== AMC ==============
        # ============== AMC ==============
        nu_a, nu_m, nu_c = 1e11, 1e13, 1e15
        model = SpectralFluxModel(nu_m, nu_c, f_peak, p, k, nu_a=nu_a)
        b1, b2, b3 = model.spectral_indices()

        seg0 = nu < nu_a
        seg1 = np.logical_and(nu_m > nu, nu >= nu_a)
        seg2 = np.logical_and(nu_m < nu, nu <= nu_c)
        seg3 = nu > nu_c

        flux = np.empty(nu.size)
        flux[seg0] = f_peak * (nu_a / nu_m) ** b1 * (nu[seg0] / nu_a) ** 2
        flux[seg1] = f_peak * (nu[seg1] / nu_m) ** b1
        flux[seg2] = f_peak * (nu[seg2] / nu_m) ** b2
        flux[seg3] = f_peak * (nu_c / nu_m) ** b2 * (nu[seg3] / nu_c) ** b3

        axs[1].loglog(nu, flux, color='black', label='AMC')
        axs[1].vlines(nu_a, ymin=0.0, ymax=f_peak * (nu_a / nu_m) ** b1, color='black', linestyle='--', alpha=0.6)
        axs[1].vlines(nu_m, ymin=0.0, ymax=f_peak, color='black', linestyle='--', alpha=0.6)
        axs[1].vlines(nu_c, ymin=0.0, ymax=f_peak * (nu_c / nu_m) ** b2, color='black', linestyle='--', alpha=0.6)

        # Annotate frequencies
        axs[1].annotate(r'$\nu_a$', xy=(1.3 * nu_a, 1e-5), xytext=(1.3 * nu_a, 1e-5), fontsize=16)
        axs[1].annotate(r'$\nu_m$', xy=(1.3 * nu_m, 1e-5), xytext=(1.3 * nu_m, 1e-5), fontsize=16)
        axs[1].annotate(r'$\nu_c$', xy=(1.3 * nu_c, 1e-5), xytext=(1.3 * nu_c, 1e-5), fontsize=16)

        # Annotate spectral indices
        axs[1].annotate('2',         xy=(5e9, 5e-3),  xytext=(5e9, 5e-3),  fontsize=14)
        axs[1].annotate('1/3',       xy=(1e12, 1),    xytext=(1e12, 1),    fontsize=14)
        axs[1].annotate('(1 - p)/2', xy=(8e13, 7e-1), xytext=(8e13, 7e-1), fontsize=14)
        axs[1].annotate('-p/2',      xy=(1e16, 1e-2), xytext=(1e16, 1e-2), fontsize=14)

        axs[1].text(0.98, 0.95, r"$\nu_a < \nu_m < \nu_c$", transform=axs[1].transAxes, ha="right", va="top",fontsize=12)
        axs[1].set_ylim(1e-6, 1e2)

        # ============== MAC ==============
        # ============== MAC ==============
        # ============== MAC ==============
        nu_m, nu_a, nu_c = 1e10, 1e13, 1e15
        model = SpectralFluxModel(nu_m, nu_c, f_peak, p, k, nu_a=nu_a)
        b1, b2, b3 = model.spectral_indices()

        seg0 = nu < nu_m
        seg1 = np.logical_and(nu <= nu_a, nu > nu_m)
        seg2 = np.logical_and(nu_a < nu, nu <= nu_c)
        seg3 = nu > nu_c

        flux = np.empty(nu.size)
        flux[seg0] = f_peak * (nu_m / nu_a) ** ((p + 4) / 2) * (nu[seg0] / nu_m) ** 2
        flux[seg1] = f_peak * (nu_a / nu_m) ** ((1 - p) / 2) * (nu[seg1] / nu_a) ** 2.5
        flux[seg2] = f_peak * (nu[seg2] / nu_m) ** ((1 - p) / 2)
        flux[seg3] = f_peak * (nu_c / nu_m) ** ((1 - p) / 2) * (nu[seg3] / nu_c) ** (-p / 2)

        axs[2].loglog(nu, flux, color='black', label='AMC')
        axs[2].vlines(nu_m, ymin=0.0, ymax=f_peak * (nu_a / nu_m) ** b2 * (nu_m / nu_a) ** b1, color='black', linestyle='--', alpha=0.6)
        axs[2].vlines(nu_a, ymin=0.0, ymax=f_peak * (nu_a / nu_m) ** b2, color='black', linestyle='--', alpha=0.6)
        axs[2].vlines(nu_c, ymin=0.0, ymax=f_peak * (nu_c / nu_m) ** b2, color='black', linestyle='--', alpha=0.6)

        # Annotate frequencies
        axs[2].annotate(r'$\nu_m$', xy=(1.3 * nu_m, 1e-13), xytext=(1.3 * nu_m, 1e-13), fontsize=16)
        axs[2].annotate(r'$\nu_a$', xy=(1.3 * nu_a, 1e-13), xytext=(1.3 * nu_a, 1e-13), fontsize=16)
        axs[2].annotate(r'$\nu_c$', xy=(1.3 * nu_c, 1e-13), xytext=(1.3 * nu_c, 1e-13), fontsize=16)

        # Annotate spectral indices
        axs[2].annotate('2',         xy=(1e9, 1e-10), xytext=(1e9, 1e-10), fontsize=14)
        axs[2].annotate('5/2',       xy=(1e11, 1e-5), xytext=(1e11, 1e-5), fontsize=14)
        axs[2].annotate('(1 - p)/2', xy=(5e13, 5e-2), xytext=(5e13, 5e-2), fontsize=14)
        axs[2].annotate('-p/2',      xy=(1e16, 1e-3), xytext=(1e16, 1e-3), fontsize=14)

        axs[2].text(0.98, 0.95, r"$\nu_m < \nu_a < \nu_c$", transform=axs[2].transAxes, ha="right", va="top",fontsize=12)
        axs[2].set_ylim(1e-14, 1e2)

        # ============== CAM ==============
        # ============== CAM ==============
        # ============== CAM ==============
        nu_c, nu_a, nu_m = 1e7, 1e10, 1e12
        model = SpectralFluxModel(nu_m, nu_c, f_peak, p, k, nu_a=nu_a)
        b1, b2, b3 = model.spectral_indices()
        flux_smooth = model.evaluate(nu)

        seg1 = nu < nu_a
        seg2 = np.logical_and(nu_a < nu, nu <= nu_m)
        seg3 = nu > nu_m

        rat = (1 / 3) * np.sqrt(nu_c / nu_a)

        flux = np.empty(nu.size)
        flux[seg1] = f_peak * (nu[seg1] / nu_a) ** 2
        flux[seg2] = f_peak * rat * (nu[seg2] / nu_a) ** -0.5
        flux[seg3] = f_peak * rat * (nu_m / nu_a) ** -0.5 * (nu[seg3] / nu_m) ** -(p / 2)

        axs[3].loglog(nu, flux, color='black', label='AMC')
        axs[3].loglog(nu, flux_smooth, color='red', label='AMC', alpha=0.6)
        axs[3].vlines(nu_c, ymin=0.0, ymax=f_peak * (nu_c / nu_m) ** b2, color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_a, ymin=0.0, ymax=f_peak * rat, color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_m, ymin=0.0, ymax=f_peak * rat * (nu_m / nu_a) ** b2, color='black', linestyle='--', alpha=0.6)

        # Annotate frequencies
        axs[3].annotate(r'$\nu_a$', xy=(1.3 * nu_a, 1e-9), xytext=(1.3 * nu_a, 1e-9), fontsize=16)
        axs[3].annotate(r'$\nu_m$', xy=(1.3 * nu_m, 1e-9), xytext=(1.3 * nu_m, 1e-9), fontsize=16)
        # axs[3].annotate(r'$\nu_c$', xy=(1.3 * nu_c, 1e-13), xytext=(1.3 * nu_c, 1e-13), fontsize=16)

        # Annotate spectral indices
        axs[3].annotate('2',    xy=(5e8, 2e-2),  xytext=(5e8, 2e-2), fontsize=14)
        axs[3].annotate('-1/2', xy=(1e11, 1e-2), xytext=(1e11, 1e-2), fontsize=14)
        axs[3].annotate('-p/2', xy=(5e13, 1e-4), xytext=(5e13, 1e-4), fontsize=14)
        # axs[3].annotate('-p/2', xy=(1e16, 1e-3), xytext=(1e16, 1e-3), fontsize=14)

        axs[3].text(0.98, 0.95, r"$\nu_c < \nu_a < \nu_m$", transform=axs[3].transAxes, ha="right", va="top",fontsize=12)
        # axs[3].set_ylim(1e-14, 1e2)

        fig.supylabel('Flux Density [mJy]', fontsize=14)
        plt.xlabel('Frequency [Hz]', fontsize=14)
        plt.xlim(1e8, 1e18)
        plt.tight_layout()
        plt.show()

        # save_plot_unique('radiation_plot', 'pdf', r'C:\Server')

    def test_plot_smoothing(self):
        """"""
        fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

        nu = np.geomspace(1e8, 1e18, 500)
        f_peak, p, k = 1, 2.25, 0.0

        # ============== FLUX SEGMENT SMOOTHING ==============
        # ============== FLUX SEGMENT SMOOTHING ==============
        # ============== FLUX SEGMENT SMOOTHING ==============
        nu_a, nu_c, nu_m = 1e10, 1e14, 1e16
        model = SpectralFluxModel(nu_m, nu_c, f_peak, p, k, nu_a=nu_a)
        flux_smooth = model(nu)
        b1, b2, b3 = model.spectral_indices()

        seg0 = nu < nu_a
        seg1 = np.logical_and(nu_c > nu, nu >= nu_a)
        seg2 = np.logical_and(nu_c < nu, nu <= nu_m)
        seg3 = nu > nu_m

        flux = np.empty(nu.size)
        flux[seg0] = f_peak * (nu_a / nu_c) ** b1 * (nu[seg0] / nu_a) ** 2
        flux[seg1] = f_peak * (nu[seg1] / nu_c) ** b1
        flux[seg2] = f_peak * (nu[seg2] / nu_c) ** b2
        flux[seg3] = f_peak * (nu_m / nu_c) ** b2 * (nu[seg3] / nu_m) ** b3

        axs[0].loglog(nu, flux, color='black', label='Sharply Broken')
        axs[0].loglog(nu, flux_smooth, color='royalblue', label='Smoothly Broken')
        axs[0].vlines(nu_a, ymin=0.0, ymax=f_peak * (nu_a / nu_c) ** b1, color='black', linestyle='--', alpha=0.6)
        axs[0].vlines(nu_c, ymin=0.0, ymax=f_peak, color='black', linestyle='--', alpha=0.6)
        axs[0].vlines(nu_m, ymin=0.0, ymax=f_peak * (nu_m / nu_c) ** b2, color='black', linestyle='--', alpha=0.6)

        # Annotate frequencies
        axs[0].annotate(r'$\nu_a$', xy=(1.3 * nu_a, 1e-5), xytext=(1.3 * nu_a, 1e-5), fontsize=16)
        axs[0].annotate(r'$\nu_c$', xy=(1.3 * nu_c, 1e-5), xytext=(1.3 * nu_c, 1e-5), fontsize=16)
        axs[0].annotate(r'$\nu_m$', xy=(1.3 * nu_m, 1e-5), xytext=(1.3 * nu_m, 1e-5), fontsize=16)

        # Annotate spectral indices
        # axs[0].annotate('2', xy=(1e9, 3e-3), xytext=(1e9, 3e-3), fontsize=14)
        # axs[0].annotate('1/3', xy=(1e12, .7e0), xytext=(1e12, .7e0), fontsize=14)
        # axs[0].annotate('-1/2', xy=(1e15, 7e-1), xytext=(1e15, 7e-1), fontsize=14)
        # axs[0].annotate('-p/2', xy=(1e17, 2.7e-2), xytext=(1e17, 2.7e-2), fontsize=14)

        axs[0].text(0.98, 0.95, r"$\nu_a < \nu_c < \nu_m$", transform=axs[0].transAxes, ha="right", va="top", fontsize=12)
        axs[0].set_ylim(1e-6, 1e2)

        fig.supylabel('Flux Density [mJy]', fontsize=14)
        plt.xlabel('Frequency [Hz]', fontsize=14)
        plt.xlim(1e8, 1e18)
        plt.tight_layout()
        plt.show()

        # ============== F2S SMOOTHING ==============
        # ============== F2S SMOOTHING ==============
        # ============== F2S SMOOTHING ==============
        # nu_a = np.full(500, 6e9)
        # nu_m = np.geomspace(1e16, 1e10, 500)
        # nu_c = np.geomspace(1e10, 1e16, 500)
        # nu = np.geomspace(1e8, 1e18, 500)
        #
        # # Define other
        # f_peak, p, k = 2e4, 2.5, 0.0
        #
        # # Get smoothed flux
        # model = SpectralFluxModel(nu_m, nu_c, f_peak, p, k, nu_a=nu_a)
        #
        # flux_sharp = model(nu)
        # flux_smooth = model(nu_m, fts=True)
        #
        # axs[1].loglog(nu, flux_sharp, label=r'$s_{23}$')
        # axs[1].loglog(nu, flux_smooth, label=r'$s_{eff}$')
        # plt.legend()
        # plt.show()