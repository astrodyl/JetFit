import time
import unittest

import numpy as np
from jetsimpy import Jet, Gaussian
from matplotlib import pyplot as plt

import jetsimpy

from jetfit.core.input import Observation
from jetfit.models.jetsim import JetSimpy


class TestJetSimpy(unittest.TestCase):
    def test_smoothing(self):
        """"""
        p = dict(
            Eiso=1e52,      # Core isotropic equivalent energy
            lf=300,         # Core Lorentz factor
            theta_c=0.1,    # half opening angle
            n0=0,           # ism number density
            A=30,           # wind number density amplitude
            eps_e=0.1,      # epsilon_e
            eps_b=0.01,     # epsilon_b
            p=2.17,         # electron power index
            theta_v=0.0,    # viewing angle (rad)
            d=474.33,    # distance (Mpc)
            z=0.1,        # redshift
        )
        jet = Jet(Gaussian(p['theta_c'], p['Eiso'], p['lf']), p['A'], p['n0'], tmax=1e12)

        tday = np.logspace(-2, 3, 200)
        tsecond = tday * 3600 * 24

        model = JetSimpy(**p)

        plt.loglog(tsecond, model.nu_m(tday), label=r'$\nu_m$')
        plt.loglog(tsecond, model.nu_c(tday), label=r'$\nu_c$')
        plt.loglog(tsecond, model.nu_a(tday), label=r'$\nu_a$')
        plt.ylabel(r'$\nu$ [Hz]')
        plt.xlabel('Time [seconds since trigger]')
        plt.legend()
        plt.show()

        plt.loglog(tsecond, jet.FluxDensity(tsecond, 1e12, p), label=r'$\nu = 1e12$ Hz')
        plt.loglog(tsecond, jet.FluxDensity(tsecond, 1e14, p), label=r'$\nu = 1e14$ Hz')
        plt.loglog(tsecond, jet.FluxDensity(tsecond, 1e15, p), label=r'$\nu = 1e15$ Hz')
        plt.loglog(tsecond, jet.FluxDensity(tsecond, 1e18, p), label=r'$\nu = 1e18$ Hz')
        plt.ylabel('Flux [mJy]')
        plt.xlabel('Time [seconds since trigger]')
        plt.legend()
        plt.show()

    def test_spectrum(self):
        """"""
        params = {
            'Eiso': 1e52,  # Isotropic equivalent energy (erg)
            'lf': 300,  # initial Lorentz factor
            'theta_c': 0.1,  # half opening angle
            'n0': 0,  # ism number density
            'A': 30,  # wind number density amplitude
            'eps_e': 0.1,  # epsilon_e
            'eps_b': 0.01,  # epsilon_b
            'p': 2.2,  # electron power index
            'theta_v': 0.0,  # viewing angle
            'd': 16_0009.4496,  # luminosity distance (Mpc)
            'z': 2.006,  # redshift
        }
        ndata = 1000

        t = 1.0
        nu = np.geomspace(1e5, 1e18, ndata)

        model = JetSimpy(**params)
        spectrum = model.spectral_flux(t, nu, model="sync")
        spectrum_sa = model.spectral_flux(t, nu, model="sync_sa")

        # Calculate the characteristics
        f_pk = model.f_peak(t)
        nu_a = model.nu_a(t, 'sync_sa') * (1 + params['z'])
        nu_m = model.nu_m(t, 'sync_sa') * (1 + params['z'])
        nu_c = model.nu_c(t, 'sync_sa') * (1 + params['z'])

        ts = np.geomspace(1e-2, 1e2, 200)
        plt.loglog(ts, model.nu_m(ts, 'sync_sa') * (1 + params['z']), label=r'$\nu_m$')
        plt.loglog(ts, model.nu_c(ts, 'sync_sa') * (1 + params['z']), label=r'$\nu_c$')
        plt.loglog(ts, model.nu_a(ts, 'sync_sa') * (1 + params['z']), label=r'$\nu_a$')
        plt.legend()
        plt.show()

        # Annotation for each break
        plt.annotate(r'$\nu_m$', xy=(1.2 * nu_m, 1e-9), xytext=(1.2 * nu_m, 1e-9), fontsize=12)
        plt.annotate(r'$\nu_a$', xy=(1.2 * nu_a, 1e-9), xytext=(1.2 * nu_a, 1e-9), fontsize=12)
        plt.annotate(r'$\nu_c$', xy=(1.2 * nu_c, 1e-9), xytext=(1.2 * nu_c, 1e-9), fontsize=12)

        plt.loglog(nu, spectrum, label='sync')
        plt.loglog(nu, spectrum_sa, label='sync_sa')
        plt.axhline(f_pk, color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_m, ymin=0.0, ymax=f_pk, color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_a, ymin=0.0, ymax=f_pk * (nu_a / nu_m) ** (1/3), color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_c, ymin=0.0, ymax=f_pk * (nu_c / nu_m) ** ((1-params['p'])/2), color='black', linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

    def test_modeling(self):
        """"""
        params = {
            'Eiso': 1e52,   # Isotropic equivalent energy (erg)
            'lf': 600,      # initial Lorentz factor
            'theta_c': 0.1, # half opening angle
            'n0': 1,        # ism number density
            'A': 0,         # wind number density amplitude
            'eps_e': 0.1,   # epsilon_e
            'eps_b': 0.01,  # epsilon_b
            'p': 2.2,      # electron power index
            'theta_v': 0.0, # viewing angle
            'd': 16_0009.4496,    # luminosity distance (Mpc)
            'z': 2.006,       # redshift
        }

        model = JetSimpy(**params)

        # Model using example from GitHub
        tday = np.logspace(-2, 3, 100)
        tsecond = tday * 3600 * 24
        nu = 1e15

        f_sync = model.spectral_flux(tday, nu)
        plt.loglog(tday, f_sync, label="sync")

        try:
            f_sync_sa = model.spectral_flux(tday, nu, model="sync_sa")
            f_sync_sa2 = model.spectral_flux(tday, nu, model="sync_sa2")
            plt.loglog(tday, f_sync_sa, label="sync_sa", linestyle='--')
            plt.loglog(tday, f_sync_sa2, label="sync_sa2", linestyle='--')
        except Exception as e:
            print(e)

        plt.legend()
        plt.show()

        plt.loglog(tday, model.nu_m(tday), label=r'$\nu_m$')
        plt.loglog(tday, model.nu_c(tday), label=r'$\nu_c$')
        plt.loglog(tday, model.nu_a(tday), label=r'$\nu_a$')
        plt.legend()
        plt.show()

        obs = Observation.from_csv(r"C:\Projects\repos\JetFit\jetfit\resources\done\050922C\050922C.csv")

        tday = obs.as_arrays.times[obs.sflux_loc]
        tsecond = tday * 86_400
        nus = obs.as_arrays.frequencies[obs.sflux_loc]

        # for nu in nus:
        start = time.time()
        fd_gaussian = jetsimpy.FluxDensity_gaussian(tsecond, nus, params)
        # plt.loglog(tday, fd_gaussian, linestyle='--')
        print(time.time() - start)
        # plt.show()

        # Model the observation using adapter class
        start = time.time()
        model = JetSimpy(**params)
        modeled = model.model(obs)[obs.sflux_loc]
        print(time.time() - start)

        plt.scatter(tsecond, modeled, color='red')
        plt.xscale('log')
        plt.yscale('log')
        # plt.show()


        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
