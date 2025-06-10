import time
import unittest

import numpy as np
from matplotlib import pyplot as plt

import jetsimpy

from jetfit.core.input import Observation
from jetfit.models.jetsim import JetSimpy


class TestJetSimpy(unittest.TestCase):
    def test_modeling(self):
        """"""
        params = {
            'Eiso': 1e53,   # Isotropic equivalent energy (erg)
            'lf': 600,      # initial Lorentz factor
            'theta_c': 0.1, # half opening angle
            'n0': 1,        # ism number density
            'A': 0,         # wind number density amplitude
            'eps_e': 0.1,   # epsilon_e
            'eps_b': 0.01,  # epsilon_b
            'p': 2.17,      # electron power index
            'theta_v': 0.4, # viewing angle
            'd': 16_0009.4496,    # luminosity distance (Mpc)
            'z': 2.006,       # redshift
        }

        obs = Observation.from_csv(r"C:\Projects\repos\JetFit\jetfit\resources\done\050922C\050922C.csv")

        # Model using example from GitHub
        tday = np.logspace(-2, 3, 30)
        tsecond = tday * 3600 * 24
        nu = 1e15

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
