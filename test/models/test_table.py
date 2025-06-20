import json
import time
import unittest

import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.input import Observation
from jetfit.core.utils import utils
from jetfit.models.boosted import BoostedFireballModel, HydroSimTable
from jetfit.scripts.plot.light_curve import LightCurvePlot


class MyTestCase(unittest.TestCase):

    def setUp(self):
        """
        """
        self.hydro_sim_table = HydroSimTable(
            utils.get_hydro_sim_table_path()
        )

        # Create a position array using log(time), asymptotic lorentz factor,
        # boost lorentz factor, and observation angle (radians).
        self.position = np.array([
            [np.log(t), 7.973, 11.000923, 0.476]
            for t in [100.0, 1000.0, 1010.0, 2000.0, 5000.0, 10000.0]
        ])

    def test_nonsense(self):
        """"""
        times = np.geomspace(1e2, 1e7, 100)

        positions = []
        labels = []
        for i in np.linspace(0, 1, 10):
            positions.append(np.array([[np.log(t), 11, 15, i] for t in times]))
            labels.append(str(round(i, 3)))

        nu_ms = []
        for pos in positions:
            csf = self.hydro_sim_table.get_characteristics(pos)
            nu_ms.append(csf[:, 2])

        for i, nu_m in enumerate(nu_ms):
            plt.loglog(times, nu_m, label=labels[i])
        plt.legend()
        plt.show()


    def test_get_characteristics(self):
        """
        """
        csf = self.hydro_sim_table.get_characteristics(self.position)

        pfs, cfs, sfs = (csf[:, 0], csf[:, 1], csf[:, 2])

        np.testing.assert_allclose(  # Peak fluxes
            pfs, [0.0, 0.0, 0.0, 6.79732613e-06, 9.79182939e-04, 1.19235458e-02], strict=True
        )

        np.testing.assert_allclose(  # Cooling frequencies
            cfs, [0.0, 0.0, 0.0, 3.16227766e+16, 3.16227766e+16, 2.85617604e+14], strict=True
        )

        np.testing.assert_allclose(  # Synchrotron frequencies
            sfs, [0.0, 0.0, 0.0, 3.82635597e+14, 3.82635597e+14, 3.82635597e+14], strict=True
        )

    # @unittest.skip("Only for performance purposes")
    def test_timing(self):
        """"""
        times = np.geomspace(1e-2, 20, 500)

        pos = np.array([[np.log(t), 10.0, 20.0, 0.0] for t in (times * 86400)])
        pfs, cfs, sfs = (self.hydro_sim_table.get_characteristics_at(pos))

        pos2 = np.array([[np.log(t), 10.0, 10.0, 0.0] for t in (times * 86400)])
        pfs2, cfs2, sfs2 = (self.hydro_sim_table.get_characteristics_at(pos2))

        pos3 = np.array([[np.log(t), 10.0, 20.0, 0.03] for t in (times * 86400)])
        pfs3, cfs3, sfs3 = (self.hydro_sim_table.get_characteristics_at(pos3))

        pos4 = np.array([[np.log(t), 10.0, 5.0, 0.0] for t in (times * 86400)])
        pfs4, cfs4, sfs4 = (self.hydro_sim_table.get_characteristics_at(pos4))

        plt.loglog(times, sfs, label=r'$\theta_{0}$ ' + f'= 0.05, ' + r'$\theta_{obs} $' + f'= 0', color='red')
        plt.loglog(times, sfs3, label=r'$\theta_{0}$ ' + f'= 0.05, ' + r'$\theta_{obs} $' + f'= 0.6' + r'$\theta_0$', color='red', linestyle='--')
        plt.loglog(times, sfs2, label=r'$\theta_{0}$ ' + f'= 0.1, ' + r'$\theta_{obs} $' + f'= 0', color='blue')
        plt.loglog(times, sfs4, label=r'$\theta_{0}$ ' + f'= 0.2, ' + r'$\theta_{obs} $' + f'= 0', color='green')
        plt.title('RHD Unscaled Synchrotron Frequency')
        plt.xlabel(r'$\tau$ (days)')
        plt.ylabel(r'$f_m$ (Hz)')
        plt.legend()
        plt.show()

        plt.loglog(times, cfs, label=r'$\theta_{0}$ ' + f'= 0.05, ' + r'$\theta_{obs} $' + f'= 0', color='red')
        plt.loglog(times, cfs3, label=r'$\theta_{0}$ ' + f'= 0.05, ' + r'$\theta_{obs} $' + f'= 0.6' + r'$\theta_0$', color='red', linestyle='--')
        plt.loglog(times, cfs2, label=r'$\theta_{0}$ ' + f'= 0.1, ' + r'$\theta_{obs} $' + f'= 0', color='blue')
        plt.loglog(times, cfs4, label=r'$\theta_{0}$ ' + f'= 0.2, ' + r'$\theta_{obs} $' + f'= 0', color='green')
        plt.title('RHD Unscaled Cooling Frequency')
        plt.xlabel(r'$\tau$ (days)')
        plt.ylabel(r'$f_c$ (Hz)')
        plt.legend()
        plt.show()

        plt.loglog(times, pfs, label=r'$\theta_{0}$ ' + f'= 0.05, ' + r'$\theta_{obs} $' + f'= 0', color='red')
        plt.loglog(times, pfs3, label=r'$\theta_{0}$ ' + f'= 0.05, ' + r'$\theta_{obs} $' + f'= 0.6' + r'$\theta_0$', color='red', linestyle='--')
        plt.loglog(times, pfs2, label=r'$\theta_{0}$ ' + f'= 0.1, ' + r'$\theta_{obs} $' + f'= 0', color='blue')
        plt.loglog(times, pfs4, label=r'$\theta_{0}$ ' + f'= 0.2, ' + r'$\theta_{obs} $' + f'= 0', color='green')
        plt.title('RHD Unscaled Peak Flux')
        plt.xlabel(r'$\tau$ (days)')
        plt.ylabel(r'$f_{peak}$ (mJy)')
        plt.legend()
        plt.show()

    def test_boosted(self):
        """"""
        times = np.geomspace(796436 /86400, 30885700.0 / 86400, 500)

        params = {
            'E': 0.15869069395227384,
            'eta': 7.973477192135503,
            'gamma_b': 11.000923300022666,
            'dL28': 0.012188,
            'eps_b': 0.01332370657126752,
            'eps_e': 0.04072783842837688,
            'n0': 0.0009871221028954489,
            'p': 2.1333493591554804,
            'theta_obs': 0.4769798916899842,
            'zeta': 1.0,
            'z': 0.00973,
            'hydro_sim_table': self.hydro_sim_table,
        }

        model = BoostedFireballModel(
            **params
        )

        nu_c = model.nu_c(times, False)

        plt.loglog(model.scale_times(times) * 86400, model.nu_m(times, False), label=r'$f_m$', color='red')
        plt.loglog(model.scale_times(times) * 86400, model.nu_c(times, False), label=r'$f_c$', color='blue')
        plt.title('GW170817: Characteristic Spectral Functions')
        plt.xlabel(r'$\tau$ (days)')
        plt.ylabel(r'$f$ (Hz)')
        plt.legend()
        plt.show()

        plt.loglog(model.scale_times(times) * 86400, model.nu_m(times, True), label=r'$\nu_m$', color='red')
        plt.loglog(model.scale_times(times) * 86400, model.nu_c(times, True), label=r'$\nu_c$', color='blue')
        plt.title('GW170817: Scaled Characteristic Frequencies')
        plt.xlabel(r'$\tau$ (days)')
        plt.ylabel(r'$f$ (Hz)')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
