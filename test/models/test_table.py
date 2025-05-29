import time
import unittest

import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.utils import nav_utils
from jetfit.models.afterglow.boosted_fireball.hydro_sim.hydro_sim import HydroSimTable


class MyTestCase(unittest.TestCase):

    def setUp(self):
        """
        """
        self.hydro_sim_table = HydroSimTable(
            nav_utils.get_hydro_sim_table_path()
        )

        # Create a position array using log(time), asymptotic lorentz factor,
        # boost lorentz factor, and observation angle (radians).
        self.position = np.array([
            [np.log(t), 7.973, 11.000923, 0.476]
            for t in [100.0, 1000.0, 1010.0, 2000.0, 5000.0, 10000.0]
        ])

    def test_get_characteristics(self):
        """
        """
        pfs, cfs, sfs = (self.hydro_sim_table.get_characteristics_at(self.position))

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

        pos = np.array([[np.log(t), 10.0, 5.0, 0.0] for t in (times * 86400)])
        pfs4, cfs4, sfs4 = (self.hydro_sim_table.get_characteristics_at(pos))

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

        # start2 = time.time()
        # for _ in range(1000):
        # chars = (self.hydro_sim_table.get_combined_characteristics(self.position))
        # pfs2, cfs2, sfs2 = chars[:, 0], chars[:, 1], chars[:, 2]
        # end2 = time.time()

        # np.testing.assert_allclose(pfs, pfs2)
        # np.testing.assert_allclose(cfs, cfs2)
        # np.testing.assert_allclose(sfs, sfs2)

        # print(end - start)
        # print(end2 - start2)

if __name__ == '__main__':
    unittest.main()
