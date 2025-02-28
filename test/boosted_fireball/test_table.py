import time
import unittest

import numpy as np

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

    @unittest.skip("Only for performance purposes")
    def test_timing(self):
        """"""
        start = time.time()
        for _ in range(1000):
            pfs, cfs, sfs = (self.hydro_sim_table.get_characteristics_at(self.position))
        end = time.time()

        start2 = time.time()
        for _ in range(1000):
            chars = (self.hydro_sim_table.get_combined_characteristics(self.position))
            pfs2, cfs2, sfs2 = chars[:, 0], chars[:, 1], chars[:, 2]
        end2 = time.time()

        np.testing.assert_allclose(pfs, pfs2)
        np.testing.assert_allclose(cfs, cfs2)
        np.testing.assert_allclose(sfs, sfs2)

        print(end - start)
        print(end2 - start2)

if __name__ == '__main__':
    unittest.main()
