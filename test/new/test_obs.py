import time
import unittest
import astropy.units as u
import numpy as np

from jetfit.core.defns.enums import DataType
from jetfit.core.utils import nav_utils
from jetfit.core.input import Observation


class FromCSV(unittest.TestCase):
    """"""
    @classmethod
    def setUp(cls):
        """"""
        cls.obs = Observation.from_csv(
            nav_utils.get_test_path() / 'resources' / 'test.csv'
        )

    def test_observation(self) -> None:
        """"""
        obs = self.obs

        self.assertEqual(obs.length, 104)
        self.assertEqual(len(obs.value_array), obs.length)
        self.assertEqual(len(obs.value_array), len(obs.error_array))

        # Check integrated flux
        t_expected = u.Quantity(104.106, 's')
        self.assertEqual(obs.data[0].time, t_expected)

        val_expected = u.Quantity(7.52e-10, 'erg cm-2 s-1')
        self.assertEqual(obs.data[0].value, val_expected)

        self.assertEqual(obs.data[0].uncertainty.lower, u.Quantity(9.78e-11, 'erg cm-2 s-1'))
        self.assertEqual(obs.data[0].uncertainty.upper, u.Quantity(9.78e-11, 'erg cm-2 s-1'))

        # Check spectral flux
        self.assertEqual(obs.data[75].value.unit, 'mJy')
        self.assertEqual(obs.data[75].uncertainty.lower.unit, 'mJy')
        self.assertEqual(obs.data[75].uncertainty.upper.unit, 'mJy')

        # Check spectral index
        self.assertEqual(obs.data[-1].value.value, -1.3)
        self.assertEqual(obs.data[-1].value.unit, u.dimensionless_unscaled)
        self.assertEqual(obs.data[-1].time_range.lower, u.Quantity(16628, 's'))
        self.assertEqual(obs.data[-1].time_range.upper, u.Quantity(446331, 's'))

    def test_speed(self):
        """"""
        obs = Observation.from_csv(nav_utils.get_test_path() / 'resources' / 'test_spectral.csv')
        types = obs.flux_types

        # Check loop speed
        start1 = time.time()
        res = np.full(len(obs.data), np.nan)

        for _ in range(100000):
            for i, t in enumerate(types):
                if t == DataType.SPECTRAL_INDEX:
                    res[i] = 1

                elif t == DataType.INTEGRATED_FLUX:
                    res[i] = 2

                else:
                    res[i] = 3

        end1 = time.time()
        print('loop time: ', end1 - start1)
        print()

        # np speed
        start2 = time.time()
        res2 = np.full(len(obs.data), np.nan)

        for _ in range(100000):
            res2[types == DataType.SPECTRAL_INDEX]  = 1
            res2[types == DataType.INTEGRATED_FLUX] = 2
            res2[types == DataType.SPECTRAL_INDEX]  = 3

        end2 = time.time()
        print('numpy time: ', end2 - start2)


if __name__ == '__main__':
    unittest.main()
