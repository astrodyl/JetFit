import unittest

import numpy as np

from jetfit.core.utils.math_utils import chi_squared


class MyTestCase(unittest.TestCase):
    """"""
    def test_chi_squared_exact(self):
        """"""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        model = np.array([1.0, 2.0, 3.0, 4.0])
        errors = np.array([0.1, 0.2, 0.3, 0.4])

        chi2 = chi_squared(model, data, errors)
        assert np.isclose(chi2, 0.0), f"Expected chi2=0, got {chi2}"

    def test_chi_squared_known_offset(self):
        data = np.array([1.0, 2.0, 3.0])
        model = np.array([2.0, 3.0, 4.0])  # offset by +1
        errors = np.array([1.0, 1.0, 1.0])

        expected_chi2 = 3 * (1.0) ** 2  # (1**2 / 1**2) summed over 3 points
        chi2 = chi_squared(model, data, errors)
        assert np.isclose(chi2, expected_chi2), f"Expected chi2={expected_chi2}, got {chi2}"

    def test_chi_squared_different_errors(self):
        data = np.array([1.0, 2.0])
        model = np.array([2.0, 2.0])
        errors = np.array([1.0, 2.0])

        # chi^2 = (1/1)^2 + (0/2)^2 = 1
        expected_chi2 = 1.0
        chi2 = chi_squared(model, data, errors)
        assert np.isclose(chi2, expected_chi2), f"Expected chi2={expected_chi2}, got {chi2}"

    def test_chi_squared_eff_slop_reduces_chi2(self):
        f = np.array([10.0])
        y = np.array([12.0])
        e = np.array([1.0])
        s_no_slop = 0.0
        s_with_slop = 0.1

        chi2_no_slop = chi_squared(f, y, e, s_no_slop)
        chi2_with_slop = chi_squared(f, y, e, s_with_slop)

        assert chi2_with_slop < chi2_no_slop, "Slop should reduce chi2_eff"


if __name__ == '__main__':
    unittest.main()
