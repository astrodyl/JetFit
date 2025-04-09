import unittest

import numpy as np

from jetfit.models2.fireball import FireballModel


class TestCharacteristicModels(unittest.TestCase):
    def test_param_relationships(self):
        """"""
        for k in (0.0, 1.0, 1.333, 2.0):
            model = FireballModel(
                E=1.0, rho0=1.0, p=2.2, k=k, z=0.0, dL=1.0,
                eps_e=0.1, eps_b=0.1, X=0.7
            )

            # Base values to compare
            base_f_peak = model.f_peak(1.0)
            base_nu_m = model.nu_m(1.0)
            base_nu_c = model.nu_c(1.0)

            # Test that a change in time behaves as expected
            t2_f_peak = model.f_peak(2.0)
            t2_nu_m = model.nu_m(2.0)
            t2_nu_c = model.nu_c(2.0)

            # Assert equal within 1%
            e_pf = -(0.5 * (k / (4 - k)))
            e_nm = -1.5
            e_nc = -(0.5 * ((4 - 3*k) / (4 - k)))
            self.assertAlmostEqual(t2_f_peak / base_f_peak, 2 ** e_pf, delta=0.01)
            self.assertAlmostEqual(t2_nu_m / base_nu_m, 2 ** e_nm, delta=0.01)
            self.assertAlmostEqual(t2_nu_c / base_nu_c, 2 ** e_nc, delta=0.01)

            # Test that a change in energy behaves as expected
            model.E = 2.0
            e2_f_peak = model.f_peak(1.0)
            e2_nu_m = model.nu_m(1.0)
            e2_nu_c = model.nu_c(1.0)
            model.E = 1.0

            # Assert equal within 1%
            e_pf = 0.5 * (8 - 3 * k) / (4 - k)
            e_nm = 0.5
            e_nc = -0.5 * (4 - 3 * k) / (4 - k)
            self.assertAlmostEqual(e2_f_peak / base_f_peak, 2 ** e_pf, delta=0.01)
            self.assertAlmostEqual(e2_nu_m / base_nu_m, 2 ** e_nm, delta=0.01)
            self.assertAlmostEqual(e2_nu_c / base_nu_c, 2 ** e_nc, delta=0.01)

            # Test that a change in density behaves as expected
            model.rho0 = 2.0
            d2_f_peak = model.f_peak(1.0)
            d2_nu_m = model.nu_m(1.0)
            d2_nu_c = model.nu_c(1.0)
            model.rho0 = 1.0

            # Assert equal within 1%
            e_pf = 2 / (4 - k)
            e_nm = 0.0
            e_nc = -4 / (4 - k)
            self.assertAlmostEqual(d2_f_peak / base_f_peak, 2 ** e_pf, delta=0.01)
            self.assertAlmostEqual(d2_nu_m / base_nu_m, 2 ** e_nm, delta=0.01)
            self.assertAlmostEqual(d2_nu_c / base_nu_c, 2 ** e_nc, delta=0.01)

            # Test that a change in redshift behaves as expected
            model.z = 1.0
            z2_f_peak = model.f_peak(1.0)
            z2_nu_m = model.nu_m(1.0)
            z2_nu_c = model.nu_c(1.0)
            model.z = 0.0

            # Assert equal within 1%
            e_pf = 0.5 * (8 - k) / (4 - k)
            e_nm = 0.5
            e_nc =-0.5 * (4 + k) / (4 - k)
            self.assertAlmostEqual(z2_f_peak / base_f_peak, 2 ** e_pf, delta=0.01)
            self.assertAlmostEqual(z2_nu_m / base_nu_m, 2 ** e_nm, delta=0.01)
            self.assertAlmostEqual(z2_nu_c / base_nu_c, 2 ** e_nc, delta=0.01)

            # Test that a change in magnetic fraction behaves as expected
            model.eps_b = 0.2
            b2_f_peak = model.f_peak(1.0)
            b2_nu_m = model.nu_m(1.0)
            b2_nu_c = model.nu_c(1.0)
            model.eps_b = 0.1

            # Assert equal within 1%
            e_pf = 0.5
            e_nm = 0.5
            e_nc = -1.5
            self.assertAlmostEqual(b2_f_peak / base_f_peak, 2 ** e_pf, delta=0.01)
            self.assertAlmostEqual(b2_nu_m / base_nu_m, 2 ** e_nm, delta=0.01)
            self.assertAlmostEqual(b2_nu_c / base_nu_c, 2 ** e_nc, delta=0.01)

            # Test that a change in electric fraction behaves as expected
            model.eps_e = 0.2
            b2_f_peak = model.f_peak(1.0)
            b2_nu_m = model.nu_m(1.0)
            b2_nu_c = model.nu_c(1.0)
            model.eps_e = 0.1

            # Assert equal within 1%
            e_pf = 0.0
            e_nm = 2.0
            e_nc = 0.0
            self.assertAlmostEqual(b2_f_peak / base_f_peak, 2 ** e_pf, delta=0.01)
            self.assertAlmostEqual(b2_nu_m / base_nu_m, 2 ** e_nm, delta=0.01)
            self.assertAlmostEqual(b2_nu_c / base_nu_c, 2 ** e_nc, delta=0.01)

    def test_vdh_ism(self):
        """
        Test that the general k-model reduces to the ISM
        (k=0) case when k is set to 0.
        """
        model = FireballModel(
            E=1.0, rho0=1.0, p=2.2, k=0.0, z=0.0, dL=1.0,
            eps_e=0.1, eps_b=0.1, X=0.7
        )

        # Modeled values
        f_peak = model.f_peak(1.0)
        nu_m = model.nu_m(1.0)
        nu_c = model.nu_c(1.0)

        # True values
        f_peak_true = 21.3 * 0.5
        nu_m_true = 8.98e11 * (0.5**0.5)
        nu_c_true = 5.98e13 * (0.5**-0.5)

        # Assert equal within 1%
        self.assertAlmostEqual(f_peak / f_peak_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_m / nu_m_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_c / nu_c_true, 1.0, delta=0.01)

    def test_vdh_wind(self):
        """
        Test that the general k-model reduces to the wind
        (k=2) case when k is set to 2.
        """
        model = FireballModel(
            E=1.0, rho0=1.0, p=2.2, k=2.0, z=0.0, dL=1.0,
            eps_e=0.1, eps_b=0.1, X=0.7
        )

        # Modeled values
        f_peak = model.f_peak(1.0)
        nu_m = model.nu_m(1.0)
        nu_c = model.nu_c(1.0)

        # Normalization since we normalize density to 1e17cm,
        # but for k=2, VDH normalizes to A=5e11 * A_x,
        norm = (1 / 5e11) * 1e34 * 1.67e-24

        # True values
        f_peak_true = 60.8 * (0.5**1.5) * norm
        nu_m_true = 1.85e12 * (0.5 ** 0.5)
        nu_c_true = 9.97e11 * (0.5 ** -1.5) / norm**2

        # Assert equal within 1%
        self.assertAlmostEqual(f_peak / f_peak_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_m / nu_m_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_c / nu_c_true, 1.0, delta=0.01)

    def test_spn_ism(self):
        """
        Test that the spectral values match Sari, Piran, & Narayan 1998.
        """
        model = FireballModel(
            E=1.0, rho0=1.0, p=2.5, k=0.0, z=0.0, dL=1.0,
            eps_e=1, eps_b=1, X=1.0
        )

        # Modeled values
        nu_c = model.nu_c(1.0)
        nu_m = model.nu_m(1.0)
        f_peak = model.f_peak(1.0) * (8 * np.pi / 9)

        # True values
        nu_c_true = 2.7e12
        nu_m_true = 5.7e14
        f_peak_true = 1.1e2

        # Assert equal within 5%
        self.assertAlmostEqual(f_peak / f_peak_true, 1.0, delta=0.05)
        self.assertAlmostEqual(nu_m / nu_m_true, 1.0, delta=0.05)
        self.assertAlmostEqual(nu_c / nu_c_true, 1.0, delta=0.05)

    def test_cl_wind(self):
        """
        Test that the spectral values match CL 2000.
        """
        z = 1.0

        model = FireballModel(
            E=1.0, rho0=1.0, p=2.5, k=2.0, z=z, dL=1.0,
            eps_e=0.1, eps_b=0.1, X=0.0
        )

        # Normalization since we normalize density to 1e17cm,
        # but for k=2, VDH normalizes to A=5e11 * A_x,
        norm = (1 / 5e11) * 1e34 * 1.67e-24

        # Modeled values
        f_peak = model.f_peak(1.0)
        nu_m = model.nu_m(1.0)
        nu_c = model.nu_c(1.0)

        # True values
        f_peak_true = 20.0 * (
            (((np.sqrt(1 + z) - 1) / (np.sqrt(2) - 1)) ** -2) *
            (((1 + z) / 2) ** 0.5) * norm)
        nu_m_true = 5e12 * ((1 + z) / 2) ** 0.5
        nu_c_true = (2e12 * ((1 + z) / 2) ** -1.5) / (norm**2)

        # Assert equal within 5%
        self.assertAlmostEqual(f_peak / f_peak_true, 1.0, delta=0.05)
        self.assertAlmostEqual(nu_m / nu_m_true, 1.0, delta=0.05)
        self.assertAlmostEqual(nu_c / nu_c_true, 1.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
