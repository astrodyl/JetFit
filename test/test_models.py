import unittest

import numpy as np

from jetfit.models.base import cooling_frequency, synchrotron_frequency
from jetfit.models.base import peak_flux, nu_a_amc_ad, nu_a_mac_ad, nu_a_acm_ad, nu_a_acm_rad


class TestSpectralFunctions(unittest.TestCase):
    """
    Tests for models!
    """
    def setUp(self):
        """
        """
        self.t_obs = np.geomspace(0.1, 10.0, 100)

    def test_peak_flux_adiabatic(self):
        """ Tests that the adiabatic peak flux model returns the correct values. """
        z = 1.0  # Vary z if you'd like
        p, eps_b, d, hmf = 2.2, 0.1, 1e28, 0.7  # Do not vary these!

        # Evaluate cooling frequency for ISM and WIND
        f_pk_ism = peak_flux(1.0e52, 1.0000, 0.0, eps_b, d, z, hmf, t_obs=1.0)
        f_pk_win = peak_flux(1.0e52, 3.0e35, 2.0, eps_b, d, z, hmf, t_obs=1.0)

        # Values taken fro VHD09
        f_pk_ism_true = 21.3 * (0.5 * (1 + z))
        f_pk_win_true = 60.8 * (0.5 * (1 + z)) ** 1.5

        # Assert equal within 1%
        self.assertAlmostEqual(f_pk_ism / f_pk_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(f_pk_win / f_pk_win_true, 1.0, delta=0.01)

    def test_peak_flux_radiative(self):
        """ Tests that the radiative peak flux model returns the correct values. """
        z = 1.5  # Vary z if you'd like
        p, eps_b, d, hmf = 2.2, 0.1, 1e28, 0.7  # Do not vary these!

        # Evaluate cooling frequency for ISM and WIND
        f_pk_ism = peak_flux(1.0e52, 1.0000, 0.0, eps_b, d, z, hmf, t_obs=1.0, adiabatic=False)
        f_pk_win = peak_flux(1.0e52, 3.0e35, 2.0, eps_b, d, z, hmf, t_obs=1.0, adiabatic=False)

        # Values taken fro VHD09
        f_pk_ism_true = 109.0 * (0.5 * (1 + z)) ** (10.0 / 7.0)
        f_pk_win_true = 170.0 * (0.5 * (1 + z)) ** (5.0 / 3.0)

        # Assert equal within 1%
        self.assertAlmostEqual(f_pk_ism / f_pk_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(f_pk_win / f_pk_win_true, 1.0, delta=0.01)

    def test_cooling_frequency_scaling_adiabatic_ism(self):
        """
            Tests that changing the values scales the cooling frequency
            in the adiabatic regime for an ISM medium properly.
        """
        # Vary the following values
        z = 2.1
        nrg = 2.5e52
        n0 = 3.2
        t = 12.11

        # Evaluate cooling frequency for ISM
        nu_c_ism = cooling_frequency(nrg, n0, 0.0, eps_b=0.1, z=z, t_obs=t)

        # Value taken fro VHD09
        nu_c_ism_true_nrg = 5.98e13 * (
            (0.5 * (1 + z)) ** -0.5 *  # redshift scaling
            (nrg / 1e52) ** -0.5 *     # energy scaling
            n0 ** -1.0 *               # density scaling
            t ** -0.5                  # time scaling
        )

        # Assert equal within 1%
        self.assertAlmostEqual(nu_c_ism / nu_c_ism_true_nrg, 1.0, delta=0.01)

    def test_cooling_frequency_scaling_radiative_ism(self):
        """
            Tests that changing the values scales the cooling frequency
            in the radiative regime for an ISM medium properly.
        """
        # Vary the following values
        z = 2.1
        nrg = 2.5e52
        n0 = 3.2
        t = 3.41

        # Evaluate cooling frequency for ISM
        nu_c_ism = cooling_frequency(nrg, n0, 0.0, eps_b=0.1, z=z, t_obs=t, adiabatic=False)

        # Value taken fro VHD09
        nu_c_ism_true_nrg = 2.01e12 * (
            (0.5 * (1 + z)) ** -(5 / 7) *   # redshift scaling
            (nrg / 1e52) ** -(4 / 7) *      # energy scaling
            n0 ** -(13 / 14) *              # density scaling
            t ** -(2/7)                     # time scaling
        )

        # Assert equal within 1%
        self.assertAlmostEqual(nu_c_ism / nu_c_ism_true_nrg, 1.0, delta=0.01)

    def test_cooling_frequency_scaling_radiative_wind(self):
        """
            Tests that changing the values scales the cooling frequency
            in the radiative regime for a wind medium properly.
        """
        # Vary the following values
        z = 2.1
        nrg = 2.5e52
        n0 = 4.2e35
        t = 3.41

        # Evaluate cooling frequency for WIND
        nu_c_win = cooling_frequency(nrg, n0, 2.0, eps_b=0.1, z=z, t_obs=t, adiabatic=False)

        # Value taken fro VHD09
        nu_c_win_true_nrg = 9.02e10 * (
            (0.5 * (1 + z)) ** -(4/3) * # redshift scaling
            (nrg / 1e52) ** (2/3) *     # energy scaling
            (n0 / 3.0e35) ** -(13/6) *  # density scaling
            t ** (1/3)                  # time scaling
        )

        # Assert equal within 1%
        self.assertAlmostEqual(nu_c_win / nu_c_win_true_nrg, 1.0, delta=0.01)

    def test_cooling_frequency_adiabatic(self):
        """ Tests that the adiabatic cooling frequency model returns the correct values. """
        z = 2.0  # Vary z if you'd like

        # Evaluate cooling frequency for ISM and WIND
        nu_c_ism = cooling_frequency(1.0e52, 1.0000, 0.0, eps_b=0.1, z=z, t_obs=1.0)
        nu_c_win = cooling_frequency(1.0e52, 3.0e35, 2.0, eps_b=0.1, z=z, t_obs=1.0)

        # Values taken fro VHD09
        nu_c_ism_true = 5.98e13 * (0.5 * (1 + z)) ** -0.5
        nu_c_win_true = 9.97e11 * (0.5 * (1 + z)) ** -1.5

        # Assert equal within 1%
        self.assertAlmostEqual(nu_c_ism / nu_c_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_c_win / nu_c_win_true, 1.0, delta=0.01)

    def test_cooling_frequency_radiative(self):
        """ Tests that the radiative cooling frequency model returns the correct values. """
        z = 2.5  # Vary z if you'd like

        # Evaluate cooling frequency for ISM and WIND
        nu_c_ism = cooling_frequency(1.0e52, 1.0000, 0.0, eps_b=0.1, z=z, t_obs=1.0, adiabatic=False)
        nu_c_win = cooling_frequency(1.0e52, 3.0e35, 2.0, eps_b=0.1, z=z, t_obs=1.0, adiabatic=False)

        # Values taken fro VHD09
        nu_c_ism_true = 2.01e12 * (0.5 * (1 + z)) ** -(5.0 / 7.0)
        nu_c_win_true = 9.02e10 * (0.5 * (1 + z)) ** -(4.0 / 3.0)

        # Assert equal within 1%
        self.assertAlmostEqual(nu_c_ism / nu_c_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_c_win / nu_c_win_true, 1.0, delta=0.01)

    def test_synchrotron_frequency_adiabatic(self):
        """ Tests that the adiabatic synchrotron frequency model returns the correct values. """
        z = 3.0  # Vary z if you'd like
        p, eps_b, eps_e, hmf = 2.2, 0.1, 0.1, 0.7  # Do not vary these!

        # Evaluate cooling frequency for ISM and WIND
        nu_m_ism = synchrotron_frequency(1.0e52, 1.0000, 0.0, p, eps_b, eps_e, z, hmf, t_obs=1.0)
        nu_m_win = synchrotron_frequency(1.0e52, 3.0e35, 2.0, p, eps_b, eps_e, z, hmf, t_obs=1.0)

        # Values taken fro VHD09
        nu_m_ism_true = 8.98e11 * (0.5 * (1 + z)) ** 0.5
        nu_m_win_true = 1.85e12 * (0.5 * (1 + z)) ** 0.5

        # Assert equal within 1%
        self.assertAlmostEqual(nu_m_ism / nu_m_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_m_win / nu_m_win_true, 1.0, delta=0.01)

    def test_synchrotron_frequency_radiative(self):
        """ Tests that the radiative synchrotron frequency model returns the correct values. """
        z = 3.5  # Vary z if you'd like
        p, eps_b, eps_e, hmf = 2.2, 0.1, 0.1, 0.7  # Do not vary these!

        # Evaluate cooling frequency for ISM and WIND
        nu_m_ism = synchrotron_frequency(1.0e52, 1.0000, 0.0, p, eps_b, eps_e, z, hmf, t_obs=1.0, adiabatic=False)
        nu_m_win = synchrotron_frequency(1.0e52, 3.0e35, 2.0, p, eps_b, eps_e, z, hmf, t_obs=1.0, adiabatic=False)

        # Values taken fro VHD09
        nu_m_ism_true = 2.94e12 * (0.5 * (1 + z)) ** (5.0 / 7.0)
        nu_m_win_true = 6.27e12 * (0.5 * (1 + z)) ** (2.0 / 3.0)

        # Assert equal within 1%
        self.assertAlmostEqual(nu_m_ism / nu_m_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_m_win / nu_m_win_true, 1.0, delta=0.01)

    def test_absorption_frequency_amc(self):
        """ Tests that the adiabatic absorption frequency model returns the correct values. """
        z = 4.0  # Vary z if you'd like
        p, eps_b, eps_e, hmf = 2.2, 0.1, 0.1, 0.7  # Do not vary these!

        # Evaluate cooling frequency for ISM and WIND
        nu_a_ism = nu_a_amc_ad(1.0e52, 1.0000, 0.0, p, eps_b, eps_e, z, hmf, t_obs=1.0)
        nu_a_win = nu_a_amc_ad(1.0e52, 3.0e35, 2.0, p, eps_b, eps_e, z, hmf, t_obs=1.0)

        # Values taken from VHD09
        nu_a_ism_true = 7.75e10 * (0.5 * (1 + z)) ** -1.0
        nu_a_win_true = 5.16e11 * (0.5 * (1 + z)) ** -0.4

        # Assert equal within 1%
        self.assertAlmostEqual(nu_a_ism / nu_a_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_a_win / nu_a_win_true, 1.0, delta=0.01)

    def test_absorption_frequency_mac(self):
        """ Tests that the adiabatic absorption frequency model returns the correct values. """
        z = 4.5  # Vary z if you'd like
        p, eps_b, eps_e, hmf = 2.2, 0.1, 0.1, 0.7  # Do not vary these!

        # Evaluate cooling frequency for ISM and WIND
        nu_a_ism = nu_a_mac_ad(1.0e52, 1.0000, 0.0, p, eps_b, eps_e, z, hmf, t_obs=1.0)
        nu_a_win = nu_a_mac_ad(1.0e52, 3.0e35, 2.0, p, eps_b, eps_e, z, hmf, t_obs=1.0)

        # Values taken from VHD09
        nu_a_ism_true = 1.13e11 * (0.5 * (1 + z)) ** -0.31
        nu_a_win_true = 4.38e11 * (0.5 * (1 + z)) ** 0.016

        # Assert equal within 1%
        self.assertAlmostEqual(nu_a_ism / nu_a_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_a_win / nu_a_win_true, 1.0, delta=0.01)

    def test_absorption_frequency_acm_adiabatic(self):
        """ Tests that the adiabatic absorption frequency model returns the correct values. """
        z = 4.5  # Vary z if you'd like
        eps_b, hmf = 0.1, 0.7  # Do not vary these!

        # Evaluate cooling frequency for ISM and WIND
        nu_a_ism = nu_a_acm_ad(1.0e52, 1.0000, 0.0, eps_b, z, hmf, t_obs=1.0)
        nu_a_win = nu_a_acm_ad(1.0e52, 3.0e35, 2.0, eps_b, z, hmf, t_obs=1.0)

        # Values taken from VHD09
        nu_a_ism_true = 1.25e09 * (0.5 * (1 + z)) ** -0.5
        nu_a_win_true = 9.23e10 * (0.5 * (1 + z)) ** 0.60

        # Assert equal within 1%
        self.assertAlmostEqual(nu_a_ism / nu_a_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_a_win / nu_a_win_true, 1.0, delta=0.01)

    def test_absorption_frequency_acm_radiative(self):
        """ Tests that the radiative absorption frequency model returns the correct values. """
        z = 4.5  # Vary z if you'd like
        eps_b, hmf = 0.1, 0.7  # Do not vary these!

        # Evaluate cooling frequency for ISM and WIND
        nu_a_ism = nu_a_acm_rad(1.0e52, 1.0000, 0.0, eps_b, z, hmf, t_obs=1.0)
        nu_a_win = nu_a_acm_rad(1.0e52, 3.0e35, 2.0, eps_b, z, hmf, t_obs=1.0)

        # Values taken from VHD09
        nu_a_ism_true = 6.56e09 * (0.5 * (1 + z)) ** -0.2
        nu_a_win_true = 3.47e10 * (0.5 * (1 + z)) ** (7.0 / 15.0)

        # Assert equal within 1%
        self.assertAlmostEqual(nu_a_ism / nu_a_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_a_win / nu_a_win_true, 1.0, delta=0.01)


if __name__ == '__main__':
    unittest.main()
