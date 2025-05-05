import unittest

import numpy as np
import astropy.units as u
import astropy.constants as const
from matplotlib import pyplot as plt

from jetfit.core.utils.math_utils import chi_squared
from jetfit.models2.basemodels import PeakFluxModel, SynchrotronFrequencyModel, CoolingFrequencyModel, \
    AbsorptionFrequencyModel, SpectralFluxModel
from jetfit.models2.fireball import FireballModel, StratifiedFireballModel


# Constants in cgs units
m_p = const.m_p.cgs  # noqa
m_e = const.m_e.cgs  # noqa
q_e = u.Quantity(4.8032e-10 * u.g**0.5 * u.cm**1.5 / u.s)
c = const.c.cgs  # noqa


class TestCharacteristicModels(unittest.TestCase):
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

    def test_plot(self):
        """"""

        model = StratifiedFireballModel(
            E=4.0, p=2.5, eps_b=0.001, eps_e=0.1, X=0.7,
            k1=2.0, k2=0.0, nt=1.0, rt=1e17, sn=3.0,
            dL=2.0, z=0.0
        )

        ts = np.geomspace(0.0012, 10, 500)  # 100s to 2 days
        radii = model.radii(ts)

        # Plot n effective
        for s in np.linspace(-3.0, 3.0, 5):
            model.sn = s
            n_eff, _ = model.smooth(ts)

            plt.loglog(radii, n_eff, label=f's = {s}')
            plt.axvline(model.rt, linestyle='--', color='black')
            plt.xlabel(r'Radius [cm]')
            plt.ylabel(r'$n_{eff}$')
            plt.legend(loc='best')
        plt.show()

        # Plot k effective
        for s in np.linspace(-3.0, 3.0, 5):
            model.sn = s
            _, k_eff = model.smooth(ts)

            plt.plot(radii, k_eff, label=f's = {s}')
            plt.axvline(model.rt, linestyle='--', color='black')
            plt.xlabel(r'Radius [cm]')
            plt.ylabel(r'$k_{eff}$')
            plt.legend(loc='best')
            plt.xscale('log')
        plt.show()

    def test_nu_a_amc(self):
        """"""
        # ISM
        af_ism = AbsorptionFrequencyModel(
            E=1.0, rho0=1.0, eps_e=0.1, eps_b=0.1, k=0.0, z=0.0, X=0.7, p=2.2
        ).evaluate_amc(1.0) / (0.5 ** -1)

        # Wind
        af_wind = AbsorptionFrequencyModel(
            E=1.0, rho0=5e11 / m_p.value / 1e34, eps_e=0.1, eps_b=0.1, k=2.0, z=0.0, X=0.7, p=2.2
        ).evaluate_amc(1.0) / (0.5 ** (-2 / 5))

        af_ism_true = 7.75e10
        af_wind_true = 5.16e11

        # Assert equal within 1%
        self.assertAlmostEqual(af_ism / af_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(af_wind / af_wind_true, 1.0, delta=0.01)

    def test_nu_a_mac(self):
        """"""
        # ISM
        af_ism = AbsorptionFrequencyModel(
            E=1.0, rho0=1.0, eps_e=0.1, eps_b=0.1, k=0.0, z=0.0, X=0.7, p=2.2
        ).evaluate_mac(1.0) / (0.5 ** -0.31)

        # Wind
        af_wind = AbsorptionFrequencyModel(
            E=1.0, rho0=5e11 / m_p.value / 1e34, eps_e=0.1, eps_b=0.1, k=2.0, z=0.0, X=0.7, p=2.2
        ).evaluate_mac(1.0) / (0.5 ** 0.016)

        af_ism_true = 1.13e11
        af_wind_true = 4.38e11

        # Assert equal within 1%
        self.assertAlmostEqual(af_ism / af_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(af_wind / af_wind_true, 1.0, delta=0.01)

    def test_linsolve_cl(self):
        """"""

        a = np.array([
            [0.5,   1.0,    0.5,    0.0],   # log(F_nu_max)
            [-0.4,  1.2,    0.2,    -1.0],  # log(nu_a)
            [0.5,   0.0,    0.5,    2.0],   # log(nu_m)
            [0.5,   -2.0,   -1.5,   0.0]    # log(nu_c)
        ])

        f_nu_max_mjy = 1.0
        nu_a_hz = 1e9
        nu_m_hz = 1e12
        nu_c_hz = 1e14

        b = np.array([
            np.log10(f_nu_max_mjy)  - np.log10(20),
            np.log10(nu_a_hz)       - np.log10(1e11),
            np.log10(nu_m_hz)       - np.log10(5e12),
            np.log10(nu_c_hz)       - np.log10(2e12)
        ])

        x = np.linalg.solve(a, b)
        y = 10 ** x

    def test_linsolve_vdh(self):
        """"""

        # k-values to evaluate
        ks = np.linspace(-3.0, 3.0, 100)
        # ks = np.linspace(0.0, 2.0, 100)

        # Define necessary params for evaluating models
        t = 1.0     # observing time in days
        p = 2.2     # electron energy index
        hmf = 0.7   # hydrogen mass fraction
                    #   ~1.0 for ISM
                    #   ~0.0 for wind
        z = 0.0     # redshift
        d = 1.0     # luminosity distance
        n = 1.0     # 5e11 / m_p.value / 1e34

        # Define reference characteristic values
        f_p_mjy = 1.0   # peak flux [mJy]
        nu_a_hz = 1e9   # absorption frequency [Hz]
        nu_m_hz = 1e12  # synchrotron frequency [Hz]
        nu_c_hz = 1e14  # cooling frequency [Hz]

        # Define the characteristic models with all physical parameters
        # of interest set to unity. When evaluating the model, this will
        # return only the pre-factor that we need to solve the system of
        # equations.
        peak_flux_model = PeakFluxModel(
            E=1.0, rho0=n, eps_b=1.0, dL=d, z=z, k=0.0, X=hmf)
        nu_m_model = SynchrotronFrequencyModel(
            E=1.0, eps_e=1.0, eps_b=1.0, k=0.0, z=z, X=hmf, p=p)
        nu_c_model = CoolingFrequencyModel(
            E=1.0, rho0=n, eps_b=1.0, k=0.0, z=z)
        nu_a_model = AbsorptionFrequencyModel(
            E=1.0, rho0=n, eps_e=1.0, eps_b=1.0, k=0.0, z=z, X=hmf, p=p)

        # For each value of k, construct and solve a system of equations for:
        # (1) energy (normalized to 1/52),
        # (2) the number density (normalized to m_p and 1e17cm),
        # (3) the electric field energy fraction, and
        # (4) the magnetic field energy fraction.

        sols = []
        for k in ks:

            # Update the models with the new k value
            peak_flux_model.k = nu_m_model.k = nu_c_model.k = nu_a_model.k = k

            # System of equations for slow cooling (nu_a < nu_m < nu_c)
            a_slow = np.array([
                # E                             n0                  eps_e       eps_b
                [0.5 * (8 - 3*k) / (4 - k),     2 / (4 - k),        0.0,        0.5 ],  # log(F_nu_max)
                [-0.5 * (4 - 3*k) / (4 - k),   -4 / (4 - k),        0.0,       -1.5 ],  # log(nu_c)
                [0.5,                           0.0,                2.0,        0.5 ],  # log(nu_m)
                [0.8 * (1 - k) / (4 - k),       2.4 / (4 - k),     -1.0,        0.2 ]   # log(nu_a_slow)
            ])

            # log(characteristics) minus log(pre-factors)
            b_slow = np.array([
                np.log10(f_p_mjy) - np.log10(peak_flux_model(t=t)),
                np.log10(nu_c_hz) - np.log10(nu_c_model(t=t)),
                np.log10(nu_m_hz) - np.log10(nu_m_model(t=t)),
                np.log10(nu_a_hz) - np.log10(nu_a_model(t=t, order='amc'))
            ])

            sols.append(10 ** np.linalg.solve(a_slow, b_slow))

        sols = np.asarray(sols)

        titles = (
            r'Energy ($E_{52}$)',
            r'Density ($n_{0}$)',
            r'Electric Field Fraction ($\epsilon_{E}$)',
            r'Magnetic Field Fraction ($\epsilon_{B}$)'
        )
        y_labels = (r'$E_{52}$', r'$n_{0}$', r'$\epsilon_{E}$', r'$\epsilon_{B}$')
        colors = ('red', 'green', 'blue', 'purple')

        for i, title in enumerate(titles):

            # Add reference values to labels
            plt.plot([], [], alpha=0, label=r'$F_{peak}$ = ' + f'1 mJy')
            plt.plot([], [], alpha=0, label=r'$\nu_{c}$ = ' + r'$10^{14}$ Hz')
            plt.plot([], [], alpha=0, label=r'$\nu_{m}$ = ' + r'$10^{12}$ Hz')
            plt.plot([], [], alpha=0, label=r'$\nu_{a}$ = ' + r'$10^{9}$ Hz')

            # Plot the data
            plt.plot(ks, sols[:, i], linewidth=0.75, color=colors[i])

            # Configure the plot
            plt.title(title)
            plt.xlabel('k')
            plt.ylabel(y_labels[i])
            plt.grid(alpha=0.4)
            plt.legend()
            plt.show()

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
        nu_a = model.nu_a(1.0, nu_m)

        # True values
        f_peak_true = 21.3 * 0.5
        nu_m_true = 8.98e11 * (0.5**0.5)
        nu_c_true = 5.98e13 * (0.5**-0.5)
        nu_a_slow_true = 7.75e10 * (0.5**-1)

        # Assert equal within 1%
        self.assertAlmostEqual(f_peak / f_peak_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_m / nu_m_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_c / nu_c_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_a / nu_a_slow_true, 1.0, delta=0.01)

    def test_vdh_wind(self):
        """
        Test that the general k-model reduces to the wind
        (k=2) case when k is set to 2.
        """

        # Correct for the different normalizations. We use a
        # number density referenced to 1e17cm.
        n0 = 5e11 / 1.67e-24 / 1e34

        model = FireballModel(
            E=1.0, rho0=n0, p=2.2, k=2.0, z=0.0, dL=1.0,
            eps_e=0.1, eps_b=0.1, X=0.7
        )

        # Modeled values
        f_peak = model.f_peak(1.0)
        nu_m = model.nu_m(1.0)
        nu_c = model.nu_c(1.0)
        nu_a = model.nu_a(1.0, nu_m)

        # True values
        f_peak_true = 60.8 * (0.5**1.5)
        nu_m_true = 1.85e12 * (0.5 ** 0.5)
        nu_c_true = 9.97e11 * (0.5 ** -1.5)
        nu_a_slow_true = 5.16e11 * (0.5 ** -0.4)

        # Assert equal within 1%
        self.assertAlmostEqual(f_peak / f_peak_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_m / nu_m_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_c / nu_c_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_a / nu_a_slow_true, 1.0, delta=0.01)

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

        # Derivation of the peak flux differs by 8pi/9
        f_peak = model.f_peak(1.0) * 8.0 * np.pi / 9

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

    def test_spectral_flux_model(self):
        """"""
        nu_m = np.array([50 * i * 1e12 for i in range(20)])
        nu_c = np.array([2e14 for _ in range(20)])
        nu_a = np.array([1e9 for _ in range(20)])
        f = np.array([1 for _ in range(20)])

        sfm = SpectralFluxModel(nu_m, nu_c, nu_a, f, 2.2, 2.0)
        em1 = sfm.evaluate(np.array([1e14 for _ in range(20)]))
        em2 = sfm.evaluate2(np.array([1e14 for _ in range(20)]))

        sfm = SpectralFluxModel(1e12, 2e14, 1e15, 1, 2.2, 2.0)
        es1 = sfm.evaluate(np.array([1e14 for _ in range(20)]))
        es2 = sfm.evaluate2(np.array([1e14 for _ in range(20)]))

        sfm = SpectralFluxModel(1e12, 2e14, 1e9, 1, 2.2, 2.0)
        ea1 = sfm.evaluate(1e14)
        ea2 = sfm.evaluate2(1e14)

        sfm = SpectralFluxModel(nu_m, nu_c, nu_a, f, 2.2, 2.0)
        ez1 = sfm.evaluate(1e14)
        ez2 = sfm.evaluate2(1e14)
        print()

if __name__ == '__main__':
    unittest.main()
