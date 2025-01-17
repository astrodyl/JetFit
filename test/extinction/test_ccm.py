import unittest

import numpy as np
import scipy.constants as u

from dust_extinction.parameter_averages import CCM89
from matplotlib import pyplot as plt

from jetfit.models.data.flux.extinction.ccm_model import CCMExtinction


class MyTestCase(unittest.TestCase):
    """
    Tests the internal CCM dust extinction implementation.

    Attributes
    ----------
    ebv_mw : float
        The Milky Way EBV value.

    ebv_sf : float
        The Source Frame EBV value.

    R_v : float
        R(V) = A(V)/E(B-V) = total-to-selective extinction
    """
    @classmethod
    def setUp(cls):
        cls.ebv_mw = 0.1
        cls.ebv_sf = 0.15
        cls.R_v = 3.1

    def test_curve_values(self):
        """
        Tests the CCM curve calculations.

        Uses the ``dust_extinction.CCM89`` model as a means of an independent
        calculation. ``CCM89(R_v)(x)`` is equivalent to
        ``CCMExtinction(_,R_v).curve(1/x)``.

        The values for x are chosen such that every regime of CCM is
        calculated.
        """
        c = (u.speed_of_light / u.micron)

        xs = (
            0.5,  # infrared
            2.0,  # optical/NIR
            5.0,  # ultraviolet (a)
            7.0,  # ultraviolet (b)
            9.0,  # far ultraviolet
        )

        for x in xs:
            curve_a = CCM89(Rv=self.R_v)(x)
            curve_b = CCMExtinction(self.R_v).curve(1 / x)
            curve_c = CCMExtinction(self.R_v).evaluate(x * c)
            self.assertAlmostEqual(curve_a, curve_b, 6)
            self.assertAlmostEqual(curve_b, curve_c, 6)

    def test_setting_ebv_to_zero(self):
        """
        Tests that an EBV value of zero does not contribute to the extinction.

        Since the extinction contributes a multiplicative factor in linear
        space, we should expect a return of `1.0` for an ebv of `0.0`.
        """
        frequency = 9e13

        self.assertEqual(CCMExtinction().evaluate(frequency, ebv=0, linear=True), 1.0)
        self.assertEqual(CCMExtinction().evaluate(frequency, ebv=0, linear=False), 0.0)

    def test_out_of_range(self):
        """
        Tests that a `ValueError` is raised when a frequency that is not
        applicable to the CCM model is passed.
        """
        valid = CCMExtinction().valid_x_range

        with self.assertRaises(ValueError):
            _ = CCMExtinction(self.R_v).curve(1.1 / valid.lower)

        with self.assertRaises(ValueError):
            _ = CCMExtinction(self.R_v).evaluate(0.9 / valid.upper)

    @unittest.skip("Test=Plot CCM Curve, Reason=For visual inspection only")
    def test_plot_curves(self):
        """
        Plots the extinction curves for visual inspection.
        """
        _, ax = plt.subplots()

        xs = np.arange(0.5, 10.0, 0.1)
        r_vs = (2.0, 3.0, 4.0, 5.0, 6.0)

        # Plot my CCMExtinction model
        for r_v in r_vs:
            ccm = CCMExtinction(r_v)
            curve = [ccm.curve(1 / x) for x in xs]
            ax.plot(xs, curve, label='R(V) = ' + str(r_v))

        # Set labels
        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # Create 2nd x-axis with lambda values
        axis_xs = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]

        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    unittest.main()
