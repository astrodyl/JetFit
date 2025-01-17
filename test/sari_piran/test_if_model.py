import unittest

from jetfit.models.data.flux.integrated.spif_model import IntegratedFluxModel


class TestSP98IModel(unittest.TestCase):
    def test_get_fast_single_segment(self) -> None:
        """
        Tests that the class is able to determine the segment in the fast
        cooling regime for a frequency range that spans only a single
        segment.
        """
        bounds = (
            (1.0e13, 1.0e14),  # B
            (2.0e15, 2.0e16),  # C
            (2.0e18, 2.0e19),  # D
        )

        for i, b in enumerate(bounds):
            model = IntegratedFluxModel(
                pf=1.0e-7,
                cf=1.0e15,
                sf=1.0e18,
                p=2.5,
                lower=b[0],
                upper=b[1]
            )
            self.assertEqual(model.segment, ['B', 'C', 'D'][i])

    def test_get_fast_joined_segment(self) -> None:
        """
        Tests that the class is able to determine the segment in the fast
        cooling regime for a frequency range that spans multiple segments.
        """
        bounds = (
            (1.0e13, 1.0e16),  # BC
            (2.0e13, 2.0e19),  # BCD
            (2.0e15, 2.0e19),  # CD
        )

        for i, b in enumerate(bounds):
            model = IntegratedFluxModel(
                pf=1.0e-7,
                cf=1.0e15,
                sf=1.0e18,
                p=2.5,
                lower=b[0],
                upper=b[1]
            )
            self.assertEqual(model.segment, ['BC', 'BCD', 'CD'][i])

    def test_get_slow_single_segment(self) -> None:
        """
        Tests that the class is able to determine the segment in the slow
        cooling regime for a frequency range that spans only a single
        segment.
        """
        bounds = (
            (1.0e13, 1.0e14),  # F
            (2.0e15, 2.0e16),  # G
            (2.0e18, 2.0e19),  # H
        )

        for i, b in enumerate(bounds):
            model = IntegratedFluxModel(
                pf=1.0e-7,
                cf=1.0e18,
                sf=1.0e15,
                p=2.5,
                lower=b[0],
                upper=b[1]
            )
            self.assertEqual(model.segment, ['F', 'G', 'H'][i])

    def test_get_slow_joined_segment(self) -> None:
        """
        Tests that the class is able to determine the segment in the slow
        cooling regime for a frequency range that spans multiple segments.
        """
        bounds = (
            (1.0e13, 1.0e16),  # FG
            (2.0e13, 2.0e19),  # FGH
            (2.0e15, 2.0e19),  # GH
        )

        for i, b in enumerate(bounds):
            model = IntegratedFluxModel(
                pf=1.0e-7,
                cf=1.0e18,
                sf=1.0e15,
                p=2.5,
                lower=b[0],
                upper=b[1]
            )
            self.assertEqual(model.segment, ['FG', 'FGH', 'GH'][i])


class TestSP98IFastCoolingFlux(unittest.TestCase):
    def test_segment_b_flux(self) -> None:
        """
        Tests that the flux calculation for the fast cooling regime on
        segment B is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Fast Cooling, Segment B is defined as: f_hi < cf < sf.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e15,
            1.2e17,
            2.5,
            1.0e13,
            1.0e14
        )

        self.assertEqual(model.get_segment_b_flux(), 3215802.545987821)
        self.assertEqual(model.get_segment_b_flux(), model.evaluate())

    def test_segment_c_flux(self) -> None:
        """
        Tests that the flux calculation for the fast cooling regime on
        segment C is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Fast Cooling, Segment C is defined as: cf < f_lo, f_hi < sf.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e15,
            1.2e17,
            2.5,
            2.0e15,
            1.0e16
        )

        self.assertEqual(model.get_segment_c_flux(), 366677018.5872534)
        self.assertEqual(model.get_segment_c_flux(), model.evaluate())

    def test_segment_d_flux(self) -> None:
        """
        Tests that the flux calculation for the fast cooling regime on
        segment D is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Fast Cooling, Segment D is defined as: cf < sf < f_lo.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e15,
            1.2e17,
            2.5,
            2.0e17,
            1.0e18
        )

        self.assertEqual(model.get_segment_d_flux(), 1339841320.432524)
        self.assertEqual(model.get_segment_d_flux(), model.evaluate())

    def test_segment_bc_flux(self) -> None:
        """
        Tests that the flux calculation for the fast cooling regime on
        segment BC is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Fast Cooling, Segment BC is defined as: f_lo < cf < f_hi < sf.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e15,
            1.2e17,
            2.5,
            1.0e13,
            1.0e16
        )

        self.assertEqual(model.get_segment_bc_flux(), 525668428.269398)
        self.assertEqual(model.get_segment_bc_flux(), model.evaluate())

    def test_segment_cd_flux(self) -> None:
        """
        Tests that the flux calculation for the fast cooling regime on
        segment BC is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Fast Cooling, Segment BC is defined as:  cf < f_lo < sf < f_hi.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e15,
            1.2e17,
            2.5,
            1.0e16,
            1.0e18
        )

        self.assertEqual(model.get_segment_cd_flux(), 3525305931.836649)
        self.assertEqual(model.get_segment_cd_flux(), model.evaluate())

    def test_segment_bcd_flux(self) -> None:
        """
        Tests that the flux calculation for the fast cooling regime on
        segment BCD is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Fast Cooling, Segment BCD is defined as: f_lo < cf < sf < f_hi.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e15,
            1.2e17,
            2.5,
            1.0e15,
            1.0e18
        )

        self.assertEqual(model.get_segment_bcd_flux(), 3978476191.9463687)
        self.assertEqual(model.get_segment_bcd_flux(), model.evaluate())


class TestSP98ISlowCoolingFlux(unittest.TestCase):
    def test_segment_f_flux(self) -> None:
        """
        Tests that the flux calculation for the slow cooling regime on
        segment F is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Slow Cooling, Segment F is defined as: f_hi < sf < cf.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e17,
            1.2e15,
            2.5,
            1.0e13,
            1.0e14
        )

        self.assertEqual(model.get_segment_f_flux(), 3123871.6928717806)
        self.assertEqual(model.get_segment_f_flux(), model.evaluate())

    def test_segment_g_flux(self) -> None:
        """
        Tests that the flux calculation for the slow cooling regime on
        segment G is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Slow Cooling, Segment G is defined as: sf < f_lo, f_hi < cf.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e17,
            1.2e15,
            2.5,
            2.0e15,
            1.0e16
        )

        self.assertEqual(model.get_segment_g_flux(), 270155941.6222191)
        self.assertEqual(model.get_segment_g_flux(), model.evaluate())

    def test_segment_h_flux(self) -> None:
        """
        Tests that the flux calculation for the slow cooling regime on
        segment H is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Slow Cooling, Segment H is defined as: sf < cf < f_lo.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e17,
            1.2e15,
            2.5,
            2.0e17,
            1.0e18
        )

        self.assertEqual(model.get_segment_h_flux(), 423695027.5774274)
        self.assertEqual(model.get_segment_h_flux(), model.evaluate())

    def test_segment_fg_flux(self) -> None:
        """
        Tests that the flux calculation for the slow cooling regime on
        segment FG is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Slow Cooling, Segment FG is defined as: f_lo < sf < f_hi < cf.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e17,
            1.2e15,
            2.5,
            1.0e13,
            1.0e16
        )

        self.assertEqual(model.get_segment_fg_flux(), 425389182.4767293)
        self.assertEqual(model.get_segment_fg_flux(), model.evaluate())

    def test_segment_gh_flux(self) -> None:
        """
        Tests that the flux calculation for the slow cooling regime on
        segment GH is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Slow Cooling, Segment GH is defined as:  sf < f_lo < cf < f_hi.
        """
        model = IntegratedFluxModel(
            1.0e-7,
            1.1e17,
            1.2e15,
            2.5,
            1.0e16,
            1.0e18
        )

        self.assertEqual(model.get_segment_gh_flux(), 1299574524.6320086)
        self.assertEqual(model.get_segment_gh_flux(), model.evaluate())

    def test_segment_fgh_flux(self) -> None:
        """
        Tests that the flux calculation for the slow cooling regime on
        segment FGH is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Slow Cooling, Segment FGH is defined as:  f_lo < sf < cf < f_hi.
        """
        p = 2.5
        pf, cf, sf = 1.0e-7, 1.1e17, 1.2e15
        lower, upper = 1.0e15, 1.0e18

        model = IntegratedFluxModel(pf, cf, sf, p, lower, upper)

        fgh_flux = (
            model.get_segment_f_flux(lower=lower, upper=sf) +
            model.get_segment_g_flux(lower=sf, upper=cf) +
            model.get_segment_h_flux(lower=cf, upper=upper)
        )

        self.assertEqual(model.get_segment_fgh_flux(), fgh_flux)
        self.assertEqual(model.get_segment_fgh_flux(), model.evaluate())
        self.assertEqual(model.get_segment_fgh_flux(), 1654538059.9925501)


if __name__ == '__main__':
    unittest.main()
