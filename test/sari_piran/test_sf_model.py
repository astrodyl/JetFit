import unittest

from jetfit.core.values.flux import SpectralFluxValue
from jetfit.models.data.flux.spectral.spsf_model import SpectralFluxModel


class TestSP98SModel(unittest.TestCase):
    def test_return_type(self) -> None:
        """
        Tests that the class returns the correct type.
        """
        model = SpectralFluxModel(
            pf=1.0e-7,
            cf=1.0e15,
            sf=1.0e18,
            p=2.5,
            frequency=1.0e14,
            units='mjy'
        )

        # Return flux as a float value
        self.assertIsInstance(
            model.evaluate(obj=False),
            float,
            msg="`evaluate` did not return a float."
        )

        # Return flux as an object
        flux_obj = model.evaluate(obj=True)

        self.assertIsInstance(flux_obj, SpectralFluxValue,
            msg="`evaluate` did not return a `SpectralFluxValue`."
        )
        self.assertIsInstance(flux_obj.value, float)

    def test_get_fast_segment(self) -> None:
        """
        Tests that the class is able to determine the segment in the fast
        cooling regime.
        """
        for i, f in enumerate([1.0e14, 1.0e16, 1.0e19]):
            model = SpectralFluxModel(
                pf=1.0e-7,
                cf=1.0e15,
                sf=1.0e18,
                p=2.5,
                frequency=f,
                units='mjy'
            )
            self.assertEqual(model.segment, ['B', 'C', 'D'][i])

    def test_get_slow_segment(self) -> None:
        """
        Tests that the class is able to determine the segment in the slow
        cooling regime.
        """
        frequencies = (1.0e14, 1.0e16, 1.0e19)

        # Test for different models
        for i, f in enumerate(frequencies):
            model = SpectralFluxModel(
                pf=1.0e-7, cf=1.0e18, sf=1.0e15,
                p=2.5, frequency=f, units='mjy'
            )
            self.assertEqual(model.segment, ['F', 'G', 'H'][i])

        # Test for same model with new frequency
        model = SpectralFluxModel(
            pf=1.0e-7, cf=1.0e18, sf=1.0e15,
            p=2.5, frequency=99, units='mjy'
        )

        for i, f in enumerate(frequencies):
            model.frequency = f
            self.assertEqual(model.segment, ['F', 'G', 'H'][i])

    def test_get_regime(self) -> None:
        """
        Tests that the class is able to determine the correct regime.
        """
        p, f = 2.5, 2.42e+17
        pf, cf, sf = 1.0e-7, 1.0e18, 1.0e15

        model = SpectralFluxModel(pf, cf, sf, p, f, 'mjy')
        self.assertEqual(model.regime, 'slow')

        model.cf, model.sf = model.sf, model.cf
        self.assertEqual(model.regime, 'fast')


class TestUnExtinguishedFlux(unittest.TestCase):
    """

    """
    def test_segment_b_flux(self) -> None:
        """
        Tests that the flux calculation for the fast cooling regime on
        segment B is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Fast Cooling, Segment B is defined as: f < cf < sf.
        """
        model = SpectralFluxModel(
            1.0e-7,
            1.1e15,
            1.2e17,
            2.5,
            2.42e+14,
            'mjy'
        )

        self.assertEqual(model.get_segment_b_flux(), 6.036810736797686e-08)
        self.assertEqual(model.get_segment_b_flux(), model.evaluate())

    def test_segment_c_flux(self) -> None:
        """
        Tests that the flux calculation for the fast cooling regime on
        segment C is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Fast Cooling, Segment C is defined as: cf < f < sf.
        """
        model = SpectralFluxModel(
            1.0e-7,
            1.1e15,
            1.2e17,
            2.5,
            2.42e+16,
            'mjy'
        )

        self.assertEqual(model.get_segment_c_flux(), 2.1320071635561042e-08)
        self.assertEqual(model.get_segment_c_flux(), model.evaluate())

    def test_segment_d_flux(self) -> None:
        """
        Tests that the flux calculation for the fast cooling regime on
        segment D is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Fast Cooling, Segment D is defined as: cf < sf < f.
        """
        model = SpectralFluxModel(
            1.0e-7,
            1.1e15,
            1.2e17,
            2.5,
            2.42e+18,
            'mjy'
        )

        self.assertEqual(model.get_segment_d_flux(), 2.240335546140111e-10)
        self.assertEqual(model.get_segment_d_flux(), model.evaluate())

    # </editor-fold>

    # <editor-fold desc="Slow Cooling Tests">
    def test_segment_f_flux(self) -> None:
        """
        Tests that the flux calculation for the slow cooling regime on
        segment F is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Slow Cooling, Segment F is defined as: f < sf < cf.
        """
        model = SpectralFluxModel(
            1.0e-7,
            1.0e18,
            1.0e15,
            2.5,
            2.42e+14,
            'mjy'
        )

        self.assertEqual(model.get_segment_f_flux(), 6.231679684369752e-08)
        self.assertEqual(model.get_segment_f_flux(), model.evaluate())

    def test_segment_g_flux(self) -> None:
        """
        Tests that the flux calculation for the slow cooling regime on
        segment G is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Slow Cooling, Segment G is defined as: sf < f < cf.
        """
        model = SpectralFluxModel(
            1.0e-7,
            1.0e18,
            1.0e15,
            2.5,
            2.42e+17,
            'mjy'
        )

        self.assertEqual(model.get_segment_g_flux(), 1.6298156192086981e-09)
        self.assertEqual(model.get_segment_g_flux(), model.evaluate())

    def test_segment_h_flux(self):
        """
        Tests that the flux calculation for the slow cooling regime on
        segment H is correct. Additionally, tests that the ``evaluate``
        method returns the correct flux.

        Slow Cooling, Segment H is defined as: sf < cf < f.
        """
        model = SpectralFluxModel(
            1.0e-7,
            1.0e18,
            1.0e15,
            2.5,
            2.42e+19,
            'mjy'
        )

        self.assertEqual(model.get_segment_h_flux(), 1.047685160387475e-11)
        self.assertEqual(model.get_segment_h_flux(), model.evaluate())
    # </editor-fold>


if __name__ == '__main__':
    unittest.main()
