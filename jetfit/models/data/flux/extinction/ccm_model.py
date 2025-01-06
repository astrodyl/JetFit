import scipy.constants

from jetfit.core.defns.mixins import BoundedMixin


class CCMExtinction:
    """
    The Cardelli, Clayton, & Mathis (CCM) extinction model.

    Cardelli, Clayton, & Mathis (CCM) is an average extinction law that is
    derived over the wavelength range 0.125 micrometer to 3.5 micrometer.
    This wavelength range is applicable for optical/NIR and infrared data.

    For UV wavelengths, the Fitzpatrick and Massa model should be used to
    model the extinction (F&M, 1986, 1988, 1990).

    Attributes
    ----------
    valid_micro_m : BoundedMixin
        The limits that the model is valid in micrometers.

    valid_freq_hz : BoundedMixin
        The limits that the model is valid in Hz.

    c : float
        The speed of light in micrometers per second. The CCM model requires
        the wavelength to be measured in micrometers, so these units are
        convenient for transforming between frequency and wavelength.

    r_v : float, optional, default=3.1
        The ratio of the total extinction to the selective extinction. The
        default is chosen to be the average value for the Milky Way.

    ebv_milky_way : float
        The milky way E(B - V) strength.

    ebv_source_frame : float
        The source frame E(B - V) strength.

    References
    ----------
    .. [1] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. (1989)
        https://articles.adsabs.harvard.edu//full/1989ApJ...345..245C/0000249.000.html
    """
    valid_micro_m = BoundedMixin(lower=0.3, upper=3.3)
    valid_freq_hz = BoundedMixin(lower=8.99e13, upper=9.89e14)
    c = scipy.constants.speed_of_light * 10 ** 6

    def __init__(self, ebv_milky_way: float, ebv_source_frame: float, r_v: float = 3.1):
        self.ebv_milky_way = ebv_milky_way
        self.ebv_source_frame = ebv_source_frame
        self.r_v = r_v

    def evaluate(self, frequency: float, linear: bool = True):
        """
        Calculates the extinction due to dust.

        Parameters
        ----------
        frequency : float
            The band frequency measured in Hz.

        linear : bool, optional, default=True

        Returns
        -------
        float
            The extinction correction factor.
        """
        if not self.valid_freq_hz.encompasses(frequency):
            raise ValueError(
                f'CCM model is not valid for {frequency}. '
                f'The valid range is {self.valid_freq_hz.bounds()}.'
            )

        sf_ext = -0.4 * self.r_v * self.ebv_source_frame * self.curve(self.c / frequency)
        mw_ext = -0.4 * self.r_v * self.ebv_milky_way * self.curve(self.c / frequency)

        return (10 ** sf_ext) * (10 ** mw_ext) if linear else sf_ext + mw_ext

    def curve(self, wavelength: float) -> float:
        """
        Calculates the ratio of the extinction in a given band to that in the
        V band.

        Parameters
        ----------
        wavelength : float
            The wavelength of the band in micrometers.

        Returns
        -------
        float
            The extinction curve.
        """
        x = 1 / wavelength
        y = x - 1.82

        # Infrared
        if 0.3 <= x <= 1.1:
            a = 0.574 * x ** 1.61
            b = -0.527 * x ** 1.61

        # Optical/NIR
        elif 1.1 < x <= 3.3:
            a = (1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 +
                 0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.7753 * y ** 6 +
                 0.32999 * y ** 7)

            b = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 -
                 5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.3026 * y ** 6 -
                 2.09002 * y ** 7)

        else:
            raise ValueError(f'CCM Extinction is not valid for {wavelength}.')

        return a + b / self.r_v
