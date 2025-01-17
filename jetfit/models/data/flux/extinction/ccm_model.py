import scipy.constants as u

from jetfit.core.defns.mixins import BoundedMixin


class CCMExtinction:
    """
    The Cardelli, Clayton, & Mathis (CCM) extinction model.

    Attributes
    ----------
    valid_x_range : BoundedMixin
        The wavelengths [micrometers] that the model is applicable.

    r_v : float, optional, default=3.1
        The ratio of the total-to-selective extinction. The  default is
        chosen to be the average value for the Milky Way.

    References
    ----------
    .. [1] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. (1989)
        https://articles.adsabs.harvard.edu//full/1989ApJ...345..245C/0000249.000.html
    """
    valid_x_range = BoundedMixin(lower=0.3, upper=10.0)

    def __init__(self, r_v: float = 3.1):
        self.r_v = r_v

    def evaluate(self, frequency: float, z: float = 0.0, ebv: float = None, linear: bool = True):
        """
        Calculates the extinction factor.

        Parameters
        ----------
        frequency : float
            The band frequency measured in Hz.

        z : float, optional, default=0.0
            The redshift. If provided, assumes that `frequency` is in the
            observer frame. The redshift is passed to `curve` where the
            observed frequency will be transformed to the source wavelength.
            Defaults to ``0`` such that source frame = observed frame.

        ebv : float, optional, default=None
            The E(B - V) value. If provided, calculates A(x) instead

        linear : bool, optional, default=True
            If ``True``, returns 10^A(x). Else, returns A(x).

        Returns
        -------
        float
            If ``ebv`` is `None`, returns A(x)/A(V).
            Elif ``linear`` is `True`, returns 10^A(x).
            Elif ``linear`` is `False` and returns A(x).
        """
        curve = self.curve((u.speed_of_light / u.micron) / frequency, z)

        if ebv is None:
            return curve  # A(x) / A(V)

        if linear:
            return 10 ** (-0.4 * self.r_v * ebv * curve)  # 10^-A(x)/2.5

        return -0.4 * self.r_v * ebv * curve  # -A(x) / 2.5

    def curve(self, wavelength: float, z: float = 0.0) -> float:
        """
        Calculates the ratio of the extinction in a given band to that in
        V band.

        Parameters
        ----------
        wavelength : float
            The wavelength of the band measured in micrometers.

        z : float, optional, default=0.0
            The redshift of the event. Used to transform the wavelength to
            the source frame from the observed frame. Defaults to ``0`` such
            that source frame = observed frame.

        Returns
        -------
        float
            The extinction curve, A(x) / A(V).
        """
        x = (1 + z) / wavelength
        y = x - 1.82

        # Infrared
        if x < 0.3:
            raise ValueError(
                f'The wave number is above the CCM valid limit: '
                f'{x} > {self.valid_x_range.upper}.'
            )

        elif x <= 1.1:
            a = 0.574 * x ** 1.61
            b = -0.527 * x ** 1.61

        # Optical/NIR
        elif x <= 3.3:
            a = (1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 +
                 0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.7753 * y ** 6 +
                 0.32999 * y ** 7)

            b = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 -
                 5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.3026 * y ** 6 -
                 2.09002 * y ** 7)

        # Ultraviolet
        elif x <= 8:
            f_a = f_b = 0

            if x >= 5.9:
                f_a = -0.04473 * (x - 5.9) ** 2 - 0.009779 * (x - 5.9) ** 3
                f_b = 0.213 * (x - 5.9) ** 2 + 0.1207 * (x - 5.9) ** 3

            a = 1.752 - (0.316 * x) - (0.104 / ((x - 4.67) ** 2 + 0.341)) + f_a
            b = -3.090 + 1.825 * x + (1.206 / ((x - 4.62) ** 2 + 0.263)) + f_b

        # Far Ultraviolet
        elif x <= 10:
            a = -1.073 - 0.628 * (x - 8) + 0.137 * (x - 8) ** 2 - 0.07 * (x - 8) ** 3
            b = 13.670 + 4.257 * (x - 8) - 0.42 * (x - 8) ** 2 + 0.374 * (x - 8) ** 3

        else:
            raise ValueError(
                f'The wave number is below the CCM valid limit: '
                f'{x} < {self.valid_x_range.lower}.'
            )

        return a + b / self.r_v  # A(x)/A(V)
