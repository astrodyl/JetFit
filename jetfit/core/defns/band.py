import scipy.constants

from jetfit.core.defns.mixins import BoundedMixin


BAND_MAP = {
    # Bounds are measured in Hz

    # Johnson Cousins Filters
    'U': {'range': BoundedMixin(lower=8.10E+14, upper=8.30E+14), 'color': 'cyan'},
    'B': {'range': BoundedMixin(lower=6.64E+14, upper=6.87E+14), 'color': 'blue'},
    'V': {'range': BoundedMixin(lower=5.34E+14, upper=5.54E+14), 'color': 'green'},
    'R': {'range': BoundedMixin(lower=4.46E+14, upper=4.66E+14), 'color': 'red'},
    'Rc': {'range': BoundedMixin(lower=4.46E+14, upper=4.66E+14), 'color': 'red'},
    'I': {'range': BoundedMixin(lower=3.62E+14, upper=3.82E+14), 'color': 'purple'},
    'Ic': {'range': BoundedMixin(lower=3.62E+14, upper=3.82E+14), 'color': 'purple'},

    # SDSS Filters
    'uprime': {'range': BoundedMixin(lower=8.38E+14, upper=8.58E+14), 'color': 'tab:cyan'},
    'gprime': {'range': BoundedMixin(lower=6.08E+14, upper=6.50E+14), 'color': 'tab:blue'},
    'rprime': {'range': BoundedMixin(lower=4.71E+14, upper=4.91E+14), 'color': 'orange'},
    'iprime': {'range': BoundedMixin(lower=3.84E+14, upper=4.04E+14), 'color': 'tab:purple'},
    'zprime': {'range': BoundedMixin(lower=3.19E+14, upper=3.39E+14), 'color': 'darkred'},

    # Near Infrared Filters
    'J': {'range': BoundedMixin(lower=2.30E+14, upper=2.50E+14), 'color': 'indianred'},
    'H': {'range': BoundedMixin(lower=1.72E+14, upper=1.92E+14), 'color': 'firebrick'},
    'K': {'range': BoundedMixin(lower=1.26E+14, upper=1.46E+14), 'color': 'brown'},
    'Ks': {'range': BoundedMixin(lower=1.26E+14, upper=1.46E+14), 'color': 'brown'},

    # Swift UVOT
    'uvot-uvw2': {'range': BoundedMixin(lower=1.45494013e+15, upper=1.65494013e+15), 'color': 'pink'},
    'uvot-uvm2': {'range': BoundedMixin(lower=1.23478387e+15, upper=1.43478387e+15), 'color': 'darkblue'},
    'uvot-uvw1': {'range': BoundedMixin(lower=1.10304792e+15, upper=1.20304792e+15), 'color': 'green'},
    'uvot-u': {'range': BoundedMixin(lower=8.55201899e+14, upper=8.75201899e+14), 'color': 'cyan'},
    'uvot-b': {'range': BoundedMixin(lower=6.72587564e+14, upper=6.92587564e+14), 'color': 'lightblue'},
    'uvot-v': {'range': BoundedMixin(lower=5.38267114e+14, upper=5.58267114e+14), 'color': 'lightgreen'},

    # XRAY
    'xray': {'range': BoundedMixin(lower=1.00E+16, upper=1.00E+19), 'color': 'black'}
}


class Band(BoundedMixin):
    """

    Attributes
    ----------
    name : str
    """
    def __init__(
            self,
            name: str,
            color: str,
            lower: float,
            upper: float,
            flux: list = None,
            times: list = None
    ):
        super().__init__(lower, upper)
        self.name = name
        self.color = color

        self.flux = flux if flux else []
        self.times = times if times else []

    @property
    def center(self) -> float:
        """ Returns the center frequency of the band. """
        return (self.lower + self.upper) / 2

    @classmethod
    def from_wavelength(cls, wavelength: float):
        """
        Instantiates a ``Band`` from a ``wavelength``.

        Parameters
        ----------
        wavelength : float
            The average wavelength of the band measured in meters.

        Returns
        -------
        Band
            The instantiated ``Band`` object.
        """
        return cls.from_frequency(scipy.constants.speed_of_light / wavelength)

    @classmethod
    def from_frequency(cls, frequency: float):
        """
        Instantiates a ``Band`` from a ``frequency``.

        Parameters
        ----------
        frequency : float
            The average frequency of the band measured in Hz.

        Returns
        -------
        Band
            The instantiated ``Band`` object.
        """
        for name, band in BAND_MAP.items():
            band_range = band.get('range')

            if band_range.encompasses(frequency):
                return cls(
                    name,
                    band.get('color'),
                    band_range.lower,
                    band_range.upper
                )

        raise ValueError('Frequency does not have a defined band.')

    @classmethod
    def from_data(cls, d):
        """"""
        instance = cls.from_frequency(d.frequency.value)
        instance.flux.append(d)
        instance.times.append(d.time)
        return instance

    @classmethod
    def from_name(cls, n: str):
        """
        Instantiates a ``Band`` from a band name.

        Parameters
        ----------
        n : str
            The name of the band.

        Returns
        -------
        Band
            The instantiated ``Band`` object.
        """
        if n not in BAND_MAP:
            raise ValueError(f'Band name "{n}" is not unsupported.')

        band = BAND_MAP[n]
        band_range = band.get('range')

        return cls(
            n,
            band.get('color'),
            band_range.lower,
            band_range.upper
        )
