

class JetFit:
    def __init__(self):
        self.mcmc = None        # settings, results (burn, run, best fit, ..)
        self.table = None       # path to file
        self.output = None      # directory
        self.photometry = None  # path to file

    @property
    def hydrodynamic_properties(self):
        """ """
        return [p for p in self.mcmc.parameters if p.group == 'hydrodynamic']

    @property
    def radiation_properties(self):
        """ """
        return [p for p in self.mcmc.parameters if p.group == 'radiation']

    @property
    def observational_properties(self):
        """ """
        return [p for p in self.mcmc.parameters if p.group == 'observational']


class MCMC:
    """ Create a MCMC object for both burn and run. Will be an object of
    bigger class like Fitter.
    """
    def __init__(self):
        self.threads = None
        self.sampler = None
        self.iterations = None
        self.num_walkers = None
        self.num_temperatures = None
        self.start_positions = None

        self.parameters = None


class MCMCValue:
    def __init__(self):
        self.name = None    # str: explosion_energy
        self.group = None   # str: hydrodynamic, radiation, observational
        self.value = None   # float
        self.scale = None   # one of log, ln, linear

    @property
    def fit(self) -> bool:
        """ Returns bool indicating that the value should not be fit to.
        """
        return False


class MCMCFittingValue(MCMCValue):
    def __init__(self):
        super().__init__()
        self.prior = None   # str: uniform, sine, cosine
        self.bounds = None  # tuple: (lower, upper)
        self.wander = None  # bool: initialize across entire bounds or around default value

    @property
    def fit(self) -> bool:
        """ Returns bool indicating that the value should be fit to.
        """
        return False


class Observation:
    def __init__(self, times: list, fluxes: list):
        self.times = None   # list[TimeValue]: value, bounds, conversions (secs, days, ..)
        self.fluxes = None  # list[FluxDensityValue | TotalFluxValue]: value, bounds, conversions (mJy, cgs, mags)


class Value:
    def __init__(self, value: float, **kw):
        self.value = value
        self.name = kw.get('name', None)
        self.units = kw.get('units', None)


class BoundedValue(Value):
    def __init__(self, value: float, bounds: tuple, **kw):
        super().__init__(value, **kw)

        self._bounds = bounds

    @property
    def lower(self) -> float:
        """ Returns the lower bound of the value. """
        return self.bounds[0]

    @property
    def upper(self) -> float:
        """ Returns the upper bound of the value. """
        return self.bounds[1]

    @property
    def bounds(self) -> tuple:
        """ Returns the bounds of the value. """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds: tuple) -> None:
        """ Sets the bounds of the value.

        :param bounds: tuple of (lower, upper)
        """
        if len(bounds) != 2:
            raise ValueError('Argument `bounds` must have exactly two elements.')

        if bounds[0] > bounds[1]:
            raise ValueError('Argument `bounds` was provided in reverse order.')

        self._bounds = bounds


class FluxValue(BoundedValue):
    def __init__(self, value: float, **kw):
        super().__init__(value, **kw)

    def convert_to_mjy(self) -> None:
        """  """
        if self.units is None:
            raise ValueError('Flux value has no units.')

        if self.units.lower() == 'mjy':
            return

        if self.units == 'cgs':
            self.value /= 1.0e-26
            self.bounds = (self.bounds[0] / 1.0e-26, self.bounds[1] / 1.0e-26)

        self.units = 'mjy'

    def convert_to_cgs(self) -> None:
        """  """
        if self.units is None:
            raise ValueError('Flux value has no units.')

        if self.units.lower() == 'cgs':
            return

        if self.units == 'mJy':
            self.value *= 1.0e-26
            self.bounds = (self.bounds[0] * 1.0e-26, self.bounds[1] * 1.0e-26)

        self.units = 'cgs'


class TotalFluxValue(BoundedValue):
    def __init__(self, value: float, bounds: tuple, interval: tuple, **kw):
        super().__init__(value, bounds, **kw)
        self.interval = interval

    @property
    def type(self) -> str:
        """ Returns the type of flux. """
        return 'total'


class SpectralFluxValue(BoundedValue):
    def __init__(self, value: float, bounds: tuple, frequency: float, **kw):
        super().__init__(value, bounds, **kw)
        self.frequency = frequency

    @property
    def type(self) -> str:
        """ Returns the type of flux. """
        return 'spectral'