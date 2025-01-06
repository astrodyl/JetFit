from enum import Enum


class ScaleType(Enum):
    """
    Types of scales that can be applied to data values.

    Attributes
    ----------
    LINEAR : ScaleType
        Indicates a linear scale.

    LOG : ScaleType
        Indicates a base-10 logarithmic scale.

    LN : ScaleType
        Indicates a natural logarithmic (base-e) scale.

    Notes
    -----
    It is often useful to fit a model in one scale, but perform MCMC sampling
    in another scale. For example, a parameter space may cover many orders of
    magnitude which could make sampling very expensive and difficult.
    """
    LINEAR = 'linear'
    LOG    = 'log'
    LN     = 'ln'


class Prior(Enum):
    """

    """
    GAUSSIAN  = 'gaussian'
    TGAUSSIAN = 'tgaussian'
    UNIFORM   = 'uniform'
    SINE      = 'sine'


class FluxType(Enum):
    """
    Type of flux measurement.

    Attributes
    ----------
    SPECTRAL : FluxType
        The amount of energy per unit area per unit frequency.

    INTEGRATED : FluxType
        The total energy per unit area calculated by summing up the spectral
        flux across all frequencies within some range.
    """
    SPECTRAL   = 'spectral'
    INTEGRATED = 'integrated'

    @classmethod
    def from_str(cls, s: str):
        """

        Parameters
        ----------
        s : str
            Type of flux.
        """
        return cls[s.lower()]


class FluxUnits(Enum):
    """
    Units of a flux measurement.

    Attributes
    ----------
    CGS : FluxUnits
        Centigrade-Gram-Seconds units.

    MJY : FluxUnits
        Milli-Jansky units.
    """
    CGS = 'cgs'
    MJY = 'mjy'


class IndexType(Enum):
    """


    Attributes
    ----------
    PHOTON : Index

    SPECTRAL : Index

    """
    PHOTON   = 'photonindex'
    SPECTRAL = 'spectralindex'


class TimeUnits(Enum):
    """
    Units of a time measurement.

    Attributes
    ----------
    SEC : TimeUnits

    HRS : TimeUnits

    DAY : TimeUnits
    """
    SEC = 'seconds'
    HRS = 'hours'
    DAY = 'days'
