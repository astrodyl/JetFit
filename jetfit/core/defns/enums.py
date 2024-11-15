from enum import Enum


class ParameterGroup(Enum):
    HYDRODYNAMIC = 'hydrodynamic'
    RADIATION = 'radiation'
    OBSERVATIONAL = 'observational'


class ScaleType(Enum):
    """ Types of scales that can be applied to data values.

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
    LOG = 'log'
    LN = 'ln'


class Prior(Enum):
    """

    """
    GAUSSIAN = 'gaussian'
    TGAUSSIAN = 'tgaussian'
    UNIFORM = 'uniform'
    SINE = 'sine'


class FluxType(Enum):
    """

    """
    SPECTRAL = 'spectral'
    INTEGRATED = 'integrated'


class FluxUnits(Enum):
    """

    """
    CGS = 'cgs'
    MJY = 'mjy'


class TimeUnits(Enum):
    """

    """
    SEC = 'seconds'
    HRS = 'hours'
    DAY = 'days'
