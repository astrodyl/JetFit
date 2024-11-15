import math
import numpy as np
import operator

from jetfit.core.defns.enums import ScaleType, TimeUnits, FluxUnits
from jetfit.core.values.bounded_value import BoundedValue


# <editor-fold desc="Error Propagation">
def bounded_operation(op: operator, b1: BoundedValue, b2: BoundedValue) -> BoundedValue:
    """
    Performs an `op` operation between `b1` and b2`.

    Assumes that the errors are uncorrelated.

    Parameters
    ----------
    op : operator
        The operation to perform.

    b1 : BoundedValue
        The value to be performed on.

    b2 : BoundedValue
        The value to do the performing.

    Returns
    -------
    BoundedValue
        The value with propagated errors.
    """
    if b1.units != b2.units:
        raise TypeError(f"Cannot perform operation between values with "
                        f"mismatched units: {b1.units} != {b2.units}.")

    value = op(b1.value, b2.value)

    if op == operator.__truediv__ or op == operator.__mul__:
        lower = value * quadrature(b1.lower / b1.value, b2.lower / b2.value)
        upper = value * quadrature(b1.upper / b1.value, b2.upper / b2.value)

    elif op == operator.__add__ or op == operator.__sub__:
        lower = value * quadrature(b1.lower, b2.lower)
        upper = value * quadrature(b1.upper, b2.upper)

    else:
        raise NotImplementedError(f"Unsupported operator for BoundedValue: "
                                  f"{op.__name__}.")

    return BoundedValue(value, lower, upper, b1.units)


def quadrature(x1: float, x2: float) -> float:
    """
    Adds `x1` and `x2` in quadrature.

    Parameters
    ----------
    x1 : float
        First value to add.

    x2 : float
        Second value to add.

    Returns
    -------

    """
    return float(np.sqrt(x1 ** 2 + x2 ** 2))
# </editor-fold>


# <editor-fold desc="Calculations">
def gaussian(x: float, mu: float, sigma: float) -> float:
    """
    Evaluates a gaussian function.

    Parameters
    ----------
    x : float
        Value to evaluate.

    mu : float
        The mean of the distribution.

    sigma : float
        The standard deviation.

    Returns
    -------
    float
        The value of the evaluated gaussian function.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
# </editor-fold>


# <editor-fold desc="Flux Conversions">
def to_flux_density(value: float, _range: tuple, from_u: str | FluxUnits,
                    to_u: str | FluxUnits) -> float:
    """
    Converts an integrated flux in CGS to a flux density in mJy.

    Parameters
    ----------
    value : float
        The value to convert.

    _range : tuple of float
        The integrated range.

    from_u : FluxUnits
        Original flux units.

    to_u : FluxUnits
        Desired flux units.

    Returns
    -------
    float
        The equivalent flux density in mJy.
    """
    if isinstance(from_u, str):
        from_u = FluxUnits(from_u)

    if isinstance(to_u, str):
        to_u = FluxUnits(to_u)

    if from_u == FluxUnits.MJY:
        return value

    if to_u == FluxUnits.CGS:
        return value / (_range[1] - _range[0])

    if to_u == FluxUnits.MJY:
        return to_mjy(value, FluxUnits.CGS) / (_range[1] - _range[0])


def convert_flux_to(value: float, from_u: str | FluxUnits,
                    to_u: str | FluxUnits) -> float:
    """ """
    if from_u == to_u:
        return value

    if isinstance(from_u, str):
        from_u = FluxUnits(from_u)

    if isinstance(to_u, str):
        to_u = FluxUnits(to_u)

    match to_u:
        case FluxUnits.MJY:
            return to_mjy(value, from_u)
        case FluxUnits.CGS:
            return to_cgs(value, from_u)


def to_mjy(value: float, units: FluxUnits) -> float:
    """
    Converts a flux value to milli-Jansky.

    Parameters
    ----------
    value : float
        The value to be converted.

    units : FluxUnits
        The units of the original value.

    Returns
    -------
    float
        The equivalent flux in mJy.
    """
    if units == FluxUnits.MJY:
        return value

    return value / 1.0e-26


def to_cgs(value: float, units: FluxUnits) -> float:
    """
    Converts a flux value to CGS units.

    Parameters
    ----------
    value : float
        The value to be converted.

    units : FluxUnits
        The units of the original value.

    Returns
    -------
    float
        The equivalent flux in CGS units.
    """
    if units == FluxUnits.CGS:
        return value

    return value * 1.0e-26
# </editor-fold>


# <editor-fold desc="Time Conversions">
def convert_time_to(value: float, from_u: str | TimeUnits,
                    to_u: str | TimeUnits) -> float:
    """
    Converts the time `value` from `from_u` to `to_u`.

    Parameters
    ----------
    value : float

    from_u : `jetfit.core.enums.TimeUnits`
        The original time units.

    to_u : `jetfit.core.enums.TimeUnits`
        The desired time units.

    Returns
    -------
    float
        The converted time value in `units`.
    """
    if from_u == to_u:
        return value

    if isinstance(from_u, str):
        from_u = TimeUnits(from_u)

    if isinstance(to_u, str):
        to_u = TimeUnits(to_u)

    match to_u:
        case TimeUnits.SEC:
            return to_seconds(value, from_u)
        case TimeUnits.HRS:
            return to_hours(value, from_u)
        case TimeUnits.DAY:
            return to_days(value, from_u)


def to_seconds(value: float, units: TimeUnits) -> float:
    """
    Converts the time `value` to seconds from `units`.

    Parameters
    ----------
    value : float
        The value to be converted.

    units : `jetfit.core.enums.TimeUnits`
        The original time units.

    Returns
    -------
    float
        The converted time value in seconds.
    """
    match units:
        case TimeUnits.SEC:
            return value
        case TimeUnits.HRS:
            return value * 3600.0
        case TimeUnits.DAY:
            return value * 86400.0


def to_hours(value: float, units: TimeUnits) -> float:
    """
    Converts the time `value` to hours from `units`.

    Parameters
    ----------
    value : float
        The value to be converted.

    units : `jetfit.core.enums.TimeUnits`
        The original time units.

    Returns
    -------
    float
        The converted time value in hours.
    """
    match units:
        case TimeUnits.SEC:
            return value / 3600.0
        case TimeUnits.HRS:
            return value
        case TimeUnits.DAY:
            return value * 24.0


def to_days(value: float, units: TimeUnits) -> float:
    """
    Converts the time `value` to days from `units`.

    Parameters
    ----------
    value : float
        The value to be converted.

    units : `jetfit.core.enums.TimeUnits`
        The original time units.

    Returns
    -------
    float
        The converted time value in days.
    """
    match units:
        case TimeUnits.SEC:
            return value / 86400.0
        case TimeUnits.HRS:
            return value / 24.0
        case TimeUnits.DAY:
            return value
# </editor-fold>


# <editor-fold desc="Scale Conversions">
def to_scale(value: float, from_s: str | ScaleType,
             to_s: str | ScaleType) -> float:
    """
    Returns the converted `value` from `from_s` to `to_s`.

    Parameters
    ----------
    value : float
        The value to convert.

    from_s : str or ScaleType
        The original scale type.

    to_s : str or ScaleType
        The new scale type.

    Returns
    -------
    float
        The converted value.
    """
    if isinstance(from_s, str):
        from_s = ScaleType(from_s)

    if isinstance(to_s, str):
        to_s = ScaleType(to_s)

    if from_s == to_s:
        return value

    match to_s:
        case ScaleType.LOG:
            return to_log(value, from_s)
        case ScaleType.LN:
            return to_ln(value, from_s)
        case ScaleType.LINEAR:
            return to_linear(value, from_s)


def to_log(value: float, scale: ScaleType) -> float:
    """
    Returns a value converted from the provided `scale` to log10.

    Parameters
    ----------
    value : float
        The value to convert.

    scale : ScaleType
        The original scale type.

    Returns
    -------
    float
        The converted value.
    """
    match scale:
        case ScaleType.LOG:
            return value
        case ScaleType.LN:
            return value / math.log(10)
        case ScaleType.LINEAR:
            return math.log10(value)


def to_ln(value: float, scale: ScaleType) -> float:
    """
    Returns a value converted from the provided `scale` to natural log.

    Parameters
    ----------
    value : float
        The value to convert.

    scale : ScaleType
        The original scale type.

    Returns
    -------
    float
        The converted value.
    """
    match scale:
        case ScaleType.LN:
            return value
        case ScaleType.LOG:
            return value * math.log(10)
        case ScaleType.LINEAR:
            return math.log(value)


def to_linear(value: float, scale: ScaleType) -> float:
    """
    Returns a value converted from the provided `scale` to linear.

    Parameters
    ----------
    value : float
        The value to convert.

    scale : ScaleType
        The original scale type.

    Returns
    -------
    float
        The converted value.
    """
    match scale:
        case ScaleType.LINEAR:
            return value
        case ScaleType.LOG:
            return 10 ** value
        case ScaleType.LN:
            return math.exp(value)
# </editor-fold>