import math
import numpy as np

from jetfit.core.defns.enums import ScaleType


# <editor-fold desc="Calculations">
def chi_squared(
        f: np.ndarray[float],
        y: np.ndarray[float],
        e: np.ndarray[float],
        s: float = None,
) -> float:
    """
    Calculates the chi-squared value.

    Parameters
    ----------
    f : np.ndarray of float
        The predicted values.

    y : np.ndarray of float
        The observed values.

    e : np.ndarray of float
        The uncertainty in the observed values.

    s : float, optional, default=None
        The slop parameter.

    Returns
    -------
    float or Quantity
        The chi-squared value.
    """
    if s is None:
        return np.sum(((y - f) / e) ** 2)

    return chi_squared_eff(f, y, e, s)


def chi_squared_eff(
        f: np.ndarray[float],
        y: np.ndarray[float],
        e: np.ndarray[float],
        s: float,
) -> float:
    """
    When the slop parameter, `s`, is provided, the chi-squared
    calculation accounts for additional unknown variances. When
    `s > 0`, the effective uncertainties increase, decreasing the
    penalty for model-data mismatches but adding a penalty for
    increasing `s` through the normalization term. When `s = 0`,
    the calculation reduces to the standard chi-squared.

    Parameters
    ----------
    f : np.ndarray of float
        The modeled values.

    y : np.ndarray of float
        The observed values.

    e : np.ndarray of float
        The uncertainty in the observed values.

    s : float
        The slop parameter.

    Returns
    -------
    float
        The effective chi-squared value.
    """
    # Convert slop to linear space
    s_lin_hi = 10 ** (np.log10(f) + s) - f
    s_lin_lo = f - 10 ** (np.log10(f) - s)

    # Force slop to be symmetric
    s_lin_avg = (s_lin_hi + s_lin_lo) / 2

    # Combine the slop and data uncertainties
    sig = np.sqrt(s_lin_avg ** 2 + e ** 2)

    # return chi-squared effective
    return np.sum(2 * np.log(sig) + ((y - f) / sig) ** 2)
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