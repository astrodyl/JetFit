import math
import numpy as np

from jetfit.core.defns.enums import ScaleType


# <editor-fold desc="Calculations">
def chi_squared(
        f: np.ndarray,
        y: np.ndarray,
        e: np.ndarray,
        s: float = None,
        const: bool = False
):
    """
    Calculates the chi-squared value.

    Parameters
    ----------
    f : np.ndarray
        The predicted values.

    y : np.ndarray
        The observed values.

    e : np.ndarray
        The uncertainty in the observed values.

    s : float, optional, default=None
        The slop parameter.

    const : bool, optional, default=False
        If ``True``, calculates the constant. When maximizing the likelihood,
        the constant is not relevant and should be skipped for efficient.

    Returns
    -------
    float or Quantity
        The chi-squared value.
    """
    if s is None:
        return np.sum(((y - f) / e) ** 2)

    return chi_squared_eff(f, y, e, s, const)


def chi_squared_eff(
        f: np.ndarray,
        y: np.ndarray,
        e: np.ndarray,
        s: float,
        const: bool = False
) -> float:
    """
    TODO: Transforms slop basis to linear. Needs to be configurable.

    When the slop parameter, ``s``, is provided, the chi-squared calculation
    accounts for additional unknown variances. When ``s > 0``, the effective
    uncertainties increase, decreasing the penalty for model-data mismatches
    but adding a penalty for increasing ``s`` through the normalization term.
    When ``s = 0``, the calculation reduces to the standard chi-squared.

    Parameters
    ----------
    f : np.ndarray
        The predicted values.

    y : np.ndarray
        The observed values.

    e : np.ndarray
        The uncertainty in the observed values.

    s : float
        The slop parameter.

    const : bool, optional, default=False
        If ``True``, calculates the constant. When maximizing the likelihood,
        the constant is not relevant and should be skipped for efficient.

    Returns
    -------
    float
        The chi-squared value.
    """
    res = np.zeros_like(f)

    def peak() -> np.ndarray[float]:
        """
        Calculates the maximum of the transformed log-normal in linear space.

        Returns
        -------
        np.ndarray of float
            The peak of the transformed log-normal model distribution.
        """
        return 10 ** (np.log10(f) - (s ** 2) * np.log(10))

    def sig_lo() -> np.ndarray[float]:
        """
        Calculates the empirical 1-sig widths of the asymmetric Gaussian.

        This calculation is only valid for s ~< 0.1 and for f < f_max.

        Returns
        -------
        np.ndarray of float
            The combined error for f < f_max.
        """
        return np.sqrt(
            (f_max * (2.3029 * s + 2.6293 * s ** 2 - 3.6945 * s ** 3)) ** 2 + e ** 2
        )

    def sig_hi() -> np.ndarray[float]:
        """
        Calculates the empirical 1-sig widths of the asymmetric Gaussian.

        This calculation is only valid for s ~< 0.1 and for f >= f_max.

        Returns
        -------
        np.ndarray of float
            The combined error for f >= f_max.
        """
        return np.sqrt(
            (f_max * (2.3027 * s - 2.6544 * s ** 2 - 4.0699 * s ** 3)) ** 2 + e ** 2
        )

    plus = f >= (f_max := peak())
    res[plus] = ((y - f_max) / sig_hi()) ** 2
    res[~plus] = ((y - f_max) / sig_lo()) ** 2

    return np.sum(res)
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