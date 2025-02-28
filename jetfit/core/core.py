import astropy.units as u
import numpy as np


def q_e():
    """ The electron charge in cgs units. """
    return u.Quantity(4.8032e-10 * u.g**0.5 * u.cm**1.5 / u.s)


def two_point_approx(y1, y2, x1, x2, log: bool = False):
    """"""
    if not log:
        return (y2 / y1) / (x2 / x1)

    return np.log(y2 / y1) / np.log(x2 / x1)


def to_array(x, unit: str | u.Unit):
    """"""
    if isinstance(x, u.Quantity):
        x = x.to_value(unit)

    return np.atleast_1d(x)


def pl(amplitude, x, x_0, beta):
    """
    1D Power law.

    If passing args as arrays, must all be the same shape.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity or array_like
        The amplitude of at the reference point, `x_0`.

    x : float or astropy.units.Quantity or array_like
        The lower integral limit.

    x_0 : float or astropy.units.Quantity or array_like
        The reference point.

    beta : float or array_like
        The power-law index.

    Notes
    -----
    The power-law index sign convention is opposite of the
    convention used in astropy's `PowerLaw1d` method.
    """
    return amplitude * np.power(x / x_0, beta)


def ipl(amplitude, lower, upper, x_0, beta):
    """
    1D Integrated power law.

    If passing args as arrays, must all be the same shape.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity or array_like
        The amplitude(s) at the reference point(s), `x_0`.

    lower : float or astropy.units.Quantity or array_like
        The lower integral limit(s).

    upper : float or astropy.units.Quantity or array_like
        The upper integral limit(s).

    x_0 : float or astropy.units.Quantity or array_like
        The reference point(s).

    beta : float or array_like
        The power-law index(ices).

    Notes
    -----
    The power-law index sign convention is opposite of the
    convention used in astropy's `PowerLaw1d` method.
    """
    constant = amplitude * np.power(x_0, -beta) / (beta + 1)
    return constant * (np.power(upper, beta + 1) - np.power(lower, beta + 1))
