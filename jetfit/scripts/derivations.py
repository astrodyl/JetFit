import astropy.units as u
import astropy.constants as const
import numpy as np


# Constants in cgs units
m_p = const.m_p.cgs  # noqa
m_e = const.m_e.cgs  # noqa
q_e = u.Quantity(4.8032e-10 * u.g**0.5 * u.cm**1.5 / u.s)
c = const.c.cgs  # noqa
tcs = const.sigma_T.cgs  # noqa


# noinspection PyPep8Naming
def peak_spectral_power(gamma, B, R, ):
    """"""
    return (4/3) * np.pi * (
        m_e * c**2 * tcs * gamma * B
    ) / (3 * q_e)


# noinspection PyPep8Naming
def characteristic_frequency(gamma, gamma_e, B) -> float | u.Quantity:
    """
    The characteristic synchrotron frequency from a randomly
    oriented electron with Lorentz factor `gamma_e` >> 1 in
    a magnetic field `B`.

    Parameters
    ----------
    gamma : float or u.Quantity
        The Lorentz factor of the shocked fluid.

    gamma_e : float or u.Quantity
        The Lorentz factor of the electron.

    B : float or u.Quantity
        The magnetic field strength.

    Returns
    -------
    float or u.Quantity['frequency']
        The characteristic synchrotron frequency.
    """
    return gamma * gamma_e**2 * q_e * B / (2 * np.pi * m_e * c)


def shock_radius(gamma, beta, t) -> float | u.Quantity:
    """
    The shock radius, R(t).

    Parameters
    ----------
    gamma : float or u.Quantity
        The Lorentz factor of the shocked fluid.

    beta : float
        The numerical constant that varies depending on the
        details of the hydrodynamic evolution and the spectrum.

    t : float or u.Quantity['time']
        The time to evaluate.

    Returns
    -------
    float or u.Quantity['length']
        The shock radius evaluated at `t`.
    """
    return beta * gamma**2 * c * t


# noinspection PyPep8Naming
def lorentz_factor(E, rho, k, t, alpha, beta):
    """
    The Lorentz factor of the shocked fluid, gamma.

    Parameters
    ----------
    E : float or u.Quantity['energy']
        The energy of the spherical shock.

    rho : float or u.Quantity['number density']
        The density.

    k : float
        The density power-law index

    t : float or u.Quantity['time']
        The time in seconds.

    alpha : float
        Numerical coefficient.

    beta : float
        Numerical coefficient.

    Returns
    -------
    float or u.Quantity['dimensionless']
        The Lorentz factor of the shocked fluid.
    """
    return (
        alpha * beta**(3-k) * np.pi *
        c**(5-k) * rho * E**-1 * t**(3-k)
    ) ** -(0.5 / (4 - k))


# noinspection PyPep8Naming
def min_lorentz_factor(E, rho, p, k, t, eps_e, alpha, beta):
    """
    Minimum Lorentz factor of the shocked electrons.

    Parameters
    ----------
    E : float or u.Quantity['energy']
        The energy.

    rho : float or u.Quantity['number density']
        The number density in cm-3.

    p : float
        The electron energy index.

    k : float
        The density power-law index

    t : float or u.Quantity['time']
        The time in seconds.

    eps_e : float
        The electron energy fraction.

    alpha : float
        Numerical coefficient.

    beta : float
        Numerical coefficient.

    Returns
    -------
    float or u.Quantity
        The minimum Lorentz factor of the shocked electrons.
    """
    return (
        eps_e * (p - 2) * (p - 1) *
        (m_p / m_e) * lorentz_factor(E, rho, k, t, beta, alpha)
    )


# noinspection PyPep8Naming
def cooling_lorentz_factor(gamma, B, t):
    """
    The cooling Lorentz factor of the shocked electrons.

    Parameters
    ----------
    gamma : float or u.Quantity
        The Lorentz factor of the shocked fluid.

    B : float or u.Quantity
        The magnetic field strength.

    t : float or u.Quantity['time']
        The time in seconds.

    Returns
    -------
    float or u.Quantity
        The minimum Lorentz factor of the shocked electrons.
    """
    return (
        6 * np.pi * m_e * c /
        (tcs * gamma * B**2 * t)
    )


class SPN98:
    """
    Sari, Piran, & Narayan 1998

    References
    ----------
    https://ui.adsabs.harvard.edu/abs/1998ApJ...497L..17S/abstract
    """
    p = 2.5
    k = 0.0
    X = 1.0
    alpha = 16/17
    beta = 4


class CL00:
    """
    Chevalier & LI 2000

    References
    ----------
    https://ui.adsabs.harvard.edu/abs/2000ApJ...536..195C/abstract
    """
    p = 2.5
    k = 2.0
    X = 0.0
    alpha = 16/9
    beta = 8


if __name__ == '__main__':

    energy  = u.Quantity(1e52, unit='erg')
    time    = u.Quantity(1.0, unit='d')
    density = u.Quantity(5e11, unit='g cm-1')

    # Chevalier & LI 2000 values
    g = lorentz_factor(energy, density, CL00.k, time, CL00.alpha, CL00.beta)
    r = shock_radius(g, CL00.beta, time)
