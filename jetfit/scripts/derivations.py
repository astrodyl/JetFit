import astropy.units as u
import astropy.constants as const
import numpy as np

from jetfit.models2.basemodels import OpeningAngleModel, ShockRadius

# Constants in cgs units
m_p = const.m_p.cgs  # noqa
m_e = const.m_e.cgs  # noqa
q_e = u.Quantity(4.8032e-10 * u.g**0.5 * u.cm**1.5 / u.s)
c = const.c.cgs  # noqa
tcs = const.sigma_T.cgs  # noqa


def a(k):
    """"""
    return 16 / (17 - k)


def b(k):
    """"""
    return 4 - k


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


def shock_radius(gamma, beta, t, z) -> float | u.Quantity:
    """
    The shock radius, R(t).

    Parameters
    ----------
    gamma : float or u.Quantity
        The Lorentz factor of the shocked fluid evaluated
        using the observer time, `t`.

    beta : float
        The numerical constant that varies depending on the
        details of the hydrodynamic evolution and the spectrum.

    t : float or u.Quantity['time']
        The observer time to evaluate.

    z : float
        The redshift.

    Returns
    -------
    float or u.Quantity['length']
        The shock radius evaluated at `t`.
    """
    return beta * gamma**2 * c * t / (1 + z)


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


def trans_radius(n1, n2, k1, k2, ref=1e17):
    """
    Calculates the transition radius of the stratified
    density.

    Parameters
    ----------
    n1 : float or u.Quantity
        The density before the transition.

    n2 : float or u.Quantity
        The density after the transition.

    k1 : float
        The density power-law index for `n1`.

    k2 : float
        The density power-law index for `n2`.

    ref : float or u.Quantity['length'], optional, default=1e17
        The characteristic radius.

    Returns
    -------
    u.Quantity['cm']
        The transition radius.
    """
    return u.Quantity(ref * (n2 / n1) ** (1/(k2 - k1)), unit='cm')


# noinspection PyPep8Naming
def trans_time(E, n, r, k, z):
    """
    Calculates the transition time of the stratified
    density.

    Parameters
    ----------
    E : float or u.Quantity['energy']
        The isotropic explosion energy.

    n : float or u.Quantity
        The density before the transition with units
        of g cm^(3-k)

    r : float or u.Quantity['length']
        The transition radius.

    k : float
        The density power-law index for `n`.

    z : float
        The redshift

    Returns
    -------
    u.Quantity['s']
        The transition time in seconds.
    """
    if isinstance(E, float):
        E = u.Quantity(E, unit='erg')

    if isinstance(r, float):
        r = u.Quantity(r, unit='cm')

    if isinstance(n, float):
        n = u.Quantity(n, unit=u.g * u.cm ** (k-3))

    return ((1 + z) * (
        (a(k) * b(k) ** (3 - k) * np.pi * c ** (5 - k) * n / E)
    ) * ((r / (b(k) * c)) ** (4 - k))).to('s')


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
    r_shock = shock_radius(g, CL00.beta, time, 0.0)

    # Transitions for GRB 080413B
    r_trans = trans_radius(0.3968, 1.182e-6, 2.3182, -4.736)

    k_trans = 2.3182
    e_trans = u.Quantity(0.25890738 * 1e52, unit='erg')
    n_trans = u.Quantity(
        0.3968289 * m_p.value * (1e17 ** k_trans),
        unit=u.g * u.cm ** (k_trans-3)
    )

    t_trans = trans_time(e_trans, n_trans, r_trans, k_trans, 1.1)

    # tests
    theta = OpeningAngleModel(1.0, 1, k=0.0, z=5.0)(1.0)
    r_sh2 = ShockRadius(1.0, 1.0, 0.0, 0.0)(1.0)

    print('R (transition): ', r_trans)
    print('t (transition): ', t_trans)
