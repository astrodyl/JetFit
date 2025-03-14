import numpy as np
import astropy.units as u


def main(ti, fi, tf, ff) -> tuple:
    """

    Parameters
    ----------
    ti : u.Quantity['time']
        The initial time.

    fi : u.Quantity['energy flux']
        The initial integrated flux.

    tf : u.Quantity['time']
        The final time.

    ff : u.Quantity['energy flux']
        The final integrated flux.

    Returns
    -------
    tuple of u.Quantity
        The effective (temporal index, time).
    """
    # Effective spectral index
    alpha = np.log(ff / fi) / np.log(tf / ti)

    # Effective time
    x1, x2 = alpha + 1, alpha + 2
    time = (x1 / x2) * (tf ** x2 - ti ** x2) / (tf ** x1 - ti ** x1)

    # return effective temporal index and time
    return alpha, time


if __name__ == '__main__':

    # Time range
    t_start = u.Quantity(98.479, unit='s')
    t_end   = u.Quantity(583.507, unit='s')

    # Associated flux values
    f_start = u.Quantity(1.16e-10, unit='erg cm-2 s-1')
    f_end   = u.Quantity(1.70e-11, unit='erg cm-2 s-1')

    # Do the stuff
    alpha_eff, time_eff = main(t_start, f_start, t_end, f_end)

    print(f'Alpha_eff.... {round(alpha_eff.value, 5)}')
    print(f'Time_eff..... {round(time_eff.value,  5)} {time_eff.unit}')
