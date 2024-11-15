from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGInterpolator

from jetfit.core.defns.enums import ScaleType


"""
    HydroSimTable.py
    
    table from paper. This is what it is.
"""


class HydroSimTable:
    """
    A Hydrodynamic Simulations Table.

    Attributes
    ----------
    path : str or Path
        Path to the Boosted Fireball hydrodynamic simulation table that is
        described in [1]_.

    peak_fluxes : `scipy.interpolate.RegularGridInterpolator`
        Rectangular grid of peak flux values.

    cooling_frequencies : `scipy.interpolate.RegularGridInterpolator`
        Rectangular grid of cooling frequency values.

    synchrotron_frequencies : `scipy.interpolate.RegularGridInterpolator`
        Rectangular grid of synchrotron frequency values.

    char_spec_params : dict

    References
    ----------
    .. [1] Y. Wu & A. MacFadyen, "Constraining the Outflow Structure of
    the Binary Neutron Star Merger Event GW170817/GRB170817A with a Markov
    Chain Monte Carlo Analysis," The Astrophysical Journal, vol. 869, pp.
    55-65, 2018.
    """
    spectral_scale = ScaleType.LN
    spectral_axes = ('f_peak', 'f_nu_c', 'f_nu_m')

    spectrum_axes = ('tau', 'Eta0', 'GammaB', 'theta_obs')
    spectrum_scales = {}

    def __init__(self, path: str | Path, load: bool = True):
        """

        Parameters
        ----------
        load : bool, optional
            If `True`, loads the table upon instantiation. If `False`, the
            table can be loaded at a later time by calling the load() method.
        """
        self.path = path
        self.peak_fluxes = None
        self.cooling_frequencies = None
        self.synchrotron_frequencies = None

        self.char_spec_params = {}

        if load:
            self.load()

    def load(self) -> None:
        """ Loads the hydrodynamic simulation table. """
        char_spec_funcs = {}

        with h5py.File(self.path, 'r') as table:

            for key in table.keys():

                if key in self.spectral_axes:
                    char_spec_funcs[key] = np.ma.log(table[key][...]).filled(-np.inf)

                elif key == 'tau':
                    self.char_spec_params[key] = np.log(table[key][...])
                    self.spectrum_scales[key] = ScaleType.LN

                elif key in self.spectrum_axes:
                    self.char_spec_params[key] = table[key][...]
                    self.spectrum_scales[key] = ScaleType.LINEAR

        char_spec_params_list = [self.char_spec_params[axis] for axis in self.spectrum_axes]

        self.peak_fluxes = RGInterpolator(char_spec_params_list, char_spec_funcs['f_peak'])
        self.cooling_frequencies = RGInterpolator(char_spec_params_list, char_spec_funcs['f_nu_c'])
        self.synchrotron_frequencies = RGInterpolator(char_spec_params_list, char_spec_funcs['f_nu_m'])

    def get_characteristics_at(self, position: np.ndarray) -> tuple:
        """ Returns a tuple of the characteristic spectral functions at the
        provided position.

        Parameters
        ----------
        position : np.ndarray of float, with shapes ???

        Returns
        -------
        tuple of np.ndarray of float with shapes ???
            (peak fluxes, cooling frequencies, synchrotron frequencies)
        """
        return (
            self.get_peak_fluxes_at(position),
            self.get_cooling_frequencies_at(position),
            self.get_synchrotron_frequencies_at(position)
        )

    def get_synchrotron_frequencies_at(self, position: np.ndarray) -> np.ndarray:
        """ Returns the synchrotron frequencies from the characteristics
        spectral functions table at the provided position.

        Parameters
        ----------
        position : np.ndarray of float, with shapes ???

        Returns
        -------
        np.ndarray
            Synchrotron frequencies at the provided position.
        """
        return self.get_characteristic_at(position, self.synchrotron_frequencies)

    def get_cooling_frequencies_at(self, position: np.ndarray) -> np.ndarray:
        """ Returns the cooling frequencies from the characteristics
        spectral functions table at the provided position.

        Parameters
        ----------
        position : np.ndarray of float, with shapes ???

        Returns
        -------
        np.ndarray
            Cooling frequencies at the provided position.
        """
        return self.get_characteristic_at(position, self.cooling_frequencies)

    def get_peak_fluxes_at(self, position: np.ndarray) -> np.ndarray:
        """ Returns the peak fluxes from the characteristics spectral
        functions table at the provided position.

        Parameters
        ----------
        position : np.ndarray of float, with shapes ???

        Returns
        -------
        np.ndarray
            Peak fluxes at the provided position.
        """
        return self.get_characteristic_at(position, self.peak_fluxes)

    def get_characteristic_at(self, position: np.ndarray, func) -> np.ndarray:
        """ Calls the provided `scipy.interpolate.RegularGridInterpolator`
        `func` and returns the characteristic spectral function at the
        provided `position` in linear space.

        Parameters
        ----------
        position : np.ndarray of float, with shapes ???

        func : `scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        np.ndarray
            Characteristic spectral function at the provided position.
        """
        try:
            match self.spectral_scale:
                case ScaleType.LN:
                    return np.exp(func(position))
                case ScaleType.LINEAR:
                    np.seterr(all='warn')
                    csf = func(position)
                    np.seterr(all='raise')
                    return csf
                case ScaleType.LOG:
                    return np.power(10.0, func(position))
        except:
            # When lorentz factor is low and observation time is early,
            # there is no detection, which is represented by nans.
            return np.array([np.nan for _ in range(len(position))])
