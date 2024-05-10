import numpy as np
import h5py as h5
from typing import List

"""
    interpolator.py

    This module performs interpolation in the boosted fireball table.
"""


class Interpolator:
    # h5 Table
    _Table = {}
    _Axis = {}

    # Interpolation function
    _f_peak = None
    _f_nu_c = None
    _f_nu_m = None

    def __init__(self, table, log_table: bool = True, log_axis: List[str] = None):
        """ Initialize Interpolator.

        :param table: directory to boosted fireball table
        :param log_table: whether Table is measured in log scale
        :param log_axis: whether certain axis is measured in log scale
        """
        if log_axis is None:
            log_axis = ['tau']

        self._load_table(table)
        self._set_scale(log_table=log_table, log_axis=log_axis)
        self._get_interpolator()

    def _load_table(self, table: str) -> None:
        """ Loads the boosted fireball Table.

        :param table: path to boosted fireball table
        """
        data = h5.File(table, 'r')

        try:
            for key in data.keys():
                if key in ['f_peak', 'f_nu_c', 'f_nu_m']:
                    self._Table[key] = data[key][...]
                else:
                    self._Axis[key] = data[key][...]

            # Convert bytes to string
            self._Axis['Axis'] = np.array([x.decode("utf-8") for x in self._Axis['Axis']])
        except Exception as e:
            raise e
        finally:
            data.close()

    def _set_scale(self, log_table: bool = True, log_axis: List[str] = None):
        """ Sets the proper scales to table and axis.

        :param log_table: whether Table is measured in log scale
        :param log_axis: whether certain axis is measured in log scale
        """
        if log_axis is None:
            log_axis = ['tau']

        self.Info = self._Axis.copy()

        if 'LogAxis' not in self._Axis.keys():
            self._Axis['LogAxis'] = log_axis
            for key in log_axis:
                if key not in self._Axis.keys():
                    raise ValueError('could not find %s in Axis' % key)
                else:
                    self._Axis[key] = np.log(self._Axis[key])

        if 'LogTable' not in self._Table.keys():
            for key in self._Table.keys():
                temp = np.ma.log(self._Table[key])
                self._Table[key] = temp.filled(-np.inf)
            self._Table['LogTable'] = True

    def _get_interpolator(self) -> None:
        """ Uses scipy.interpolate.RegularGridInterpolator to perform
        interpolation.
        """
        from scipy.interpolate import RegularGridInterpolator

        axes = [self._Axis[key] for key in self._Axis['Axis']]
        self._f_peak = RegularGridInterpolator(axes, self._Table['f_peak'])
        self._f_nu_c = RegularGridInterpolator(axes, self._Table['f_nu_c'])
        self._f_nu_m = RegularGridInterpolator(axes, self._Table['f_nu_m'])

    def get_table_info(self) -> dict:
        """ Gets the table Information.

        :return: dictionary of table information
        """
        return self.Info

    def get_value(self, position):
        """ Gets the characteristic function values at the position.

        :param position: (tau, Eta0, GammaB, theta_obs) (linear scale)
        :return: (float, float, float) characteristic function values
        """
        scaled_position = position.copy()

        # Convert linear scale to log scale
        for key in self._Axis['LogAxis']:
            idx = np.where(self._Axis['Axis'] == key)[0][0]
            scaled_position[:, idx] = np.log(scaled_position[:, idx])

        # When lorentz factor is low and observation time is early,
        # there is no detection, which is represented by nans.
        try:
            if self._Table['LogTable']:
                f_peak = np.exp(self._f_peak(scaled_position))
                f_nu_c = np.exp(self._f_nu_c(scaled_position))
                f_nu_m = np.exp(self._f_nu_m(scaled_position))
            else:
                np.seterr(all='ignore')
                f_peak = self._f_peak(scaled_position)
                f_nu_c = self._f_nu_c(scaled_position)
                f_nu_m = self._f_nu_m(scaled_position)
                np.seterr(all='raise')
            return f_peak, f_nu_c, f_nu_m
        except:
            nans = [np.nan for _ in range(len(position))]
            f_peak, f_nu_c, f_nu_m = nans, nans, nans
            return f_peak, f_nu_c, f_nu_m
