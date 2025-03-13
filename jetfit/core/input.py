from collections import defaultdict
from pathlib import Path

import numpy as np

from jetfit.core.defns.band import Band
from jetfit.core.defns.enums import DataType
from jetfit.core.utils.csv_utils import CSVReader
from jetfit.core.values import IntegratedFlux, SpectralFlux, SpectralIndex


class Observation:
    """
    Time series of flux measurements.

    Attributes
    ----------
    data : np.ndarray

    value_array : np.ndarray
        The measurement values.
    """
    def __init__(self, data, offsets: defaultdict = None):
        self._data = data

        # Map fitting offset from MCMC to data groups
        self.cal_offsets = offsets

        # Create arrays that can be efficiently accessed.
        self.value_array = np.full(len(data), np.nan, dtype=np.float64)
        self.error_array = np.full(len(data), np.nan, dtype=np.float64)
        self.time_array = np.full(len(data), np.nan, dtype=np.float64)
        self.wave_number_array = np.empty(len(data), dtype=np.float64)
        self.flux_types = np.full(len(data), np.nan, dtype=DataType)

        for i, f in enumerate(data):
            if f.type != DataType.SPECTRAL_INDEX:
                self.time_array[i] = f.time.to_value('s')

            if f.type == DataType.SPECTRAL_FLUX:
                self.wave_number_array[i] = 1 / f.wavelength.to_value('um')

            self.value_array[i] = f.value.value
            self.error_array[i] = f.avg_uncertainty.value
            self.flux_types[i] = f.type

        self.length = len(data)

    @classmethod
    def from_csv(cls, path: str | Path):
        """  """
        csv = CSVReader(path, live_dangerously=True)
        data = np.empty(len(csv.df), dtype=object)
        offsets = defaultdict(list)

        for row in csv.rows():
            data_type = row.ValueType.lower()

            if data_type == DataType.INTEGRATED_FLUX.value:
                data[row.Index] = IntegratedFlux.from_csv_row(row)

            elif data_type == DataType.SPECTRAL_FLUX.value:
                data[row.Index] = SpectralFlux.from_csv_row(row)

            elif data_type == DataType.SPECTRAL_INDEX.value:
                data[row.Index] = SpectralIndex.from_csv_row(row)

            else:
                raise IOError(
                    f'Row {row.Index} has an invalid data type: '
                    f'{row.ValueType}.'
                )

            if hasattr(row, 'CalGroup') and isinstance(row.CalGroup, str):
                offsets[row.CalGroup].append(row.Index)

        return cls(data, offsets)

    @property
    def data(self):
        """ Returns the list of data. """
        return self._data

    @property
    def flux_loc(self) -> np.ndarray:
        """ Returns an array of flux indices. """
        return np.where(self.flux_types != DataType.SPECTRAL_INDEX)[0]

    @property
    def spectral_flux_loc(self) -> np.ndarray:
        """ Returns an array of spectral flux indices. """
        return np.where(self.flux_types == DataType.SPECTRAL_FLUX)[0]

    @property
    def integrated_flux_loc(self) -> np.ndarray:
        """ Returns an array of integrated flux indices. """
        return np.where(self.flux_types == DataType.INTEGRATED_FLUX)[0]

    @property
    def spectral_index_loc(self) -> np.ndarray:
        """ Returns an array of spectral index indices. """
        return np.where(self.flux_types == DataType.SPECTRAL_INDEX)[0]

    def get_bands(self) -> list[Band]:
        """
        Organizes the flux data into a list of `Band`s.

        Returns
        -------
        list of `Band`
        """
        bands = []

        for value in self.data:
            if value.type == DataType.SPECTRAL_INDEX:
                continue

            for band in bands:
                # Band already exists
                if band.encompasses(value.frequency.value):
                    band.flux.append(value)
                    band.times.append(value.time)
                    break
            else:
                # First time seeing band
                bands.append(Band.from_data(value))

        return bands
