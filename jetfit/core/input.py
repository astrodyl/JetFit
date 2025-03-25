from collections import defaultdict
from pathlib import Path

import numpy as np

from jetfit.core.defns.enums import DataType
from jetfit.core.utils.csv_utils import CSVReader
from jetfit.core.values import IntegratedFlux, SpectralFlux, SpectralIndex


class ObsArray:
    """
    Container for storing `Observation` data as arrays.

    Useful for MCMC fitting when speed is important. All
    arrays have the same size even if a particular array
    is only applicable to a subset of the observation. In
    these cases, any value can be stored in the other
    non-applicable elements (e.g., np.nan). This is done
    so that masks are more easily applied.

    Parameters
    ----------
    values : np.ndarray
        The measurement values (e.g., spectral flux measured
        in mJy, integrated flux measured in erg cm-2 s-1 and
        spectral index (dimensionless)).

    errors : np.ndarray
        The errors associated with the measurement values.
        Measured in same units as its value.

    times : np.ndarray
        The times associated with the measurements. Measured
        in days since trigger.

    types : np.ndarray
        The types associated with the measurements.

    filters : np.ndarray
        The filters associated with the measurements.

    frequencies : np.ndarray
        The frequencies associated with the spectral flux
        values. Measured in Hz.

    if_lower_freqs, if_upper_freqs : np.ndarray
        The lower and upper frequencies associated with the
        integrated flux values. Measured in Hz.

   si_lower_freqs, si_upper_freqs : np.ndarray
        The lower and upper frequencies associated with the
        spectral index values. Measured in Hz.

    wave_numbers : np.ndarray
        The wave numbers corresponding to the frequencies.
        Measured in micro-meters.
    """
    if_units = 'erg cm-2 s-1'
    sf_units = 'mJy'
    time_units = 'd'

    def __init__(
            self,
            values,
            errors,
            times,
            types,
            filters,
            frequencies,
            if_lower_freqs,
            if_upper_freqs,
            si_lower_freqs,
            si_upper_freqs,
            wave_numbers
    ):
        self.values = values
        self.errors = errors
        self.times = times
        self.types = types
        self.filters = filters
        self.frequencies = frequencies
        self.if_lower_freqs = if_lower_freqs
        self.if_upper_freqs = if_upper_freqs
        self.si_lower_freqs = si_lower_freqs
        self.si_upper_freqs = si_upper_freqs
        self.wave_numbers = wave_numbers

        # Static locations of data
        self.flux_loc = np.where(self.types != DataType.SPECTRAL_INDEX)[0]
        self.sflux_loc = np.where(self.types == DataType.SPECTRAL_FLUX)[0]
        self.iflux_loc = np.where(self.types == DataType.INTEGRATED_FLUX)[0]
        self.sindex_loc = np.where(self.types == DataType.SPECTRAL_INDEX)[0]

        # Units


    @classmethod
    def from_data(cls, data):
        """
        Instantiates an `ObsArray` from an array of data values.

        Parameters
        ----------
        data :
            DataTypes `{SpectralIndex, SpectralFlux, IntegratedFlux}`

        Returns
        -------
        ObsArray
            Instantiated from a data array.
        """

        # Initializes info for all data types
        values = np.full(len(data), np.nan, dtype=np.float64)
        errors = np.full(len(data), np.nan, dtype=np.float64)
        times  = np.full(len(data), np.nan, dtype=np.float64)
        types  = np.full(len(data), np.nan, dtype=DataType)

        # Initializes info for flux data types
        filters = np.full(len(data), np.nan, dtype='U10')

        # Initializes info for spectral flux data type
        frequencies = np.full(len(data), np.nan, dtype=np.float64)
        wave_numbers = np.empty(len(data), dtype=np.float64)

        # Initializes info for integrated flux data type
        if_lower_freqs = np.full(len(data), np.nan, dtype=np.float64)
        if_upper_freqs = np.full(len(data), np.nan, dtype=np.float64)

        # Initializes info for spectral index data type
        si_lower_freqs = np.full(len(data), np.nan, dtype=np.float64)
        si_upper_freqs = np.full(len(data), np.nan, dtype=np.float64)

        for i, f in enumerate(data):
            times[i] = f.time.to_value(cls.time_units)
            types[i] = f.type

            if f.type != DataType.SPECTRAL_INDEX:
                filters[i] = f.filter

            if f.type == DataType.SPECTRAL_FLUX:
                values[i] = f.value.to_value(cls.sf_units)
                errors[i] = f.avg_uncertainty.to_value(cls.sf_units)
                frequencies[i] = f.frequency.to_value('Hz')
                wave_numbers[i] = 1 / f.wavelength.to_value('um')

            elif f.type == DataType.INTEGRATED_FLUX:
                values[i] = f.value.to_value(cls.if_units)
                errors[i] = f.avg_uncertainty.to_value(cls.if_units)
                if_lower_freqs[i] = f.int_range.lower.to_value('Hz')
                if_upper_freqs[i] = f.int_range.upper.to_value('Hz')

            elif f.type == DataType.SPECTRAL_INDEX:
                values[i] = f.value.value
                errors[i] = f.avg_uncertainty.value
                si_lower_freqs[i] = f.int_range.lower.to_value('Hz')
                si_upper_freqs[i] = f.int_range.upper.to_value('Hz')

        # return ObsArray
        return cls(
            values=values,
            errors=errors,
            times=times,
            types=types,
            filters=filters,
            frequencies=frequencies,
            if_lower_freqs=if_lower_freqs,
            if_upper_freqs=if_upper_freqs,
            si_lower_freqs=si_lower_freqs,
            si_upper_freqs=si_upper_freqs,
            wave_numbers=wave_numbers
        )


class Observation:
    """
    Time series of flux measurements.

    Attributes
    ----------
    ??
    """
    def __init__(self, data, offsets: defaultdict = None, host=None):
        self._data = data
        self._as_arrays = ObsArray.from_data(data)

        # Map fitting offset from MCMC to data groups
        self.cal_offsets = offsets
        self.host_corr = host

        self.length = len(data)

    @classmethod
    def from_csv(cls, path: str | Path):
        """
        Instantiates an `Observation` from a CSV.

        Parameters
        ----------
        path : str | Path
            CSV file path.

        Returns
        -------
        Observation
            Instantiated from a CSV file path.
        """
        csv = CSVReader(path, live_dangerously=True)

        data = []
        offsets, host = defaultdict(list), defaultdict(list)

        for row in csv.rows():
            data_type = row.ValueType.lower()

            if data_type == DataType.INTEGRATED_FLUX.value:
                data.append(IntegratedFlux.from_csv_row(row))

            elif data_type == DataType.SPECTRAL_FLUX.value:
                data.append(SpectralFlux.from_csv_row(row))

            elif data_type == DataType.SPECTRAL_INDEX.value:
                data.append(SpectralIndex.from_csv_row(row))

            else:
                raise IOError(
                    f'Row {row.Index} has an invalid data type: '
                    f'{row.ValueType}.'
                )

            if hasattr(row, 'CalGroup') and isinstance(row.CalGroup, str):
                offsets[row.CalGroup].append(row.Index)

            if hasattr(row, 'HostGroup') and isinstance(row.HostGroup, str):
                host[row.HostGroup].append(row.Index)

        return cls(np.asarray(data, dtype=object), offsets, host)

    @property
    def data(self) -> np.ndarray:
        """
        Returns the list of data. No setter is defined
        so that locations can remain static. Otherwise,
        the locations would need to be determined on call
        which would slow down MCMC.

        Returns
        -------
        np.ndarray
        """
        return self._data

    @property
    def as_arrays(self) -> ObsArray:
        """
        Returns the data as arrays. No setter is defined
        so that locations can remain static. Otherwise,
        the locations would need to be determined on call
        which would slow down MCMC.

        Returns
        -------
        ObsArray
        """
        return self._as_arrays

    @property
    def flux_loc(self):
        """ Returns the locations of the flux values. """
        return self.as_arrays.flux_loc

    @property
    def sflux_loc(self):
        """ Returns the locations of the flux values. """
        return self.as_arrays.sflux_loc

    @property
    def iflux_loc(self):
        """ Returns the locations of the flux values. """
        return self.as_arrays.iflux_loc

    @property
    def sindex_loc(self):
        """ Returns the locations of the flux values. """
        return self.as_arrays.sindex_loc