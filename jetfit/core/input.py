from pathlib import Path

import numpy as np
import astropy.units as u

from jetfit.core.defns.enums import DataType
from jetfit.core.utils.csv_utils import CSVReader
from jetfit.core.values import IntegratedFlux, SpectralFlux, SpectralIndex, Bound


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
    values : np.ndarray of float64
        The measurement values (e.g., spectral flux measured
        in mJy, integrated flux measured in erg cm-2 s-1 and
        spectral index (dimensionless)).

    errors : np.ndarray of float64
        The errors associated with the measurement values.
        Measured in same units as its value.

    times : np.ndarray of float64
        The times associated with the measurements. Measured
        in days since trigger.

    types : np.ndarray
        The types associated with the measurements.

    filters : np.ndarray of str
        The filters associated with the measurements.

    frequencies : np.ndarray of float
        The frequencies associated with the spectral flux
        values. Measured in Hz.

    if_lower_freqs, if_upper_freqs : np.ndarray of float
        The lower and upper frequencies associated with the
        integrated flux values. Measured in Hz.

   si_lower_freqs, si_upper_freqs : np.ndarray of float
        The lower and upper frequencies associated with the
        spectral index values. Measured in Hz.

    wave_numbers : np.ndarray of float
        The wave numbers corresponding to the frequencies.
        Measured in micro-meters.
    """

    # Units
    if_units = 'erg cm-2 s-1'
    sf_units = 'mJy'
    time_units = 'd'
    freq_units = 'Hz'

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
            wave_numbers,
            extinguishable
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
        self.extinguishable = extinguishable

        # Static truth arrays of data
        self.flux_loc = self.types != DataType.SPECTRAL_INDEX
        self.sflux_loc = self.types == DataType.SPECTRAL_FLUX
        self.iflux_loc = self.types == DataType.INTEGRATED_FLUX
        self.sindex_loc = self.types == DataType.SPECTRAL_INDEX

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
        extinguishable = np.full(len(data), False, dtype=bool)

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

                if 3e13 < f.frequency.to_value('Hz') < 1e15:
                    extinguishable[i] = True

            if f.type == DataType.SPECTRAL_FLUX:
                values[i] = f.value.to_value(cls.sf_units)
                errors[i] = f.avg_uncertainty.to_value(cls.sf_units)
                frequencies[i] = f.frequency.to_value(cls.freq_units)
                wave_numbers[i] = 1 / f.wavelength.to_value('um')

            elif f.type == DataType.INTEGRATED_FLUX:
                values[i] = f.value.to_value(cls.if_units)
                errors[i] = f.avg_uncertainty.to_value(cls.if_units)
                if_lower_freqs[i] = f.int_range.lower.to_value(cls.freq_units)
                if_upper_freqs[i] = f.int_range.upper.to_value(cls.freq_units)

            elif f.type == DataType.SPECTRAL_INDEX:
                values[i] = f.value.value
                errors[i] = f.avg_uncertainty.value
                si_lower_freqs[i] = f.int_range.lower.to_value(cls.freq_units)
                si_upper_freqs[i] = f.int_range.upper.to_value(cls.freq_units)

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
            extinguishable=extinguishable,
            wave_numbers=wave_numbers,
        )


class Observation:
    """
    Time series of flux measurements.

    Attributes
    ----------
    ??
    """
    def __init__(self, data, offsets=None, host=None, groups=None):
        self._as_arrays = ObsArray.from_data(data)
        self._data = data

        # Groups
        self.cal_groups = offsets
        self.data_groups = groups
        self.host_groups = host

        self.data_regimes = {}

        # Set the valid time bounds for each data group
        if groups is not None:
            for group, pos in groups.items():
                times = self._as_arrays.times[pos]

                self.data_regimes[group] = Bound(
                    times.min() * u.d,  # type: ignore
                    times.max() * u.d,  # type: ignore
                )

        self.length = len(data)

    @classmethod
    def from_csv(cls, path: str | Path):
        """
        Instantiates an `Observation` from a CSV.

        Parameters
        ----------
        path : str | Path
            The CSV file path.

        Returns
        -------
        Observation
            Instantiated from a CSV file path.
        """
        csv = CSVReader(path, live_dangerously=True)

        def init_dict(group: str) -> dict:
            """ Initialize group dictionary. """
            return {
                cg : [False for _ in range(len(csv.df))]
                for cg in csv.df[group].unique() if isinstance(cg, str)
            }

        # Handle optional columns
        groups, offsets, hosts = None, None, None

        if 'CalGroup' in csv.df.columns.values:
            offsets = init_dict('CalGroup')

        if 'DataGroup' in csv.df.columns.values:
            groups = init_dict('DataGroup')

        if 'HostGroup' in csv.df.columns.values:
            hosts = init_dict('HostGroup')

        data = []
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

            if offsets and isinstance(row.CalGroup, str):
                offsets[row.CalGroup][row.Index] = True

            if hosts and isinstance(row.HostGroup, str):
                hosts[row.HostGroup][row.Index] = True

            if groups and isinstance(row.DataGroup, str):
                groups[row.DataGroup][row.Index] = True

        return cls(np.asarray(data, dtype=object), offsets, hosts, groups)

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
            The array representation of the observation.
        """
        return self._as_arrays

    @property
    def flux_loc(self) -> np.array:
        """ Returns bools indicating the flux locations. """
        return self.as_arrays.flux_loc

    @property
    def sflux_loc(self) -> np.array:
        """ Returns bools indicating the spectral flux locations. """
        return self.as_arrays.sflux_loc

    @property
    def iflux_loc(self) -> np.array:
        """ Returns bools indicating the integrated flux locations. """
        return self.as_arrays.iflux_loc

    @property
    def sindex_loc(self) -> np.array:
        """ Returns bools indicating the spectral index locations. """
        return self.as_arrays.sindex_loc

    @property
    def extinguishable(self) -> np.array:
        """ Returns bools indicating the flux affected by dust extinction locations. """
        return self.as_arrays.extinguishable
