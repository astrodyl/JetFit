from pathlib import Path

from jetfit.core.defns.enums import FluxType, TimeUnits, FluxUnits
from jetfit.core.utilities.io import csv
from jetfit.core.values.flux_value import IntegratedFluxValue
from jetfit.core.values.flux_value import SpectralFluxValue
from jetfit.core.values.flux_value import FluxValue
from jetfit.core.values.time_value import TimeValue


class Observation:
    """
    Time series of flux measurements.

    Attributes
    ----------
    times : list of `TimeValue`
        List of times associated with a flux measurement.

    fluxes : list of `FluxValue`
        List of `SpectralFluxValue` and `IntegratedFluxValue` objects.
    """
    def __init__(self, times: list[TimeValue], fluxes: list):
        """
        Initializes the instance using a path to an observation file.

        Notes
        -----
        The CSV file requires the following columns:

        - `Time` : Time of the flux measurement.
        - `TimeUnits` : One of `jetfit.core.enums.TimeUnits`
        - `Flux` : Value of the flux measurement.
        - `FluxUnits` : One of `jetfit.core.enums.FluxUnits`
        - `FluxTypes` : One of `jetfit.core.enums.FluxTypes`
        - `FluxLowerErrors` : Lower error of the flux measurement.
        - `FluxUpperErrors` : Upper error of the flux measurement.

        The CSV file conditionally contains the following columns:

        - `Frequency` : Center frequency if `FluxType` is `spectral`.
        - `FluxLower` : Lower frequency if `FluxType` is `integrated`.
        - `FluxUpper` : Upper frequency if `FluxType` is `integrated`.

        If times, fluxes, and csv_path are all provided, then the values in
        the CSV will overwrite the provided lists.
        """
        self.times = times if times else []
        self.fluxes = fluxes if fluxes else []

    @classmethod
    def from_csv(cls, path: str | Path):
        """
        Populates `times` and `fluxes` from a CSV file.

        Parameters
        ----------
        path : str or Path, optional
            Path to a CSV file containing observation info.

        Raises
        ------
        ValueError
            Parsed an unsupported flux type.
        """
        times: list[TimeValue] = []
        fluxes: list[FluxValue] = []

        for row in csv.read(path).itertuples():
            times.append(TimeValue.from_csv_row(row))

            if row.FluxType.lower() == FluxType.INTEGRATED.value:
                fluxes.append(IntegratedFluxValue.from_csv_row(row))

            elif row.FluxType.lower() == FluxType.SPECTRAL.value:
                fluxes.append(SpectralFluxValue.from_csv_row(row))

            else:
                raise ValueError(f'Unsupported flux type: {row.flux_type}. '
                                 f'Supported flux types include '
                                 f'{[f.value for f in FluxType]}.')

        return cls(times, fluxes)

    def numpyify(self):
        """ Converts the time and flux value objects into numpy arrays. """
        pass
