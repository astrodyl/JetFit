from pathlib import Path

from jetfit.core.defns.enums import FluxType
from jetfit.core.utilities.io import csv
from jetfit.core.values.flux import IntegratedFluxValue
from jetfit.core.values.flux import SpectralFluxValue
from jetfit.core.values.flux import FluxValue
from jetfit.core.values.time import TimeValue


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

        Raises
        ------
        ValueError
            Parsed an unsupported flux type.
        """
        times: list[TimeValue] = []
        fluxes: list[FluxValue] = []

        for row in csv.read(path).itertuples():
            times.append(TimeValue.from_csv_row(row))

            if row.ValueType.lower() == FluxType.INTEGRATED.value:
                fluxes.append(IntegratedFluxValue.from_csv_row(row))

            elif row.ValueType.lower() == FluxType.SPECTRAL.value:
                fluxes.append(SpectralFluxValue.from_csv_row(row))

            else:
                raise ValueError(f'Unsupported flux type: {row.ValueType}. '
                                 f'Supported flux types include '
                                 f'{[f.value for f in FluxType]}.')

        return cls(times, fluxes)
