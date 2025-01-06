import numpy as np
from pathlib import Path

from jetfit.core.defns.band import Band
from jetfit.core.defns.enums import FluxType, IndexType, TimeUnits
from jetfit.core.utilities.exceptions import EvidenceTypeError
from jetfit.core.utilities.io.csv import CSVReader
from jetfit.core.values.flux import IntegratedFluxValue
from jetfit.core.values.flux import SpectralFluxValue
from jetfit.core.values.index import SpectralIndexValue
from jetfit.core.values.time import TimeValue


class Measurement:
    """
    ???
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Evidence:
    """
    Evidence for MCMC analysis.

    Values are additionally stored as `optimized` arrays. These arrays are
    useful for when calculations need to be fast. For example, in MCMC
    fitting. Note that the optimized arrays don't necessarily have the
    same length.

    To prevent additional queries to the hydrodynamic simulation grid when
    interpolating the spectral functions, the lower and upper time bounds
    for spectral index measurements are stored in ``optimized_x`` but only
    a single index value is stored in the ``optimized_y`` array suc that
    len(optimized_x) == len(optimized_y) + # of spectral index values.

    Although non-intuitive, this decreases the time required to calculate the
    log posterior by nearly a factor of 2.

    Attributes
    ----------
    values : list of `Measurement`
        ????
    """
    def __init__(self, evidence: list[Measurement]):
        self.values = evidence

        self.optimized_x = []
        self.optimized_y = []
        self.optimized_err = []
        self._set_optimized_arrays()

    def _set_optimized_arrays(self) -> None:
        """ Creates and sets the optimized arrays from ``values``. """
        x, y, err = [], [], []

        for v in self.values:
            y.append(v.y.value)
            err.append(v.y.avg_error)

            if v.y.type == IndexType.SPECTRAL:
                x.append(v.y.start.value)
                x.append(v.y.stop.value)

            elif v.y.type in FluxType:
                x.append(v.x.value)

        self.optimized_x = np.asarray(x)
        self.optimized_y = np.asarray(y)
        self.optimized_err = np.asarray(err)

    @classmethod
    def from_csv(cls, path: str | Path):
        """
        Populates `times` and `values` from a CSV file.

        Parameters
        ----------
        path : str or Path, optional
            Path to a CSV file containing evidence for the event.

        Notes
        -----
        The CSV file requires the following columns:

        - `Time` : Time of the flux measurement.
        - `TimeLower` : Time of the flux measurement.
        - `TimeUpper` : Time of the flux measurement.
        - `TimeUnits` : One of `jetfit.core.enums.TimeUnits`

        - `Value` : Value of the flux measurement.
        - `ValueLowerErrors` : Lower error of the flux measurement.
        - `ValueUpperErrors` : Upper error of the flux measurement.
        - `ValueUnits` : One of `jetfit.core.enums.FluxUnits`
        - `ValueType` : One of `jetfit.core.enums.FluxUnits`
        - `Model` : One of `jetfit.core.enums.FluxTypes`

        The CSV file conditionally contains the following columns:

        - `Frequency` : Center frequency if `FluxType` is `spectral`.
        - `FrequencyLower` : Lower frequency if `FluxType` is `integrated`.
        - `FrequencyUpper` : Upper frequency if `FluxType` is `integrated`.
        - `FrequencyUnits` : Upper frequency if `FluxType` is `integrated`.

        Raises
        ------
        ValueError
            Parsed an unsupported flux type.
        """
        csv = CSVReader(path)

        measurements = []
        for row in csv.rows():
            time = TimeValue.from_csv_row(row, TimeUnits.SEC)

            match row.ValueType.lower():
                case FluxType.INTEGRATED.value:
                    value = IntegratedFluxValue.from_csv_row(row)

                case FluxType.SPECTRAL.value:
                    value = SpectralFluxValue.from_csv_row(row)

                case IndexType.SPECTRAL.value:
                    value = SpectralIndexValue.from_csv_row(row)

                case _:
                    raise EvidenceTypeError(row)

            measurements.append(Measurement(time, value))

        return cls(measurements)

    def filter(self, allowed_types: list | set):
        """
        Returns a new Evidence instance containing only values of the
        specified types.

        Parameters
        ----------
        allowed_types : list or set
            The types to include in the new container.

        Returns
        -------
            Container: A new instance of Container with filtered objects.
        """
        return Evidence(
            [v for v in self.values if v.y.type in allowed_types]
        )

    def get_bands(self) -> list:
        """

        Returns
        -------
        list
        """
        bands = []

        for value in self.values:

            if value.y.type not in FluxType:
                continue

            for band in bands:
                if band.encompasses(value.y.frequency):
                    band.flux.append(value.y)
                    band.times.append(value.x)
                    break
            else:
                bands.append(Band.from_measurement(value))

        return bands
