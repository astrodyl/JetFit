import csv

import numpy as np

from jetfit.core.defns.band import Band
from jetfit.core.utilities.io.csv import CSVReader
from jetfit.core.values.time import TimeValue


ZERO_POINTS = {
    # Vega flux in mJy
    'U': 1_790_000,
    'B': 4_063_000,
    'V': 3_636_000,
    'R': 3_064_000,
    'I': 2_416_000,
    'J': 1_589_000,
    'H': 1_021_000,
    'K': 640_000,
    'Ks': 640_000,

    # AB flux in mJy
    'uprime': 3_631_000,
    'gprime': 3_631_000,
    'rprime': 3_631_000,
    'iprime': 3_631_000,
    'zprime': 3_631_000,
}


SYS_OFFSET = {
    # m_AB - m_Vega
    'U': 0.79,
    'B': -0.09,
    'V': 0.02,
    'R': 0.21,
    'I': 0.45,
    'J': 0.91,
    'H': 1.39,
    'K': 1.85,
    'Ks': 1.85,

    'uprime': 0.91,
    'gprime': -0.08,
    'rprime': 0.16,
    'iprime': 0.37,
    'zprime': 0.54,
}


def main(input_path: str,output_path: str,  xrt_path: str = None, ) -> None:
    """"""
    input_csv = CSVReader(input_path, live_dangerously=True)
    xrt_csv = CSVReader(xrt_path, live_dangerously=True) if xrt_path else None

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        write_headers(writer)

        # Convert the optical CSV
        for row in input_csv.rows():
            time = TimeValue(row.Time, row.TimeUnits)
            time.to_seconds()

            if row.Filter == 'J':
                print()

            flux, flux_error = mag_to_flux(row.Mag, row.MagError, row.MagSys, row.Filter)
            frequency = filter_to_frequency(row.Filter)

            writer.writerow(
                [time.value, None, None, time.units.value,
                 flux, flux_error, flux_error, 'mjy', 'Spectral',
                 'SpectralFlux', frequency, None, None,
                 'hz'
                 ]
            )

        # Convert the XRT CSV
        if xrt_csv:
            for row in xrt_csv.rows():
                writer.writerow(
                    [row.Times, None, None, 'seconds',
                     row.Fluxes, row.FluxErrs, row.FluxErrs, 'cgs', 'Integrated',
                     'IntegratedFlux', None, 7.25E+16, 2.42E+18,
                     'hz'
                     ]
                )


def write_headers(writer) -> None:
    """"""
    writer.writerow(
        ('Time', 'TimeLower', 'TimeUpper', 'TimeUnits', 'Value',
         'ValueLower', 'ValueUpper', 'ValueUnits', 'ValueType',
        'Model', 'Frequency', 'FrequencyLower', 'FrequencyUpper',
        'FrequencyUnits'
         )
    )


def mag_to_flux(m: float, e: float, s: str, f: str) -> tuple:
    """
    Converts a magnitude value to flux with units of mJy.

    Parameters
    ----------
    m : float
        The magnitude value.

    e : float
        The magnitude error.

    s : str
        The magnitude system name.

    f : str
        The filter name.

    Returns
    -------
    tuple of float
        The flux and flux error with units of mJy.

    Raises
    ------
    ValueError
        If the filter value is not a key of `ZERO_POINTS`.
    """
    f = f.strip()

    if f not in ZERO_POINTS:
        raise ValueError(f'Unsupported filter: {f}')

    # Convert mags to the corresponding zero point system
    if type(s) == str:
        if s.lower() == 'ab' and 'prime' not in f:
            m = m - SYS_OFFSET[f]

        elif s.lower() == 'vega' and 'prime' in f:
            m = m + SYS_OFFSET[f]

    flux = ZERO_POINTS[f] * 10 ** (-0.4 * m)
    flux_error = flux * 0.4 * np.log(10) * e

    return flux, flux_error


def filter_to_frequency(f: str) -> float:
    """
    Converts a filter name to a frequency value.

    Parameters
    ----------
    f : str
        The filter name.

    Returns
    -------
    float
        The average frequency value.
    """
    f = f.strip()

    return Band.from_name(f).center


if __name__ == '__main__':

    event = '080413B'

    args = {
        'input_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_in.csv",

        'xrt_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_xrt.csv",

        'output_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_out.csv",
    }

    main(**args)





























