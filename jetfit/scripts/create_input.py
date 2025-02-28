import csv

import astropy.units as u
from synphot import SpectralElement

from jetfit.core.defns.band import Band
from jetfit.core.utils.csv_utils import CSVReader


# m_AB - m_Vega = SYS_OFFSET[filter_name]
SYS_OFFSET = {
    'U': 0.79, 'B': -0.09, 'V': 0.02, 'R': 0.21, 'I': 0.45,
    'J': 0.91, 'H': 1.39, 'K': 1.85, 'Ks': 1.85, 'Ic': 0.45,
    'Rc': 0.21,

    'uprime': 0.91, 'gprime': -0.08, 'rprime': 0.16,
    'iprime': 0.37, 'zprime': 0.54
}


# Effective wavelengths in Angstrom
EFF_WL = {
    'U': SpectralElement.from_filter('johnson_u').pivot(),
    'B': SpectralElement.from_filter('johnson_b').pivot(),
    'V': SpectralElement.from_filter('johnson_v').pivot(),
    'R': SpectralElement.from_filter('johnson_r').pivot(),
    'I': SpectralElement.from_filter('johnson_i').pivot(),
    'J': SpectralElement.from_filter('bessel_j').pivot(),
    'H': SpectralElement.from_filter('bessel_h').pivot(),
    'K': SpectralElement.from_filter('bessel_k').pivot(),
    'Ks': SpectralElement.from_filter('bessel_k').pivot(),
    'Rc': SpectralElement.from_filter('cousins_r').pivot(),
    'Ic': SpectralElement.from_filter('cousins_i').pivot()
}


def main(input_path: str, output_path: str,  xrt_path: str = None) -> None:
    """"""
    input_csv = CSVReader(input_path, live_dangerously=True)
    xrt_csv = CSVReader(xrt_path, live_dangerously=True) if xrt_path else None

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        write_headers(writer)

        # Convert the optical/NIR CSV
        for row in input_csv.rows():
            # read the time
            time = u.Quantity(row.Time, unit=row.TimeUnits)

            # convert mag to flux
            flux, flux_err = mag_to_flux(row.Mag, row.MagError, row.Filter, row.MagSys)

            # get frequency of filter
            frequency = filter_to_frequency(row.Filter)

            # write values to csv
            writer.writerow(
                [
                    # [Time, TimeLower, TimeUpper, TimeUnit]
                    time.to_value('s'), None, None, 's',

                    # [Value, ValueLower, ValueUpper, ValueUnit, ValueType]
                    flux.value, flux_err.value, flux_err.value, flux.unit, 'Spectral Flux',

                    # [Wave, WaveLower, WaveUpper, WaveUnit]
                    frequency.value, None, None, frequency.unit
                 ]
            )

        # Convert the XRT CSV
        if xrt_csv:
            for row in xrt_csv.rows():
                # write values to csv
                writer.writerow(
                    [
                        # [Time, TimeLower, TimeUpper, TimeUnit]
                        row.Times, None, None, 's',

                        # [Value, ValueLower, ValueUpper, ValueUnit, ValueType]
                        row.Fluxes, row.FluxErrs, row.FluxErrs, 'erg cm-2 s-1', 'Integrated Flux',

                        # [Wave, WaveLower, WaveUpper, WaveUnit]
                        None, 7.25E+16, 2.42E+18, 'Hz'
                     ]
                )


def write_headers(writer) -> None:
    """ Writes the headers to the CSV. """
    writer.writerow(
        (
            'Time', 'TimeLower', 'TimeUpper', 'TimeUnits', 'Value',
            'ValueLower', 'ValueUpper', 'ValueUnits', 'ValueType',
            'Wave', 'WaveLower', 'WaveUpper', 'WaveUnits'
        )
    )


def filter_to_frequency(dfilter: str) -> u.Quantity:
    """
    Maps a filter name to a frequency.

    Parameters
    ----------
    dfilter : str
        The filter name.

    Returns
    -------
    float
        The effective frequency of the filer.
    """
    try:
        f = EFF_WL[dfilter].to('Hz', equivalencies=u.spectral())
    except KeyError:
        f = u.Quantity(Band.from_name(dfilter).center, unit='Hz')

    return f

def mag_to_flux(mag, mag_error, dfilter, system):
    """
    Converts a magnitude and magnitude error to a flux.

    Parameters
    ----------
    mag : float
        The magnitude (AB or Vega).

    mag_error : float
        The uncertainty of the magnitude.

    dfilter : str
        The filter name.

    system : str, {'vega', 'ab'}

    Returns
    -------
    tuple of u.Quantity
        The flux and flux error in mJy.
    """
    # Convert all mags to AB system
    if system.lower() == 'vega':
        mag = mag + SYS_OFFSET[dfilter]

    # Convert to flux
    flux = (mag * u.ABmag).to('mJy')
    flux_error = abs(((mag + mag_error) * u.ABmag).to('mJy') - flux)

    # return flux and flux uncertainty
    return flux, flux_error


if __name__ == '__main__':

    event = '090424'

    args = {
        'input_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_in.csv",

        'xrt_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_xrt.csv",

        'output_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_out.csv",
    }

    main(**args)
