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


# Converts from Vega to AB for UVOT data
# https://swift.gsfc.nasa.gov/analysis/uvot_digest/zeropts.html
UVOT_OFFSET = {
    'uvot-uvw2': 1.73,
    'uvot-uvm2': 1.69,
    'uvot-uvw1': 1.51,
    'uvot-u': 1.02,
    'uvot-b': -0.13,
    'uvot-v': -0.01,
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
    'Ic': SpectralElement.from_filter('cousins_i').pivot(),

    # Swift-UVOT wavelengths
    'uvot-uvw2': u.Quantity(1928.0, unit='AA'),
    'uvot-uvm2': u.Quantity(2246.0, unit='AA'),
    'uvot-uvw1': u.Quantity(2600.0, unit='AA'),
    'uvot-u': u.Quantity(3465.0, unit='AA'),
    'uvot-b': u.Quantity(4392.0, unit='AA'),
    'uvot-v': u.Quantity(5468.0, unit='AA'),
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
            dfilter = row.Filter.strip()

            # read the time
            time = u.Quantity(row.Time, unit=row.TimeUnits)

            # convert mag to flux
            flux, flux_err = mag_to_flux(row.Mag, row.MagError, dfilter, row.MagSys)

            # get frequency of filter
            frequency = filter_to_frequency(dfilter)

            # write values to csv
            writer.writerow(
                [
                    # [Time, TimeLower, TimeUpper, TimeUnit]
                    time.to_value('s'), None, None, 's',

                    # [Value, ValueLower, ValueUpper, ValueUnit, ValueType]
                    flux.value, flux_err.value, flux_err.value, flux.unit, 'Spectral Flux',

                    # [Wave, WaveLower, WaveUpper, WaveUnit, Filter]
                    frequency.value, None, None, frequency.unit, dfilter
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

                        # [Wave, WaveLower, WaveUpper, WaveUnit, Filter]
                        None, 7.25E+16, 2.42E+18, 'Hz', 'xray'
                     ]
                )


def write_headers(writer) -> None:
    """ Writes the headers to the CSV. """
    writer.writerow(
        (
            'Time', 'TimeLower', 'TimeUpper', 'TimeUnits', 'Value',
            'ValueLower', 'ValueUpper', 'ValueUnits', 'ValueType',
            'Wave', 'WaveLower', 'WaveUpper', 'WaveUnits', 'Filter'
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
        mag += SYS_OFFSET[dfilter]

    elif system.lower == 'uvot':
        mag += UVOT_OFFSET[dfilter]

    # Convert to flux
    flux = (mag * u.ABmag).to('mJy')
    flux_error = abs(((mag + mag_error) * u.ABmag).to('mJy') - flux)

    # return flux and flux uncertainty
    return flux, flux_error


if __name__ == '__main__':

    event = '210905A'

    # mags = [
    #     15.34,
    #     15.29,
    #     15.80,
    #     15.97,
    #     15.92,
    #     15.94,
    #     15.99,
    #     18.55,
    #     19.48,
    #     20.17,
    #     20.38,
    #     20.78,
    #     20.63,
    # ]
    #
    # mag_errs = [
    #     0.10,
    #     0.09,
    #     0.12,
    #     0.13,
    #     0.12,
    #     0.12,
    #     0.13,
    #     0.05,
    #     0.08,
    #     0.17,
    #     0.15,
    #     0.20,
    #     0.27,
    # ]
    #
    # for i, mag in enumerate(mags):
    #     flux11, flux_err11 = mag_to_flux(mag, mag_errs[i], 'uvot-v', 'uvot')
    #     print(round(flux11.value, 6), '\t', flux_err11.value)

    args = {
        'input_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_in.csv",

        'xrt_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_xrt.csv",

        'output_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_out.csv",
    }

    main(**args)
