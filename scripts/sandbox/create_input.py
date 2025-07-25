import csv

import numpy as np
import astropy.units as u
from dust_extinction.parameter_averages import CCM89
from synphot import SpectralElement

from jetfit.core.utils import CSVReader


# m_AB - m_Vega = SYS_OFFSET[filter_name]
SYS_OFFSET = {
    'U': 0.79, 'B': -0.09, 'V': 0.02, 'R': 0.21, 'I': 0.45,
    'J': 0.91, 'H': 1.39, 'K': 1.85, 'Ks': 1.85, 'Ic': 0.45,
    'Rc': 0.21,

    'uprime': 0.91, 'gprime': -0.08, 'rprime': 0.16,
    'iprime': 0.37, 'zprime': 0.54
}

# Aliases
SYS_OFFSET['g'] = SYS_OFFSET['gprime']
SYS_OFFSET['r'] = SYS_OFFSET['rprime']
SYS_OFFSET['i'] = SYS_OFFSET['iprime']
SYS_OFFSET['z'] = SYS_OFFSET['zprime']


# Converts from UVOT Vega to AB for UVOT data
# https://swift.gsfc.nasa.gov/analysis/uvot_digest/zeropts.html
UVOT_OFFSET = {
    'uvw2': 1.73,
    'uvm2': 1.69,
    'uvw1': 1.51,
    'uvot-u': 1.02,
    'uvot-b': -0.13,
    'uvot-v': -0.01,
}


# Effective wavelengths in Angstrom
EFF_WL = {
    'U': SpectralElement.from_filter('johnson_u').pivot(),
    'B': SpectralElement.from_filter('johnson_b').pivot(),
    'V': SpectralElement.from_filter('johnson_v').pivot(),
    'R': SpectralElement.from_filter('johnson_r').pivot(),  # 6899 AA
    'I': SpectralElement.from_filter('johnson_i').pivot(),
    'J': SpectralElement.from_filter('bessel_j').pivot(),
    'H': SpectralElement.from_filter('bessel_h').pivot(),
    'K': SpectralElement.from_filter('bessel_k').pivot(),
    'Rc': SpectralElement.from_filter('cousins_r').pivot(),
    'Ic': SpectralElement.from_filter('cousins_i').pivot(),

    # SDSS
    'u' : u.Quantity(3540.0, unit='AA'),
    'g' : u.Quantity(4770.0, unit='AA'),
    'r' : u.Quantity(6231.0, unit='AA'),
    'i' : u.Quantity(7625.0, unit='AA'),
    'z' : u.Quantity(9134.0, unit='AA'),

    # Swift-UVOT wavelengths
    'uvw2': u.Quantity(1928.0, unit='AA'),
    'uvm2': u.Quantity(2246.0, unit='AA'),
    'uvw1': u.Quantity(2600.0, unit='AA'),
    'uvot-u': u.Quantity(3465.0, unit='AA'),
    'uvot-b': u.Quantity(4392.0, unit='AA'),
    'uvot-v': u.Quantity(5468.0, unit='AA'),
}

# aliases
EFF_WL['Ks'] = EFF_WL['K']
EFF_WL['uprime'] = EFF_WL['u']
EFF_WL['gprime'] = EFF_WL['g']
EFF_WL['rprime'] = EFF_WL['r']
EFF_WL['iprime'] = EFF_WL['i']
EFF_WL['zprime'] = EFF_WL['z']
EFF_WL['uvot-uvw2'] = EFF_WL['uvw2']
EFF_WL['uvot-uvm2'] = EFF_WL['uvm2']
EFF_WL['uvot-uvw1'] = EFF_WL['uvw1']


def main(input_path: str, output_path: str,  xrt_path: str = None, before = None, after = None) -> None:
    """
    Creates a CSV for use with the AMPy.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file.

    output_path : str
        Path to save the output CSV file.

    xrt_path : str, optional
        Path to the XRT input CSV.

    before : u.Quantity['time'], optional
        Exclude data after `before` time.

    after : u.Quantity['time'], optional
        Exclude data before `after` time.
    """
    input_csv = CSVReader(input_path, sort=False)
    xrt_csv = CSVReader(xrt_path, sort=False) if xrt_path else None

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        write_headers(writer)

        # Convert the optical/NIR/UV CSV
        for row in input_csv.rows():
            dfilter = row.Filter.strip()

            # Read the time
            time = u.Quantity(row.Time, unit=row.TimeUnits)

            if before is not None and time > before:
                continue

            if after is not None and time < after:
                continue

            # If ebv is provided, we need to de-redden the data
            ebv = None
            if hasattr(row, 'Ebv') and not np.isnan(row.Ebv):
                ebv = row.Ebv

            # Get frequency of filter
            frequency = EFF_WL[dfilter].to('Hz', equivalencies=u.spectral())

            # Check if fluxes were provided already
            if hasattr(row, 'Flux') and not np.isnan(row.Flux):
                try:
                    flux = u.Quantity(row.Flux, unit=row.FluxUnit).to('mJy')
                    flux_err = u.Quantity(row.FluxError, unit=row.FluxUnit).to('mJy')
                except u.UnitConversionError as e:
                    flux = u.Quantity(row.Flux, unit=row.FluxUnit).to('mJy', equivalencies=u.spectral_density(frequency))
                    flux_err = u.Quantity(row.FluxError, unit=row.FluxUnit).to('mJy', equivalencies=u.spectral_density(frequency))


            else:
                flux, flux_err = mag_to_flux(row.Mag, row.MagError, dfilter, row.MagSys, ebv)

            # Format the CalOffset
            cal_offset = dfilter + '_offset'
            if hasattr(row, 'DataSource'):
                cal_offset = dfilter + str(row.DataSource) + '_offset'

            # write values to csv
            writer.writerow(
                [
                    # [Time, TimeLower, TimeUpper, TimeUnit]
                    time.to_value('s'), None, None, 's',

                    # [Value, ValueLower, ValueUpper, ValueUnit, ValueType]
                    flux.value, flux_err.value, flux_err.value, flux.unit, 'Spectral Flux',

                    # [Wave, WaveLower, WaveUpper, WaveUnit, Filter, CalOffset]
                    frequency.value, None, None, frequency.unit, dfilter, cal_offset
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
            'Wave', 'WaveLower', 'WaveUpper', 'WaveUnits', 'Filter', 'CalGroup'
        )
    )


def mag_to_flux(mag, mag_error, dfilter, system, ebv=None):
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

    ebv : float, optional

    Returns
    -------
    tuple of u.Quantity
        The flux and flux error in mJy.
    """
    # Convert all mags to AB system
    if system.lower().strip() == 'vega':
        mag += SYS_OFFSET[dfilter]

    elif system.lower().strip() == 'uvot':
        mag += UVOT_OFFSET[dfilter]

    # Convert to flux
    flux = (mag * u.ABmag).to('mJy')
    flux_error = abs(((mag + mag_error) * u.ABmag).to('mJy') - flux)

    # De-redden
    if ebv is not None:
        flux /= CCM89(Rv=3.1).extinguish(EFF_WL[dfilter], Ebv=ebv)

    # return flux and flux uncertainty
    return flux, flux_error


if __name__ == '__main__':

    event = '140506A'

    # mags = [24.71,24.24,23.66,23.13,24.27,24.43,23.7,23.71]
    # errors = [0.11,0.1,0.17,0.17,0.14,0.21,0.15,0.23]
    # filters = ['gprime','rprime','iprime','zprime','white','rprime','iprime','zprime',]
    #
    #
    # for mag, error, filts in zip(mags, errors, filters):
    #     f, fe = mag_to_flux(mag, error, filts, 'ab')
    #     print(round(f, 8), '\t', round(fe, 8))
    #
    # exit()

    args = {
        'input_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_in.csv",

        'xrt_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_xrt.csv",

        'output_path':
            rf"C:\Users\Dylan\Documents\GRB_DATA\{event}\{event}_out.csv",

        'before':  # Include data before this time
            None, #u.Quantity(10.0, unit='d'),

        'after':  # Include data after this time
            None, #u.Quantity(4000, unit='s')
    }

    main(**args)