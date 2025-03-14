import json
import astropy.units as u
import numpy as np
from dust_extinction.parameter_averages import CCM89
from matplotlib import pyplot as plt

from jetfit.core.defns.enums import DataType
from jetfit.core.input import Observation
from jetfit.core.utils import nav_utils
from jetfit.models2.basemodels import SpectralFluxModel
from jetfit.models2.fireball import FireballModel
from jetfit.plot.light_curve import LightCurve


def sec_to_days(x):
    """ Used for plotting axes. """
    return x / 86400


def days_to_sec(x):
    """ Used for plotting axes. """
    return x * 86400


def plot_critical_frequencies(nu_ms, nu_cs, times) -> None:
    """
    Plots the critical frequencies as a function of time.

    Useful for visualizing if they cross.

    Parameters
    ----------
    nu_ms : np.ndarray of float
        The synchrotron frequencies.

    nu_cs : np.ndarray of float
        The cooling frequencies.

    times: np.ndarray
        The times for the frequencies.
    """
    _, ax = plt.subplots()

    ax.loglog(times, nu_ms, color='blue', label=r'$\nu_{m}$')
    ax.loglog(times, nu_cs, color='orange', label=r'$\nu_{c}$')
    ax.set_title('Critical Frequencies')
    ax.set_xlabel('Time Since Trigger (days)')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend(loc='best')
    ax.grid(alpha=0.5)

    ax2 = ax.secondary_xaxis('top', functions=(days_to_sec, sec_to_days))
    ax2.set_xlabel("Time Since Trigger (seconds)")

    plt.show()


def plot_spectrum(
        frequencies, flux, time, redshift, ebv_mw, ebv_sf
) -> None:
    """
    Plots the GRB spectrum.

    Parameters
    ----------
    frequencies : np.ndarray
        The frequencies spanning the spectrum.

    flux : np.ndarray
        The flux values at each frequency.

    time : float
        The time the spectrum is evaluated.

    redshift : float
        The redshift of the event.

    ebv_mw : float
        The EBV for milky.

    ebv_sf : float
        The EBV for the source frame.
    """
    if isinstance(time, u.Quantity):
        time_val  = time.value
        time_unit = time.unit
    else:
        time_val  = time
        time_unit = 's'

    # Calculate source frequencies
    src_frequencies = (1 + redshift) * frequencies * u.Hz

    # Plot the un-extinguished spectrum with at z = 0
    plt.loglog(
        frequencies, flux,
        '--', linewidth=1.5, label=f'unextinguished (z = 0)'
    )

    # Plot the extinguished spectrum with EBV (milky way, z = 0) and (source frame, z = redshift).
    fully_extinguished_flux = flux * (
        CCM89(3.1).extinguish(src_frequencies, Ebv=ebv_sf) *
        CCM89(3.1).extinguish(src_frequencies, Ebv=ebv_mw)
    )
    plt.loglog(
        frequencies, fully_extinguished_flux,
        '--', linewidth=1.5, label=f'MW + SF Corrected (z = {redshift})'
    )

    # Plot vertical lines corresponding to filter frequencies
    plt.axvline(x=5.44e14, color='green',  linestyle=':', alpha=0.3)  # V
    plt.axvline(x=4.56e14, color='red',    linestyle=':', alpha=0.3)  # R
    plt.axvline(x=3.74e14, color='indigo', linestyle=':', alpha=0.3)  # I

    plt.title(f'Synchrotron Spectrum at t = {time_val} {time_unit}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Flux Density (mJy)')
    plt.legend(loc='best')
    plt.show()


def main(event, obs, model_params, time):
    """"""

    # Get optical observation frequencies
    obs_frequencies = []
    for i, d in enumerate(obs.data):
        if d.type == DataType.SPECTRAL_FLUX:
            obs_frequencies.append(d.frequency)
    obs_frequencies = np.array(obs_frequencies, dtype=object)

    # Define the model
    model = FireballModel(**model_params)

    # Define the spectral characteristic values
    f_peak, nu_c, nu_m = (
        model.f_peak(time), model.nu_c(time), model.nu_m(time)
    )

    # Define the flux models
    spectral_model = SpectralFluxModel(nu_m, nu_c, f_peak, model.p, model.k)

    # Generate flux in log space
    frequencies = np.logspace(
        np.log10(obs_frequencies.min().to_value('Hz')),
        np.log10(obs_frequencies.max().to_value('Hz')),
        num=500
    )

    # Evaluate the spectrum at all frequencies
    modeled_spectral_flux, modeled_spectral_segments = [], []
    for f in frequencies:
        modeled_spectral_segments.append(SpectralFluxModel(nu_m, nu_c, f_peak, model.p, model.k).segment(f).name)
        modeled_spectral_flux.append(spectral_model.evaluate_smooth(f))
    modeled_spectral_flux = np.array(modeled_spectral_flux)

    # Print out useful information
    print('[ SPECTRAL INFORMATION ]')
    slope, _ = np.polyfit(np.log10(frequencies), np.log10(modeled_spectral_flux), 1)
    print('Peak Flux...............', round(f_peak, 3), 'mJy')
    print('Synchrotron Frequency...', round(nu_m, 3), 'Hz')
    print('Cooling Frequency.......', round(nu_c, 3), 'Hz')
    print('Spectral Index, b.......', round(slope, 3))
    print('Electron Index, p.......', round(model.p, 3))
    print('Regime..................', 'slow' if nu_m < nu_c else 'fast')

    # Determine if there is a break in the spectral plot
    spectral_break = False
    for i in range(1, len(modeled_spectral_segments)):
        pre = modeled_spectral_segments[i - 1]
        post = modeled_spectral_segments[i]

        if pre != post:
            spectral_break = True
            print('Break at index:', f'{i}.', f'{pre} -> {post}', '\n')

    if not spectral_break:
        print('Segment.................', modeled_spectral_segments[0], '\n')

    # Plot flux vs. frequency
    plot_spectrum(
        frequencies, modeled_spectral_flux, time,
        model.z, model.ebv_mw, model.ebv_sf
    )

    # Transition to Light Curve Analysis Beyond this Point
    print('[ LIGHT CURVE INFORMATION ]')
    modeled_times = np.logspace(
        np.log10(obs.time_array[obs.flux_loc].min() / 86400.0),
        np.log10(obs.time_array[obs.flux_loc].max() / 86400.0),
        num=200
    )

    # Plot critical frequencies
    plot_critical_frequencies(
        nu_ms=model.nu_m(modeled_times),
        nu_cs=model.nu_c(modeled_times),
        times=modeled_times
    )

    segments = np.full(len(modeled_times), '', dtype=str)

    # Get all the modeled segment names
    for i, t in enumerate(modeled_times):
        segments[i] = SpectralFluxModel(
            model.nu_m(t), model.nu_c(t), model.f_peak(t), model.p, model.k
        ).segment(obs_frequencies[0].value).name

    # Determine if there is a break in the light curve
    spectral_break = False
    for i in range(1, len(segments)):
        pre = segments[i - 1]
        post = segments[i]

        if pre != post:
            spectral_break = True
            print('Break at index, time:', f'{i}, {int(modeled_times[i] * 86_400)}s.', f'{pre} -> {post}', '\n')
    if not spectral_break:
        print('Segment.................', segments[0])

    # Plot light curve with extinction
    lc = LightCurve(FireballModel, model_params, obs, title='Light Curve')
    lc.plot(show=True)

    # Plot the CCM Extinction curve
    obs_wn = np.unique(obs.wave_number_array[obs.spectral_flux_loc])
    src_wn = (1 + model.z) * obs_wn

    curve_mw = CCM89(Rv=3.1)(obs_wn)
    curve_sf = CCM89(Rv=3.1)(src_wn)

    _, ax = plt.subplots()
    ax.plot(src_wn, curve_sf, '--', label='R(V) = ' + str(3.1))

    # Set labels
    ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
    ax.set_ylabel(r'$A(x)/A(V)$')

    ax.set_title('CCM Extinction Curves')
    ax.legend(loc='best')
    plt.show()


if __name__ == "__main__":

    event_name = '050922C'

    # Paths to input files
    observation_path = nav_utils.get_input_csv_path('new', event_name)
    best_params_path = r"C:\skynet-server\PARI\jetfit_5\050922C\best_fit.json"

    # Read in the model parameters
    with open(best_params_path, "r") as jf:
        best_params = json.load(jf)

    main(**{
        # Name of the event
        'event': event_name,

        # Observation object used for modeling
        'obs': Observation.from_csv(observation_path),

        # Model parameters to evaluate
        'model_params': best_params,

        # Time to evaluate GRB spectrum
        'time': u.Quantity(value=300, unit='s')
    })
