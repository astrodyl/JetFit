import json
import astropy.units as u
import numpy as np
from dust_extinction.parameter_averages import CCM89
from matplotlib import pyplot as plt
from synphot import SpectralElement

from jetfit.core.defns.enums import DataType
from jetfit.core.input import Observation
from jetfit.core.utils import nav_utils
from jetfit.models2.basemodels import SpectralFluxModel
from jetfit.models2.fireball import FireballModel
from jetfit.plot.light_curve import LightCurve

if __name__ == "__main__":
    # Time to evaluate spectral plot
    time = u.Quantity(value=300, unit='s')

    # Input paths
    event = '050922C'
    observation_path = nav_utils.get_input_csv_path('new', event)
    # best_params_path = rf"C:\Projects\skynet\JetFit\jetfit\results\{event}_1\best_fit.json"
    best_params_path = r"C:\skynet-server\PARI\jetfit_3\050922C\best_fit.json"

    # Read in observation data
    obs = Observation.from_csv(observation_path)

    # Get flux locations
    obs_spectral_flux = obs.data[obs.spectral_flux_loc]
    obs_integrated_flux = obs.data[obs.integrated_flux_loc]

    # Get optical observation frequencies
    obs_frequencies = []
    for i, d in enumerate(obs.data):
        if d.type == DataType.SPECTRAL_FLUX:
            obs_frequencies.append(d.frequency)
    obs_frequencies = np.array(obs_frequencies, dtype=object)

    # Read in the model parameters
    with open(best_params_path, "r") as jf:
        params = json.load(jf)

    # Define the model
    model = FireballModel(**params)

    # Define the spectral characteristic values
    f_peak, nu_c, nu_m = (
        model.f_peak(time), model.nu_c(time), model.nu_m(time)
    )

    # Define the flux models
    spectral_model = SpectralFluxModel(nu_m, nu_c, f_peak, model.p)

    # Generate flux in log space
    frequencies = np.logspace(
        np.log10(obs_frequencies.min().to_value('Hz')),
        np.log10(obs_frequencies.max().to_value('Hz')),
        num=500
    )

    # Evaluate the spectrum at all frequencies
    modeled_spectral_flux, modeled_spectral_segments = [], []
    for f in frequencies:
        modeled_spectral_segments.append(SpectralFluxModel(nu_m, nu_c, f_peak, model.p).segment(f).name)
        modeled_spectral_flux.append(spectral_model.evaluate(f))
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
            print('Break at index:', f'{i}.', f'{pre} -> {post}', '\n')
            # break
    # else:
    #     print('Segment.................', modeled_spectral_segments[0], '\n')

    # Plot flux vs. frequency
    src_frequencies = (1 + model.z) * frequencies * u.Hz
    plt.loglog(frequencies, modeled_spectral_flux, '--', linewidth=1.5, label=f'unextinguished (z = 0)')
    plt.loglog(frequencies, modeled_spectral_flux * CCM89(3.1).extinguish(src_frequencies, Ebv=model.ebv_mw), '--', linewidth=1.5, label=f'MW Corrected (z = 0)')
    plt.loglog(frequencies, modeled_spectral_flux * CCM89(3.1).extinguish(src_frequencies, Ebv=model.ebv_sf) * CCM89(3.1).extinguish(src_frequencies, Ebv=model.ebv_mw), '--', linewidth=1.5, label=f'MW + SF Corrected (z = {model.z})')
    plt.axvline(x=3.74e14, color='indigo', linestyle=':', alpha=0.3)
    plt.axvline(x=4.56e14, color='red', linestyle=':', alpha=0.3)
    plt.axvline(x=5.44e14, color='green', linestyle=':', alpha=0.3)
    plt.title(f'Synchrotron Spectrum at t = {time.value} {time.unit}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Flux Density (mJy)')
    plt.legend(loc='best')
    # plt.show()

    # Transition to Light Curve Analysis Beyond this Point
    print('[ LIGHT CURVE INFORMATION ]')
    modeled_times = np.logspace(
        np.log10(obs.time_array[obs.flux_loc].min() / 86400.0),
        np.log10(obs.time_array[obs.flux_loc].max() / 86400.0),
        num=200
    )
    segments = np.full(len(modeled_times), '', dtype=str)

    # Get all the modeled segment names
    for i, t in enumerate(modeled_times):
        segments[i] = SpectralFluxModel(
            model.nu_m(t), model.nu_c(t), model.f_peak(t), model.p
        ).segment(obs_frequencies[0].value).name

    # Determine if there is a break in the light curve
    for i in range(1, len(segments)):
        pre = segments[i - 1]
        post = segments[i]

        if pre != post:
            print('Break at index, time:', f'{i}, {int(modeled_times[i] * 86_400)}s.', f'{pre} -> {post}', '\n')
    # else:
    #     print('Segment.................', segments[0])

    # Plot light curve with extinction
    lc = LightCurve(FireballModel, params, obs, title='Light Curve')
    plt.axvline(x=297, color='green', linestyle=':', alpha=0.4)
    lc.plot(show=True)

    # Plot light curve with MW extinction only
    # params['ebv_sf'] = None
    # lc = LightCurve(FireballModel, params, obs, title='Light Curve Without EBV_SF')
    # lc.plot(show=True)

    # Plot light curve with no extinction
    # params['ebv_mw'] = None
    # lc = LightCurve(FireballModel, params, obs, title='Light Curve Without Extinction')
    # lc.plot(show=True)

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
