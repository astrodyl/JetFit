import json
from collections import defaultdict
from dust_extinction.parameter_averages import CCM89
from matplotlib import pyplot as plt
import numpy as np
import astropy.units as u

from jetfit.ampy import Ampy
from jetfit.core.utils import save_plot_unique
from jetfit.models.fireball import FireballModel, StratifiedFireballModel
from scripts.plot.base import OPTION_MAP
from scripts.plot.visualize import LightCurvePlot


import scienceplots
plt.style.use(['science'])


def flatten(v):
    """ Flatten v. """
    s = list(v.shape[1:])
    s[0] = np.prod(v.shape[:2])
    return v.reshape(s)


def sample_light_curve(samples, ampy, lcg, times):
    """"""
    modeled = defaultdict(list)

    for i, sample in enumerate(samples):
        params = ampy.mcmc.params.samples_to_dict(sample)

        # Generate the spectral flux
        for ds in lcg.get_spectral_data():
            sflux = lcg.model_spectral_flux(
                np.atleast_1d(ds), params, times, ext_model=CCM89(Rv=3.1)
            )
            modeled[ds.band].append(sflux)

        # Generate the integrated flux
        for di in lcg.get_integrated_data():
            iflux = lcg.model_integrated_flux(np.atleast_1d(di), params, times)

            # Temporary: Force conversion to mJy
            iflux_q = u.Quantity(iflux, unit=lcg.observation.as_arrays.if_units)
            modeled[di.band].append((iflux_q / di.int_range.width).to_value('mJy'))

    return modeled


def pretty_plot(model, chain, best_path, params_path, obs_path, out_dir, event):
    """"""
    ampy = Ampy(obs_path, params_path)

    with open(best_path, 'r') as f:
        best = json.load(f)

    # Load in the file with the spacing values
    with open(r"C:\Projects\repos\JetFit\scripts\events\spacing.json", "r") as f:
        scales = json.load(f)[event]

    lcg = LightCurvePlot(model, best, ampy.obs)

    # Get the times to plot
    ranges = ampy.obs.epoch(ampy.obs.flux_loc)
    times = np.geomspace(ranges[0] / 2, ranges[1] * 2, num=200)
    # times = np.geomspace(9e-4, 10, num=200)

    # Generate LCs from randomly samples sets
    samples = chain[np.random.randint(len(chain), size=100)]
    modeled = sample_light_curve(samples, ampy, lcg, times)

    # Plot the best fitting LC
    best_modeled = lcg.model_fluxes(best, times, ext_model=CCM89(Rv=3.1))

    for band, fluxes in best_modeled.items():
        lcg.ax.loglog(times, fluxes * scales[band], color=OPTION_MAP[band]['color'])

    # Plot the distribution of LCs
    for band, fluxes in modeled.items():

        # Get the 1-sigmas
        p16, _, p84 = np.percentile(fluxes, [16, 50, 84], axis=0)

        lcg.ax.fill_between(
            times, p16 * scales[band], p84 * scales[band],
            color=OPTION_MAP[band]['color'], alpha=0.2
        )

    lcg.plot_observation(best, spreads=scales, offset=True, excluded=True)
    lcg.ax.set_xlim(times.min(), times.max())
    # lcg.ax.set_xlim(8e-4, times.max())
    # lcg.ax.set_ylim(8e-7)
    lcg.ax.set_ylabel('Arbitrarily Scaled Flux Density')
    save_plot_unique('light_curve_cal', 'pdf', str(out_dir))


if __name__ == '__main__':

    import astropy.units as u
    import numpy as np

    s_event = '220101A'
    p_best = rf"C:\Server\FINAL\analytic\{s_event}\minimized\minimized.json"
    p_obs = rf"C:\Projects\repos\JetFit\jetfit\resources\grbs\{s_event}\{s_event}.csv"
    p_params = rf"C:\Projects\repos\JetFit\jetfit\resources\grbs\{s_event}\parameters.toml"
    p_chain = rf"C:\Server\FINAL\analytic\{s_event}\chain.npz"

    pretty_plot(**{
        'model': FireballModel,

        'chain': flatten(np.load(p_chain)['chain']),

        'best_path': p_best,

        'obs_path': p_obs,

        'out_dir': rf"C:\Server\FINAL\analytic\{s_event}\paper",

        'params_path': p_params,

        'event': s_event
    })
