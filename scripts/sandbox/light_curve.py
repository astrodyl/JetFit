import json
from collections import defaultdict
from dust_extinction.parameter_averages import CCM89
from matplotlib import pyplot as plt

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


def pretty_plot(model, best_path, params_path, obs_path, out_dir, event):
    """"""
    ampy = Ampy(obs_path, params_path)

    with open(best_path, 'r') as f:
        best = json.load(f)

    chain = flatten(np.load(rf"C:\Server\FINAL\analytic\221009A\chain.npz")['chain'])
    # chain = chain[chain[:, 6] > 1.0]

    lcg = LightCurvePlot(model, best, ampy.obs)

    # Get the times to plot
    ranges = ampy.obs.epoch(ampy.obs.flux_loc)
    times = np.geomspace(ranges[0] / 2, ranges[1] * 2, num=200)
    # times = np.geomspace(9e-4, 10, num=200)

    samples = chain[np.random.randint(len(chain), size=100)]
    modeled = sample_light_curve(samples, ampy, lcg, times)

    options = {
        "130612A" : dict(xray=50, V=1, R=3, I=6),
        "090618"  : dict(xray=1, B=1, V=2, R=4, i=6),
        "131030A" : dict(xray=1, B=1, r=2, R=6, i=20, z=40, J=80, H=180, S=100),
        "140506A" : {"xray":1, "uvw2":3, "uvm2":6, "uvw1":12, "uvot-u":24, "uvot-b":60, 'g': 80, "uvot-v":180, 'r':360, 'R':800, 'i':2000, 'z':4000, 'J':6000, 'H':10_000, 'K':20_000},
        "050525A" : {"xray":1, "uvw2":1, "uvm2":6, "uvw1":18, "uvot-u":60, "uvot-b":200, "uvot-v":600, 'V':2000, 'R':4000, 'I':10_000, 'J':40_000, 'H':80_000},
        "160131A" : {"xray":10, "uvw2":1, "uvm2":1, "uvw1":2, "uvot-u":3, "uvot-b":5, 'g': 8, "uvot-v":16, 'r':40, 'i':80, 'z':200},
        "171010A" : {"xray":1, 'g':1, 'r':2, 'i':4, 'z':8},
        "111228A" : {"xray":1, "uvw2":1, "uvm2":3, "uvw1":8, "uvot-u":20, 'B':40, "uvot-b":100, 'g': 200, "uvot-v":400, 'V':1_000, 'r':2_000, 'R':4000, 'i':8000, 'z':15_000, 'J':20_000, 'H':30_000, 'K':40_000},
        "080413B" : {"xray":1, "uvm2":1, "uvw1":5, "uvot-u":10, "uvot-b":40, 'g': 100, "uvot-v":300, 'r':1_000, 'R':3000, 'i':6000, 'I':12000, 'z':3e4, 'J':1e5, 'H':5e5, 'K':1e6},
        "210905A" : {"xray":1, 'i':1, 'Ic':3, 'z':6, 'J':12, 'H':24, 'K':60},
        "220101A" : {"xray":1, 'r': 10, 'R':20, 'i':40, 'F775W': 80, 'I': 160, 'z':400, 'J':800, 'F125W': 2000, 'H':4000, 'K':10_000},
        "221009A" : {"xray":1, "uvot-u":10, "uvot-b":40, 'g': 100, "uvot-v":300, 'r': 10, 'i':40,  'z':400, 'Ka': 1, 'Kb': 1, 'Kc': 1, 'Kd': 1},
    }
    scales = options[event]

    # BEST FIT PLOTTING
    best_modeled = lcg.model_fluxes(best, times, ext_model=CCM89(Rv=3.1))

    for band, fluxes in best_modeled.items():
        lcg.ax.loglog(times, fluxes * scales[band], color=OPTION_MAP[band]['color'])


    # DISTRIBUTION PLOTTING
    for band, fluxes in modeled.items():
        lc = np.stack(fluxes, axis=0)

        p16, p50, p84 = np.percentile(lc, [16, 50, 84], axis=0)

        lcg.ax.fill_between(
            times, p16 * scales[band], p84 * scales[band],
            color=OPTION_MAP[band]['color'], alpha=0.2
        )

    lcg.plot_observation(best, spreads=scales, offset=True, excluded=True)
    lcg.ax.set_xlim(times.min(), times.max())
    save_plot_unique('light_curve_cal', 'pdf', str(out_dir))


def main(model, best_path, obs_path, params_path, out_dir, event):
    """"""
    pretty_plot(model, best_path, params_path, obs_path, out_dir, event)


if __name__ == '__main__':

    import astropy.units as u
    import numpy as np

    s_event = '221009A'
    p_best = rf"C:\Server\FINAL\analytic\{s_event}\minimized\minimized.json"
    p_obs = rf"C:\Projects\repos\JetFit\jetfit\resources\grbs\{s_event}\{s_event}.csv"
    p_params = rf"C:\Projects\repos\JetFit\jetfit\resources\grbs\{s_event}\parameters.toml"

    main(**{
        'model': FireballModel,

        'best_path': p_best,

        'obs_path': p_obs,

        'out_dir': rf"C:\Server\FINAL\analytic\{s_event}\paper",

        'params_path': p_params,

        'event': s_event
    })
