import argparse
import json
import os.path
from pathlib import Path

import emcee
import numpy as np
from dust_extinction.parameter_averages import CCM89
from matplotlib import pyplot as plt

from jetfit.core.input import Observation
from jetfit.mcmc.mcmc import MCMC, MCMCModels
from jetfit.core import utils
from jetfit.mcmc.parameters import Parameters
from jetfit.models.boosted import HydroSimTable as HydroSimTable
from jetfit.models.boosted import BoostedFireballModel
from jetfit.models.fireball import FireballModel, StratifiedFireballModel
from jetfit.models.jetsim import JetSimpy
from jetfit.scripts.plot.freq import FrequencyPlotter
from jetfit.scripts.plot.sfbm import SFBMDensityProfiler
from jetfit.scripts.plot.dist import DistributionPlot
from jetfit.scripts.plot.dist2 import SpectralIndexPlot, DensityProfilePlot
from jetfit.scripts.plot.light_curve import LightCurvePlot
from jetfit.scripts.plot.posterior import PosteriorPlot


def plot_mcmc_diagnostics(mcmc: MCMC, results_dir: Path | str):
    """
    Plot corners and trace plots.

    Parameters
    ----------
    mcmc : MCMC
        The finished MCMC object.

    results_dir : Path or str
        The directory to save the results.
    """
    import arviz as az

    # Plot corner
    corner = PosteriorPlot(mcmc.sampler, mcmc.params.fitting, mcmc.param_pos)
    corner.plot(out_dir=results_dir)

    # Use arviz style
    az.style.use("arviz-darkgrid")

    # Create the production inference data object
    var_names = [p.name for p in mcmc.params.fitting]
    inf_data = az.from_emcee(mcmc.sampler, var_names=var_names)

    # Can't save the burn sampler due to multiprocessing issues.
    chain = np.transpose(mcmc.burn_chain, (1, 0, 2))
    burn = {name: chain[..., i] for i, name in enumerate(var_names)}
    inf_data_burn = az.from_dict(posterior=burn)

    # Save summary statistics to a csv
    az.summary(inf_data).to_csv(results_dir / "summary.csv")

    # Plot the trace plot
    az.plot_trace(inf_data)
    plt.savefig(results_dir / "trace.png")

    # Plot the burn-in trace plot
    az.plot_trace(inf_data_burn)
    plt.savefig(results_dir / "trace_burn.png")

    try:  # Optional stats
        print(f"Acceptance Fraction..{mcmc.sampler.acceptance_fraction}\n")
        print(f"Autocorrelation......{mcmc.sampler.acor}\n")
    except Exception as e:
        print(e)


def main(
        event: str,
        mcmc_path: Path,
        model_path: Path,
        data_path: Path,
        results_dir: Path
) -> None:
    """
    Runs the MCMC sampling routine.

    Parameters
    ----------
    event : str
        The name of the event.

    mcmc_path : Path
        The path to the MCMC settings file.

    model_path : Path
        The path to the model parameter file.

    data_path : Path
        The path to the input file.

    results_dir : Path
        The directory where the results will be saved.
    """

    # -----------------------------------------------------------------
    # -------------------------- Directories --------------------------
    # -----------------------------------------------------------------
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # -----------------------------------------------------------------
    # ----------------------------- I/O -------------------------------
    # -----------------------------------------------------------------
    parameters  = Parameters.from_toml(model_path)
    observation = Observation.from_csv(data_path)
    mcmc_params = utils.MCMCSettingsReader(mcmc_path)

    if parameters.has('nt'):
        model = StratifiedFireballModel
    elif parameters.has('A'):
        model = JetSimpy
    else:
        model = FireballModel

    if model.__name__ == 'BoostedFireballModel':
        meta = {
            'hydro_sim_table': HydroSimTable(
                utils.get_hydro_sim_table_path()
            ),
        }
    else:
        meta = None

    # -----------------------------------------------------------------
    # ---------------------- Observed Flux Model ----------------------
    # -----------------------------------------------------------------
    # Pre-compute extinction values (if applicable)
    ebv = {'ebv_milky_way': None}
    wn = observation.as_arrays.wave_numbers[observation.extinguishable]
    extinction_model = CCM89(Rv=3.1)

    for p in parameters.fixed:
        if p.name in ebv.keys():
            ebv[p.name] = extinction_model.extinguish(wn, Ebv=p.value)

    if parameters.has('rv_milky_way'):
        ebv['ebv_milky_way'] = None

    # -----------------------------------------------------------------
    # ----------------------------- MCMC ------------------------------
    # -----------------------------------------------------------------
    # Define a filename to save the sampler to disk.
    # Warning: The sampler files are very large ~1 GB each.
    sampler_kw = {}
    filename = str(results_dir / f'{event}_chain.h5')

    if filename is not None:
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(mcmc_params.num_walkers, len(parameters.fitting))
        sampler_kw['backend'] = backend

    # Create the MCMC object and run. See you in a few hours!
    sampler_name = mcmc_params.data['sampler']['name']
    run_kw = {}

    if sampler_name == 'ensemble':
        run_kw = {'progress': True}

    mcmc = MCMC(
        model=MCMCModels(observation, model, meta, CCM89, ext_mw_pc=ebv['ebv_milky_way']),
        observation=observation,
        parameters=parameters,
    )
    mcmc.run(
        nwalkers=mcmc_params.num_walkers,
        iterations=mcmc_params.run_length,
        burn=mcmc_params.burn_length,
        sampler=sampler_name,
        workers=mcmc_params.workers,
        ntemps=mcmc_params.ntemps,
        run_kw=run_kw,
        sampler_kw=sampler_kw,
    )

    # -----------------------------------------------------------------
    # ----------------------------- PLOT ------------------------------
    # -----------------------------------------------------------------
    # Plot the light curves
    best_params = mcmc.get_best_params()

    if model.__name__ == 'StratifiedFireballModel':
        profiler = SFBMDensityProfiler(mcmc.sampler, parameters)

        profiler.profile(
            observation.times().min(),
            observation.times().max(),
        )
        profiler.plot_profile(results_dir)

    elif model.__name__ == 'FireballModel':
        # Plot the density profiles
        density_plotter = DensityProfilePlot(
            mcmc.sampler, parameters)

        density_plotter.plot(
            observation.times().min(),
            observation.times().max(),
            out_dir=results_dir
        )

        # # Plot distributions
        dist_plotter = DistributionPlot(mcmc.sampler, parameters, observation)

        # # Plot the opening angle and energy distribution
        if parameters.has('tj'):
            dist_plotter.beaming(out_dir=results_dir)

        # Plot the spectral index distribution
        spectral_index_plotter = SpectralIndexPlot(
            mcmc.sampler, parameters, model)

        spectral_index_plotter.model(
            observation.data[observation.sindex_loc], out_dir=results_dir)

    # -----------------------------------------------------------------
    # ---------------------------- LOGGING ----------------------------
    # -----------------------------------------------------------------
    out_params = mcmc.get_best_params()
    out_params['chi_squared'] = -2 * mcmc.sampler.get_log_prob(flat=True).max()
    out_params['mcmc'] = {
        'sampler': sampler_name,
        'prod_len': mcmc_params.run_length,
        'burn_len': mcmc_params.burn_length,
        'nwalkers': mcmc_params.num_walkers,
        'model': model.__name__,
    }

    with open(results_dir / "best_fit.json", "w") as jf:
        json.dump(out_params, jf, indent=4)

    # Plot frequencies
    fp = FrequencyPlotter(mcmc.sampler, parameters, model, meta)
    fp.plot_all(observation, out_dir=results_dir)

    # Plot light curve
    lc = LightCurvePlot(
        model=model,
        params=best_params,
        observation=observation,
        title=f'{event} Light Curve',
        meta=meta
    )
    lc.plot(
        out_dir=results_dir,
        ext_model=extinction_model,
    )

    # -----------------------------------------------------------------
    # -------------------------- DIAGNOSTICS --------------------------
    # -----------------------------------------------------------------
    plot_mcmc_diagnostics(mcmc, results_dir)

    plt.close()
    print(f'AMPy completed modeling of {event} successfully.')


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser(description="JetFit Parameters")

    parser.add_argument('--event', help='Event directory name.')
    parser.add_argument('--mcmc',  help='Path to the MCMC settings.toml file.')
    parser.add_argument('--model', help='Path to the model defaults.toml file.')
    parser.add_argument('--data',  help='Path to the input data file.')
    parser.add_argument('--results', help='Path the the results directory.')

    args = parser.parse_args()

    sub_dir = 'grbs'

    if args.event is None:
        # Specify the events to run
        events = [
            # '050525A',
            # '050922C',
            '080413B',
            # '080319B_nature_mix_1',
            # '090424',
            # '090618',
            # '111228A',
            # '130612A',
            # '130612A_1',
            # '131030A',
            # '140506A',
            # '160131A',
            # '171010A',
            # '210905A',
            # '220101A',
            # '221009A',
            # '250129A',
            # '170817'
        ]
    else:
        events = [args.event]

    # Run each event
    for event in events:

        main(
            **{
                'event':
                    event,

                'mcmc_path':
                    Path(args.mcmc)
                    if args.mcmc is not None
                    else utils.get_mcmc_settings_path(),

                'model_path':
                    Path(args.model)
                    if args.model is not None
                    else utils.get_event_path(sub_dir, event) / 'parameters.toml',

                'data_path':
                    Path(args.data)
                    if args.data is not None
                    else utils.get_input_csv_path(sub_dir, event),

                'results_dir':
                    Path(args.results)
                    if args.results is not None
                    else utils.get_results_path() / event,
            }
        )
