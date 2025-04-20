import argparse
import json
import os.path
from pathlib import Path

import numpy as np
from dust_extinction.parameter_averages import CCM89
from matplotlib import pyplot as plt

from jetfit.core.input import Observation
from jetfit.mcmc.mcmc import MCMC
from jetfit.core.utils import nav_utils
from jetfit.mcmc.parameters.parameters import Parameters
from jetfit.mcmc.settings.reader import MCMCSettingsReader
from jetfit.models.afterglow.boosted_fireball.hydro_sim.hydro_sim import HydroSimTable
from jetfit.models2.basemodels import ObservedFluxModel
from jetfit.models2.boosted import BoostedFireballModel
from jetfit.models2.fireball import FireballModel
from jetfit.plot.dist import DistributionPlot
from jetfit.plot.dist2 import SpectralIndexPlot, StratifiedDensityProfilePlot
from jetfit.plot.light_curve import LightCurvePlot, FrequencyPlot
from jetfit.plot.posterior import PosteriorPlot


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
    parameters = Parameters.from_toml(model_path)
    observation = Observation.from_csv(data_path)
    mcmc_params = MCMCSettingsReader(mcmc_path)

    # -----------------------------------------------------------------
    # ---------------------- Observed Flux Model ----------------------
    # -----------------------------------------------------------------
    # Pre-compute extinction values (if applicable)
    ebv = {'ebv_source_frame': None, 'ebv_milky_way': None}
    wn = observation.as_arrays.wave_numbers[observation.sflux_loc]
    extinction_model = CCM89(Rv=3.1)

    for p in parameters.fixed:
        if p.name in ebv.keys():
            ebv[p.name] = extinction_model.extinguish(wn, Ebv=p.value)

    # Store pre-computed values in the extrinsic model
    observed_flux_model = ObservedFluxModel(
        FireballModel, extinction_model,
        ext_sf=ebv['ebv_source_frame'],
        ext_mw=ebv['ebv_milky_way'],
        dynamic=True,
    )

    # -----------------------------------------------------------------
    # ----------------------------- MCMC ------------------------------
    # -----------------------------------------------------------------
    # Define a filename to save the sampler to disk.
    # Warning: The sampler files are very large ~1 GB each.
    filename = None  # str(results_dir / f'{event}_chain.h5')

    # Create the MCMC object and run. See you in a few hours!
    mcmc = MCMC(
        **mcmc_params.data['sampler'],
        model=observed_flux_model,
        observation=observation,
        parameters=parameters,
        filename=filename
    )
    mcmc.run()

    # -----------------------------------------------------------------
    # ----------------------------- PLOT ------------------------------
    # -----------------------------------------------------------------
    # Plot the light curves
    best_params = mcmc.get_best_params()

    # Plot the spectral index distribution
    spectral_index_plotter = SpectralIndexPlot(
        mcmc.sampler, parameters, FireballModel, observation.data_regimes)

    spectral_index_plotter.model(
        observation.data[observation.sindex_loc], out_dir=results_dir)

    # Plot the density profiles
    density_plotter = StratifiedDensityProfilePlot(
        mcmc.sampler, parameters, observation.data_regimes)

    density_plotter.plot(
        observation.as_arrays.times.min(),
        observation.as_arrays.times.max(),
        out_dir=results_dir
    )

    # Plot distributions
    dist_plotter = DistributionPlot(mcmc.sampler, parameters, observation)

    # Plot the opening angle and energy distribution
    if parameters.has('tj'):
        dist_plotter.beaming(out_dir=results_dir)

    # Plot frequencies
    fp = FrequencyPlot(mcmc.sampler, parameters, observed_flux_model.dynamic)
    fp.plot(
        model=observed_flux_model.afterglow_model,
        obs=observation,
        out_dir=results_dir
    )
    fp.plot_best(
        model=observed_flux_model.afterglow_model,
        obs=observation,
        out_dir=results_dir
    )

    # Plot light curve
    lc = LightCurvePlot(
        model=observed_flux_model.afterglow_model,
        params=best_params,
        observation=observation,
        title=f'{event} Light Curve',
        dynamic=observed_flux_model.dynamic
    )
    lc.plot(
        out_dir=results_dir,
        ext_model=observed_flux_model.extinction_model,
    )

    # Plot chi squared
    # _, ax = plt.subplots()
    #
    # cs_vals = -2 * mcmc.sampler.get_log_prob(flat=False)
    #
    # for i in range(len(cs_vals[0])):
    #     plt.plot(np.log10(cs_vals[:, i]), alpha=0.4)
    #
    # plt.xlabel("Step")
    # plt.ylabel(r"$\chi^2$")
    # plt.title("Chi-squared traces per walker")
    # plt.savefig(results_dir / 'chi-squared.png')

    # Plot corner
    corner = PosteriorPlot(mcmc.sampler, mcmc.params.fitting, mcmc.param_pos)
    corner.plot(out_dir=results_dir)

    # -----------------------------------------------------------------
    # ---------------------------- LOGGING ----------------------------
    # -----------------------------------------------------------------
    out_params = mcmc.get_best_params()
    out_params['chi_squared'] = -2 * mcmc.sampler.get_log_prob(flat=True).max()

    with open(results_dir / "best_fit.json", "w") as jf:
        json.dump(out_params, jf, indent=4)

    # -----------------------------------------------------------------
    # -------------------------- DIAGNOSTICS --------------------------
    # -----------------------------------------------------------------
    # import arviz as az

    # az.style.use("arviz-darkgrid")
    # inf_data = az.from_emcee(mcmc.sampler, var_names=[p.name for p in mcmc.params.fitting])
    # inf_data_burn = az.from_emcee(mcmc.burn_sampler, var_names=[p.name for p in mcmc.params.fitting])

    # Save summary statistics to a csv
    # az.summary(inf_data).to_csv(results_dir / "summary.csv")

    # Plot the trace plot
    # az.plot_trace(inf_data)
    # plt.savefig(results_dir / "trace.png")

    # Plot the burn-in trace plot
    # az.plot_trace(inf_data_burn)
    # plt.savefig(results_dir / "trace_burn.png")

    # try:  # Optional stats
    #     print(f"Autocorrelation........{mcmc.sampler.acor}\n")
    #     print(f"Acceptance Fraction....{mcmc.sampler.acceptance_fraction}\n")
    # except Exception as e:
    #     pass

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

    sub_dir = 'newer'

    if args.event is None:
        # Specify the events to run
        events = [
            # '050525A',
            # '050922C',
            # '080413B',
            # '080319B_early',
            # '080319B_mid',
            # '080319B',
            # '080413B_early',
            # '080413B_late',
            # '090424',
            # '090618',
            # '111228A',
            # '111228A_early',
            # '111228A_late',
            # '130612A',
            # '131030A',
            # '140506A',
            # '160131A',
            # '171010A',
            # '210905A',
            '220101A',
            # '221009A',
            # '231118A',
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
                    else nav_utils.get_mcmc_settings_path(),

                'model_path':
                    Path(args.model)
                    if args.model is not None
                    else nav_utils.get_event_path(sub_dir, event) / 'parameters.toml',

                'data_path':
                    Path(args.data)
                    if args.data is not None
                    else nav_utils.get_input_csv_path(sub_dir, event),

                'results_dir':
                    Path(args.results)
                    if args.results is not None
                    else nav_utils.get_results_path() / event,
            }
        )
