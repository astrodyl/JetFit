import argparse
import json
import os.path
from pathlib import Path

from matplotlib import pyplot as plt

from jetfit.core.input import Observation
from jetfit.mcmc.mcmc import MCMC
from jetfit.core.utils import nav_utils
from jetfit.mcmc.settings.reader import MCMCSettingsReader
from jetfit.models.afterglow.boosted_fireball.hydro_sim.hydro_sim import HydroSimTable
from jetfit.models.afterglow.boosted_fireball.parameters.reader import BFParamsReader
from jetfit.models2.boosted import BoostedFireballModel
from jetfit.models2.fireball import FireballModel
from jetfit.plot.light_curve import LightCurvePlot
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
    event : Path

    mcmc_path : Path
        The path to the MCMC settings file.

    model_path : Path
        The path to the model parameter file.

    data_path : Path
        The path to the input file.

    results_dir : Path
        The directory where the results will be saved.
    """
    # -------- PLOTTING ----------
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # TEST FOR GENERIC FIREBALL MODEL
    observation = Observation.from_csv(data_path)

    # -------- NEW COOLER WAY OF DOING THINGS --------
    mcmc_params = MCMCSettingsReader(mcmc_path)
    model_params = BFParamsReader(model_path)

    mcmc = MCMC(
        burn_length=mcmc_params.burn_length,
        run_length=mcmc_params.run_length,
        num_walkers=mcmc_params.num_walkers,
        model=FireballModel,
        observation=observation,
        fixed_params=model_params.fixed,
        fitting_params=model_params.fitting,
        filename=str(results_dir / 'chain.h5'),
        # meta={
        #     'hydro_sim_table':
        #           HydroSimTable(nav_utils.get_hydro_sim_table_path())
        #       }
    )

    mcmc.run()

    # Plot the light curves
    best_params = mcmc.get_best_params()

    lc = LightCurvePlot(
        model=mcmc.model,
        params=best_params.get('model'),
        observation=mcmc.observation,
        title=f'{event} Light Curve'
    )
    lc.plot(out_dir=results_dir, host_corr=best_params.get('host'))

    # Plot the corner plot
    corner = PosteriorPlot(mcmc.sampler, mcmc.fitting_params, mcmc.param_pos)
    corner.plot(out_dir=results_dir)

    # -------- LOGGING ---------
    with open(results_dir / "best_fit.json", "w") as jf:
        json.dump(mcmc.get_best_params(), jf, indent=4)

    # -------- DIAGNOSTICS ---------
    import arviz as az

    az.style.use("arviz-darkgrid")
    idata = az.from_emcee(mcmc.sampler, var_names=[p.name for p in mcmc.fitting_params])
    idata_burnin = az.from_emcee(mcmc.burn_sampler, var_names=[p.name for p in mcmc.fitting_params])

    # Save summary statistics to a csv
    az.summary(idata).to_csv(results_dir / "summary.csv")

    # Plot the trace plot
    az.plot_trace(idata)
    plt.savefig(results_dir / "trace.png")

    # Plot the burn trace plot
    az.plot_trace(idata_burnin)
    plt.savefig(results_dir / "trace_burn.png")

    # print(f"Autocorrelation........{mcmc.sampler.acor}\n")
    # print(f"Acceptance Fraction....{mcmc.sampler.acceptance_fraction}\n")


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

    sub_dir = 'new'

    if args.event is None:
        # Specify the events to run
        events = [
            # '050922C',
            # '080413B',
            # '080413B_early',
            # '080413B_late',
            # '090424',
            # '090618',
            # '111228A',
            # '130612A',
            # '131030A',
            # '160131A',
            # '171010A',
            # '220101A',
            # '210905A',
            '221009A',
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
