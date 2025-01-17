import argparse
import json
import os.path
from pathlib import Path

from matplotlib import pyplot as plt

from jetfit.core.defns.evidence import Evidence
from jetfit.mcmc.mcmc import MCMC
from jetfit.core.utils import paths
from jetfit.mcmc.settings.reader import MCMCSettingsReader
from jetfit.models.afterglow.boosted_fireball.parameters.reader import BFParamsReader
from jetfit.plot.light_curve import LightCurve
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
    # -------- NEW COOLER WAY OF DOING THINGS --------
    mcmc_params = MCMCSettingsReader(mcmc_path)
    model_params = BFParamsReader(model_path)
    evidence = Evidence.from_csv(data_path)

    mcmc = MCMC(
        burn_length=mcmc_params.burn_length,
        run_length=mcmc_params.run_length,
        num_walkers=mcmc_params.num_walkers,
        model=mcmc_params.model,
        evidence=evidence,
        fixed_params=model_params.fixed,
        fitting_params=model_params.fitting,
    )

    mcmc.run()

    # -------- PLOTTING ----------
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    lc = LightCurve(mcmc.model, mcmc.get_best_params(), mcmc.evidence)
    lc.plot(out_dir=results_dir)

    corner = PosteriorPlot(mcmc.sampler, mcmc.evidence, mcmc.fitting_params)
    corner.plot(out_dir=results_dir)

    # -------- LOGGING ---------
    with open(results_dir / "best_fit.json", "w") as jf:
        json.dump(vars(mcmc.get_best_params()), jf, indent=4)

    # -------- DIAGNOSTICS ---------
    import arviz as az

    az.style.use("arviz-darkgrid")

    idata = az.from_emcee(mcmc.sampler, var_names=[p.name for p in mcmc.fitting_params])
    az.summary(idata).to_csv(results_dir / "summary.csv")

    # print(mcmc.sampler.acor)
    # print(mcmc.sampler.acceptance_fraction)
    # print(f"Effective Sample Size (ESS):\n{az.ess(idata)}\n")
    # print(f"Gelman-Rubin Statistic (R Hat):\n{az.rhat(idata)}\n")
    # print(f"Acceptance Fraction:\n{mcmc.sampler.acceptance_fraction}\n")

    az.plot_trace(idata)
    plt.savefig(results_dir / "trace.png")


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

    sub_dir = 'final'

    if args.event is None:
        # Specify the events to run
        events = [
            # '080413B',
            # '090424',
            '090618',
            # '111228A',
            # '130612A',
            # '160131A',
            # '171010A',
            # '220101A',
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
                    else paths.get_mcmc_settings_path(),

                'model_path':
                    Path(args.model)
                    if args.model is not None
                    else paths.get_event_path(sub_dir, event) / 'parameters.toml',

                'data_path':
                    Path(args.data)
                    if args.data is not None
                    else paths.get_input_csv_path(sub_dir, event),

                'results_dir':
                    Path(args.results)
                    if args.results is not None
                    else paths.get_results_path() / event,
            }
        )
