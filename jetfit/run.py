import argparse
import json
import os.path
from pathlib import Path

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
from jetfit.plot.light_curve import LightCurvePlot, FrequencyPlot
from jetfit.plot.posterior import PosteriorPlot
from jetfit.plot.spectral import CriticalFrequenciesPlot


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
        ext_mw=ebv['ebv_milky_way']
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

    fp = FrequencyPlot(mcmc.sampler, parameters)
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

    lc = LightCurvePlot(
        model=observed_flux_model.afterglow_model,
        params=best_params,
        observation=observation,
        title=f'{event} Light Curve'
    )
    lc.plot(
        out_dir=results_dir,
        ext_model=observed_flux_model.extinction_model,
    )

    # Plot the critical frequencies
    # cf = CriticalFrequenciesPlot(
    #     mcmc.model.afterglow_model(**best_params.get('model')),
    #     observation.as_arrays.times[observation.flux_loc].min(),
    #     observation.as_arrays.times[observation.flux_loc].max()
    # )
    # cf.plot(out_dir=results_dir, title=f'{event} Critical Frequencies')

    # Plot the corner plot
    corner = PosteriorPlot(mcmc.sampler, mcmc.params.fitting, mcmc.param_pos)
    corner.plot(out_dir=results_dir)

    # -------- LOGGING ---------
    with open(results_dir / "best_fit.json", "w") as jf:
        json.dump(mcmc.get_best_params(), jf, indent=4)

    # -------- DIAGNOSTICS ---------
    import arviz as az

    az.style.use("arviz-darkgrid")
    idata = az.from_emcee(mcmc.sampler, var_names=[p.name for p in mcmc.params.fitting])
    idata_burnin = az.from_emcee(mcmc.burn_sampler, var_names=[p.name for p in mcmc.params.fitting])

    # Save summary statistics to a csv
    # az.summary(idata).to_csv(results_dir / "summary.csv")

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

    sub_dir = 'newer'

    if args.event is None:
        # Specify the events to run
        events = [
            # '050922C',
            '080413B',
            # '080413B_early',
            # '080413B_late',
            # '090424',
            # '090618',
            # '111228A',
            # '111228A_early',
            # '111228A_late',
            # '140506A',
            # '130612A',
            # '131030A',
            # '160131A',
            # '171010A',
            # '220101A',
            # '210905A',
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
