import argparse

import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.defns.observation import Observation
from jetfit.core.fit import JetFit
from jetfit.mcmc.diagnose.plot import Plot
from jetfit.mcmc.mcmc import MCMC
from jetfit.scalers.hydro_sim import HydroSimTable
from jetfit.scalers.scaler import Scaler
from jetfit.core.utilities import utils

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="JetFit Parameters")

    parser.add_argument('--event', help='Corresponding resource directory.')

    # Configurations
    parser.add_argument('--parameters', help='path to the parameter config file')
    parser.add_argument('--mcmc', help='path to the mcmc config file')

    # Inputs
    parser.add_argument('--table', help='path to the numerical table file')   # Table.h5
    parser.add_argument('--input', help='path to the input csv file')         #

    args = parser.parse_args()

    # TODO: move redshift-distance code here
    # TODO: Finish parser and verify args (for now, just basic reqs). later will add options to overwrite defaults
    # TODO: Check that values are same (table, flux values, scales, etc.)
    # TODO: Write plotting routine
    # TODO: Write logging routine (probably different pickles for convenience + one lightweight log)

    mcmc = MCMC.from_toml(
        utils.get_mcmc_settings_path(),
        utils.get_mcmc_params_path()
    )

    observation = Observation.from_csv(
        utils.get_input_csv_path(event='231118A')
    )

    scaler = Scaler(
        HydroSimTable(utils.get_hydro_sim_table_path()),
        observation.times
    )

    jetfit = JetFit(mcmc, observation, scaler)
    jetfit.run()

    # ------------------------------
    plot = Plot(jetfit.sampler, jetfit.mcmc.fitting_parameters, jetfit.observation, r"C:\Projects\repos\JetFit\jetfit\results")
    plot.light_curve(jetfit)
    plot.corner()
    plt.show()

    import arviz as az

    idata = az.from_emcee(jetfit.sampler, var_names=[p.name for p in jetfit.mcmc.fitting_parameters])

    print(f"Effective Sample Size (ESS):\n{az.ess(idata)}\n")
    print(f"Gelman-Rubin Statistic (R Hat):\n{az.rhat(idata)}\n")
    print(f"Acceptance Fraction:\n{jetfit.sampler.acceptance_fraction}\n")

    az.style.use("arviz-darkgrid")
    az.plot_posterior(idata, var_names=["boost_lorentz_factor", "obs_angle", "asymptotic_lorentz_factor"], hdi_prob=.68)
    az.plot_trace(idata)

    plt.show()