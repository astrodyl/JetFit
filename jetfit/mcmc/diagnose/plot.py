import os
from pathlib import Path

import numpy as np
import corner
from matplotlib import pyplot as plt
from scipy.stats import norm

from jetfit.core.defns.enums import ScaleType
from jetfit.core.defns.observation import Observation
from jetfit.core.fit import JetFit
from jetfit.core.values.time_value import TimeValue
from jetfit.model import generator
from jetfit.scalers.scaler import Scaler


class Plot:
    """

    Attributes
    ----------
    sampler : `emcee.EnsembleSampler`

    parameters : list of `jetfit.core.values.mcmc_value.MCMCFittingParameters`

    output : str or pathlib.Path
        The directory to save the plots.
    """
    def __init__(self, sampler, params: list, obs: Observation, output: str | Path):
        self.sampler = sampler
        self.parameters = params
        self.observation = obs
        self.output = output

        self.flat_chain = sampler.get_chain(flat=True)

    def light_curve(self, fit: JetFit) -> None:
        """
        """
        _, ax = plt.subplots(figsize=(8, 8))

        fluxes = [f.value for f in self.observation.fluxes]
        flux_errors = [f.avg_error for f in self.observation.fluxes]
        times = [t.value for t in self.observation.times]

        ax.errorbar(times, fluxes, yerr=flux_errors, fmt='.', label='XRT')
        ax.set_yscale('log'), ax.set_xscale('log'), ax.set_ylabel('Flux (CGS)')
        ax.set_xlabel('Time Since Trigger (seconds)')

        # Plot the model
        log_probs = self.sampler.get_log_prob(flat=True)    # Get all log-probabilities
        max_index = np.nanargmax(log_probs)                 # Index of max log-probability
        best_position = self.flat_chain[max_index]

        params = fit.get_model_parameters(best_position, ScaleType.LINEAR)

        times = np.linspace(
                self.observation.times[0].value,
                self.observation.times[-1].value * 2,
                200
            )

        time_values = [TimeValue(t, self.observation.times[0].units) for t in times]

        scaler = Scaler(fit.scaler.hydro_sim_table, time_values)

        peak_fluxes, cooling_frequencies, synchrotron_frequencies = (
            scaler.scaled_characteristics(params)
        )

        modeled_fluxes = []
        for i, peak_flux in enumerate(peak_fluxes):
            modeled_fluxes.append(
                generator.generate(
                    self.observation.fluxes[0],
                    peak_flux,
                    cooling_frequencies[i],
                    synchrotron_frequencies[i],
                    params.electron_index
                ).value
            )

        plt.loglog(
            times,
            np.array(modeled_fluxes),
            '--',
            linewidth=1.5
        )

        plt.savefig(os.path.join(self.output, 'light_curve.png'))

    def corner(self) -> None:
        """
        """

        labels, ranges, bins = [], [], []
        for i, p in enumerate(self.parameters):
            ranges.append((p.prior.lower, p.prior.upper))
            labels.append(self.get_pretty_label(p.name))
            bins.append(50)

        # Create the two-tailed sigma levels to plot
        sigma_fractions = [self.sigma_to_fraction(sigma) for sigma in [1, 2, 3]]

        fig = corner.corner(
            self.flat_chain,
            bins=bins,
            color='mediumblue',
            labels=labels,
            label_size=20,
            show_titles=True,
            plot_datapoints=False,
            quantiles=[0.16, 0.5, 0.84],
            label_kwargs={'fontsize': 18},
            title_kwargs={"fontsize": 18},
            fill_contours=True,
            smooth=0.75,
            smooth1d=0.75,
            range=ranges,
            levels=sigma_fractions
        )

        # Plot dashed lines corresponding to the median for the 2D plots
        medians = [np.median(self.flat_chain[:, i]) for i in range(len(self.flat_chain[0]))]
        corner.overplot_lines(fig, medians, linestyle='--', color="black")

        fig.savefig(os.path.join(self.output, 'corner.png'))

    @staticmethod
    def get_pretty_label(key: str) -> str | None:
        """
        """
        try:
            return {
                'explosion_energy': r'$log_{10}E_{j,50}$',
                'circumburst_density': r'$log_{10}n_{0,0}$',
                'asymptotic_lorentz_factor': r'$\eta_0$',
                'boost_lorentz_factor': r'$\gamma_B$',
                'obs_angle': r'$\theta_{obs}$',
                'electron_energy_fraction': r'$log_{10}\epsilon_e$',
                'magnetic_energy_fraction': r'$log_{10}\epsilon_B$',
                'electron_energy_index': r'$p$'
            }[key]
        except KeyError:
            return None

    @staticmethod
    def sigma_to_fraction(sigma: float) -> float:
        """
        Converts a sigma level to a two-tailed fraction.

        Parameters
        ----------
        sigma : float
            The sigma level.

        Returns
        -------
        float
            The fractional sigma level.
        """
        return 2.0 * norm.cdf(sigma) - 1.0
