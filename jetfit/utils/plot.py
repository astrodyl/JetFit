import os
from typing import Union

import emcee
import numpy as np
from corner import corner, overplot_lines
from matplotlib import pyplot as plt
import scipy.stats as stats


class Plot:
    def __init__(self, chain: np.ndarray, fitter, bp: np.ndarray, **kwargs):
        self.chain = chain
        self.fitter = fitter
        self.best_params = bp
        self.log_params = kwargs['log']
        self.fitted_params = kwargs['fit']
        self.log_type = kwargs['log_type']

        self.options = {
            'colors': ['orange', 'red', 'g', 'b'],
            'scale': [6.0, 1.0, 100.0, 800.0],
            'legend': True,
            'x_axis_day': True
        }

        if not os.path.exists('./jetfit/results/'):
            os.mkdir('./jetfit/results/')

    def plot_autocorrelation(self, path: str):
        """ Estimate the normalized autocorrelation function of a 1-D series.
        """
        fig, axes = plt.subplots(len(self.fitted_params), figsize=(10, 7), sharex='all')

        def next_pow_two(n):
            i1 = 1
            while i1 < n:
                i1 = i1 << 1
            return i1

        def autocorr_func_1d(x, norm: bool = True):
            x = np.atleast_1d(x)
            if len(x.shape) != 1:
                raise ValueError("Invalid dimensions for 1D autocorrelation function")
            n = next_pow_two(len(x))

            # Compute the FFT and then the auto-correlation function
            f = np.fft.fft(x - np.mean(x), n=2 * n)
            acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
            acf /= 4 * n

            if norm:  # Optionally normalize
                acf /= acf[0]

            return acf

        labels = [self.get_pretty_label(param) for param in self.fitted_params]

        for i in range(len(self.fitted_params)):
            axes[i].plot(autocorr_func_1d(self.chain[:, :, i].reshape(-1)), "k")

            axes[i].set_xlim(0, len(self.chain[i]))
            axes[i].set_ylabel(labels[i])

        axes[-1].set_xlabel("Lag")
        fig.savefig(path)

    def plot_light_curves(self, data_frame, default_params: dict, path: str) -> None:
        """ Plots the generated synthetic light curves.

        :param data_frame: pd.DataFrame object
        :param default_params: {param_name: default value}
        :param path: path to save plot
        """
        best_params = self.overwrite_defaults(default_params.copy())

        _, ax = plt.subplots(figsize=(8, 8))

        colors = self.options['colors']
        scales = self.options['scale']

        frequencies = data_frame['Freqs'].unique()

        for freq, color, scale in zip(frequencies, colors, scales):
            sub_df = data_frame[data_frame['Freqs'] == freq]

            times = sub_df['Times']
            if self.options['x_axis_day']:
                times = times / 24. / 3600

            label = '%.1e x %d' % (freq, scale) if max(scales) > 1.0 else '%.1e' % freq
            ax.errorbar(times, sub_df['Fluxes'] * scale, yerr=sub_df['FluxErrs'] * scale,
                        color=color, fmt='.', label=label)

        ax.set_yscale('log'), ax.set_xscale('log'), ax.set_ylabel('Flux density (mJy)')
        ax.set_xlabel('Time (day)') if self.options['x_axis_day'] else ax.set_xlabel('Time (s)')

        if self.options['legend']:
            ax.legend(loc=0)

        for i, freq in enumerate(data_frame['Freqs'].unique()):
            new_times = np.linspace(data_frame['Times'].min() * 1.0, data_frame['Times'].max() * 2.0, 200)
            new_frequencies = np.ones(len(new_times)) * freq

            flux_model = np.asarray(self.fitter.flux_generator.get_spectral(new_times, new_frequencies, best_params))
            plt.loglog(new_times / 24. / 3600., flux_model * scales[i], '--', color=colors[i], linewidth=1.5)

        plt.savefig(path)

    def plot_markov_chain(self, path: str) -> None:
        """ Plots the markov chain for all walkers and saves the figure
        to the result directory.

        :param path: path to save plot
        """
        fig, axes = plt.subplots(len(self.fitted_params), figsize=(10, 7), sharex='all')
        fig.suptitle("Markov Chain", fontsize=16)

        labels = [self.get_pretty_label(param) for param in self.fitted_params]

        for i in range(len(self.fitted_params)):
            for j in range(len(self.chain[:, :, i])):
                # Over plot each walker for the given parameter
                axes[i].plot(self.chain[:, :, i][j], "k", alpha=0.3)

            axes[i].set_xlim(0, len(self.chain[i]))
            axes[i].set_ylabel(labels[i])

        axes[-1].set_xlabel("step number")

        fig.savefig(path)

    def plot_corner_plot(self, bounds: dict, path: str) -> None:
        """ Plots the corner plot of parameter distributions.

        :param bounds: dictionary of upper and lower bounds
        :param path: path to save plot
        """
        # Concatenate all the walkers
        chain = self.chain.reshape((-1, len(self.fitted_params)))

        labels, ranges, bins = [], [], []
        for i, param in enumerate(self.fitted_params):
            # Convert to the proper bounds
            if param in self.log_params:
                func = np.log10 if self.log_type == 'Log10' else np.log
                bounds[param][0] = func(bounds[param][0])
                bounds[param][1] = func(bounds[param][1])

            # Store the plot ranges a list of tuples
            ranges.append(tuple(bounds[param]))

            # Create LaTeX labels for each parameter
            labels.append(self.get_pretty_label(param))

            # Determine the number of bins for each parameter set
            bins.append(self.freedman_diaconis(chain[:, i]))

        # Create the two-tailed sigma levels to plot
        sigma_fractions = [self.sigma_to_fraction(sigma) for sigma in [0.25, 1, 2, 3]]

        fig = corner(chain,
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
                     smooth=True,
                     smooth1d=True,
                     range=ranges,
                     levels=sigma_fractions
                     )

        # Plot dashed lines corresponding to the median for the 2D plots
        medians = [np.median(chain[:, i]) for i in range(len(chain[0]))]
        overplot_lines(fig, medians, linestyle='--', color="black")

        fig.savefig(path)

    def log_to_linear(self) -> np.ndarray:
        """ Converts all fitting parameters that were passed in as log
        to linear.
        """
        linear = []
        for i, key in enumerate(self.fitted_params):
            value = self.best_params[i]

            if key in self.log_params:
                value = np.power(10.0, value) if self.log_type == 'Log10' else np.exp(value)

            linear.append(value)

        return np.array(linear)

    def overwrite_defaults(self, default_params: dict) -> dict:
        """ Overwrites default values with best fitted values.

        :param default_params: {param_name: default value}
        """
        best_linear_params = self.log_to_linear()

        for i, key in enumerate(self.fitted_params):
            default_params[key] = best_linear_params[i]

        return default_params

    @staticmethod
    def get_pretty_label(key: str) -> Union[str, None]:
        """ Returns LaTeX label for the provided fitting parameter. If the
        provided key is not valid, returns None.

        :param key: fitting parameter name
        :return: LaTeX formatted str or None
        """
        try:
            return {
                'explosion_energy': r'$log_{10}E_{j,50}$',
                'circumburst_density': r'$log_{10}n_{0,0}$',
                'asymptotic_lorentz_factor': r'$\eta_0$',
                'boost_lorentz_factor': r'$\gamma_B$',
                'observation_angle': r'$\theta_{obs}$',
                'electron_energy_fraction': r'$log_{10}\epsilon_e$',
                'magnetic_energy_fraction': r'$log_{10}\epsilon_B$',
                'spectral_index': r'$p$'
            }[key]
        except KeyError:
            return None

    @staticmethod
    def freedman_diaconis(data: np.ndarray) -> int:
        """ Implements the Freedman–Diaconis rule which can be used to
        select the width of the bins to be used in a histogram.

        :param data: 1d numpy array of data to be binned
        :return: integer number of bins
        """
        q25, q75 = np.percentile(data, [25, 75])
        bin_width = 2 * (q75 - q25) * len(data) ** (-1 / 3)
        return int((np.max(data) - np.min(data)) / bin_width)

    @staticmethod
    def sigma_to_fraction(sigma: float) -> float:
        """ Converts a sigma level to a two-tailed fraction.

        :param sigma: sigma level
        """
        return 2.0 * stats.norm.cdf(sigma) - 1.0
