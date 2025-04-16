from pathlib import Path

import corner
import numpy as np
from matplotlib import pyplot as plt


class PosteriorPlot:
    """

    """
    def __init__(self, sampler, params: list, param_pos):
        self.sampler = sampler
        self.params = params
        self.param_pos = param_pos

    def plot(self, show: bool = False, out_dir: str | Path = None):
        """
        Creates the corner plot of 1D and 2D posteriors.

        Parameters
        ----------
        show : bool, optional
            If ``True``, calls `plt.show()`.

        out_dir : str or Path, optional
            The directory to save `corner.png.`
        """
        chain = self.sampler.get_chain(flat=True)

        plot_kw = {
            'label_size': 16, 'show_titles': True,
            'plot_datapoints': False, 'quantiles': [0.16, 0.5, 0.84],
            'label_kwargs': {'fontsize': 14}, 'title_kwargs': {"fontsize": 14},
            'fill_contours': True, 'smooth': 0.75, 'smooth1d': 0.75,
        }

        # Split parameters into physical and non-physical
        ranges, bins, labels, pos = [], [], [], []
        h_ranges, h_bins, h_labels, h_pos = [], [], [], []
        np_ranges, np_bins, np_labels, np_pos = [], [], [], []

        param_pos = {}
        for i, p in enumerate(self.params):
            name = p.name if p.group is None else f'{p.name} ({p.group})'
            param_pos[name] = i

        # Split the parameters into groups (GRB physics, statistical, host)
        for p in self.params:
            name = p.name if p.group is None else f'{p.name} ({p.group})'

            if '_offset' in name or 'slop' in name:
                np_ranges.append((p.prior.lower, p.prior.upper))
                np_labels.append(self.get_pretty_label(name))
                np_pos.append(param_pos[name])
                np_bins.append(50)

            elif '_host' in name:
                h_ranges.append((p.prior.lower, p.prior.upper))
                h_labels.append(self.get_pretty_label(name))
                h_pos.append(param_pos[name])
                h_bins.append(50)

            else:
                ranges.append((p.prior.lower, p.prior.upper))
                labels.append(self.get_pretty_label(name))
                pos.append(param_pos[name])
                bins.append(50)

        if ranges:
            fig = corner.corner(
                chain[:, pos], bins=bins, color='mediumblue',
                labels=labels, range=ranges, **plot_kw,
            )
            if out_dir:
                fig.savefig(out_dir / 'corner.png')

        if np_ranges:
            np_fig = corner.corner(
                chain[:, np_pos], bins=np_bins, color='mediumblue',
                labels=np_labels, range=np_ranges, **plot_kw
            )
            if out_dir:
                np_fig.savefig(out_dir / 'corner_np.png')

        if h_ranges:
            h_fig = corner.corner(
                chain[:, h_pos], bins=h_bins, color='mediumblue',
                labels=h_labels, range=h_ranges, **plot_kw
            )
            if out_dir:
                h_fig.savefig(out_dir / 'corner_host.png')

        # Plot dashed lines corresponding to the median for the 2D plots
        # medians = [np.median(chain[:, i]) for i in range(len(chain[0]))]
        # corner.overplot_lines(fig, medians, linestyle='--', color="black")

        if show:
            plt.show()

    @staticmethod
    def get_pretty_label(key: str):
        """
        Returns LaTeX label for the provided fitting parameter.

        :param key: fitting parameter name
        :return: LaTeX formatted str or None
        """
        if '_offset' in key:
            return r'$\delta_{' + f'{key.split('_')[0]}' r'}$'

        if 'rho0' in key:
            return key.replace('rho0', r'$log_{10}n_{17}$')

        try:
            return {
                # Boosted Fireball Model
                'explosion_energy': r'$log_{10}E_{j,50}$',
                'circumburst_density': r'$log_{10}n_{0,0}$',
                'asymptotic_lorentz_factor': r'$\eta_0$',
                'boost_lorentz_factor': r'$\gamma_B$',
                'obs_angle': r'$\theta_{obs}$',
                'electron_energy_fraction': r'$log_{10}\epsilon_e$',
                'magnetic_energy_fraction': r'$log_{10}\epsilon_B$',
                'electron_energy_index': r'$p$',
                'ebv_source_frame': r'$ebv_{sf}$',

                # Generic Fireball Model
                'E': r'$log_{10}E_{52}$',
                'eps_e': r'$log_{10}\epsilon_e$',
                'eps_b': r'$log_{10}\epsilon_B$',
                'ebv_sf': r'$E(B-v)_{sf}$',
                'rho0': r'$log_{10}n$',
            }[key]
        except KeyError:
            return key
