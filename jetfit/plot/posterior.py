from pathlib import Path

import corner
from matplotlib import pyplot as plt


class PosteriorPlot:
    """

    """
    def __init__(self, sampler, params: list, param_pos):
        self.sampler = sampler
        self.params = params
        self.param_pos = param_pos

    def plot(self, show: bool = False, out_dir: str | Path = None) -> None:
        """

        Parameters
        ----------
        show : bool, optional
            If ``True``, calls `plt.show()`.

        out_dir : str or Path, optional
            The directory to save `corner.png.`
        """
        chain = self.sampler.get_chain(flat=True)

        # Split parameters into physical and non-physical
        ranges, bins, labels, np_pos = [], [], [], []
        np_ranges, np_bins, np_labels, pos = [], [], [], []

        param_pos = {}
        for i, p in enumerate(self.params):
            name = p.name if p.group is None else f'{p.name}_{p.group}'
            param_pos[name] = i

        for p in self.params:
            name = p.name if p.group is None else f'{p.name}_{p.group}'

            # Non-physical parameters
            if '_offset' in name or '_host' in name or 'slop' in name:
                np_ranges.append((p.prior.lower, p.prior.upper))
                np_labels.append(self.get_pretty_label(name))
                np_pos.append(param_pos[name])
                np_bins.append(50)

            else:  # Physical parameters
                ranges.append((p.prior.lower, p.prior.upper))
                labels.append(self.get_pretty_label(name))
                pos.append(param_pos[name])
                bins.append(50)

        # Plot for physical parameters
        fig = corner.corner(
            chain[:, pos],
            bins=bins,
            color='mediumblue',
            labels=labels,
            label_size=16,
            show_titles=True,
            plot_datapoints=False,
            quantiles=[0.16, 0.5, 0.84],
            label_kwargs={'fontsize': 14},
            title_kwargs={"fontsize": 14},
            fill_contours=True,
            smooth=0.75,
            smooth1d=0.75,
            range=ranges
        )

        # Plot non-physical parameters
        np_fig = corner.corner(
            chain[:, np_pos],
            bins=np_bins,
            color='mediumblue',
            labels=np_labels,
            label_size=16,
            show_titles=True,
            plot_datapoints=False,
            quantiles=[0.16, 0.5, 0.84],
            label_kwargs={'fontsize': 14},
            title_kwargs={"fontsize": 14},
            fill_contours=True,
            smooth=0.75,
            smooth1d=0.75,
            range=np_ranges
        )

        # Plot dashed lines corresponding to the median for the 2D plots
        # medians = [np.median(chain[:, i]) for i in range(len(chain[0]))]
        # corner.overplot_lines(fig, medians, linestyle='--', color="black")

        if show:
            plt.show()

        if out_dir is not None:
            fig.savefig(out_dir / 'corner.png')
            np_fig.savefig(out_dir / 'corner_np.png')

    @staticmethod
    def get_pretty_label(key: str):
        """
        Returns LaTeX label for the provided fitting parameter.

        :param key: fitting parameter name
        :return: LaTeX formatted str or None
        """
        if '_offset' in key:
            return r'$\delta_{' + f'{key.split('_')[0]}' r'}$'

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