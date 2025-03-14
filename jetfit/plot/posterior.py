from pathlib import Path

import corner
import numpy as np
from matplotlib import pyplot as plt


class PosteriorPlot:
    """

    """
    def __init__(self, sampler, params: list):
        self.sampler = sampler
        self.params = params

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

        ranges = [(p.prior.lower, p.prior.upper) for p in self.params]
        bins = [50 for _ in range(len(self.params))]
        labels = [self.get_pretty_label(p.name) for p in self.params]

        fig = corner.corner(
            chain,
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

        # Plot dashed lines corresponding to the median for the 2D plots
        medians = [np.median(chain[:, i]) for i in range(len(chain[0]))]
        corner.overplot_lines(fig, medians, linestyle='--', color="black")

        if show:
            plt.show()

        if out_dir is not None:
            fig.savefig(out_dir / 'corner.png')

    @staticmethod
    def get_pretty_label(key: str):
        """
        Returns LaTeX label for the provided fitting parameter.

        :param key: fitting parameter name
        :return: LaTeX formatted str or None
        """
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
                'ebv_sf': r'$Ebv_{sf}$',
                'rho0': r'$log_{10}n$',
                'V_offset': r'$\delta_V$',
                'R_offset': r'$\delta_R$',
                'I_offset': r'$\delta_I$',
            }[key]
        except KeyError:
            return key