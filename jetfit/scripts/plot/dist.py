import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.utils import save_plot_unique
from jetfit.models.basemodels import OpeningAngleModel


class DistributionPlot:
    """"""
    def __init__(self, sampler, params, obs):
        self.sampler = sampler
        self.params = params
        self.obs = obs

    def sample(self, thin=1, nsamps=100):
        """
        Randomly draws `nsamps` sets of samples from
        the `sampler`.

        Parameters
        ----------
        thin : int, optional, default=1
            Take only every `thin` steps from the chain.

        nsamps : int, optional, default=100
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            The randomly drawn sets of sampled values.
        """
        flat_chain = self.sampler.get_chain(flat=True, thin=thin)
        indices = np.random.randint(len(flat_chain), size=nsamps)
        return flat_chain[indices]

    def get_best_params(self, as_dict=True, **kwargs):
        """
        Returns the highest likelihood set of parameters.

        Parameters
        ----------
        as_dict : bool, optional, default=True
            If True, return the sampled values as a dictionary.

        kwargs :
            Any args passed to `params.samples_to_dict()`.

        Returns
        -------
        np.ndarray or dict
            The highest likelihood set of parameters.
        """
        max_index = np.nanargmax(self.sampler.get_log_prob(flat=True))
        params = self.sampler.get_chain(flat=True)[max_index]

        if as_dict:
            return self.params.samples_to_dict(params, **kwargs)

        return params

    def beaming(self, out_dir=None):
        """
        Calculates and plots the spectral index distribution
        for each provided spectral index.

        Parameters
        ----------
        out_dir : Path, optional
            The output directory to save the figures.
        """
        samples = self.sample(thin=10, nsamps=200)

        angles = np.full(len(samples), np.nan)
        energies = np.full(len(samples), np.nan)

        # Get the jet break time
        tj = self.get_best_params(cat='model')['tj']

        # Calculate the distribution of values
        for i, samp in enumerate(samples):
            samp = self.params.samples_to_dict(samp)['model']

            # Model the jet opening angle
            angles[i] = OpeningAngleModel(
                samp['E'], samp['rho0'], samp['k'], samp['z']
            )(tj)

            # Calculate the beaming-corrected energy
            energies[i] = (1 - np.cos(angles[i])) * samp['E']

        # Calculate the most likely opening angle
        best_samp = self.get_best_params(as_dict=True)['model']

        best_ang = OpeningAngleModel(
            best_samp['E'], best_samp['rho0'], best_samp['k'], best_samp['z']
        )(tj)

        # Calculate the most likely energy
        best_en = (1 - np.cos(best_ang)) * best_samp['E']

        # Plot the jet opening angle distribution
        title = f"Jet Opening Angle Distribution"

        self.plot_histogram(
            angles, round(best_ang, 3), title,
            'Jet Opening Angle', 'angle_dist', out_dir
        )

        # Plot the beaming-corrected energy distribution
        title = f"Beaming-Corrected Energy Distribution"

        self.plot_histogram(
            np.log10(energies), round(np.log10(best_en), 3), title,
            r'$log_{10}E_{j, 52}$', 'energy_dist', out_dir
        )

    @staticmethod
    def plot_histogram(dist, best, title=None, x_label=None, fn=None, out_dir=None):
        """
        Plots the spectral index distribution.

        Parameters
        ----------
        dist : np.ndarray
            The distribution of spectral indices.

        best : float
            The modeled spectral index using the highest
            likelihood set of parameters.

        title : str, optional
            The title of the plot.

        x_label : str, optional
            The x-axis label.

        fn : str, optional
            The output filename.

        out_dir : str or Path, optional
            The output directory to save the figures.
        """
        cts, bins, _ = plt.hist(  # Plot the value distribution
            dist, bins='auto', facecolor='#2ab0ff',
            edgecolor='#169acf', linewidth=0.5, alpha=0.5
        )

        plt.axvline(  # Plot the best fit value
            best, color='red', linestyle='--',
            linewidth=2, label=f'Best-fit Value: {best}'
        )

        # Plot the count above each bar
        for ct, l, r in zip(cts, bins[:-1], bins[1:]):
            if int(ct) != 0:
                plt.text((l + r) / 2, ct + 0.1, str(int(ct)), ha='center', va='bottom')

        # Configure the plot
        plt.title(title if title else 'Distribution')
        plt.xlabel(x_label)
        plt.ylabel('Count')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)

        if out_dir is not None:
            save_plot_unique(fn, 'png', str(out_dir))
        else:
            plt.show()
        plt.close()
