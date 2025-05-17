import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.core import save_plot_unique
from jetfit.models2.fireball import StratifiedFireballModel
from jetfit.plot.base import Profiler


class SFBMDensityProfiler(Profiler):
    """
    Density profiler for the StratifiedFireballModel class.

    Parameters
    ----------
    sampler : emcee.sampler
        The sampler used when running MCMC.

    params : `Parameters`
        The parameters object.

    Attributes
    ----------
    n0 : dict
        The density normalizations [cm-3].

    k : dict
        The density power-law indices.

    r : dict
        The blast wave radii [cm].

    r_trans : dict
        The transition radius [cm].
    """
    afterglow_model = StratifiedFireballModel

    r_trans_options = {
        'alpha': 0.5, 'linestyle': '--',
        'color': 'black', 'label': r'$R_{t}$',
    }

    def __init__(self, sampler, params):
        super().__init__(sampler, params)

        self.n0 = {'best': [], 'dist': []}
        self.k = {'best': [], 'dist': []}
        self.r = {'best': [], 'dist': []}
        self.r_trans = {'best': [], 'dist': []}

    def profile(self, start, stop, thin=10, nsamps=200):
        """
        Generates a profile for a random distribution of
        samples drawn from `sampler`. Over plots with the
        highest likelihood profile.

        Parameters
        ----------
        start, stop : float
            The start, stop time [days].

        thin : int, optional, default=1
            Take only every `thin` steps from the chain.

        nsamps : int, optional, default=100
            Number of samples to draw.
        """
        times = np.geomspace(start, stop, 500)
        samples = self.draw(thin, nsamps)

        for s in samples:
            # Model and store using random distribution of params
            params = self.params.samples_to_dict(s).get('model')
            self.model(times, params, 'dist')

        # Model and store using the best fitting params
        best_params = self.best().get('model')
        self.model(times, best_params, 'best')

    def model(self, times, params, loc):
        """
        Models the blast wave radii, densities, and
        power-law indices.

        Parameters
        ----------
        times : np.ndarray
            Times since trigger [days].

        params : dict
            The model parameters.

        loc : str, {'dist', 'best'}
            Where to store the modeled values.
        """
        model = self.afterglow_model(**params)

        # Calculate the radii
        n_eff, k_eff = model.smooth(times)
        radii = model.radii(times)

        # Store the interesting values
        self.r[loc].append(radii)
        self.n0[loc].append(n_eff)
        self.k[loc].append(k_eff)
        self.r_trans[loc].append(model.rt)

    def plot_profile(self, out_dir=None):
        """
        Plots the profiles.

        Parameters
        ----------
        out_dir : Path or str, optional
            The directory to output the plots.
        """
        self.plot_k(out_dir)
        self.plot_n(out_dir)
        self.plot_n0(out_dir)

    def plot_n(self, out_dir=None):
        """"
        Plots the density profile.

        Parameters
        ----------
        out_dir : Path or str, optional
            The directory to output the plot.
        """
        dist = []

        for i in range(len(self.n0['dist'])):
            dist.append(
                np.array(self.n0['dist'][i]) *
                (np.array(self.r['dist'][i]) / np.array(self.r_trans['dist'][i])) ** -np.array(self.k['dist'][i])
            )

        best = (
            np.array(self.n0['best'][0]) *
            (np.array(self.r['best'][0]) / np.array(self.r_trans['best'][0])) ** -np.array(self.k['best'][0])
        )

        ax = self.plot(
            self.r['dist'], dist,
            self.r['best'][0], best
        )

        ax.axvline(self.r_trans['best'][0], **self.r_trans_options)
        ax.set_title(r'Number Density Profile')
        ax.set_ylabel(r'$n [cm^{3-k}]$')
        ax.set_xlabel(r'Radius [cm]')

        if out_dir:
            save_plot_unique('n_profile', 'png', str(out_dir))
        plt.close()

    def plot_n0(self, out_dir=None):
        """"
        Plots the density normalization profile.

        Parameters
        ----------
        out_dir : Path or str, optional
            The directory to output the plot.
        """
        ax = self.plot(
            self.r['dist'], self.n0['dist'],
            self.r['best'][0], self.n0['best'][0]
        )

        ax.axvline(self.r_trans['best'][0], **self.r_trans_options)
        ax.set_title(r'Number Density Normalization Profile')
        ax.set_ylabel(r'$n_{0} [cm^{-3}]$')
        ax.set_xlabel(r'Radius [cm]')

        if out_dir:
            save_plot_unique('n0_profile', 'png', str(out_dir))
        plt.close()

    def plot_k(self, out_dir=None):
        """"
        Plots the density power-law index profile.

        Parameters
        ----------
        out_dir : Path or str, optional
            The directory to output the plot.
        """
        ax = self.plot(
            self.r['dist'], self.k['dist'],
            self.r['best'][0], self.k['best'][0],
            log_scale=False
        )

        ax.axvline(self.r_trans['best'][0], **self.r_trans_options)
        ax.set_title('Power-Law Index Profile')
        ax.set_ylabel(r'Power-Law Index k')
        ax.set_xlabel(r'Radius [cm]')
        ax.set_xscale('log')

        if out_dir:
            save_plot_unique('k_profile', 'png', str(out_dir))
        plt.close()


class SFBMIndexProfiler(Profiler):
    """
    Models a distribution of spectral indexes and provides
    methods to generates plots.

    Parameters
    ----------
    sampler : emcee.sampler
        The sampler used when running MCMC.

    params : `Parameters`
        The parameters object.
    """
    afterglow_model = StratifiedFireballModel

    dist_options = {
        'facecolor': '#2ab0ff', 'edgecolor': '#169acf',
        'bins': 'auto', 'linewidth': 0.5, 'alpha': 0.5,
    }

    best_options = {
        'color': 'red', 'linewidth': 2, 'linestyle': '--'
    }

    truth_options = {
        'value': best_options | {'color': 'tab:purple'},
        'error': {'color': 'tab:purple', 'alpha': 0.2}
    }

    def __init__(self, sampler, params):
        super().__init__(sampler, params)
        self.indices = {'best': [], 'dist': []}

    def profile(self, times, lowers, uppers, thin=10, nsamps=200):
        """
        Generates a profile for a random distribution of
        samples drawn from `sampler`. Over plots with the
        highest likelihood profile.

        Parameters
        ----------
        times : array_like
            The times of the index measurements [days].

        lowers, uppers : array_like
            The lower/upper integration bounds [Hz].

        thin : int, optional, default=1
            Take only every `thin` steps from the chain.

        nsamps : int, optional, default=100
            Number of samples to draw.
        """
        samples = self.draw(thin, nsamps)

        for i, t in enumerate(times):
            distribution = np.full(len(samples), np.nan)

            for j, s in enumerate(samples):
                # Model and store using random distribution of params
                params = self.params.samples_to_dict(s).get('model')
                distribution[j] = self.model(t, lowers[i], uppers[i], params)  # noqa

            # Model and store using the best fitting params
            best_params = self.best().get('model')
            best = self.model(t, lowers[i], uppers[i], best_params)  # noqa

            # Store
            self.indices['dist'].append(distribution)
            self.indices['best'].append(best)

    def model(self, t, lower, upper, params):
        """
        Models the spectral index distribution.

        Parameters
        ----------
        t : array_like

        lower: array_like

        upper: array_like

        params : dict

        """
        return self.afterglow_model(
            **params).spectral_index(t, lower, upper)

    def plot_profile(self, truth=None, lower=None, upper=None, out_dir=None):
        """
        Plots the profiles.

        Parameters
        ----------
        truth : array_like, optional

        lower : array_like, optional

        upper : array_like, optional

        out_dir : Path or str, optional
            The directory to output the plots.
        """
        dist = self.indices['dist']
        best = self.indices['best']

        for i, d in enumerate(dist):
            _, _, _ = plt.hist(d, **self.dist_options)
            plt.axvline(best[i], label=f'Best-fit: {best}', **self.best_options)

            if truth is not None:
                label = f'Truth: {truth} (+{upper}, -{lower})'
                plt.axvline(truth, label=label, **self.truth_options['value'])
                plt.axvspan(truth - lower, truth + upper, **self.truth_options['error'])

            # Configure the plot
            plt.title('Spectral Index Distribution')
            plt.xlabel('Spectral Index')
            plt.ylabel('Count')
            plt.legend(loc='best')
            plt.grid(alpha=0.3)

            if out_dir is not None:
                save_plot_unique('index_profile', 'png', str(out_dir))
            plt.close()
