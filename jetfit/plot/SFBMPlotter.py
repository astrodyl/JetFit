import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.core import save_plot_unique
from jetfit.models2.basemodels import BlastWaveModel
from jetfit.models2.fireball import StratifiedFireballModel
from jetfit.plot.light_curve import sec_to_days


class SFBMDensityProfiler:
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
    n : dict
        The density normalizations.

    k : dict
        The density power-law indices.

    r : dict
        The blast wave radii.

    r_trans : float
        The transition radius [cm].
    """
    def __init__(self, sampler, params):
        self.afterglow_model = StratifiedFireballModel
        self.sampler = sampler
        self.params = params

        self.n = {'best': [], 'dist': []}
        self.k = {'best': [], 'dist': []}
        self.r = {'best': [], 'dist': []}
        self.r_trans = None

    def draw(self, thin=1, nsamps=100):
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

    def best(self, **kwargs):
        """
        Returns the highest likelihood set of parameters.

        Parameters
        ----------
        kwargs :
            Any args passed to `params.samples_to_dict()`.

        Returns
        -------
        np.ndarray or dict
            The highest likelihood set of parameters.
        """
        max_index = np.nanargmax(self.sampler.get_log_prob(flat=True))
        params = self.sampler.get_chain(flat=True)[max_index]
        return self.params.samples_to_dict(params, **kwargs)

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
            self.model(times, params, 'dist')  # type: ignore

        # Model and store using the best fitting params
        best_params = self.best().get('model')
        self.model(times, best_params, 'best')  # type: ignore

    def model(self, times, params, loc):
        """
        Models the blast wave radii, densities, and
        power-law indices.

        Parameters
        ----------
        times : np.ndarray of float
            The times to model[days].

        params : dict
            The model parameters.

        loc : str, {'dist', 'best'}
            Where to store the modeled values.
        """
        model = self.afterglow_model(**params)

        # Store the transition radius
        if loc == 'best':
            self.r_trans = model.rt

        # Generate modeled times
        blastwave_model = BlastWaveModel(
            model.E, model.nt, model.k1
        )

        # Use the modeled times to generate a dense plot
        radii = blastwave_model.shock_radius(
            model.z, times, sec_to_days(blastwave_model.decel_time())
        )

        # Model the densities and indices
        n_eff, k_eff = model.smooth(times, radii)

        # Store the densities and indices
        self.r[loc].append(radii)
        self.n[loc].append(n_eff)
        self.k[loc].append(k_eff)

    def as_dict(self):
        """ Returns the inferred parameters as a dict. """
        return {
            'n': self.n, 'k': self.k,
            'r': self.r, 'r_trans': self.r_trans,
        }


class SFBMDensityPlotter:
    """
    Plots the density profile for `SFBMDensityProfiler`.

    Parameters
    ----------
    n : dict
        The density normalizations.

    k : dict
        The density power-law indices.

    r : dict
        The blast wave radii.

    r_trans : float
        The transition radius [cm].
    """
    dist_options = {
        'alpha': 0.3, 'linewidth': 0.5,
        'linestyle': '-', 'color': 'tab:purple'
    }

    best_options = {
        'linewidth': 2, 'linestyle': '-',
        'color': 'tab:orange'
    }

    def __init__(self, n, k, r, r_trans=None):
        self.n = n
        self.k = k
        self.r = r
        self.r_trans = r_trans

    def plot(self, out_dir=None):
        """
        Plots the profiles.

        Parameters
        ----------
        out_dir : Path, optional
            The directory to output the plots.
        """
        self.plot_n(out_dir)
        self.plot_k(out_dir)

    def plot_n(self, out_dir=None):
        """"
        Plots the density normalization profile.

        Parameters
        ----------
        out_dir : Path, optional
            The directory to output the plot.
        """
        _, ax = plt.subplots()

        # Plot the n distributions and best
        for i, n in enumerate(self.n['dist']):
            ax.loglog(self.r['dist'][i], n, **self.dist_options)
        ax.loglog(self.r['best'][0], self.n['best'][0], label='Best n profile', **self.best_options)

        if self.r_trans:
            ax.axvline(self.r_trans, linestyle='--', color='black', alpha=0.5, label=r'$R_{t}$')

        ax.set_title(r'Density Normalization Profile')
        ax.set_ylabel(r'$n_{0} [cm^{-3}]$')
        ax.set_xlabel(r'Radius [cm]')
        ax.grid(alpha=0.5)
        ax.legend(loc='best')

        if out_dir:
            save_plot_unique('n_profile', 'png', str(out_dir))
        plt.close()

    def plot_k(self, out_dir=None):
        """"
        Plots the density power-law index profile.

        Parameters
        ----------
        out_dir : Path, optional
            The directory to output the plot.
        """
        _, ax = plt.subplots()

        # Plot the n distributions and best
        for i, k in enumerate(self.k['dist']):
            ax.loglog(self.r['dist'][i], k, **self.dist_options)
        ax.loglog(self.r['best'][0], self.k['best'][0], label='Best k profile', **self.best_options)

        if self.r_trans:
            ax.axvline(self.r_trans, linestyle='--', color='black', alpha=0.5, label=r'$R_{t}$')

        ax.set_title('Power-Law Index Profile')
        ax.set_ylabel(r'Power-Law Index k')
        ax.set_xlabel(r'Radius [cm]')
        ax.set_xscale('log')
        ax.legend(loc='best')
        ax.grid(alpha=0.5)

        if out_dir:
            save_plot_unique('k_profile', 'png', str(out_dir))
        plt.close()
