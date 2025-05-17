import numpy as np
from matplotlib import pyplot as plt


class Profiler:
    """

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The emcee sampler to draw samples from.

    params : Parameters
        The `Parameter` object used when running the
        MCMC with `sampler`.
    """
    dist_options = {
        'alpha': 0.3, 'linewidth': 0.5,
        'linestyle': '-', 'color': 'tab:purple'
    }

    best_options = {
        'linewidth': 2, 'linestyle': '-',
        'color': 'tab:orange', 'label': 'Best Profile'
    }

    def __init__(self, sampler, params):
        self.sampler = sampler
        self.params  = params

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

    def plot(
        self, dist_x, dist_y, best_x=None, best_y=None,
        log_scale=True, dist_kw=None, best_kw=None
    ):
        """
        Creates the plot for an arbitrary distribution.

        Parameters
        ----------
        dist_x, dist_y : array_like of array_like
            The distribution of x, y values.

        best_x, best_y : array_like of float, optional
            The best distribution.

        log_scale : bool, optional, default=True
            Should the plot be in log scale?

        dist_kw, best_kw : dict, optional
            All parameters supported by `plt.plot` for
            the distribution and/or best curves. Defaults
            are `.dist_options` and `.best_options`.

        Returns
        -------
        matplotlib.axes.Axes
            The figure object.
        """
        _, ax = plt.subplots()

        dist_options = self.dist_options | dist_kw if dist_kw else self.dist_options
        best_options = self.best_options | best_kw if dist_kw else self.best_options

        for i, dx in enumerate(dist_x):
            ax.plot(dx, dist_y[i], **dist_options)  # noqa

        if best_x is not None and best_y is not None:
            ax.plot(best_x, best_y, **best_options)

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.legend(loc='best')
        ax.grid(alpha=0.5)

        return ax
