import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
import astropy.constants as const

from jetfit.core.core import save_plot_unique
from jetfit.core.values import SpectralIndex
from jetfit.models2.basemodels import SpectralIndexModel, ShockRadius

# Constants in cgs units
m_p = const.m_p.cgs  # noqa
m_e = const.m_e.cgs  # noqa
q_e = u.Quantity(4.8032e-10 * u.g**0.5 * u.cm**1.5 / u.s)
c = const.c.cgs  # noqa
tcs = const.sigma_T.cgs  # noqa


class Distribution:
    """

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The emcee sampler to draw samples from.

    params : Parameters
        The `Parameter` object used when running the
        MCMC with `sampler`.

    regimes : dict, optional
        The data groups dict that maps the group
        name it's valid temporal regime.
    """
    def __init__(self, sampler, params, regimes=None):
        self.sampler = sampler
        self.params  = params
        self.regimes = regimes

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

    def group(self, t):
        """
        Returns the name of the group that encompasses `t`.

        Parameters
        ----------
        t : float or astropy.Quantity[time']
            The time to check.

        Returns
        -------
        str or None
            The group name that encompasses `t`.
        """
        for group, regime in self.regimes.items():
            if regime.encompasses(t):
                return group

    def best(self, as_dict=True, **kwargs):
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


class SpectralIndexPlot(Distribution):
    # TODO: Split this into plotting and modeling class
    """
    Models a distribution of spectral indexes and provides
    methods to generates plots.

    Parameters
    ----------
    model : ??
        The afterglow model with spectral index support.
    """
    def __init__(self, sampler, params, model, regimes=None):
        super().__init__(sampler, params, regimes)
        self.afterglow_model = model

    def __call__(self, *args, **kwargs):
        """ Calls the model method. """
        return self.model(*args, **kwargs)

    def evaluate(self, time, lower, upper, thin=10, nsamps=200):
        """
        Evaluates the spectral index model for each
        randomly drawn set of parameters from the
        sampler.

        Parameters
        ----------
        time : float
            The time to evaluate.

        lower : float
            The lower integration bound.

        upper : float
            The upper integration bound.

        thin : int, optional, default=1
            Take only every `thin` steps from the chain.

        nsamps : int, optional, default=100
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            The evaluated spectral index values.
        """
        samples = self.draw(thin, nsamps)
        modeled = np.full(len(samples), np.nan)

        for i, s in enumerate(samples):
            p = self.params.samples_to_dict(
                s, group=self.group(time))

            model = self.afterglow_model(**p.get('model'))

            modeled[i] = SpectralIndexModel(
                model.nu_m(time), model.nu_c(time),
                model.f_peak(time), model.p, model.k
            )(lower, upper)

        return modeled

    def evaluate_best(self, time, lower, upper):
        """
        Evaluates the spectral index model using the
        maximum likelihood values.

        Parameters
        ----------
        time : float
            The time to evaluate.

        lower : float
            The lower integration bound.

        upper : float
            The upper integration bound.

        Returns
        -------
        float
            The most likely spectral index value.
        """
        best = self.best(group=self.group(time)).get('model')
        model = self.afterglow_model(**best)

        return SpectralIndexModel(
            model.nu_m(time), model.nu_c(time),
            model.f_peak(time), model.p, model.k
        )(lower, upper)

    def model(self, indices, out_dir=None):
        """
        Calculates and plots the spectral index distribution
        for each provided spectral index.

        Parameters
        ----------
        indices : array_like of `SpectralIndex`
            The `SpectralIndex` objects to model using
            the randomly sampled values.

        out_dir : Path, optional
            The output directory to save the figures.
        """
        for index in indices:

            # Convert to floats with expected units
            time = index.time.to_value('d')
            lower = index.int_range.lower.to_value('Hz')
            upper = index.int_range.upper.to_value('Hz')

            # Get a distribution of values and the best value
            best = self.evaluate_best(time, lower, upper)
            distribution = self.evaluate(time, lower, upper)

            # Create the various AMAZING plots
            title = f"Spectral Index Distribution"
            if self.group(time) is not None:
                title += f" (DataGroup='{self.group(time)}')"

            self.plot(distribution, best, index, title, out_dir)

    def plot(self, dist, best, truth, title=None, out_dir=None):
        """
        Plots the distribution of spectral index values.
        Over-plots with the best fit value and truth value.

        Parameters
        ----------
        dist : np.ndarray
            The distribution of spectral indices.

        best : float
            The best fit spectral index.

        truth : SpectralIndex
            The accepted spectral index.

        title : str, optional
            The title of the plot.

        out_dir : Path, optional
            The directory to save the figures.
        """
        self.plot_distribution(dist)
        self.plot_best(best)

        self.plot_truth(
            truth.value.value,
            truth.uncertainty.lower.value,
            truth.uncertainty.upper.value,
        )

        # Configure the plot
        plt.title(title if title else 'Spectral Index Distribution')
        plt.xlabel('Spectral Index')
        plt.ylabel('Count')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)

        if out_dir is not None:
            save_plot_unique('index_dist', 'png', str(out_dir))
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_truth(val, l, u, **kwargs):
        """
        Over-plots the accepted value.

        Parameters
        ----------
        val : float
            The spectral index value.

        l: float
            The lower uncertainty range.

        u: float
            The upper uncertainty range.

        kwargs
            Any kwargs accepted by `plt.axvline`.
        """

        options = {
            'color': 'tab:purple',
            'linewidth': 2,
            'linestyle': '--',
        } | kwargs

        plt.axvline(
            val, label=f'True Value: {val} (+{u}, -{l})', **options
        )

        # Plot the uncertainty as a shaded region
        plt.axvspan(val - l, val + u, color=options.get('color'), alpha=0.2)

    @staticmethod
    def plot_best(val, **kwargs):
        """
        Over-plots the best fit value.

        Parameters
        ----------
        val : float
            The spectral index value.

        kwargs
            Any kwargs accepted by `plt.axvline`.
        """

        options = {
            'color': 'red',
            'linewidth': 2,
            'linestyle': '--'
        } | kwargs

        plt.axvline(val, label=f'Best-fit Value: {val}', **options)

    @staticmethod
    def plot_distribution(dist, **kwargs):
        """
        Over-plots the accepted value.

        Parameters
        ----------
        dist : np.ndarray
            The distribution of spectral index values.

        kwargs
            Any kwargs accepted by `plt.hist`.
        """

        options = {
            'facecolor': '#2ab0ff',
            'edgecolor': '#169acf',
            'bins': 'auto',
            'linewidth': 0.5,
            'alpha': 0.5,
        } | kwargs

        cts, bins, _ = plt.hist(dist, **options)


class DensityProfilePlot(Distribution):
    """

    Parameters
    ----------

    """
    def __init__(self, sampler, params, regimes=None):
        super().__init__(sampler, params, regimes)

    def evaluate(self, start, stop, thin=10, nsamps=100):
        """
        Evaluates the spectral index model for each
        randomly drawn set of parameters from the
        sampler.

        Parameters
        ----------

        thin : int, optional, default=1
            Take only every `thin` steps from the chain.

        nsamps : int, optional, default=100
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            The evaluated spectral index values.
        """
        samples = self.draw(thin, nsamps)
        modeled = np.full(len(samples), np.nan)
        times = np.logspace(start, stop, 100)

        for i, s in enumerate(samples):

            for time in times:

                # Get the model parameters
                p = self.params.samples_to_dict(
                    s, group=self.group(time)
                ).get('model')

                modeled[i] = ShockRadius(
                    p['E'], p['rho0'], p['k'], p['z']
                )(time)

        return modeled

    def evaluate_best(self, time, lower, upper):
        """
        Evaluates the spectral index model using the
        maximum likelihood values.

        Parameters
        ----------
        time : float
            The time to evaluate.

        lower : float
            The lower integration bound.

        upper : float
            The upper integration bound.

        Returns
        -------
        float
            The most likely spectral index value.
        """
        best = self.best(group=self.group(time)).get('model')
        model = self.afterglow_model(**best)

        return SpectralIndexModel(
            model.nu_m(time), model.nu_c(time),
            model.f_peak(time), model.p, model.k
        )(lower, upper)

    def model(self, indices, out_dir=None):
        """
        Calculates and plots the spectral index distribution
        for each provided spectral index.

        Parameters
        ----------
        indices : array_like of `SpectralIndex`
            The `SpectralIndex` objects to model using
            the randomly sampled values.

        out_dir : Path, optional
            The output directory to save the figures.
        """
        for index in indices:

            # Convert to floats with expected units
            time = index.time.to_value('d')
            lower = index.int_range.lower.to_value('Hz')
            upper = index.int_range.upper.to_value('Hz')

            # Get a distribution of values and the best value
            best = self.evaluate_best(time, lower, upper)
            distribution = self.evaluate(time, lower, upper)

            # Create the various AMAZING plots
            title = f"Spectral Index Distribution"
            if self.group(time) is not None:
                title += f" (DataGroup='{self.group(time)}')"

            self.plot(distribution, best, index, title, out_dir)









class JetBeamingPlot(Distribution):
    """
    Models a distribution of jet-opening angles and
    beam-corrected energies and provides methods to
    generates their plots.
    """
    def __init__(self, sampler, params, regimes=None):
        super().__init__(sampler, params, regimes)

    def plot(self):
        """"""
        pass

class FrequencyDistribution(Distribution):
    """"""
    def __init__(self, sampler):
        super().__init__(sampler)

    def plot(self):
        """"""
        pass
