import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
import astropy.constants as const

from jetfit.core.core import save_plot_unique
from jetfit.core.values import SpectralIndex
from jetfit.models.basemodels import SpectralIndexModel, BlastWaveModel

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
    def __init__(self, sampler, params, regimes=None, dynamic=False):
        self.sampler = sampler
        self.params  = params
        self.regimes = regimes
        self.dynamic = dynamic

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
        t : float or astropy.Quantity['time']
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
            p = self.params.samples_to_dict(s, group=self.group(time))
            model = self.afterglow_model(**p.get('model'))

            # Model the spectral index
            modeled[i] = SpectralIndexModel(
                model.nu_m(time), model.nu_c(time), model.f_peak(time),
                model.p, model.k,  model.nu_a(time)
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

        # Use the uncorrected obs time to eval model
        return SpectralIndexModel(
            model.nu_m(time), model.nu_c(time), model.f_peak(time),
            model.p, model.k, model.nu_a(time)
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
    def plot_truth(val, l, up, **kwargs):
        """
        Over-plots the accepted value.

        Parameters
        ----------
        val : float
            The spectral index value.

        l: float
            The lower uncertainty range.

        up: float
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
            val, label=f'True Value: {val} (+{up}, -{l})', **options
        )

        # Plot the uncertainty as a shaded region
        plt.axvspan(val - l, val + up, color=options.get('color'), alpha=0.2)

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
    Plots n(r / r17) vs. r / r17
    """
    def __init__(self, sampler, params, regimes=None):
        super().__init__(sampler, params, regimes)

        self.r = {'best': [], 'dist': []}
        self.k = {'best': [], 'dist': []}
        self.n17 = {'best': [], 'dist': []}
        self.n017 = {'best': [], 'dist': []}
        self.times = {'best': [], 'dist': []}

    def model(self, p, start, stop, loc):
        """"""
        times = np.geomspace(start, stop, 200)

        # Initialize return arrays
        ks = np.full(len(times), p['k'])
        n017s = np.full(len(times), p['rho0'])

        # Create the blast wave model
        model = BlastWaveModel(p['E'], p['rho0'], p['k'])

        # Calculate the burst-frame deceleration time [d]
        t_dec = model.decel_time() / 86_400

        # Calculate the shock radius [cm]
        radii = model.shock_radius(p['z'], times, t_dec)

        # Store the things
        self.n17[loc].append(n017s * (radii / 1e17) ** -ks)
        self.n017[loc].append(n017s)
        self.r[loc].append(radii)
        self.k[loc].append(ks)

    def model_dist(self, start, stop, thin=10, nsamps=100):
        """

        Parameters
        ----------
        start : float
            The start time measured in days.

        stop : float
            The stop time measured in days.

        thin : int, optional, default=1
            Take only every `thin` steps from the chain.

        nsamps : int, optional, default=100
            Number of samples to draw.

        """
        for s in self.draw(thin, nsamps):
            if (p := self.params.samples_to_dict(s).get('model')) is None:
                p = self.params.samples_to_dict(s).get('shared').get('model')
            self.model(p, start, stop, 'dist')

    def model_best(self, start, stop):
        """

        Parameters
        ----------
        start : float
            The start time measured in days.

        stop : float
            The stop time measured in days.
        """
        if (p := self.best().get('model')) is None:
            p = self.best().get('shared').get('model')

        self.model(p, start, stop, 'best')

    def plot(self, start, stop, thin=10, nsamps=100, out_dir=None):
        """

        Parameters
        ----------
        start : float
            The start time measured in days.

        stop : float
            The stop time measured in days.

        thin : int, optional, default=1
            Take only every `thin` steps from the chain.

        nsamps : int, optional, default=100
            Number of samples to draw.

        out_dir : Path, optional
            The output directory to save the figures.
        """
        self.model_dist(start, stop, thin, nsamps)
        self.model_best(start, stop)

        # Distribution plotting options
        dist_options = {
            'alpha': 0.3, 'linewidth': 0.5, 'linestyle': '-', 'color': 'tab:purple'}
        best_options = {
            'linewidth': 2, 'linestyle': '-', 'color': 'tab:orange'}

        # Plot the density profile
        for i, n17 in enumerate(self.n17['dist']):
            plt.loglog(self.r['dist'][i], n17, **dist_options)
        plt.loglog(self.r['best'][0], self.n17['best'][0], label=r'Best $n_{17}$', **best_options)

        plt.title('Density Profile')
        plt.xlabel('Shock Radius R [cm]')
        plt.ylabel(r'$n_{17}$ [$cm^{-3}$]')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)

        save_plot_unique('n17_profile', 'png', str(out_dir))
        plt.close()

        # Plot the density normalization
        for i, n017 in enumerate(self.n017['dist']):
            plt.loglog(self.r['dist'][i], n017, **dist_options)
        plt.loglog(self.r['best'][0], self.n017['best'][0], label=r'Best $n_{0,17}$', **best_options)

        plt.title('Density Normalization Profile')
        plt.xlabel(r'Shock Radius R [cm]')
        plt.ylabel(r'$n_{0,17}$ [$cm^{-3}$]')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)

        save_plot_unique('n017_profile', 'png', str(out_dir))
        plt.close()

        # # Plot the density power law index
        for i, k in enumerate(self.k['dist']):
            plt.plot(self.r['dist'][i], k, **dist_options)
        plt.plot(self.r['best'][0], self.k['best'][0], label=r'Best k', **best_options)

        plt.title('Density Power Law Index')
        plt.xlabel(r'Shock Radius R [cm]')
        plt.xscale('log')
        plt.ylabel('k')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)

        save_plot_unique('k_profile', 'png', str(out_dir))
        plt.close()
