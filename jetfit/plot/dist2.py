import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
import astropy.constants as const

from jetfit.core.core import save_plot_unique
from jetfit.core.values import SpectralIndex
from jetfit.models2.basemodels import SpectralIndexModel, BlastWaveModel
from jetfit.scripts.derivations import trans_radius, trans_time

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
    Plots n(r / r17) vs. r / r17
    """
    def __init__(self, sampler, params, regimes=None):
        super().__init__(sampler, params, regimes)

    @staticmethod
    def model_stratified(p_early, p_late, start, stop):
        """
        Models stratified density profiles.

        Parameters
        ----------
        p_early : dict

        p_late : dict

        start : float

        stop : float

        Returns
        -------
        tuple of length 3
        """
        times = np.geomspace(start, stop, 200)

        # Initialize return arrays
        ks = np.full(len(times), np.nan)
        radii = np.full(len(times), np.nan)
        modeled = np.full(len(times), np.nan)

        # Calculate the transition radius [cm]
        r_trans = trans_radius(
            n1=p_early['rho0'], n2=p_late['rho0'],
            k1=p_early['k'], k2=p_late['k']
        )

        # Calculate the observer frame transition time [d]
        t_trans = trans_time(
            E=1e52 * p_early['E'],
            n=m_p.value * p_early['rho0'] * 1e17 ** p_early['k'],
            r=r_trans, k=p_early['k'], z=p_early['z']
        ).to_value('d')

        # Define the early time model
        early_model = BlastWaveModel(
            E=p_early['E'], n0=p_early['rho0'], k=p_early['k']
        )

        # Calculate the (observer frame) deceleration time [d]
        t_dec = early_model.decel_time(gamma=300) / 86_400

        # Make sure units in [cm] and is a float
        r_trans = r_trans.to_value('cm')

        for i, t_obs in enumerate(times):
            if t_obs < t_trans:  # noqa
                r = early_model.shock_radius(p_early['z'], t_obs, t_dec) / 1e17
                rho = p_early['rho0'] * r ** -p_early['k']
                ks[i] = p_early['k']
            else:
                r = r_trans * ((t_dec + t_obs) / t_trans) ** (1 / (4 - p_late['k'])) / 1e17 # noqa
                rho = p_late['rho0'] * r ** -p_late['k']
                ks[i] = p_late['k']

            radii[i], modeled[i] = r, rho

        return radii, modeled, ks

    @staticmethod
    def model_single(p, start, stop):
        """"""
        times = np.geomspace(start, stop, 200)

        # Initialize return arrays
        ks = np.full(len(times), np.nan)
        radii = np.full(len(times), np.nan)
        modeled = np.full(len(times), np.nan)

        # Create the blast wave model
        model = BlastWaveModel(p['E'], p['rho0'], p['k'])

        # Calculate the deceleration time [d]
        t_dec = model.decel_time(gamma=300) / 86_400

        for i, t_obs in enumerate(times):
            ks[i] = p['k']

            # Calculate the shock radius
            radii[i] = model.shock_radius(p['z'], t_obs, t_dec) / 1e17

            # Calculate the density [cm-3]
            modeled[i] = p['rho0'] * (radii[i] ** -p['k'])

        return radii, modeled, ks

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
        self.plot_dist(start, stop, thin, nsamps)
        self.plot_best(start, stop)

        # Configure the plot
        plt.title('Density Profile')
        plt.xlabel(r'Shock Radius $R / R_{17}$')
        plt.ylabel(r'Number Density [$cm^{-3}$]')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)

        if out_dir is not None:
            save_plot_unique('density_dist', 'png', str(out_dir))
        else:
            plt.show()
        plt.close()

        # Plots the k distributions
        self.plot_ks(start, stop, thin, nsamps)
        self.plot_ks_best(start, stop)

        # Configure the plot
        plt.title('Density Power Law Index')
        plt.xscale('log')
        plt.xlabel(r'Shock Radius $R / R_{17}$')
        plt.ylabel('k')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)

        if out_dir is not None:
            save_plot_unique('k_dist', 'png', str(out_dir))
        else:
            plt.show()
        plt.close()

    def plot_dist(self, start, stop, thin=10, nsamps=100, **kwargs):
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

        kwargs
            Any optional args accepted by `plt.loglog`.
        """

        options = {
            'alpha': 0.3,
            'linewidth': 0.5,
            'linestyle': '-',
            'color': 'tab:purple'
        } | kwargs

        # Draw the samples
        samples = self.draw(thin, nsamps)

        for s in samples:

            if self.regimes and 'early' in self.regimes.keys():
                p_early = self.params.samples_to_dict(s, group='early').get('model')
                p_late = self.params.samples_to_dict(s, group='late').get('model')
                radii, modeled, _ = self.model_stratified(p_early, p_late, start, stop)

            else:
                p = self.params.samples_to_dict(s).get('model')
                radii, modeled, _ = self.model_single(p, start, stop)

            # Plot rho(R / R17) vs. R / R17
            plt.loglog(radii, modeled, **options)

    def plot_best(self, start, stop, **kwargs):
        """
        Evaluates the spectral index model using the
        maximum likelihood values.

        Parameters
        ----------
        start : float
            The start time measured in days.

        stop : float
            The stop time measured in days.

        kwargs
            Any optional args accepted by `plt.loglog`.
        """

        options = {
            'linewidth': 2,
            'linestyle': '-',
            'color': 'tab:orange',
        } | kwargs

        if self.regimes and 'early' in self.regimes.keys():
            p_early = self.best(group='early').get('model')
            p_late  = self.best(group='late').get('model')

            radii, modeled, _ = self.model_stratified(
                p_early, p_late, start, stop
            )

        else:
            p = self.best().get('model')
            radii, modeled, _ = self.model_single(p, start, stop)

        # Plot best rho(R / R17) vs. R / R17
        plt.loglog(radii, modeled, label='Best n', **options)

    def plot_ks(self, start, stop, thin, nsamps, **kwargs):
        """"""
        options = {
            'alpha': 0.3,
            'linewidth': 0.5,
            'linestyle': '-',
            'color': 'tab:purple'
        } | kwargs

        # Draw the samples
        samples = self.draw(thin, nsamps)

        for s in samples:

            if self.regimes and 'early' in self.regimes.keys():
                p_early = self.params.samples_to_dict(s, group='early').get('model')
                p_late = self.params.samples_to_dict(s, group='late').get('model')
                radii, _, ks = self.model_stratified(p_early, p_late, start, stop)

            else:
                p = self.params.samples_to_dict(s).get('model')
                radii, _, ks = self.model_single(p, start, stop)

            # Plot k vs. R / R17
            plt.plot(radii, ks, **options)

    def plot_ks_best(self, start, stop, **kwargs):
        """"""
        options = {
              'linewidth': 2,
              'linestyle': '-',
              'color': 'tab:orange',
          } | kwargs

        if self.regimes and 'early' in self.regimes.keys():
            p_early = self.best(group='early').get('model')
            p_late = self.best(group='late').get('model')

            radii, _, ks = self.model_stratified(
                p_early, p_late, start, stop
            )

        else:
            p = self.best().get('model')
            radii, _, ks = self.model_single(p, start, stop)

        # Plot best k vs. R / R17
        plt.plot(radii, ks, label='Best n', **options)


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
