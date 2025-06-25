import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from jetfit.core.structs import DataType
from jetfit.core.utils import save_plot_unique, days_to_sec, sec_to_days
from jetfit.models.basemodels import has_fts_transition
from scripts.plot.base import OPTION_MAP, Profiler


def plot_frequencies_ampy(ampy, out_dir=None):
    """
    Plot a distribution of characteristic frequencies using
    randomly indexed MCMC samples from a completed Ampy object.

    Parameters
    ----------
    ampy : Ampy
        The completed Ampy object.

    out_dir : Path, optional
        The output directory.
    """
    return plot_frequencies(
        ampy.mcmc.sampler, ampy.obs, ampy.mcmc.params, ampy.afterglow_model,
        model_kw=ampy.mcmc.models.afg_kw, out_dir=out_dir
    )


def plot_light_curve_ampy(ampy, title=None, out_dir=None):
    """
    Plot the best fitting light curve from a completed Ampy object.

    Parameters
    ----------
    ampy : Ampy
        The completed Ampy object.

    title : str, optional
        The title of the plot.

    out_dir : Path, optional
        The output directory.
    """
    # Light curve plotter takes an extinction object
    ext_model = None
    if ampy.extinction_model is not None:
        ext_model = ampy.extinction_model(Rv=3.1)

    plot_light_curve(
        ampy.afterglow_model, ampy.get_best_params(), ampy.obs,
        model_kw=ampy.mcmc.models.afg_kw, title=title,
        out_dir=out_dir, ext_model=ext_model
    )


def plot_density_profile_ampy(ampy, out_dir=None):
    """
    Plot the density profile as a function of radius.

    Parameters
    ----------
    ampy : Ampy
        The completed Ampy object.

    out_dir : Path, optional
        The output directory.
    """
    plot_density_profile(
        ampy.mcmc.sampler, ampy.mcmc.params, ampy.obs, ampy.afterglow_model,
        model_kw=ampy.mcmc.models.afg_kw, out_dir=out_dir
    )


def plot_frequencies(sampler, obs, params, model, model_kw=None, out_dir=None):
    """
    Plot a distribution of characteristic frequencies using
    randomly indexed MCMC samples.

    Parameters
    ----------
    sampler :
        The MCMC sampler.

    obs : Observation
        The observational data.

    params : Parameters
        The model parameters.

    model :
        The afterglow model class.

    model_kw : dict
        Any kwargs used in ``model`` constructor.

    out_dir : Path
        The output directory.
    """
    fp = FrequencyPlotter(sampler, params, model, model_kw)
    fp.plot_all(obs, out_dir=out_dir)
    plt.close()

def plot_light_curve(model, params, obs, model_kw=None, title='Light Curve', out_dir=None, ext_model=None):
    """
    Plot the best fitting light curve over the data.

    Parameters
    ----------
    model :
        The afterglow model class.

    params : Parameters
        The model parameters.

    obs : Observation
        The observational data.

    model_kw : dict
        Any kwargs used in ``model`` constructor.

    title : str, optional, default='Light Curve'
        The title of the plot.

    out_dir : Path
        The output directory.

    ext_model : , optional
        The dust extinction model object.
    """
    lc = LightCurvePlot(model, params, obs, model_kw, title)
    lc.plot(out_dir=out_dir, ext_model=ext_model)
    plt.close()


def plot_density_profile(sampler, params, obs, model, model_kw=None, out_dir=None):
    """
    Plot the density profile of the external medium.

    Parameters
    ----------
    sampler :
        The MCMC sampler.

    params : Parameters
        The model parameters.

    obs : Observation
        The observational data.

    model :
        The afterglow model class.

    model_kw : dict
        Any kwargs used in ``model`` constructor.

    out_dir : Path
        The output directory.
    """
    if model.__name__ in ('FireballModel', 'StratifiedFireballModel'):
        profiler = DensityProfiler(sampler, params, model, model_kw)
        profiler.profile(obs.times().min(), obs.times().max(),)
        profiler.plot_profile(out_dir)
        plt.close()


# <editor-fold desc="Light Curve">
class LightCurvePlot:
    """ Plots the modeled light curve. """
    def __init__(self, model, params, observation, meta=None, title='Light Curve'):
        self.model = model
        self.params = params
        self.observation = observation
        self.meta = meta if meta is not None else {}

        self.ax = None
        self._set_axes(title)

    def _set_axes(self, title: str):
        """ Sets the plotting axes. """
        _, ax = plt.subplots(figsize=(8, 8))

        ax.set_title('Light Curve')
        ax.set_ylabel('Flux (mJy)')
        ax.set_xlabel(f'Time Since Trigger (s)')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(title)

        # Add secondary x-axis
        ax2 = ax.secondary_xaxis('top', functions=(sec_to_days, days_to_sec))
        ax2.set_xlabel("Time Since Trigger (days)")

        self.ax = ax

    def plot(self, out_dir=None, spread=None, **kwargs) -> None:
        """

        Parameters
        ----------
        out_dir : Path, optional
            The directory to save `light_curve.png.`

        spread : dict, optional

        kwargs : dict
            Optional args for `plot_model(show, **kwargs)`.
        """
        self.plot_model(spread, **kwargs)
        self.plot_observation(spread)

        if out_dir is not None:
            save_plot_unique('light_curve', 'png', str(out_dir), dpi=1200)

    def plot_model(
            self, spread=None, ext_model=None, ndata: int = 200
    ) -> None:
        """
        Plots the model as a light curve. Converts all flux to
        flux density. Flux is plotted in `mJy` and the time is
        displayed in both days and seconds since trigger.

        Parameters
        ----------
        spread : dict, optional

        ext_model : dust_extinction model, optional
            Extinction model to use.

        ndata : int, optional
            The number of data points to plot.
        """

        def model_spectral_fluxes(p: dict, freq):
            """ Model the spectral fluxes. """
            afterglow_model = self.model(**p.get('model'), **self.meta)

            # fts check
            fts = has_fts_transition(
                afterglow_model.nu_m(times),
                afterglow_model.nu_c(times)
            )
            return afterglow_model.spectral_flux(times, freq, fts=fts)

        def model_integrated_fluxes(p: dict, low, upp):
            """ Model the integrated fluxes. """
            afterglow_model = self.model(**p.get('model'), **self.meta)

            # fts check
            fts = has_fts_transition(
                afterglow_model.nu_m(times),
                afterglow_model.nu_c(times)
            )

            return afterglow_model.integrated_flux(times, low, upp, fts=fts)

        # Only plot flux values
        flux_mask = self.observation.flux_loc

        # Get the flux times in seconds
        flux_times = self.observation.as_arrays.times[flux_mask]

        # Get the first data point for each band
        filters, filter_loc = np.unique(
            self.observation.as_arrays.bands[flux_mask], return_index=True
        )
        data = self.observation.data[flux_mask][filter_loc]
        spectral_data = data[self.observation.as_arrays.types[flux_mask][filter_loc] == DataType.SPECTRAL_FLUX]
        integrated_data = data[self.observation.as_arrays.types[flux_mask][filter_loc] == DataType.INTEGRATED_FLUX]

        # Modeling time [days]
        times = np.logspace(np.log10(flux_times.min()), np.log10(flux_times.max() * 2), num=ndata)

        # Plot the spectral flux for each t in `time`
        for sdata in spectral_data:
            # Define values in appropriate units
            frequency = sdata.frequency.to_value('Hz')
            wavelength = sdata.wavelength.to_value('um')

            # Model the spectral flux
            sflux = model_spectral_fluxes(self.params, frequency)

            z = self.params.get('model').get('z')
            host_corr = self.params.get('host')
            ebv_sf = self.params.get('extinction').get('ebv_source_frame')
            ebv_mw = self.params.get('extinction').get('ebv_milky_way')
            rv_milky_way = self.params.get('extinction').get('rv_milky_way')

            # Apply source dust extinction before host galaxy correction
            if 9e13 <= frequency <= 2.99e15:
                if ext_model is not None and ebv_sf is not None:
                    sflux *= ext_model.extinguish((1 + z) / wavelength, Ebv=ebv_sf)

            # Add host galaxy contribution before Milky Way dust correction
            filter_host = sdata.band + '_host'
            if host_corr is not None and filter_host in host_corr:
                sflux += host_corr[filter_host]

            # Apply Milky Way dust extinction
            if 9e13 <= frequency <= 2.99e15:
                if ext_model is not None:
                    model = ext_model

                    if rv_milky_way is not None:
                        model = ext_model.__class__(Rv=rv_milky_way)

                    sflux *= model.extinguish(1 / wavelength, Ebv=ebv_mw)

            if spread is not None:
                if sdata.band + '_offset' in spread:
                    sflux *= spread[sdata.band + '_offset']

            # Plot the modeled spectral flux
            self.ax.loglog(days_to_sec(times), sflux, '--', linewidth=1.0, color=OPTION_MAP[sdata.band]['color'])

        # Plot the integrated flux for each t in `time`
        for idata in integrated_data:
            # Define values in appropriate units
            lower = idata.int_range.lower.to_value('Hz')
            upper = idata.int_range.upper.to_value('Hz')

            # Model the integrated flux
            iflux = model_integrated_fluxes(self.params, lower, upper)

            # Convert to flux density [mJy]
            iflux_quant = u.Quantity(iflux, unit=self.observation.as_arrays.if_units)
            sflux = (iflux_quant / idata.int_range.width).to_value('mJy')

            # Plot the modeled integrated flux as a spectral flux
            self.ax.loglog(days_to_sec(times), sflux, '--', linewidth=1.0, color=OPTION_MAP[idata.band]['color'])

        self.ax.set_xlim(days_to_sec(times[0]/3), days_to_sec(times[-1]*1.5))

    def plot_observation(self, spread=None, offsets=False):
        """
        Plots the observational data including error bars.

        Parameters
        ----------
        spread : dict, optional

        offsets : bool, optional, default=False
        """
        flux_mask = self.observation.flux_loc
        arrays = self.observation.as_arrays

        # Plot each band
        filters = np.unique(arrays.bands[flux_mask])

        for dfilter in filters:

            flux, times, errors = [], [], []
            data = self.observation.data[flux_mask][arrays.bands[flux_mask] == dfilter]

            for d in data:
                if d.type == DataType.INTEGRATED_FLUX:
                    d = d.to_spectral('mJy')

                times.append(d.time.to_value('s'))

                if offsets:
                    offset = self.params.get('offsets').get(f'{dfilter}_offset')
                    if offset is not None:
                        d.value *= 10.0 ** (0.4 * offset)

                if spread is not None:
                    spread_val = spread.get(f'{dfilter}_offset')
                    if spread_val is not None:
                        d.value *= spread_val

                if d.value.to_value('mJy') != 0.0:
                    flux.append(d.value.to_value('mJy'))
                    errors.append(d.uncertainty.center.to_value('mJy'))

                # Upper limits
                else:
                    # Assumes error is 3-sigma limit
                    flux.append(d.uncertainty.center.to_value('mJy') * 3)
                    errors.append(0.0)

            # Handle options
            ms = 0.6 if OPTION_MAP[dfilter]['marker'] != '.' else 3.0

            # Plot the band
            self.ax.errorbar(times, flux, yerr=errors, fmt='.', label=dfilter, **OPTION_MAP[dfilter], markersize=ms, elinewidth=0.5)

        self.ax.legend(loc='best')
        self.ax.grid(alpha=0.5)
# </editor-fold>


# <editor-fold desc="Frequencies">
def model_freqs(model, t, params, **kwargs):
    """
    Models the critical frequencies.

    Parameters
    ----------
    model :
        The afterglow model class.

    t : np.ndarray
        The observer-frame times [d].

    params : Parameters
        The model parameters.

    kwargs :
        Any kwargs used in ``model`` constructor.
    """
    afterglow_model = model(**params.get('model'), **kwargs)

    return (
        nu_m := afterglow_model.nu_m(t),
        nu_c := afterglow_model.nu_c(t),
        model_nu_a(afterglow_model, t, nu_m, nu_c)
    )


def model_nu_a(model, t, nu_m, nu_c):
    """
    Models self-absorption which can be optional.

    Parameters
    ----------
    model :
        The afterglow model class.

    t : np.ndarray
        The observer-frame times [d].

    nu_m : np.ndarray of float
        The synchrotron frequencies [Hz].

    nu_c : np.ndarray of float
        The cooling frequencies [Hz].

    Returns
    -------
    np.ndarray of float or None
        The self-absorption frequencies [Hz].
    """
    if hasattr(model, 'nu_a'):
        if hasattr(model, 'use_sa') and not model.use_sa:
            return None
        return model.nu_a(t, nu_m=nu_m, nu_c=nu_c)


class FrequencyPlotter(Profiler):
    """
    Plots and models the characteristic frequencies.

    Parameters
    ----------
    sampler :
        The MCMC sampler.

    params : Parameters
        The model parameters.

    model :
        The afterglow model class.

    model_kw : dict
        Any kwargs used in ``model`` constructor.
    """
    def __init__(self, sampler, params, model, model_kw=None):
        super().__init__(sampler, params)
        self.model = model
        self.model_kw = model_kw

        self.ax = None
        self._set_axes()

    def _set_axes(self) -> None:
        """ Set plot axes. """
        _, ax = plt.subplots(figsize=(8, 8))

        ax.set_title('Critical Frequencies')
        ax.set_xlabel('Time Since Trigger [d]')
        ax.set_ylabel('Frequency [Hz]')

        # Define secondary axis
        ax2 = ax.secondary_xaxis('top', functions=(days_to_sec, sec_to_days))
        ax2.set_xlabel("Time Since Trigger [s]")
        self.ax = ax

    def plot_all(self, obs, out_dir=None, show=False):
        """
        Plots everything!

        Parameters
        ----------
        obs : Observation
            The observational data.

        out_dir : Path
            The output directory.

        show : bool
            Show the plot via ``plt.show()``?
        """
        times = np.geomspace(
            obs.times()[obs.flux_loc].min(),
            obs.times()[obs.flux_loc].max(),
            num=200
        )

        # Plot the frequencies
        self.plot_dist(times)
        self.plot_data(obs)

        if show:
            plt.show()

        if out_dir is not None:
            save_plot_unique('frequencies', 'png', str(out_dir), dpi=1200)

    def plot_dist(self, times):
        """
        Plot the distribution of frequencies.

        Parameters
        ----------
        times : np.ndarray
            The observer-frame times [d].
        """
        # Model the frequencies for each randomly sampled set
        for sample in self.draw(thin=10, nsamps=100):
            params = self.params.samples_to_dict(sample, cat='model')

            nu_m, nu_c, nu_a = model_freqs(
                self.model, times, params, **(self.model_kw or {})
            )

            # Plot the critical frequencies
            self.ax.loglog(times, nu_m, color='blue', alpha=0.1)
            self.ax.loglog(times, nu_c, color='orange', alpha=0.1)

            if nu_a is not None:
                self.ax.loglog(times, nu_a, color='green', alpha=0.1)

        self.plot_best(times)

    def plot_best(self, times):
        """
        Plot the best frequencies.

        Parameters
        ----------
        times : np.ndarray
            The observer-frame times [d].
        """
        # Model the most likely frequencies
        best_nu_ms, best_nu_cs, best_nu_as = model_freqs(
            self.model, times, self.best(cat='model'), **(self.model_kw or {})
        )

        # Over-plot with the most likely frequencies
        self.ax.loglog(times, best_nu_ms, color='purple', linewidth=2)
        self.ax.loglog(times, best_nu_cs, color='red', linewidth=2)

        if best_nu_as is not None:
            self.ax.loglog(times, best_nu_as, color='green', linewidth=2)

    def plot_data(self, obs):
        """
        Plot observed frequencies vs time.

        Parameters
        ----------
        obs : Observation
            The observational data.
        """
        for f in np.unique(obs.as_arrays.bands[obs.flux_loc]):
            t, nu = [], []

            for d in obs.data[obs.flux_loc]:
                if d.band == f:
                    t.append(d.time.to_value('d'))
                    nu.append(d.frequency.to_value('Hz'))

            # Plot the band
            self.ax.scatter(t, nu, label=f, **OPTION_MAP[f])

        self.ax.legend(loc='best')
        self.ax.grid(alpha=0.5)
# </editor-fold>


# <editor-fold desc="Single Density Profile"
class DensityProfiler(Profiler):
    """
    Density profiler.

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

    r_ref : dict
        The transition radius [cm].
    """
    r_ref_options = {
        'alpha': 0.5, 'linestyle': '--',
        'color': 'black', 'label': r'$R_{t}$',
    }

    def __init__(self, sampler, params, model, model_kw=None):
        super().__init__(sampler, params)

        self.afterglow_model = model
        self.afterglow_model_kw = model_kw or {}

        self.n0 = {'best': [], 'dist': []}
        self.k = {'best': [], 'dist': []}
        self.r = {'best': [], 'dist': []}
        self.r_ref = {'best': [], 'dist': []}

    def profile(self, start, stop, thin=10, nsamps=200):
        """
        Generates a profile for a random distribution of
        samples drawn from ``sampler``. Over plots with the
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
        model = self.afterglow_model(**params, **self.afterglow_model_kw)

        # Calculate the radii
        n_eff, k_eff = model.smooth(times)
        radii = model.radii(times)

        # Store the interesting values
        self.r[loc].append(radii)
        self.n0[loc].append(n_eff)
        self.k[loc].append(k_eff)
        self.r_ref[loc].append(model.ref_radius)

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
                (np.array(self.r['dist'][i]) / np.array(self.r_ref['dist'][i])) ** -np.array(self.k['dist'][i])
            )

        best = (
            np.array(self.n0['best'][0]) *
            (np.array(self.r['best'][0]) / np.array(self.r_ref['best'][0])) ** -np.array(self.k['best'][0])
        )

        ax = self.plot(
            self.r['dist'], dist,
            self.r['best'][0], best
        )

        ax.axvline(self.r_ref['best'][0], **self.r_ref_options)
        ax.set_title(r'Number Density Profile')
        ax.set_ylabel(r'$n [cm^{-3}]$')
        ax.set_xlabel(r'Radius [cm]')

        if out_dir:
            save_plot_unique('n_profile', 'png', str(out_dir), dpi=1200)
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

        ax.axvline(self.r_ref['best'][0], **self.r_ref_options)
        ax.set_title(r'Number Density Normalization Profile')
        ax.set_ylabel(r'$n_{0,ref} [cm^{-3}]$')
        ax.set_xlabel(r'Radius [cm]')

        if out_dir:
            save_plot_unique('n0_profile', 'png', str(out_dir), dpi=1200)
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

        ax.axvline(self.r_ref['best'][0], **self.r_ref_options)
        ax.set_title('Power-Law Index Profile')
        ax.set_ylabel(r'Power-Law Index k')
        ax.set_xlabel(r'Radius [cm]')
        ax.set_xscale('log')

        if out_dir:
            save_plot_unique('k_profile', 'png', str(out_dir), dpi=1200)
        plt.close()
# </editor-fold>

























