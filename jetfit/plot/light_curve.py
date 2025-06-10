from pathlib import Path

import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.defns.enums import DataType
from jetfit.models.basemodels import has_fts_transition
from jetfit.models.fireball import StratifiedFireballModel


OPTION_MAP = {
    # JC Optical/NIR (circles)
    'U': {'color': '#8601AF', 'marker': '.'},
    'B': {'color': '#0247FE', 'marker': '.'},
    'V': {'color': '#66B032', 'marker': '.'},
    'R': {'color': '#FE2712', 'marker': '.'},
    'I': {'color': '#4424D6', 'marker': '.'},
    'J': {'color': '#66B032', 'marker': '.'},
    'H': {'color': '#FC600A', 'marker': '.'},
    'K': {'color': '#FE2712', 'marker': '.'},

    # SDSS Optical (squares)
    'u': {'color': 'tab:purple', 'marker': 's'},
    'g': {'color': 'tab:blue',   'marker': 's'},
    'r': {'color': 'tab:orange', 'marker': 's'},
    'i': {'color': 'tab:red',    'marker': 's'},
    'z': {'color': 'tab:pink',   'marker': 's'},

    # Swift Optical/UV/XRAY (diamonds, hexagons)
    'uvot-u': {'color': 'cyan',       'marker': '.'},
    'uvot-b': {'color': 'lightblue',  'marker': '.'},
    'uvot-v': {'color': 'lightgreen', 'marker': '.'},
    'uvw2': {'color': 'pink',         'marker': '.'},
    'uvm2': {'color': 'darkblue',     'marker': '.'},
    'uvw1': {'color': 'green',        'marker': '.'},
    'xray': {'color': 'black',        'marker': '.'},

    # HST
    'F775W': {'color': 'yellow', 'marker': '.'},
    'F125W': {'color': 'grey',   'marker': '.'},

    # Radio
    'C': {'color': 'royalblue', 'marker': '.'},
    'C2': {'color': 'purple', 'marker': '.'},
    'Ka': {'color': 'peachpuff', 'marker': '.'},
    'Kb': {'color': 'peru', 'marker': '.'},
    'Kc': {'color': 'palevioletred', 'marker': '.'},
    'Kd': {'color': 'lightcoral', 'marker': '.'},
    'W': {'color': 'teal', 'marker': '.'},
    'S': {'color': 'teal', 'marker': '.'},
}

# Aliases
OPTION_MAP['Rc'] = OPTION_MAP['R']
OPTION_MAP['Ic'] = OPTION_MAP['I']
OPTION_MAP['Ks'] = OPTION_MAP['K']
OPTION_MAP['uprime'] = OPTION_MAP['u']
OPTION_MAP['gprime'] = OPTION_MAP['g']
OPTION_MAP['rprime'] = OPTION_MAP['r']
OPTION_MAP['iprime'] = OPTION_MAP['i']
OPTION_MAP['zprime'] = OPTION_MAP['z']


def sec_to_days(x):
    """ Used for plotting axes. """
    return x / 86400


def days_to_sec(x):
    """ Used for plotting axes. """
    return x * 86400


def get_best_params(sampler, params, **kwargs):
    """"""
    max_index = np.nanargmax(sampler.get_log_prob(flat=True))
    return params.samples_to_dict(sampler.get_chain(flat=True)[max_index], **kwargs)


def map_groups(observation, gen_times):
    """"""
    group_map = {
        g: np.full(len(gen_times), False, dtype=bool)
        for g in observation.groups
    }

    for group in group_map:
        group_times = observation.as_arrays.times[observation.groups[group]]

        for i, t in enumerate(gen_times):
            if (group_times.min() - 1e-6) <= t <= (group_times.max() + 1e-6):
                group_map[group][i] = True

    return group_map


class FrequencyPlot:
    """"""
    def __init__(self, sampler, params):
        self.sampler = sampler
        self.parameters = params

        self.ax = None
        self._set_axes()

    def _set_axes(self) -> None:
        """"""
        _, ax = plt.subplots(figsize=(8, 8))

        ax.set_title('Critical Frequencies')
        ax.set_xlabel('Time Since Trigger (days)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(1e13, 1e18)

        # Define secondary axis
        ax2 = ax.secondary_xaxis('top', functions=(days_to_sec, sec_to_days))
        ax2.set_xlabel("Time Since Trigger (seconds)")

        self.ax = ax

    def plot(self, model, obs, out_dir=None, show=False, model_kw=None):
        """"""
        self.plot_frequencies(model, obs, model_kw)
        self.plot_data(obs)

        if show:
            plt.show()

        if out_dir is not None:
            plt.savefig(out_dir / 'frequency_dist.png', dpi=600)

    def plot_frequencies(self, model, obs, model_kw):
        """"""
        if model_kw is None:
            model_kw = {}

        def model_frequencies(p: dict):
            """ Model the critical frequencies using data groups. """
            nu_ms_all = np.full(times.size, np.nan)
            nu_cs_all = np.full(times.size, np.nan)
            nu_as_all = np.full(times.size, np.nan)

            if p.get('shared'):
                groups = map_groups(obs, times)

                for group, pos in groups.items():
                    model_params = p.get(group).get('model')
                    afterglow_model = model(**model_params, **model_kw)
                    nu_ms_all[pos] = afterglow_model.nu_m(times[pos])
                    nu_cs_all[pos] = afterglow_model.nu_c(times[pos])
                    nu_as_all[pos] = afterglow_model.nu_a(times[pos], nu_ms_all[pos], nu_cs_all[pos])
            else:
                afterglow_model = model(**p.get('model'), **model_kw)

                # Handle Stratified model
                if isinstance(afterglow_model, StratifiedFireballModel):
                    n_eff, k_eff = afterglow_model.smooth(times)
                    nu_ms_all = afterglow_model.nu_m(times, k_eff)
                    nu_cs_all = afterglow_model.nu_c(times, n_eff, k_eff)
                    nu_as_all = afterglow_model.nu_a(times, n_eff, k_eff, nu_ms_all, nu_cs_all)

                # Handle regular model
                else:
                    nu_ms_all = afterglow_model.nu_m(times)
                    nu_cs_all = afterglow_model.nu_c(times)
                    nu_as_all = afterglow_model.nu_a(times, nu_ms_all, nu_cs_all)

            return nu_ms_all, nu_cs_all, nu_as_all

        # Get random locations from flattened chain
        flat_chain = self.sampler.get_chain(flat=True, thin=10)
        indices = np.random.randint(len(flat_chain), size=100)

        # Define time range for plot
        times = np.logspace(
            start=np.log10(obs.as_arrays.times[obs.flux_loc].min()),
            stop=np.log10(obs.as_arrays.times[obs.flux_loc].max()),
            num=200
        )

        for idx in indices:

            # Model params for each group
            params = self.parameters.samples_to_dict(
                flat_chain[idx], cat='model'
            )

            # For each data group...
            nu_ms, nu_cs, nu_as = model_frequencies(params)

            # Finally plot them.
            self.ax.loglog(times, nu_ms, color='blue',   alpha=0.1)
            self.ax.loglog(times, nu_cs, color='orange', alpha=0.1)
            if nu_as is not None:
                self.ax.loglog(times, nu_as, color='green',  alpha=0.1)

        best_params = get_best_params(self.sampler, self.parameters, cat='model')
        best_nu_ms, best_nu_cs, best_nu_as = model_frequencies(best_params)

        # Plot best frequencies
        self.ax.loglog(times, best_nu_ms, color='purple', linewidth=2)
        self.ax.loglog(times, best_nu_cs, color='red',    linewidth=2)

        if best_nu_as is not None:
            self.ax.loglog(times, best_nu_as, color='green',  linewidth=2)

    def plot_best(self, obs, model, out_dir=None, model_kw=None):
        """"""
        if model_kw is None:
            model_kw = {}

        _, ax = plt.subplots()

        def model_frequencies(p: dict):
            """ Model the critical frequencies using data groups. """
            nu_ms_all = np.full(times.size, np.nan)
            nu_cs_all = np.full(times.size, np.nan)
            nu_as_all = np.full(times.size, np.nan)

            if p.get('shared'):
                groups = map_groups(obs, times)

                for group, pos in groups.items():
                    model_params = p.get(group).get('model')
                    afterglow_model = model(**model_params, **model_kw)
                    nu_ms_all[pos] = afterglow_model.nu_m(times[pos])
                    nu_cs_all[pos] = afterglow_model.nu_c(times[pos])
                    nu_as_all[pos] = afterglow_model.nu_a(times[pos], nu_ms_all[pos], nu_cs_all[pos])
            else:
                afterglow_model = model(**p.get('model'), **model_kw)
                if isinstance(afterglow_model, StratifiedFireballModel):
                    n_eff, k_eff = afterglow_model.smooth(times)
                    nu_ms_all = afterglow_model.nu_m(times, k_eff)
                    nu_cs_all = afterglow_model.nu_c(times, n_eff, k_eff)
                    nu_as_all = afterglow_model.nu_a(times, n_eff, k_eff, nu_ms_all, nu_cs_all)
                else:
                    nu_ms_all = afterglow_model.nu_m(times)
                    nu_cs_all = afterglow_model.nu_c(times)
                    nu_as_all = afterglow_model.nu_a(times, nu_ms_all, nu_cs_all)
            return nu_ms_all, nu_cs_all, nu_as_all

        times = np.logspace(
            start=np.log10(obs.as_arrays.times[obs.flux_loc].min()),
            stop=np.log10(obs.as_arrays.times[obs.flux_loc].max()),
            num=200
        )

        best_params = get_best_params(self.sampler, self.parameters, cat='model')
        nu_ms, nu_cs, nu_as = model_frequencies(best_params)

        # Include indices in label
        ax.loglog(times, nu_ms, color='blue', label=r'$\nu_{m}$')
        ax.loglog(times, nu_cs, color='orange', label=r'$\nu_{c}$')
        if nu_as is not None:
            ax.loglog(times, nu_as, color='green', label=r'$\nu_{a}$')

        # Plot horizontal lines roughly corresponding to optical/xray
        plt.axhline(y=5e14, color='green', linewidth=10, alpha=0.5)
        plt.axhline(y=5e17, color='black', linewidth=10, alpha=0.5)

        ax.set_title('Critical Frequencies')
        ax.set_xlabel('Time Since Trigger (days)')
        ax.set_ylabel('Frequency (Hz)')
        ax.legend(loc='best')
        ax.grid(alpha=0.5)

        ax2 = ax.secondary_xaxis('top', functions=(days_to_sec, sec_to_days))
        ax2.set_xlabel("Time Since Trigger (seconds)")

        if out_dir is not None:
            plt.savefig(out_dir / 'frequency_best.png')

    def plot_data(self, obs):
        """ Plot data as frequency vs time """
        filters = np.unique(obs.as_arrays.filters[obs.flux_loc])

        for f in filters:
            times, frequencies = [], []

            # Get data for band == f
            data = obs.data[obs.flux_loc][obs.as_arrays.filters[obs.flux_loc] == f]

            for d in data:
                times.append(d.time.to_value('d'))
                frequencies.append(d.frequency.to_value('Hz'))

            # Plot the band
            self.ax.scatter(times, frequencies, label=f, **OPTION_MAP[f])

        self.ax.legend(loc='best')
        self.ax.grid(alpha=0.5)


class LightCurvePlot:
    """"""
    def __init__(self, model, params, observation, meta=None, title='Light Curve'):
        self.model = model
        self.params = params
        self.observation = observation
        self.meta = meta if meta is not None else {}

        self.ax = None
        self._set_axes(title)

    def _set_axes(self, title: str) -> None:
        """"""
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

    def plot(self, show: bool = False, out_dir: str | Path = None, **kwargs) -> None:
        """

        Parameters
        ----------
        show : bool, optional
            If ``True``, calls `plt.show()`.

        out_dir : str or Path, optional
            The directory to save `light_curve.png.`

        kwargs : dict
            Optional args for `plot_model(show, **kwargs)`.
        """
        self.plot_model(show, **kwargs)
        self.plot_observation(show)

        if show:
            plt.show()

        if out_dir is not None:
            plt.savefig(out_dir / 'light_curve.png', dpi=1200)

    def plot_model(
            self, show: bool = False, ext_model=None, ndata: int = 200
    ) -> None:
        """
        Plots the model as a light curve. Converts all flux to
        flux density. Flux is plotted in `mJy` and the time is
        displayed in both days and seconds since trigger.

        Parameters
        ----------
        show : bool, optional
            If `True`, calls `plt.show()`.

        ext_model : dust_extinction model, optional
            Extinction model to use.

        ndata : int, optional
            The number of data points to plot.
        """

        def model_spectral_fluxes(p: dict, freq):
            """ """
            modeled = np.full(times.size, np.nan)

            if p.get('shared') is not None:
                # TODO: fts check
                groups = map_groups(self.observation, times)

                for group, pos in groups.items():
                    model_params = p.get(group).get('model')
                    afterglow_model = self.model(**model_params, **self.meta)
                    modeled[pos] = afterglow_model.spectral_flux(times[pos], freq)
            else:
                afterglow_model = self.model(**p.get('model'), **self.meta)

                # fts check
                fts = has_fts_transition(
                    afterglow_model.nu_m(times),
                    afterglow_model.nu_c(times)
                )

                modeled = afterglow_model.spectral_flux(times, freq, fts)
            return modeled

        def model_integrated_fluxes(p: dict, low, upp):
            """ """
            modeled = np.full(times.size, np.nan)

            if p.get('shared') is not None:
                groups = map_groups(self.observation, times)

                for group, pos in groups.items():
                    model_params = p.get(group).get('model')
                    afterglow_model = self.model(**model_params, **self.meta)
                    modeled[pos] = afterglow_model.integrated_flux(times[pos], low, upp)
            else:
                afterglow_model = self.model(**p.get('model'), **self.meta)

                # fts check
                fts = has_fts_transition(
                    afterglow_model.nu_m(times),
                    afterglow_model.nu_c(times)
                )

                modeled = afterglow_model.integrated_flux(times, low, upp, fts)
            return modeled

        # Only plot flux values
        flux_mask = self.observation.flux_loc

        # Get the flux times in seconds
        flux_times = self.observation.as_arrays.times[flux_mask]

        # Get the first data point for each band
        filters, filter_loc = np.unique(
            self.observation.as_arrays.filters[flux_mask], return_index=True
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

            if 'shared' in self.params:
                params = self.params['shared']
            else:
                params = self.params

            z = params.get('model').get('z')
            host_corr = params.get('host')
            ebv_sf = params.get('extinction').get('ebv_source_frame')
            ebv_mw = params.get('extinction').get('ebv_milky_way')
            rv_milky_way = params.get('extinction').get('rv_milky_way')

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

        if show:
            plt.show()

    def plot_observation(self, show: bool = False) -> None:
        """
        Plots the observational data including error bars.

        Parameters
        ----------
        show : bool, optional, default=False
            If ``True``, calls `plt.show()`.
        """
        flux_mask = self.observation.flux_loc
        arrays = self.observation.as_arrays

        # Plot each band
        for dfilter in np.unique(arrays.filters[flux_mask]):
            flux, times, errors = [], [], []
            data = self.observation.data[flux_mask][arrays.filters[flux_mask] == dfilter]

            for d in data:
                if d.type == DataType.INTEGRATED_FLUX:
                    d = d.to_spectral('mJy')

                times.append(d.time.to_value('s'))

                # offset = self.params.get('offsets').get(f'{dfilter}_offset')
                # if offset is not None:
                #     d.value *= 10.0 ** (0.4 * offset)

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

        if show:
            plt.show()
