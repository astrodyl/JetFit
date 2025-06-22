from pathlib import Path

import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.structs import DataType
from jetfit.scripts.plot.helpers import sec_to_days, days_to_sec, OPTION_MAP
from jetfit.models.basemodels import has_fts_transition
from jetfit.models.fireball import StratifiedFireballModel


class LightCurvePlot:
    """ Plots the modeled light curve. """
    def __init__(self, model, params, observation, meta=None, title='Light Curve'):
        self.model = model
        self.params = params
        self.observation = observation
        self.meta = meta if meta is not None else {}

        self.ax = None
        self._set_axes(title)

    def _set_axes(self, title: str) -> None:
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

    def plot(self, show: bool = False, out_dir: str | Path = None, spread=None, **kwargs) -> None:
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
        self.plot_model(show, spread, **kwargs)
        self.plot_observation(show, spread)

        if show:
            plt.show()

        if out_dir is not None:
            plt.savefig(out_dir / 'light_curve.png', dpi=1200)

    def plot_model(
            self, show: bool = False, spread=None, ext_model=None, ndata: int = 200
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

        # if show:
        #     plt.show()

    def plot_observation(self, show: bool = False, spread=None) -> None:
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
        filters = np.unique(arrays.bands[flux_mask])

        for dfilter in filters:

            flux, times, errors = [], [], []
            data = self.observation.data[flux_mask][arrays.bands[flux_mask] == dfilter]

            for d in data:
                if d.type == DataType.INTEGRATED_FLUX:
                    d = d.to_spectral('mJy')

                times.append(d.time.to_value('s'))

                # offset = self.params.get('offsets').get(f'{dfilter}_offset')
                # if offset is not None:
                #     d.value *= 10.0 ** (0.4 * offset)

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

        # if show:
        #     plt.show()
