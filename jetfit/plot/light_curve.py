from pathlib import Path

import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.defns.enums import DataType


COLOR_MAP = {
    # JC Optical
    'U': '#8601AF', 'B': '#0247FE', 'V': '#66B032', 'R': '#FE2712',
    'I': '#4424D6', 'J': '#66B032', 'H': '#FC600A', 'K': '#FE2712',

    # SDSS Optical
    "u": '#4424D6', "g": '#347C98', "r": '#FC600A', "i": '#8601AF', "z": '#0247FE',

    # Swift Optical/UV/XRAY
    'uvot-u': 'cyan', 'uvot-b': 'lightblue', 'uvot-v': 'lightgreen',
    'uvw2': 'pink', 'uvm2': 'darkblue', 'uvw1': 'green',
    'xray': 'black'
}

# Aliases
COLOR_MAP['Rc'] = COLOR_MAP['R']
COLOR_MAP['Ic'] = COLOR_MAP['I']
COLOR_MAP['Ks'] = COLOR_MAP['K']
COLOR_MAP['uprime'] = COLOR_MAP['u']
COLOR_MAP['gprime'] = COLOR_MAP['g']
COLOR_MAP['rprime'] = COLOR_MAP['r']
COLOR_MAP['iprime'] = COLOR_MAP['i']
COLOR_MAP['zprime'] = COLOR_MAP['z']


def sec_to_days(x):
    """ Used for plotting axes. """
    return x / 86400


def days_to_sec(x):
    """ Used for plotting axes. """
    return x * 86400


class FrequencyPlot:
    """"""
    def __init__(self, mcmc, model, observation):
        self.mcmc = mcmc
        self.model = model
        self.observation = observation

    def plot(self, t_start, t_stop, out_dir):
        """"""
        _, ax = plt.subplots()

        flat_chain = self.mcmc.sampler.get_chain(flat=True, thin=10)
        inds = np.random.randint(len(flat_chain), size=100)

        times = np.logspace(np.log10(t_start), np.log10(t_stop), num=200)

        for ind in inds:
            params = self.mcmc.samples_to_dict(flat_chain[ind], model=True)

            # Calculate the critical frequencies
            grb_model = self.model(**params)
            nu_ms = grb_model.nu_m(times)
            nu_cs = grb_model.nu_c(times)

            ax.loglog(times, nu_ms, color='blue', alpha=0.1)
            ax.loglog(times, nu_cs, color='orange', alpha=0.1)

        # Plot the best fit frequencies
        best_params = self.mcmc.get_best_params(model=True)
        best_grb_model = self.model(**best_params)
        best_nu_ms = best_grb_model.nu_m(times)
        best_nu_cs = best_grb_model.nu_c(times)

        # Plot median frequencies
        ax.loglog(times, best_nu_ms, color='purple', linewidth=2)
        ax.loglog(times, best_nu_cs, color='red', linewidth=2)

        # Plot data as frequency vs time
        flux_mask = self.observation.flux_loc
        arrays = self.observation.as_arrays

        # Plot each band
        for dfilter in np.unique(arrays.filters[flux_mask]):
            times, frequencies = [], []
            data = self.observation.data[flux_mask][arrays.filters[flux_mask] == dfilter]

            for d in data:
                times.append(d.time.to_value('d'))
                frequencies.append(d.frequency.to_value('Hz'))

            # Plot the band
            ax.scatter(times, frequencies, label=dfilter, color=COLOR_MAP[dfilter])

        # Plot options
        ax.set_title('Critical Frequencies')
        ax.set_xlabel('Time Since Trigger (days)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(1e13, 2e18)
        ax2 = ax.secondary_xaxis('top', functions=(days_to_sec, sec_to_days))
        ax2.set_xlabel("Time Since Trigger (seconds)")
        ax.legend(loc='best')
        ax.grid(alpha=0.5)

        if out_dir is not None:
            plt.savefig(out_dir / 'frequency_scatter.png')


class LightCurvePlot:
    """"""
    def __init__(self, model, params, observation, title='Light Curve'):
        self.model = model
        self.params = params
        self.observation = observation

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
            plt.savefig(out_dir / 'light_curve.png')

    def plot_model(
            self, show: bool = False, host_corr: dict = None,
            ext_model=None, ebv_mw=None, ebv_sf=None, ndata: int = 200
    ) -> None:
        """
        Plots the model as a light curve. Converts all flux to
        flux density. Flux is plotted in `mJy` and the time is
        displayed in both days and seconds since trigger.

        Parameters
        ----------
        show : bool, optional
            If `True`, calls `plt.show()`.

        host_corr : dict, optional
            Key value pairs of 'filter' : value

        ext_model : dust_extinction model, optional
            Extinction model to use.

        ebv_mw : float, optional
            The E(B - V) Milky Way value.

        ebv_sf : float, optional
            The E(B - V) source frame value.

        ndata : int, optional
            The number of data points to plot.
        """
        model = self.model(**self.params)

        # Only plot flux values
        flux_mask = self.observation.flux_loc

        # Get the flux times in seconds
        flux_times = self.observation.as_arrays.times[flux_mask]

        # Get the first data point for each band
        filters, filter_loc = np.unique(
            self.observation.as_arrays.filters[flux_mask], return_index=True
        )
        data = self.observation.data[flux_mask][filter_loc]
        spectral_data = data[self.observation.as_arrays.types[filter_loc] == DataType.SPECTRAL_FLUX]
        integrated_data = data[self.observation.as_arrays.types[filter_loc] == DataType.INTEGRATED_FLUX]

        # Modeling time [days]
        times = np.logspace(
            np.log10(flux_times.min()), np.log10(flux_times.max() * 2),
            num=ndata
        )

        # Plot the spectral flux for each t in `time`
        for sdata in spectral_data:
            # Define values in appropriate units
            frequency = sdata.frequency.to_value('Hz')
            wavelength = sdata.wavelength.to_value('um')

            # Model the spectral flux
            sflux = model.evaluate_spectral_flux(times, frequency)

            # Apply source dust extinction before host galaxy correction
            if ext_model is not None and ebv_sf is not None:
                sflux *= ext_model.extinguish((1 + model.z) / wavelength, Ebv=ebv_sf)

            # Add host galaxy contribution before Milky Way dust correction
            filter_host = sdata.filter + '_host'
            if host_corr is not None and filter_host in host_corr:
                sflux += host_corr[filter_host]

            # Apply Milky Way dust extinction
            if ext_model is not None and ebv_mw is not None:
                sflux *= ext_model.extinguish(1 / wavelength, Ebv=ebv_mw)

            # Plot the modeled spectral flux
            self.ax.loglog(days_to_sec(times), sflux, '--', linewidth=1.5, color=COLOR_MAP[sdata.filter])

        # Plot the integrated flux for each t in `time`
        for idata in integrated_data:
            # Define values in appropriate units
            lower = idata.int_range.lower.to_value('Hz')
            upper = idata.int_range.upper.to_value('Hz')

            # Model the integrated flux
            iflux = model.evaluate_integrated_flux(times, lower, upper)

            # Convert to flux density [mJy]
            iflux_quant = u.Quantity(iflux, unit=self.observation.as_arrays.if_units)
            sflux_quant = (iflux_quant / idata.int_range.width).to('mJy')

            # Plot the modeled integrated flux as a spectral flux
            self.ax.loglog(days_to_sec(times), sflux_quant.value, '--', linewidth=1.5, color=COLOR_MAP[idata.filter])

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
                flux.append(d.value.to_value('mJy'))
                errors.append(d.uncertainty.center.to_value('mJy'))

            # Plot the band
            self.ax.errorbar(times, flux, yerr=errors, fmt='.', label=dfilter, color=COLOR_MAP[dfilter])

        self.ax.legend(loc='best')
        self.ax.grid(alpha=0.5)

        if show:
            plt.show()
