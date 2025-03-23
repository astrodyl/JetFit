from pathlib import Path

import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.defns.enums import DataType
from jetfit.core.input import Observation


def sec_to_days(x):
    """ Used for plotting axes. """
    return x / 86400


def days_to_sec(x):
    """ Used for plotting axes. """
    return x * 86400


class LightCurve:
    """ """
    def __init__(
            self,
            model,
            model_params,
            obs,
            cal_offset = None,
            host_corr = None,
            x_scale: str = 'log',
            y_scale: str = 'log',
            title: str = 'Light Curve',
    ):
        self.model = model
        self.model_params = model_params
        self.host_corr = host_corr
        self.cal_offset = cal_offset
        self.observation = obs
        self.bands = obs.get_bands()

        self.ax = None
        self.set_axes(x_scale, y_scale, title)

    def plot(self, show: bool = False, out_dir: str | Path = None) -> None:
        """

        Parameters
        ----------
        show : bool, optional
            If ``True``, calls `plt.show()`.

        out_dir : str or Path, optional
            The directory to save `light_curve.png.`
        """
        self.plot_model()
        self.plot_obs()

        if show:
            plt.show()

        if out_dir is not None:
            plt.savefig(out_dir / 'light_curve.png')

    def set_axes(self, x_scale: str, y_scale: str, title: str) -> None:
        """"""
        _, ax = plt.subplots(figsize=(8, 8))

        ax.set_title(title)
        ax.set_yscale(x_scale)
        ax.set_xscale(y_scale)
        ax.set_ylabel('Flux (mJy)')
        ax.set_xlabel(f'Time Since Trigger (s)')

        self.ax = ax

    def plot_model(self, show: bool = False) -> None:
        """

        Parameters
        ----------
        show : bool, optional
            If ``True``, calls `plt.show()`.
        """
        model = self.model(**self.model_params)

        flux_times = self.observation.time_array[
            self.observation.flux_types != DataType.SPECTRAL_INDEX]

        modeled_times = np.logspace(
            np.log10(flux_times.min()),
            np.log10(flux_times.max() * 2.0),
            num=300
        )

        for band in self.bands:
            data, fluxes = [], []

            for t in modeled_times:
                datum = band.flux[0].copy()
                datum.time = u.Quantity(t, u.s)
                data.append(datum)

            # Model the data at the new times
            modeled_fluxes = model.model(Observation(data))

            # Apply host galaxy correction
            host_name = band.name + '_host'
            if host_name in self.host_corr:
                modeled_fluxes += self.host_corr[host_name]

            # Apply cal offsets
            # offset_name = band.name + '_offset'
            # if offset_name in self.cal_offset:
            #     modeled_fluxes *= 10.0 ** -(0.4 * self.cal_offset[offset_name])

            # Convert integrated flux to a flux density in mJy
            if band.flux[0].type == DataType.INTEGRATED_FLUX:
                frequency_range = band.flux[0].int_range.upper.value - band.flux[0].int_range.lower.value
                modeled_fluxes = modeled_fluxes / (frequency_range * 1.0e-26)

            # Plot the model
            self.ax.loglog(modeled_times, modeled_fluxes, '--', linewidth=1.5, color=band.color)

        if show:
            plt.show()

    def plot_obs(self, show: bool = False) -> None:
        """
        Plots the observational data including error bars.

        Parameters
        ----------
        show : bool, optional, default=False
            If ``True``, calls `plt.show()`.
        """
        for band in self.bands:
            times, fluxes, errors = [], [], []

            for i, t in enumerate(band.times):
                # Convert integrated flux to a flux density in mJy
                if (f := band.flux[i].copy()).type == DataType.INTEGRATED_FLUX:
                        f = f.to_spectral()

                times.append(t.to_value('s'))
                fluxes.append(f.value.to_value('mJy'))
                errors.append(f.uncertainty.center.to_value('mJy'))

            self.ax.errorbar(times, fluxes, yerr=errors, fmt='.', label=band.name, color=band.color)

        self.ax.legend(loc='best')
        self.ax.grid(alpha=0.5)
        ax2 = self.ax.secondary_xaxis('top', functions=(sec_to_days, days_to_sec))
        ax2.set_xlabel("Time Since Trigger (days)")

        if show:
            plt.show()
