from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.defns.enums import FluxType, TimeUnits, FluxUnits
from jetfit.core.defns.evidence import Measurement, Evidence
from jetfit.core.utils.physics import TimeConversions, FluxConversions
from jetfit.core.values.time import TimeValue


class LightCurve:
    """

    Attributes
    ----------
    observation : Evidence

    model_params : ??

    model : ??

    ax : ??

    flux_units : FluxUnits, optional, default=FluxUnits.MJY
        The flux units to plot.

    time_units : FluxUnits, optional, default=TimeUnits.SEC
        The time units to plot.
    """
    def __init__(
            self,
            model,
            model_params,
            obs,
            flux_units: FluxUnits = FluxUnits.MJY,
            time_units: FluxUnits = TimeUnits.SEC,
            x_scale: str = 'log',
            y_scale: str = 'log'
    ):
        self.model = model
        self.model_params = model_params
        self.observation = obs
        self.bands = obs.get_bands()

        self.flux_units = flux_units
        self.time_units = time_units

        self.ax = None
        self.set_axes(x_scale, y_scale)

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

    def set_axes(self, x_scale: str, y_scale: str) -> None:
        """"""
        _, ax = plt.subplots(figsize=(8, 8))

        ax.set_yscale(x_scale)
        ax.set_xscale(y_scale)
        ax.set_ylabel(f'Flux ({self.flux_units.value})')
        ax.set_xlabel(f'Time Since Trigger ({self.time_units.value})')

        self.ax = ax

    def plot_model(self, show: bool = False) -> None:
        """

        Parameters
        ----------
        show : bool, optional
            If ``True``, calls `plt.show()`.
        """
        modeled_times = np.logspace(
            self.observation.optimized_x.min(),
            self.observation.optimized_x.max() * 2.0,
            num=500
        )

        for band in self.bands:
            data, fluxes = [], []

            for t in modeled_times:
                data.append(Measurement(
                    TimeValue(t, TimeUnits.SEC), band.flux[0])
                )

            # Model the data at the new times
            modeled_fluxes = self.model.evaluate(
                Evidence(data), self.model_params
            )

            if band.flux[0].type == FluxType.INTEGRATED:
                if self.flux_units == FluxUnits.MJY:
                    frequency_range = band.flux[0].frequency_range[1] - band.flux[0].frequency_range[0]
                    modeled_fluxes = modeled_fluxes / (frequency_range * FluxConversions.cgs_mjy)

            # Plot the model
            plt.loglog(
                modeled_times, modeled_fluxes,
                '--', linewidth=1.5, color=band.color
            )

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
                # Convert integrated flux to spectral flux if plotting in MJY
                if (f := band.flux[i]).type == FluxType.INTEGRATED:
                    if self.flux_units == FluxUnits.MJY:
                        f = f.get_spectral()

                # Convert the values to the desired units
                times.append(TimeConversions.convert_to(t.value, t.units, self.time_units))
                fluxes.append(FluxConversions.convert_to(f.value, f.units, self.flux_units))
                errors.append(FluxConversions.convert_to(f.avg_error, f.units, self.flux_units))

            self.ax.errorbar(times, fluxes, yerr=errors, fmt='.', label=band.name, color=band.color)

        plt.legend(loc='best')

        if show:
            plt.show()
