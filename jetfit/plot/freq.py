import numpy as np
from matplotlib import pyplot as plt

from jetfit.core.core import save_plot_unique
from jetfit.plot.helpers import days_to_sec, sec_to_days,  OPTION_MAP
from jetfit.plot.base import Profiler


def model_freqs(model, t, params, **kwargs):
    """ Models the critical frequencies. """
    if params.get('shared') is not None:
        params = params['shared']

    afterglow_model = model(**params.get('model'), **kwargs)

    return (
        nu_m := afterglow_model.nu_m(t),
        nu_c := afterglow_model.nu_c(t),
        model_nu_a(afterglow_model, t, nu_m, nu_c)
    )


def model_nu_a(model, t, nu_m, nu_c):
    """ Models self-absorption which can be optional. """
    if hasattr(model, 'use_sa') and not model.use_sa:
        return None
    return model.nu_a(t, nu_m, nu_c)


class FrequencyPlotter(Profiler):
    """ Plots the critical frequencies."""
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
        ax.set_xlabel('Time Since Trigger (days)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(1e13, 1e18)

        # Define secondary axis
        ax2 = ax.secondary_xaxis('top', functions=(days_to_sec, sec_to_days))
        ax2.set_xlabel("Time Since Trigger [s]")
        self.ax = ax

    def plot_all(self, obs, out_dir=None, show=False):
        """ Plot everything! """
        times = np.geomspace(
            obs.as_arrays.times[obs.flux_loc].min(),
            obs.as_arrays.times[obs.flux_loc].max(),
            num=200
        )

        # Plot the frequencies
        self.plot_dist(times)
        self.plot_data(obs)

        if show:
            plt.show()

        if out_dir is not None:
            save_plot_unique(
                'frequency_dist', 'png', str(out_dir)
            )

        self.plot_legacy(times)

        if out_dir is not None:
            save_plot_unique(
                'frequency_best', 'png', str(out_dir)
            )

    def plot_dist(self, times):
        """ Plot the distribution of frequencies."""
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
        """ Plot the best frequencies."""
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
        """ Plot data as frequency vs time """
        for f in np.unique(obs.as_arrays.filters[obs.flux_loc]):
            t, nu = [], []

            for d in obs.data[obs.flux_loc]:
                if d.band == f:
                    t.append(d.time.to_value('d'))
                    nu.append(d.frequency.to_value('Hz'))

            # Plot the band
            self.ax.scatter(t, nu, label=f, **OPTION_MAP[f])

        self.ax.legend(loc='best')
        self.ax.grid(alpha=0.5)

    def plot_legacy(self, times):
        """ Plot legacy style. """
        _, ax = plt.subplots()

        # Model the frequencies
        nu_m, nu_c, nu_a = model_freqs(
            self.model, times, self.best(cat='model'), **(self.model_kw or {})
        )

        # Over-plot with the most likely frequencies
        ax.loglog(times, nu_m, color='blue', label=r'$\nu_{m}$')
        ax.loglog(times, nu_c, color='orange', label=r'$\nu_{c}$')

        if nu_a is not None:
            ax.loglog(times, nu_a, color='green', label=r'$\nu_{a}$')

        ax.set_title('Critical Frequencies')
        ax.set_xlabel('Time Since Trigger [d]')
        ax.set_ylabel('Frequency [Hz]')
        ax.legend(loc='best')
        ax.grid(alpha=0.5)

        ax2 = ax.secondary_xaxis('top', functions=(days_to_sec, sec_to_days))
        ax2.set_xlabel("Time Since Trigger [d]")
