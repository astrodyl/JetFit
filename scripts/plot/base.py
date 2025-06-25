import numpy as np
from matplotlib import pyplot as plt


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


def latex(key: str) -> str:
    """
    Returns LaTeX label for the provided fitting parameter.

    Parameters
    ----------
    key : str
        The parameter name

    Returns
    -------
    str
        The LaTeX formatted ``key`` or just ``key``
    """
    if '_offset' in key:
        return r'$\delta_{' + f'{key.split('_')[0]}' r'}$'

    if 'rho0' in key:
        return key.replace('rho0', r'$log_{10}n_{17}$')

    try:
        return {
            # Jetsimpy
            'Eiso': r'$log10_{E_{iso}}$',
            'lf': r'$\Gamma$',
            'theta_c': r'$\theta_c$',
            'theta_v': r'$\theta_v$',

            # Boosted Fireball Model
            'eta': r'$\eta_0$',
            'gamma_b': r'$\gamma_B$',
            'obs_angle': r'$\theta_{obs}$',

            # Stratified Fireball Model
            'nt': r'$log_{10}n_{0, t}$',
            'rt': r'$log_{10}R_t$',
            'k1': r'$k_{pre}$',
            'k2': r'$k_{post}$',
            'sn': r'$s_n$',
            'sj': r'$s_j$',
            'tj': r'$log_{10}t_j$',

            # Generic Fireball Model
            'E': r'$log_{10}E_{52}$',
            'eps_e': r'$log_{10}\epsilon_e$',
            'eps_b': r'$log_{10}\epsilon_B$',
            'rv_milky_way': r'$log_{10}R_{v}^{MW}$',
            'ebv_source_frame': r'$E(B-V)_{sf}$',
            'ebv_milky_way': r'$E(B-V)_{MW}$',
            'rho0': r'$log_{10}n$',
        }[key]
    except KeyError:
        return key


class Profiler:
    """

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The emcee sampler to draw samples from.

    params : Parameters
        The `Parameter` object used when running the
        MCMC with `sampler`.
    """
    dist_options = {
        'alpha': 0.3, 'linewidth': 0.5,
        'linestyle': '-', 'color': 'tab:purple'
    }

    best_options = {
        'linewidth': 2, 'linestyle': '-',
        'color': 'tab:orange', 'label': 'Best Profile'
    }

    def __init__(self, sampler, params):
        self.sampler = sampler
        self.params  = params

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

    def best(self, **kwargs):
        """
        Returns the highest likelihood set of parameters.

        Parameters
        ----------
        kwargs :
            Any args passed to `params.samples_to_dict()`.

        Returns
        -------
        np.ndarray or dict
            The highest likelihood set of parameters.
        """
        max_index = np.nanargmax(self.sampler.get_log_prob(flat=True))
        params = self.sampler.get_chain(flat=True)[max_index]
        return self.params.samples_to_dict(params, **kwargs)

    def plot(
        self, dist_x, dist_y, best_x=None, best_y=None,
        log_scale=True, dist_kw=None, best_kw=None
    ):
        """
        Creates the plot for an arbitrary distribution.

        Parameters
        ----------
        dist_x, dist_y : array_like of array_like
            The distribution of x, y values.

        best_x, best_y : array_like of float, optional
            The best distribution.

        log_scale : bool, optional, default=True
            Should the plot be in log scale?

        dist_kw, best_kw : dict, optional
            All parameters supported by `plt.plot` for
            the distribution and/or best curves. Defaults
            are `.dist_options` and `.best_options`.

        Returns
        -------
        matplotlib.axes.Axes
            The figure object.
        """
        _, ax = plt.subplots()

        dist_options = self.dist_options | dist_kw if dist_kw else self.dist_options
        best_options = self.best_options | best_kw if dist_kw else self.best_options

        for i, dx in enumerate(dist_x):
            ax.plot(dx, dist_y[i], **dist_options)  # noqa

        if best_x is not None and best_y is not None:
            ax.plot(best_x, best_y, **best_options)

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.legend(loc='best')
        ax.grid(alpha=0.5)

        return ax
