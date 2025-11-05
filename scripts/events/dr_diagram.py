import argparse
import os
from pathlib import Path

import numpy as np
import corner
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import scienceplots
import seaborn as sns
import pandas as pd

plt.style.use(['science'])


def flatten(v):
    """ Flatten v. """
    s = list(v.shape[1:])
    s[0] = np.prod(v.shape[:2])
    return v.reshape(s)


def load_sampler(p_sampler):
    """ Load the sampler from a file. """
    return np.load(p_sampler)


def load_chain(p_sampler, flat=True):
    """ Load in the chain from a sampler file."""
    chain = load_sampler(p_sampler)['chain']
    return flatten(chain) if flat else chain


def plot_dr_diagram(events, out_dir=None, kwargs=None):
    """"""
    # Format the color map
    cmap = get_cmap("turbo")
    colors = [cmap(i / (len(events) - 1)) for i in range(len(events))]

    # Fractional percent levels
    sigmas = np.array([0.5, 1])
    levels = 1.0 - np.exp(-0.5 * sigmas ** 2)

    # Assign an alpha to each level (need 0 for hist2d compatibility)
    # See: https://github.com/dfm/corner.py/issues/112
    alphas = np.linspace(0.0, 0.8, len(levels) + 1)

    for i, event in enumerate(events):
        flat_chain = load_chain(rf"C:\Server\FINAL\analytic\{event}\chain.npz")

        if event in ('080319B', '080413B'):
            # Stratified medium
            k_pre = flat_chain[:, 7]
            k_pos = flat_chain[:, 8]
            x = np.concatenate((k_pre, k_pos))
            y = np.concatenate((flat_chain[:, 2], flat_chain[:, 2]))
        else:
            # Non-stratified medium
            x = flat_chain[:, 6]
            y = flat_chain[:, 2]

        corner.hist2d(
            x=x, y=y, color=colors[i], levels=levels, new_fig=i == 0,
            contourf_kwargs= {"alpha": alphas}, **kwargs
        )

    # Wrap it up
    plt.xlabel('k', fontsize=18)
    plt.ylabel(r'$log_{10}n_{0,17}$', fontsize=18)
    plt.xlim((-6.0, 3.5))
    plt.ylim((-6.0, 6.0))
    plt.grid(alpha=0.3)
    plt.show()


def plot_swarm(events, thin=10_000, q=None, out_dir=None):
    """"""
    k_posteriors = []
    for i, event in enumerate(events):
        flat_chain = load_chain(rf"C:\Server\FINAL\analytic\{event}\chain.npz")

        if event in ('080319B', '080413B'):
            # Stratified medium
            k_pre = flat_chain[:, 7][::thin]
            k_pos = flat_chain[:, 8][::thin]
            k_posteriors.append(np.concatenate((k_pre, k_pos)))
        else:
            # Non-stratified medium
            k_posteriors.append(flat_chain[:, 6][::thin])

    # Mask k values based on sigma from mean
    if q is not None:
        filtered_k_posteriors = []
        for k_vals in k_posteriors:
            lo, hi = np.percentile(k_vals, q)
            in_1sigma = k_vals[(k_vals >= lo) & (k_vals <= hi)]
            filtered_k_posteriors.append(in_1sigma)
    else:
        filtered_k_posteriors = k_posteriors

    # Format the data for seaborn
    df = pd.DataFrame({
        "k": np.concatenate(filtered_k_posteriors),
        "event": np.repeat(events, [len(k) for k in filtered_k_posteriors])
    })

    sns.set_theme()
    plt.figure(figsize=(10, 5))
    sns.swarmplot(x="event", y="k", data=df, palette=sns.color_palette("husl", 15), size=2)
    plt.ylabel("k")
    plt.xlabel("Event")
    plt.tight_layout()
    plt.show()


def main(events, out_dir):
    """"""
    plot_dr_diagram(events, out_dir=out_dir, kwargs=dict(
        plot_datapoints=False, fill_contours=True, bins=50, smooth=2.0,
    ))

    plot_swarm(events, q=[2.3, 97.7], out_dir=out_dir)


if __name__ == '__main__':
    # Run the Publisher via the command line
    parser = argparse.ArgumentParser(description="Create publication ready plots")
    parser.add_argument('--sampler', help='The MCMC chain or sampler.')
    parser.add_argument('--results', help='The results directory.')
    args = parser.parse_args()

    # Output directory
    d_results = Path(r"C:\Server\FINAL\analytic")

    event_names = [
        event for event in os.listdir(r"C:\Server\FINAL\analytic")
    ]

    main(
        **{
            'events': event_names,
            'out_dir': args.results or d_results,
        }
    )
