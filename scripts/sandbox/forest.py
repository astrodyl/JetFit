import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from jetfit.core.utils import save_plot_unique
from jetfit.mcmc.parameters import Parameters
from scripts.plot.diagnose import plot_corner

import scienceplots
plt.style.use(['science'])


def flatten(v):
    """ Flatten v. """
    s = list(v.shape[1:])
    s[0] = np.prod(v.shape[:2])
    return v.reshape(s)


def forest(p_data, k_data, p_samples):
    """"""
    excluded = {}
    events = [e for e in sorted(p_data) if e in k_data and e not in excluded]
    x = np.arange(len(events))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for ax, data, ylabel in ((axes[0], p_data, "p"), (axes[1], k_data, "k")):
        means, lowers, uppers = [], [], []


        for ev in events:

            if ev in ('080319B', '080413B') and ylabel=='k':
                p16, p50, p84 = data[ev][0]
                means.append(p50)
                lowers.append(p50 - p16)
                uppers.append(p84 - p50)

            else:
                p16, p50, p84 = data[ev]
                means.append(p50)
                lowers.append(p50 - p16)
                uppers.append(p84 - p50)


        yerr = np.vstack([lowers, uppers])

        ax.errorbar(
            x, means, yerr=yerr,
            fmt='o', linestyle='none', capsize=3, color='black'
        )
        ax.set_ylabel(ylabel, fontsize=18)
        ax.grid(alpha=0.3, axis='x')

    # Hardcode stratified events
    k_19B = k_data['080319B'][1]
    axes[1].errorbar(
        2, k_19B[1], yerr=np.vstack([k_19B[1] - k_19B[0], k_19B[2] - k_19B[1]]),
        fmt='o', linestyle='none', capsize=3, color='black'
    )

    k_13B = k_data['080413B'][1]
    axes[1].errorbar(
        3, k_19B[1], yerr=np.vstack([k_13B[1] - k_13B[0], k_13B[2] - k_13B[1]]),
        fmt='o', linestyle='none', capsize=3, color='black'
    )

    # add shaded region
    p16, p50, p84 = np.percentile(p_samples, [16, 50, 84], axis=0)

    axes[0].axhline(y=p50, c='k', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0, c='k',   linestyle='--', alpha=0.5)
    axes[1].axhline(y=2, c='k',   linestyle='--', alpha=0.5)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(events, rotation=270, ha='left', fontsize=14)

    fig.tight_layout()
    save_plot_unique('kpdist', ext='pdf', directory=r'C:\Server')
    # plt.show()


def replot_corner(chain, params):
    """"""
    # Flatten the chain
    s = list(chain.shape[1:])
    s[0] = np.prod(chain.shape[:2])
    flat_chain = chain.reshape(s)

    plot_corner(flat_chain, params)
    plt.show()


if __name__ == '__main__':

    directories = os.listdir(r"C:\Server\FINAL\analytic")

    p_samples = []
    ps, ks = {}, {}
    for directory in directories:
        data = np.load(os.path.join(r"C:\Server\FINAL\analytic", directory, "chain.npz"))["chain"]



        if directory not in ('080319B', '080413B'):
            p16, p50, p84 = np.percentile(flatten(data)[:, 5], [16, 50, 84], axis=0)
            k16, k50, k84 = np.percentile(flatten(data)[:, 6], [16, 50, 84], axis=0)
            p_samples.extend(flatten(data)[:, 5])
            ps[directory] = [p16, p50, p84]
            ks[directory] = [k16, k50, k84]

        else:
            p16, p50, p84 = np.percentile(flatten(data)[:, 6], [16, 50, 84], axis=0)
            k16_1, k50_1, k84_1 = np.percentile(flatten(data)[:, 7], [16, 50, 84], axis=0)
            k16_2, k50_2, k84_2 = np.percentile(flatten(data)[:, 8], [16, 50, 84], axis=0)
            p_samples.extend(flatten(data)[:, 6])
            ps[directory] = [p16, p50, p84]
            ks[directory] = [[k16_1, k50_1, k84_1], [k16_2, k50_2, k84_2]]

    # ks = {
    #     # name, median, upper, lower
    #     '050525A': [2.94,  0.03, 0.08],
    #     '050922C': [1.76,  0.08, 0.09],
    #     '090424':  [2.06,  0.05, 0.05],
    #     '090618':  [1.49,  0.06, 0.07],
    #     '131030A': [1.81,  0.07, 0.12],
    #     '130612A': [-0.89, 1.22, 1.42],
    #     '140506A': [-2.41, 0.05, 0.05],
    #     '161031A': [1.72,  0.08, 0.09],
    #     '171010A': [1.80,  0.36, 0.34],
    #     '210905A': [-4.27, 0.76, 0.84],
    #     '220101A': [0.71,  0.15, 0.18],
    #     '221009A': [2.19,  0.03, 0.03],
    # }
    forest(ps, ks, p_samples)

    # replot_corner(
    #     chain=np.load(r"C:\Server\62725\130612A\chain.npz").get('chain'),
    #     params=Parameters.from_toml(r"C:\Projects\repos\JetFit\jetfit\resources\grbs\130612A\parameters.toml").fitting
    # )
