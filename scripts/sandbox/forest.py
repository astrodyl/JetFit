import matplotlib.pyplot as plt
import numpy as np

from jetfit.mcmc.parameters import Parameters
from scripts.plot.diagnose import plot_corner


def forest(data):
    """"""
    events = data.keys()

    means, upper, lower = [], [], []
    for key, val in data.items():
        means.append(val[0])
        upper.append(val[1])
        lower.append(val[2])

    stds = [lower, upper]

    y_pos = np.arange(len(events))
    plt.figure(figsize=(6, 4))

    plt.errorbar(
        means, y_pos,
        xerr=stds, fmt='.', capsize=3, color='black'
    )

    # Label each y‐tick with the corresponding event name
    plt.yticks(y_pos, events)

    # invert y‐axis so the first event is at the top
    plt.gca().invert_yaxis()

    plt.xlabel("Power Law Index (k)")
    plt.ylabel("GRB")
    plt.title(r"Power Law Index Forest ($\mu \pm 1\sigma$)")
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()


def replot_corner(chain, params):
    """"""
    # Flatten the chain
    s = list(chain.shape[1:])
    s[0] = np.prod(chain.shape[:2])
    flat_chain = chain.reshape(s)

    plot_corner(flat_chain, params)
    plt.show()


if __name__ == '__main__':

    ks = {
        # name, median, upper, lower
        '050525A': [2.94,  0.03, 0.08],
        '050922C': [1.76,  0.08, 0.09],
        '090424':  [2.06,  0.05, 0.05],
        '090618':  [1.49,  0.06, 0.07],
        '131030A': [1.81,  0.07, 0.12],
        '130612A': [-0.89, 1.22, 1.42],
        '140506A': [-2.41, 0.05, 0.05],
        '161031A': [1.72,  0.08, 0.09],
        '171010A': [1.80,  0.36, 0.34],
        '210905A': [-4.27, 0.76, 0.84],
        '220101A': [0.71,  0.15, 0.18],
        '221009A': [2.19,  0.03, 0.03],
    }
    forest(ks)

    replot_corner(
        chain=np.load(r"C:\Server\62725\130612A\chain.npz").get('chain'),
        params=Parameters.from_toml(r"C:\Projects\repos\JetFit\jetfit\resources\grbs\130612A\parameters.toml").fitting
    )
