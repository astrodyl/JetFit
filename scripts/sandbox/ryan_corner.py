import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner


OPTIONS = {
    'label_size': 16, 'show_titles': True,
    'plot_datapoints': False, 'quantiles': [0.16, 0.5, 0.84],
    'label_kwargs': {'fontsize': 12}, 'title_kwargs': {"fontsize": 12},
    'fill_contours': True, 'smooth': 0.75, 'smooth1d': 0.75,
}


def load_flat_2d(h5_path, burn_frac=0, thin=1):
    with h5py.File(h5_path, "r") as f:
        chain = f["chain"][...]
        lp    = f["logprob"][...]
    b0    = int(chain.shape[1]*burn_frac) if 0 < burn_frac < 1 else int(burn_frac)
    four  = chain[:, b0::thin, :][:, :, :4].reshape(-1, 4)
    # mask  = np.isfinite(lp[:, b0::thin].reshape(-1))
    return four

# fig = plt.figure(figsize=(6, 10))
# subfigs = fig.subfigures(2, 1)

labels = (r"$\log_{10} E_{\rm iso}$", r"$\log_{10} n_0$", r"$\theta_0$", r"$\theta_{\rm obs}/ \theta_0$")

sampA = load_flat_2d(r"C:\Users\Dylan\Downloads\050525A_noex_samples.h5")
sampB = load_flat_2d(r"C:\Users\Dylan\Downloads\050525A_ex_samples.h5")

figA = corner.corner(sampA, labels=labels, bins=50, color='mediumblue', **OPTIONS)
# figA.suptitle("Run A")

# figB = corner.corner(sampB, labels=labels, bins=50, fig=figA, color='salmon', **OPTIONS)
figB = corner.corner(sampB, labels=labels, bins=50, color='mediumblue', **OPTIONS)
# figB.suptitle("Run B")

plt.show()