import numpy as np
from matplotlib import pyplot as plt


# import warnings
# warnings.filterwarnings('ignore')

def visualize_con(con, label_names, n_eigenmodes, ax=None, title=None, cbar_kw={}, cbarlabel="", cmap='seismic'):
    lh = []
    rh = []
    for i, label in enumerate(label_names):
        lh.extend(list(range(n_eigenmodes * i, n_eigenmodes * (i + 1)))) if label[-2:] == 'lh' else rh.extend(
            list(range(n_eigenmodes * i, n_eigenmodes * (i + 1))))
    lh_label_names = []
    rh_label_names = []
    for label in label_names:
        lh_label_names.append(label) if label[-2:] == 'lh' else rh_label_names.append(label)
    a_f = np.vstack(map(np.hstack, ((con[lh][:, lh], con[lh][:, rh]), (con[rh][:, lh], con[rh][:, rh]))))
    nx = len(label_names)
    if not ax:
        ax = plt.gca()
    vmax = max(a_f.max(), -a_f.min())
    vmin = - vmax
    im = ax.matshow(a_f, cmap=cmap, vmin=vmin, vmax=vmax)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ticks = np.asarray(range(n_eigenmodes, nx * n_eigenmodes + 1, n_eigenmodes)) - n_eigenmodes / 2 - 0.5
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    # ... and label them with the respective list entries.
    ax.set_yticklabels(lh_label_names+rh_label_names, fontdict={'fontsize':6})
    ax.set_xticklabels(lh_label_names+rh_label_names, fontdict={'fontsize':6})

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(ticks, minor=True)
    ax.set_yticks(ticks, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title)

    return im, cbar
