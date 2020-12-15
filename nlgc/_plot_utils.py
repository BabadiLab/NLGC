# Author: Proloy Das <proloy@umd.edu>

import numpy as np
from matplotlib import pyplot as plt


def visualize_connectivity(con, label_names, n_eigenmodes, ax=None, title=None, cbar_kw={}, cbarlabel="", cmap='seismic'):
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


def visualize_connectome(adjacency_matrix, src, subject, subjects_dir, edge_threshold=None, edge_cmap='hot',
        symmetric_cmap=True, linewidth=6.0, node_size=3.0, colorbar=True, colorbar_height=0.5, colorbar_fontsize=25,
        title=None, title_fontsize=25,):
    """
    Insert a 3d plot of a connectome into an HTML page using `nilearn.plotting.view_connectome()`

    Parameters
    ----------
    adjacency_matrix : ndarray, shape=(n_nodes, n_nodes)
        the weights of the edges.
    src : mne.SourceSpaces | mne.Forward
        the source space providing the nodes for the edges.
    subject : str
        subject identifier.
    subjects_dir : str
        mri directory
    node_coords : ndarray, shape=(n_nodes, 3)
        the coordinates of the nodes in MNI space.
    edge_threshold : str, number or None, optional (default=None)
        If None, no thresholding.
        If it is a number only connections of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only connections of amplitude above the
        given percentile will be shown.
    edge_cmap : str or matplotlib colormap, optional
    symmetric_cmap : bool, optional (default=True)
        Make colormap symmetric (ranging from -vmax to vmax).
    linewidth : float, optional (default=6.)
        Width of the lines that show connections.
    node_size : float, optional (default=3.)
        Size of the markers showing the seeds in pixels.
    colorbar : bool, optional (default=True)
        add a colorbar
    colorbar_height : float, optional (default=.5)
        height of the colorbar, relative to the figure height
    colorbar_fontsize : int, optional (default=25)
        fontsize of the colorbar tick labels
    title : str, optional (default=None)
        title for the plot
    title_fontsize : int, optional (default=25)
        fontsize of the title

    Returns
    -------
    ConnectomeView : plot of the connectome.
    It can be saved as an html page or rendered (transparently) by the
    Jupyter notebook. Useful methods are :

    - 'resize' to resize the plot displayed in a Jupyter notebook
    - 'save_as_html' to save the plot to a file
    - 'open_in_browser' to save the plot and open it in a web browser.
    """
    import mne
    from nilearn import plotting
    if isinstance(src, mne.Forward):
        src = src['src']

    vertnos = []
    hemis = []
    for i, ss in enumerate(src):
        vertnos.append(ss['vertno'])
        hemis.append(i * np.ones_like(ss['vertno']))
    vertnos = np.concatenate(vertnos)
    hemis = np.concatenate(hemis)
    node_coords = mne.vertex_to_mni(vertnos, hemis, subject, subjects_dir)

    c = plotting.view_connectome(adjacency_matrix, node_coords, edge_threshold=edge_threshold, edge_cmap=edge_cmap,
                                 symmetric_cmap=symmetric_cmap, linewidth=linewidth, node_size=node_size,
                                 colorbar=colorbar, colorbar_height=colorbar_height, colorbar_fontsize=colorbar_fontsize,
                                 title=title, title_fontsize=title_fontsize,)
    return c