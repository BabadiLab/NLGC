import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

if __name__ == "__main__":

    # fig, ax = plt.subplots(nrows=2, ncols=2)
    #
    # ####################################################################################################################
    # # AR(1), $\delta=1, \alpha=0$
    # table_h = np.array([[0.96, 0.92, 0.9, 0.86], [1, 1, 0.92, 0.88], [0.96, 1, 1, 0.92], [0.96, 0.96, 0.96, 0.92]])
    # table_f = np.array([[0, 0.09, 0.11, 0.11], [0, 0, 0.08, 0.08], [0, 0.04, 0.04, 0.08], [0, 0.06, 0.04, 0.06]])
    #
    # im0, cbar0 = heatmap(table_h, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
    #         '4 in-act.', '6 in-act.'], ax=ax[0, 0], cmap="YlGn", cbarlabel="Hit rate", vmin=0.75, vmax=1)
    #
    # ax[0, 0].set_title(r'AR(1), $\delta=1, \alpha=0$')
    #
    # im1, cbar1 = heatmap(table_f, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
    #         '4 in-act.', '6 in-act.'], ax=ax[1, 0], cmap="YlGn", cbarlabel="False-detection rate", vmin=0, vmax=0.25)
    #
    # texts = annotate_heatmap(im0, valfmt="{x:.2f}")
    # texts = annotate_heatmap(im1, valfmt="{x:.2f}")
    #
    # # AR(1), $\delta=1, \alpha=0.25$
    # table_h = np.array([[1, 0.92, 0.96, 0.96], [1, 1, 0.96, 0.92], [1, 1, 0.96, 0.92], [1, 0.96, 0.96, 0.89]])
    # table_f = np.array([[0, 0.18, 0.18, 0.21], [0, 0.12, 0.13, 0.12], [0, 0.14, 0.12, 0.09], [0, 0.16, 0.16, 0.11]])
    #
    # im2, cbar2 = heatmap(table_h, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
    #         '4 in-act.', '6 in-act.'], ax=ax[0, 1], cmap="YlGn", cbarlabel="Hit rate", vmin=0.75, vmax=1)
    #
    # ax[0, 1].set_title(r'AR(1), $\delta=1, \alpha=0.25$')
    #
    # im3, cbar3 = heatmap(table_f, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
    #         '4 in-act.', '6 in-act.'], ax=ax[1, 1], cmap="YlGn", cbarlabel="False-detection rate", vmin=0, vmax=0.25)
    #
    # texts = annotate_heatmap(im2, valfmt="{x:.2f}")
    # texts = annotate_heatmap(im3, valfmt="{x:.2f}")
    #
    # fig.tight_layout()
    # fig.show()
    #
    #
    #
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    #
    #
    # # AR(1), $\delta=1, \alpha=0.5$
    # table_h = np.array([[1, 0.9, 0.9, 0.89], [1, 0.9, 0.92, 0.82], [1, 0.96, 0.9, 0.82], [0.96, 0.9, 0.86, 0.82]])
    # table_f = np.array([[0, 0.23, 0.18, 0.1], [0, 0.16, 0.21, 0.2], [0, 0.12, 0.21, 0.16], [0, 0.14, 0.16, 0.17]])
    #
    # im0, cbar0 = heatmap(table_h, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
    #                                                                          '4 in-act.', '6 in-act.'], ax=ax[0, 0],
    #                      cmap="YlGn", cbarlabel="Hit rate", vmin=0.75, vmax=1)
    #
    # ax[0, 0].set_title(r'AR(1), $\delta=1, \alpha=0.5$')
    #
    # im1, cbar1 = heatmap(table_f, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
    #                                                                          '4 in-act.', '6 in-act.'], ax=ax[1, 0],
    #                      cmap="YlGn", cbarlabel="False-detection rate", vmin=0, vmax=0.25)
    #
    # texts = annotate_heatmap(im0, valfmt="{x:.2f}")
    # texts = annotate_heatmap(im1, valfmt="{x:.2f}")
    #
    # # AR(1), $\delta=1, \alpha=1$
    # table_h = np.array([[1, 0.8, 0.8, 0.76], [1, 0.9, 0.92, 0.82], [1, 0.96, 0.9, 0.82], [0.96, 0.9, 0.86, 0.82]])
    # table_f = np.array([[0.004, 0.16, 0.21, 0.20], [0.004, 0.14, 0.18, 0.16], [0.008, 0.21, 0.18, 0.18], [0, 0.21, 0.16, 0.17]])
    #
    # im2, cbar2 = heatmap(table_h, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
    #                                                                          '4 in-act.', '6 in-act.'], ax=ax[0, 1],
    #                      cmap="YlGn", cbarlabel="Hit rate", vmin=0.75, vmax=1)
    #
    # ax[0, 1].set_title(r'AR(1), $\delta=1, \alpha=1$')
    #
    # im3, cbar3 = heatmap(table_f, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
    #                                                                          '4 in-act.', '6 in-act.'], ax=ax[1, 1],
    #                      cmap="YlGn", cbarlabel="False-detection rate", vmin=0, vmax=0.25)
    #
    # texts = annotate_heatmap(im2, valfmt="{x:.2f}")
    # texts = annotate_heatmap(im3, valfmt="{x:.2f}")
    #
    # fig.tight_layout()
    # fig.show()




    fig, ax = plt.subplots(nrows=2, ncols=2)

    ####################################################################################################################
    # AR(1), $\delta=20%, \alpha=0$
    table_h = np.array([[1, 0.95, 0.85, 0.85], [0.875, 0.85, 0.9, 0.875], [0.93, 0.95, 0.95, 0.85], [0.875, 0.92, 0.92, 0.82]])
    table_f = np.array([[0, 0.04, 0.08, 0.08], [0, 0.16, 0.17, 0.1], [0, 0.02, 0.06, 0.09], [0, 0.003, 0.08, 0.12]])

    im0, cbar0 = heatmap(table_h, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
                                                                             '4 in-act.', '6 in-act.'], ax=ax[0, 0],
                         cmap="YlGn", cbarlabel="Hit rate", vmin=0.75, vmax=1)

    ax[0, 0].set_title(r'AR(1), $\delta=20\%, \alpha=0$')

    im1, cbar1 = heatmap(table_f, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
                                                                             '4 in-act.', '6 in-act.'], ax=ax[1, 0],
                         cmap="YlGn", cbarlabel="False-detection rate", vmin=0, vmax=0.25)

    texts = annotate_heatmap(im0, valfmt="{x:.2f}")
    texts = annotate_heatmap(im1, valfmt="{x:.2f}")

    # AR(1), $\delta=20%, \alpha=0.25$
    table_h = np.array([[1, 1, 1, 0.94], [0.975, 0.825, 0.84, 0.875], [0.98, 0.9, 0.91, 0.85], [0.85, 0.96, 0.81, 0.79]])
    table_f = np.array([[0, 0.16, 0.08, 0.12], [0, 0.19, 0.12, 0.13], [0, 0.11, 0.14, 0.12], [0.003, 0.04, 0.13, 0.1]])

    im2, cbar2 = heatmap(table_h, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
                                                                             '4 in-act.', '6 in-act.'], ax=ax[0, 1],
                         cmap="YlGn", cbarlabel="Hit rate", vmin=0.75, vmax=1)

    ax[0, 1].set_title(r'AR(1), $\delta=20\%, \alpha=0.25$')

    im3, cbar3 = heatmap(table_f, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
                                                                             '4 in-act.', '6 in-act.'], ax=ax[1, 1],
                         cmap="YlGn", cbarlabel="False-detection rate", vmin=0, vmax=0.25)

    texts = annotate_heatmap(im2, valfmt="{x:.2f}")
    texts = annotate_heatmap(im3, valfmt="{x:.2f}")

    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(nrows=2, ncols=2)

    # AR(1), $\delta=20%, \alpha=0.5$
    table_h = np.array([[1, 0.92, 1, 0.96], [1, 0.94, 0.875, 0.82], [0.96, 0.92, 0.9, 0.85], [0.96, 0.875, 0.85, 0.81]])
    table_f = np.array([[0, 0, 0.08, 0.17], [0, 0.08, 0.21, 0.17], [0.04, 0.12, 0.1, 0.19], [0, 0.14, 0.15, 0.17]])

    im0, cbar0 = heatmap(table_h, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
                                                                             '4 in-act.', '6 in-act.'], ax=ax[0, 0],
                         cmap="YlGn", cbarlabel="Hit rate", vmin=0.75, vmax=1)

    ax[0, 0].set_title(r'AR(1), $\delta=20\%, \alpha=0.5$')

    im1, cbar1 = heatmap(table_f, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
                                                                             '4 in-act.', '6 in-act.'], ax=ax[1, 0],
                         cmap="YlGn", cbarlabel="False-detection rate", vmin=0, vmax=0.25)

    texts = annotate_heatmap(im0, valfmt="{x:.2f}")
    texts = annotate_heatmap(im1, valfmt="{x:.2f}")

    # AR(1), $\delta=20%, \alpha=1$
    table_h = np.array([[1, 0.9, 0.82, 0.82], [0.875, 0.82, 0.875, 0.79], [0.93, 0.85, 0.875, 0.82], [0.96, 0.92, 0.85, 0.76]])
    table_f = np.array([[0.09, 0.1, 0.14, 0.16], [0, 0.18, 0.14, 0.19], [0, 0.19, 0.1, 0.17], [0.08, 0.14, 0.1, 0.21]])

    im2, cbar2 = heatmap(table_h, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
                                                                             '4 in-act.', '6 in-act.'], ax=ax[0, 1],
                         cmap="YlGn", cbarlabel="Hit rate", vmin=0.75, vmax=1)

    ax[0, 1].set_title(r'AR(1), $\delta=20\%, \alpha=1$')

    im3, cbar3 = heatmap(table_f, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
                                                                             '4 in-act.', '6 in-act.'], ax=ax[1, 1],
                         cmap="YlGn", cbarlabel="False-detection rate", vmin=0, vmax=0.25)

    texts = annotate_heatmap(im2, valfmt="{x:.2f}")
    texts = annotate_heatmap(im3, valfmt="{x:.2f}")

    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(nrows=2, ncols=1)

    ####################################################################################################################
    # AR(1), $\delta=20%, \alpha=0$
    table_h = np.array([[0.6, 0.5, 0.5, 0.5], [0.625, 0.42, 0.46, 0.4],      [0.41, 0.4, 0.38, 0.5], [0.28, 0.25, 0.31, 0.39]])
    table_f = np.array([[0.4, 0.45, 0.38, 0.41], [0.3875, 0.35, 0.42, 0.42], [0.23, 0.29, 0.42, 0.44], [0.18, 0.17, 0.28, 0.46]])

    im0, cbar0 = heatmap(table_h, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
                                                                             '4 in-act.', '6 in-act.'], ax=ax[0],
                         cmap="YlGn", cbarlabel="Hit rate", vmin=0.4, vmax=1)

    ax[0].set_title(r'AR(1), $\delta=20\%, \alpha=10$')
    im1, cbar1 = heatmap(table_f, ['2 act.', '3 act.', '4 act.', '5 act.'], ['0 in-act.', '2 in-act.',
                                                                             '4 in-act.', '6 in-act.'], ax=ax[1],
                         cmap="YlGn", cbarlabel="False-detection rate", vmin=0, vmax=0.5)

    texts = annotate_heatmap(im0, valfmt="{x:.2f}")
    texts = annotate_heatmap(im1, valfmt="{x:.2f}")

    fig.tight_layout()
    fig.show()