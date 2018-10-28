#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
General plotting functions for all data sources.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_basemap(basemap, lon, lat, z=None, plot_type='contourf', plot_kwargs=None,
                 title=None, colorbar=True, colorbar_label=None, draw_grids=True,
                 save_file=None, save_kwargs={}, width=6, height=4, ):
    """
    Function for plot data on a given Basemap object.

    :param lon: ndarray: 2-D longitude array
    :param lat: ndarray: 2-D latitude array
    :param z: ndarray: 2-D field to plot
    :param basemap: Basemap: Basemap object on which to plot
    :param plot_type: str: type of plot, e.g. contour or contourf
    :param plot_kwargs: dict: kwargs passed to the plot function. Use 'caxis' for plot contour levels.
    :param title: str: title of plot
    :param colorbar: bool: if True, plots a color bar
    :param colorbar_label: str: name label for the color bar
    :param draw_grids: bool: draw meridians/parallels
    :param save_file: str: full path of file to save image to
    :param save_kwargs: dict: kwargs passed to save function
    :param width: int or float: width of output image
    :param height: int or float: height of output image
    :return: pyplot Figure object
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    fig = plt.figure()
    plt.clf()
    plot_function = getattr(basemap, plot_type)
    if 'caxis' in plot_kwargs:
        c = plot_function(lon, lat, z, plot_kwargs.pop('caxis'), latlon=True, **plot_kwargs)
    else:
        c = plot_function(lon, lat, z, latlon=True, **plot_kwargs)
    if colorbar:
        cb = basemap.colorbar()
        if colorbar_label is not None:
            cb.set_label(colorbar_label)
    basemap.drawcoastlines(linewidth=0.7)
    basemap.drawcountries(linewidth=0.7)
    basemap.drawstates(linewidth=0.4)
    if draw_grids:
        basemap.drawmeridians(np.arange(0, 360, 10), linecolor='0.5')
        basemap.drawparallels(np.arange(-90, 90, 10), linecolor='0.5')
    if title is not None:
        plt.title(title)
    fig.set_size_inches(width, height)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, **save_kwargs)
    return fig


def slp_contour(fig, m, slp, lons, lats, window=100):
    """
    Add sea-level pressure labels to a contour map. I don't remember where I found the code for this function
    some time in the past, but I wish I could attribute it.

    :param fig:
    :param m:
    :param slp:
    :param lons:
    :param lats:
    :param window:
    :return:
    """
    def extrema(mat, mode='wrap', w=10):
        """
        Find the indices of local extrema (min and max)
        in the input array.
        """

        from scipy.ndimage.filters import minimum_filter, maximum_filter

        mn = minimum_filter(mat, size=w, mode=mode)
        mx = maximum_filter(mat, size=w, mode=mode)
        return np.nonzero(mat == mn), np.nonzero(mat == mx)

    caxisP = np.arange(900, 1050, 4)
    c2 = m.contour(lons, lats, slp, caxisP, latlon=True, linewidth=1.0, colors='black')
    plt.clabel(c2, c2.levels, inline=True, fmt='%0.0f')
    # Plot highs and lows for slp
    local_min, local_max = extrema(slp, mode='wrap', w=window)
    x, y = m(lons, lats)
    xlows = x[local_min]
    xhighs = x[local_max]
    ylows = y[local_min]
    yhighs = y[local_max]
    lowvals = slp[local_min]
    highvals = slp[local_max]
    # Plot lows
    xyplotted = []
    yoffset = 0.022 * (m.ymax - m.ymin)
    dmin = 20.0 * yoffset
    for x, y, p in zip(xlows, ylows, lowvals):
        if (m.xmax - dmin > x > m.xmin + dmin and m.ymax - dmin > y > m.ymin + dmin):
            dist = [np.sqrt((x - x0) ** 2 + (y - y0) ** 2) for x0, y0 in xyplotted]
            if not dist or min(dist) > dmin:
                plt.text(x, y, 'L', fontsize=14, fontweight='bold', ha='center', va='center', color='r')
                plt.text(x, y - yoffset, repr(int(p)), fontsize=9, ha='center', va='top', color='r',
                         bbox=dict(boxstyle="square", ec='None', fc=(1, 1, 1, 0.5)))
                xyplotted.append((x, y))
    # Plot highs
    xyplotted = []
    for x, y, p in zip(xhighs, yhighs, highvals):
        if (m.xmax - dmin > x > m.xmin + dmin and m.ymax - dmin > y > m.ymin + dmin):
            dist = [np.sqrt((x - x0) ** 2 + (y - y0) ** 2) for x0, y0 in xyplotted]
            if not dist or min(dist) > dmin:
                plt.text(x, y, 'H', fontsize=14, fontweight='bold', ha='center', va='center', color='b')
                plt.text(x, y - yoffset, repr(int(p)), fontsize=9, ha='center', va='top', color='b',
                         bbox=dict(boxstyle="square", ec='None', fc=(1, 1, 1, 0.5)))
                xyplotted.append((x, y))
    return fig
