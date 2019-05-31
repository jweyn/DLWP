#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Collection of random plotting functions. Unfortunately these are not very robust or well-documented, but I thought it
a bit cleaner to place them here than have plotting functions defined in every user-facing script.
"""

import numpy as np
from matplotlib import pyplot as plt
from .util import remove_chars


def plot_basemap(basemap, lon, lat, z=None, plot_type='contourf', plot_kwargs=None,
                 title=None, colorbar=True, colorbar_label=None, draw_grids=True,
                 save_file=None, save_kwargs=None, width=6, height=4, ):
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
    plot_kwargs = plot_kwargs or {}
    save_kwargs = save_kwargs or {}
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
        basemap.drawmeridians(np.arange(0, 361, 30), linecolor='0.5')
        basemap.drawparallels(np.arange(-90, 91, 30), linecolor='0.5')
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


def plot_movie(m, lat, lon, val, pred, dates, model_title='', plot_kwargs=None, out_directory=None):
    """
    Plot a series of images for a forecast and the verification.

    :param m: Basemap object
    :param lat: ndarray (lat, lon): latitude values
    :param lon: ndarray (lat, lon): longitude values
    :param val: ndarray (t, lat, lon): verification
    :param pred: ndarray (t, lat, lon): predicted forecast
    :param dates: array-like: datetime objects of verification datetimes
    :param model_title: str: name of the model, e.g., 'Neural net prediction'
    :param plot_kwargs: dict: passed to the plot pcolormesh() method
    :param out_directory: str: folder in which to save image files
    """
    if (len(dates) != val.shape[0]) and (len(dates) != pred.shape[0]):
        raise ValueError("'val' and 'pred' must have the same first (time) dimension as 'dates'")
    plot_kwargs = plot_kwargs or {}
    fig = plt.figure()
    fig.set_size_inches(6, 4)
    x, y = m(lon, lat)
    dt = dates[1] - dates[0]
    for d, date in enumerate(dates):
        hours = (d + 1) * dt.total_seconds() / 60 / 60
        ax = plt.subplot(211)
        m.pcolormesh(x, y, val[d], **plot_kwargs)
        m.drawcoastlines()
        m.drawparallels(np.arange(0., 91., 45.))
        m.drawmeridians(np.arange(0., 361., 90.))
        ax.set_title('Verification (%s)' % date)
        ax = plt.subplot(212)
        m.pcolormesh(x, y, pred[d], **plot_kwargs)
        m.drawcoastlines()
        m.drawparallels(np.arange(0., 91., 45.))
        m.drawmeridians(np.arange(0., 361., 90.))
        ax.set_title('%s at $t=%d$ (%s)' % (model_title, hours, date))
        plt.savefig('%s/%05d.png' % (out_directory, d), bbox_inches='tight', dpi=150)
        fig.clear()


def history_plot(train_hist, val_hist, model_name='', out_directory=None):
    """
    Plot the training history of a model.

    :param train_hist: array-like: training loss history
    :param val_hist: array-like: validation loss history
    :param model_name: str: name of model
    :param out_directory: str: if not None, save the figure to this directory
    :return: plt.Figure
    """
    fig = plt.figure()
    fig.set_size_inches(6, 4)
    plt.plot(train_hist, label='train MAE', linewidth=2)
    plt.plot(val_hist, label='val MAE', linewidth=2)
    plt.grid(True, color='lightgray', zorder=-100)
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.legend(loc='best')
    plt.title('%s training history' % model_name)
    if out_directory is not None:
        plt.savefig('%s/%s_history.pdf' % (out_directory, remove_chars(model_name)), bbox_inches='tight')
    return fig


def forecast_example_plot(base, verif, forecast, f_hour, model_name='', plot_diff=True, out_directory=None):
    """
    Plot the initial, verification, and forecast states for a model at a given forecast hour.

    :param base: 2d DataArray with dimensions 'lat', 'lon': initial state
    :param verif: 2d DataArray with dimensions 'lat', 'lon': verification state
    :param forecast: 2d DataArray with dimensions 'lat', 'lon': forecast state
    :param f_hour: int: forecast hour (for title purposes)
    :param model_name: str: name of the model
    :param plot_diff: bool: if True, add a filled contour for the difference between the forecast and verification
    :param out_directory: str: if not None, save the figure to this directory
    :return: plt.Figure
    """
    # Plot an example forecast
    from mpl_toolkits.basemap import Basemap
    lons, lats = np.meshgrid(base.lon, base.lat)
    fig = plt.figure()
    fig.set_size_inches(9, 9)
    m = Basemap(llcrnrlon=0., llcrnrlat=0., urcrnrlon=360., urcrnrlat=90.,
                resolution='l', projection='cyl', lat_0=40., lon_0=0.)
    x, y = m(lons, lats)
    ax = plt.subplot(311)
    if plot_diff:
        m.contour(x, y, base, np.arange(-2.5, 1.6, 0.5), cmap='jet')
    else:
        m.pcolormesh(x, y, base, vmin=-2.5, vmax=1.5, cmap='YlGnBu_r')
    m.drawcoastlines()
    m.drawparallels(np.arange(0., 91., 45.))
    m.drawmeridians(np.arange(0., 361., 90.))
    ax.set_title('$t=0$ predictors')
    ax = plt.subplot(312)
    if plot_diff:
        m.contour(x, y, verif, np.arange(-2.5, 1.6, 0.5), cmap='jet')
    else:
        m.pcolormesh(x, y, verif, vmin=-2.5, vmax=1.5, cmap='YlGnBu_r')
    m.drawcoastlines()
    m.drawparallels(np.arange(0., 91., 45.))
    m.drawmeridians(np.arange(0., 361., 90.))
    ax.set_title('$t=%d$ verification (%s)' % f_hour)
    ax = plt.subplot(313)
    if plot_diff:
        m.contour(x, y, forecast, np.arange(-2.5, 1.6, 0.5), cmap='jet')
        m.pcolormesh(x, y, forecast - verif, vmin=-1, vmax=1, cmap='seismic')
    else:
        m.pcolormesh(x, y, forecast, vmin=-2.5, vmax=1.5, cmap='YlGnBu_r')
    # plt.colorbar(orientation='horizontal')
    m.drawcoastlines()
    m.drawparallels(np.arange(0., 91., 45.))
    m.drawmeridians(np.arange(0., 361., 90.))
    ax.set_title('$t=%d$ forecast' % f_hour)
    if out_directory is not None:
        plt.savefig('%s/%s_example_%d.pdf' % (out_directory, remove_chars(model_name), f_hour), bbox_inches='tight')
    return fig


def zonal_mean_plot(obs_mean, obs_std, pred_mean, pred_std, f_hour, model_name='', out_directory=None):
    """
    Plot the zonal mean and standard deviation of observed and predicted forecast states.

    :param obs_mean: 1d DataArray with dimension 'lat': observed zonal mean
    :param obs_std: 1d DataArray with dimension 'lat': observed zonal std
    :param pred_mean: 1d DataArray with dimension 'lat': forecast zonal mean
    :param pred_std: 1d DataArray with dimension 'lat': forecast zonal std
    :param f_hour: int: forecast hour of the prediction
    :param model_name: str: name of the model
    :param out_directory: str: if not None, save the figure to this directory
    :return:
    """
    fig = plt.figure()
    fig.set_size_inches(4, 6)
    plt.fill_betweenx(obs_mean.lat, obs_mean - obs_std, obs_mean + obs_std,
                      facecolor='C0', zorder=-50, alpha=0.3)
    plt.fill_betweenx(pred_mean.lat, pred_mean - pred_std, pred_mean + pred_std,
                      facecolor='C1', zorder=-40, alpha=0.3)
    plt.plot(obs_mean, obs_mean.lat, label='observed', color='C0')
    plt.plot(pred_mean, pred_mean.lat, label='%d-hour prediction' % f_hour, color='C1')
    plt.legend(loc='best')
    plt.grid(True, color='lightgray', zorder=-100)
    plt.plot(pred_mean - pred_std, pred_mean.lat, 'k:', linewidth=0.7)
    plt.plot(pred_mean + pred_std, pred_mean.lat, 'k:', linewidth=0.7)
    plt.xlabel('zonal mean height')
    plt.ylabel('latitude')
    plt.ylim([0., 90.])
    plt.savefig('%s/%s_zonal_climo.pdf' % (out_directory, remove_chars(model_name)), bbox_inches='tight')
    plt.show()
