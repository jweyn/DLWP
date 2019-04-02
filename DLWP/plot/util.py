#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Plotting utilities.
"""
import re

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def rotate_vector_r(basemap, uin, vin, lons, lats, returnxy=False):
    """
    I got this method from StackOverflow at some point, but didn't document its origin. Work is not my own.

    Similar to Basemap.rotate_vector except that it returns map-projected
    vectors to horizontal grid vectors. The following is the docstring
    for Basemap.rotate_vector. Note this function takes the basemap instance
    as an argument.

    Rotate a vector field (``uin,vin``) on a rectilinear grid
    with longitudes = ``lons`` and latitudes = ``lats`` from
    geographical (lat/lon) into map projection (x/y) coordinates.

    Differs from transform_vector in that no interpolation is done.
    The vector is returned on the same grid, but rotated into
    x,y coordinates.

    The input vector field is defined in spherical coordinates (it
    has eastward and northward components) while the output
    vector field is rotated to map projection coordinates (relative
    to x and y). The magnitude of the vector is preserved.

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    uin, vin         input vector field on a lat/lon grid.
    lons, lats       Arrays containing longitudes and latitudes
                     (in degrees) of input data in increasing order.
                     For non-cylindrical projections (those other than
                     ``cyl``, ``merc``, ``cyl``, ``gall`` and ``mill``) lons
                     must fit within range -180 to 180.
    ==============   ====================================================

    Returns ``uout, vout`` (rotated vector field).
    If the optional keyword argument
    ``returnxy`` is True (default is False),
    returns ``uout,vout,x,y`` (where ``x,y`` are the map projection
    coordinates of the grid defined by ``lons,lats``).
    """

    # if lons,lats are 1d and uin,vin are 2d, and
    # lats describes 1st dim of uin,vin, and
    # lons describes 2nd dim of uin,vin, make lons,lats 2d
    # with meshgrid.
    if lons.ndim == lats.ndim == 1 and uin.ndim == vin.ndim == 2 and \
            uin.shape[1] == vin.shape[1] == lons.shape[0] and \
            uin.shape[0] == vin.shape[0] == lats.shape[0]:
        lons, lats = np.meshgrid(lons, lats)
    else:
        if not lons.shape == lats.shape == uin.shape == vin.shape:
            raise TypeError("shapes of lons,lats and uin,vin don't match")
    x, y = basemap(lons, lats)
    # rotate from geographic to map coordinates.

    # Map the (lon, lat) vector in the complex plane.
    uvc = uin + 1j * vin
    uvmag = np.abs(uvc)
    theta = np.angle(uvc)

    # Define a displacement (dlon, dlat) that moves all
    # positions (lons, lats) a small distance in the
    # direction of the original vector.
    dc = 1E-5 * np.exp(theta * 1j)
    dlat = dc.imag * np.cos(np.radians(lats))
    dlon = dc.real

    # Deal with displacements that overshoot the North or South Pole.
    farnorth = np.abs(lats + dlat) >= 90.0
    somenorth = farnorth.any()
    if somenorth:
        dlon[farnorth] *= -1.0
        dlat[farnorth] *= -1.0

    # Add displacement to original location and find the native coordinates.
    lon1 = lons + dlon
    lat1 = lats + dlat
    xn, yn = basemap(lon1, lat1)

    # Determine the angle of the displacement in the native coordinates.
    vecangle = np.arctan2(yn - y, xn - x)
    if somenorth:
        vecangle[farnorth] += np.pi

    # Reverse the direction of vecangle about the original vector
    vecdiff = vecangle - theta
    vecangle -= 2. * vecdiff

    # Compute the x-y components of the original vector.
    uvcout = uvmag * np.exp(1j * vecangle)
    uout = uvcout.real
    vout = uvcout.imag

    if returnxy:
        return uout, vout, x, y
    else:
        return uout, vout


def radar_colormap():
    """
    Function to output a matplotlib color map object for reflectivity based on
    the National Weather Service color scheme.
    """

    nws_reflectivity_colors = [
        #    "#646464", # ND
        #    "#ccffff", # -30
        #    "#cc99cc", # -25
        #    "#996699", # -20
        #    "#663366", # -15
        #    "#cccc99", # -10
        #    "#999966", # -5
        #    "#646464", # 0
        "#ffffff",  # 0 white
        "#04e9e7",  # 5
        "#019ff4",  # 10
        "#0300f4",  # 15
        "#02fd02",  # 20
        "#01c501",  # 25
        "#008e00",  # 30
        "#fdf802",  # 35
        "#e5bc00",  # 40
        "#fd9500",  # 45
        "#fd0000",  # 50
        "#d40000",  # 55
        "#bc0000",  # 60
        "#f800fd",  # 65
        "#9854c6",  # 70
        #    "#fdfdfd" # 75
    ]
    return ListedColormap(nws_reflectivity_colors)


def blue_red_colormap(size=100, reverse=False, white_padding=1, ):
    size = size - (size % 4)
    ll = size // 4
    clist = []
    s1 = [0.5 + 0.5 / ll * x for x in range(ll)]
    s2 = [1.0 / ll * x for x in range(ll)]
    for x in s1:
        clist.append([0.0, 0.0, x])
    for x in s2:
        clist.append([x, x, 1.0])
    for x in range(white_padding):
        clist.append([1.0, 1.0, 1.0])
    for x in range(ll):
        clist.append([1.0, s2[-x - 1], s2[-x - 1]])
    for x in range(ll):
        clist.append([s1[-x - 1], 0.0, 0.0])
    if reverse:
        clist = clist[::-1]

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('BuRd', clist)
    return cmap


def rgb_colormap(color='blue', size=100, reverse=False, white_padding=1, ):
    ll = size - white_padding
    clist = []
    r = 1
    g = 1
    b = 1
    if color == 'red':
        r = 0.5
    elif color == 'green':
        g = 0.5
    elif color == 'blue':
        b = 0.5
    else:
        raise ValueError('Select "red", "green", or "blue" for color.')
    for x in range(white_padding):
        clist.append([1.0, 1.0, 1.0])
    for x in range(white_padding, size, 1):
        y = x - white_padding
        clist.append([1.0 - 1.0 * r * y / ll, 1.0 - 1.0 * g * y / ll, 1.0 - 1.0 * b * y / ll])
    if reverse:
        clist = clist[::-1]

    cmap = LinearSegmentedColormap.from_list(color, clist)
    return cmap


def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max requiring the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin)).
          For example, for a data range from -15.0 to +5.0 and
          the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    for rj, sj in zip(reg_index, shift_index):
        r, g, b, a = cmap(rj)
        cdict['red'].append((sj, r, r))
        cdict['green'].append((sj, g, g))
        cdict['blue'].append((sj, b, b))
        cdict['alpha'].append((sj, a, a))
    new_cmap = LinearSegmentedColormap(name, cdict)
    # plt.register_cmap(cmap=new_cmap)
    return new_cmap


def remove_chars(s):
    """
    Remove characters from a string that have unintended effects on file paths.

    :param s: str
    :return: str
    """
    return ''.join(re.split('[$/\\\\]', s))
