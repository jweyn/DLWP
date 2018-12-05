"""A spectral transforms engine using pyspharm."""
# (c) Copyright 2016 Andrew Dawson.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import (absolute_import, division, print_function)  #noqa

import numpy as np
try:
    from spharm import Spharmt, getspecindx, gaussian_lats_wts
except ImportError:
    raise ImportError('pyspharm is required to use this transforms engine')


class TransformsEngine(object):
    """A spectral transforms engine based on pyspharm."""

    def __init__(self, nlon, nlat, truncation, radius=6371200.):
        """
        Initialize the spectral transforms engine.
        Arguments:
        * nlon: int
            Number of longitudes in the transform grid.
        * nlat: int
            Number of latitudes in the transform grid.
        * truncation: int
            The spectral truncation (triangular). This is the maximum
            number of spherical harmonic modes retained in the discrete
            truncation. More modes means higher resolution.
        """
        self.sh = Spharmt(nlon, nlat, gridtype='regular', rsphere=radius)
        self.radius = radius
        self.nlon = nlon
        self.nlat = nlat
        self.truncation = truncation

    def vrtdiv_spec_from_uv_grid(self, u, v):
        """
        Compute spectral vorticity and divergence from grid u and v.
        """
        try:
            vrt, div = self.sh.getvrtdivspec(u, v, ntrunc=self.truncation)
        except ValueError:
            msg = ('u and v must be 2d or 3d arrays with shape ({y}, {x}) '
                   'or ({y}, {x}, :)'.format(y=self.nlat, x=self.nlon))
            raise ValueError(msg)
        return vrt, div

    def uv_grid_from_vrtdiv_spec(self, vrt, div):
        """
        Compute grid u and v from spectral vorticity and divergence.
        """
        try:
            u, v = self.sh.getuv(vrt, div)
        except ValueError:
            nspec = (self.truncation + 1) * (self.truncation + 2) // 2
            msg = ('vrt and div must be 1d or 2d arrays with shape '
                   '(n) or (n, :) where n <= {}'.format(nspec))
            raise ValueError(msg)
        return u, v

    def spec_to_grid(self, scalar_spec):
        """
        Transform a scalar field from spectral to grid space.
        """
        try:
            scalar_grid = self.sh.spectogrd(scalar_spec)
        except ValueError:
            nspec = (self.truncation + 1) * (self.truncation + 2) // 2
            msg = ('scalar_spec must be a 1d or 2d array with shape '
                   '(n) or (n, :) where n <= {}'.format(nspec))
            raise ValueError(msg)
        return scalar_grid

    def grid_to_spec(self, scalar_grid):
        """
        Transform a scalar field from grid to spectral space.
        """
        try:
            scalar_spec = self.sh.grdtospec(scalar_grid,
                                            ntrunc=self.truncation)
        except ValueError:
            msg = ('scalar_grid must be a 2d or 3d array with shape '
                   '({y}, {x}) or ({y}, {x}, :)'.format(y=self.nlat,
                                                        x=self.nlon))
            raise ValueError(msg)
        return scalar_spec

    def wavenumbers(self):
        """
        Wavenumbers corresponding to the spectral fields.
        """
        return getspecindx(self.truncation)

    def grid_latlon(self):
        """
        Return the latitude and longitude coordinate vectors of the
        model grid.
        """
        lats, _ = gaussian_lats_wts(self.nlat)
        lons = np.arange(0., 360., 360. / self.nlon)
        return lats, lons
