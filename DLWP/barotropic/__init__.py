"""
A package for building and running simple atmospheric models.
The package contains code for a spectral barotropic model, with spectral
transforms provided by pyspharm. It also provides code for writing model
state to NetCDF files.
"""
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

# Some of the code has been modified by me (@jweyn). For the original
# model, please visit https://github.com/ajdawson/barotropic

from __future__ import (absolute_import, division, print_function)  #noqa

from .model import BarotropicModel, BarotropicModelPsi
