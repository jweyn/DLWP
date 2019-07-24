#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Utilities for retrieving and processing ERA5 reanalysis data using XArray.
"""

import os
import warnings
import itertools as it
import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
try:
    import cdsapi
except ImportError:
    warnings.warn("module 'cdsapi' not found; retrieval of ERA5 data unavailable.")


# ==================================================================================================================== #
# Universal parameters and functions
# ==================================================================================================================== #

def _check_exists(file_name, path=False):
    if os.path.exists(file_name):
        exists = True
        local_file = file_name
    else:
        exists = False
        local_file = None
    if path:
        return exists, local_file
    else:
        return exists


# For some reason, multiprocessing.Pool.map is placing arguments passed to the function inside another length-1 tuple.
# Much clearer programming would have required arguments of obj, m, month, *args here so that the user knows to include
# the CFS object, month index, month dates, and other arguments correctly.
def call_process_month(args):
    obj = args[0]
    obj._process_month(*args[1:])


def call_fetch(args):
    obj = args[0]
    obj._fetch(*args[1:])


# Format strings for files to write
netcdf_file_format = ''

# Start and end dates of available data
data_start_date = datetime(1979, 1, 1)
data_end_date = datetime(2018, 12, 31)
reforecast_start_date = datetime(1999, 1, 1)
reforecast_end_date = datetime(2009, 12, 31, 18)

# netCDF fill value
fill_value = np.array(nc.default_fillvals['f4']).astype(np.float32)

# Dictionary mapping request variables to netCDF variable naming conventions
var_names = {
    'geopotential': 'z',
    'temperature': 't'
}


# ==================================================================================================================== #
# ERA5Reanalysis object class
# ==================================================================================================================== #

class ERA5Reanalysis(object):
    """
    Class for manipulating ERA5 Reanalysis data with xarray. Class methods include functions to download,
    process, and export data.
    """

    def __init__(self, root_directory=None, file_id=''):
        """
        Initialize an instance of the ERA5Reanalysis class.

        :param root_directory: str: local directory where raw files are stored. If None, defaults to ~/.era5
        :param file_id: str: prepended to the processed file names. Useful if files for the same dates will be created
            with different parameters, i.e., hours or variables or levels.
        """
        self.raw_files = []
        self.dataset_variables = []
        self.dataset_levels = []
        if root_directory is None:
            self._root_directory = '%s/.era5' % os.path.expanduser('~')
        else:
            self._root_directory = root_directory
        self._file_id = file_id
        self._delete_temp = False
        self.level_coord = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
                            500, 550, 600, 650, 700, 750] + list(range(775, 1001, 25))
        self.inverse_lat = True
        # Data
        self.Dataset = None
        self.basemap = None
        self._lat_array = None
        self._lon_array = None

    @property
    def lat(self):
        if self._lat_array is not None:
            return self._lat_array
        try:
            lat = self.Dataset.variables['lat'][:]
            if len(lat.shape) > 2:
                self._lat_array = lat[0, ...].values
                return self._lat_array
            else:
                self._lat_array = lat.values
                return self._lat_array
        except AttributeError:
            raise AttributeError('Call to lat method is only valid after data are opened.')
        except KeyError:
            return

    @property
    def lon(self):
        if self._lon_array is not None:
            return self._lon_array
        try:
            lon = self.Dataset.variables['lon'][:]
            if len(lon.shape) > 2:
                self._lon_array = lon[0, ...].values
                return self._lon_array
            else:
                self._lon_array = lon.values
                return self._lon_array
        except AttributeError:
            raise AttributeError('Call to lon method is only valid after data are opened.')
        except KeyError:
            return

    def set_variables(self, variables):
        """
        Set the variables to retrieve or open in the dataset. Overridden by arguments to the 'retrieve' method.

        :param variables: list of string variable names
        :return:
        """
        for v in variables:
            try:
                assert str(v) in list(var_names.keys())
            except TypeError:
                raise TypeError('variables must be convertible to string types')
            except AssertionError:
                raise ValueError('variables must be within the available levels for the dataset (%s)' %
                                 list(var_names.keys()))
        self.dataset_variables = sorted(list(variables))

    def set_levels(self, levels):
        """
        Set the levels to retrieve or open in the dataset. Overridden by arguments to the 'retrieve' method.

        :param levels: list of integer pressure height levels (mb / hPa)
        :return:
        """
        for l in levels:
            try:
                assert int(l) in self.level_coord
            except TypeError:
                raise ValueError('levels must be integers in hPa')
            except AssertionError:
                raise ValueError('levels must be within the available levels for the dataset (%s)' % self.level_coord)
        self.dataset_levels = sorted(list(levels))

    def closest_lat_lon(self, lat, lon):
        """
        Find the grid-point index of the closest point to the specified latitude and longitude values in loaded
        CFS reanalysis data.

        :param lat: float or int: latitude in degrees
        :param lon: float or int: longitude in degrees
        :return:
        """
        if lon < 0.:
            lon += 360.
        distance = (self.lat - lat) ** 2 + (self.lon - lon) ** 2
        min_dist = 2.5
        if np.min(distance) > min_dist:
            raise ValueError('no latitude/longitude points close to requested lat/lon!')
        return np.unravel_index(np.argmin(distance, axis=None), distance.shape)

    def _set_file_names(self):
        # Sets a list of file names.
        for variable in self.dataset_variables:
            for level in self.dataset_levels:
                self.raw_files.append('%s/%s%s_%s.nc' % (self._root_directory, self._file_id, variable, level))

    def retrieve(self, variables, levels, years='all', months='all', days='all', hourly=3, n_proc=4, verbose=False,
                 request_kwargs=None, delete_temporary=False):
        """
        Retrieve netCDF files of ERA5 reanalysis data. Must specify the variables and pressure levels desired.
        Iterates over variable/level pairs for each API request. Note that with 3-hourly data, one variable/level pair
        can be retrieved with a single API request for all dates between 1979-2018. If more dates or higher hourly
        resolution is required, it is currently up to the user to perform separate retrieval requests. DO NOT use the
        same retrieve function in the same instance of a class to request more dates as this will overwrite
        previously downloaded files. Instead, create a new instance of ERA5Reanalysis, give a different file_id, and
        then manually concatenate the datasets loaded on each instance.

        :param variables: iterable of str: variables to retrieve, one at a time
        :param levels: iterable of int: pressure levels to retrieve, one at a time
        :param years: iterable: years of data. If 'all', use 1979-2018.
        :param months: iterable: months of data. If 'all', get all months.
        :param days: iterable: month days of data. If 'all', get all days.
        :param hourly: int: hourly time resolution; e.g., 6 for data every 6 hours.
        :param n_proc: int: number of processes for parallel retrieval
        :param verbose: bool: if True, print progress statements. The API already lists progress statements.
        :param request_kwargs: dict: other keywords passed to the retrieval. For example, 'grid' can be used to modify
            the lat/lon resolution.
        :param delete_temporary: bool: if True, delete the temporary files from the server in favor of the edited
            files with correct dimensions. May be risky to delete the raw files.
        """
        # Parameter checks
        request_kwargs = {} or request_kwargs
        self.set_variables(variables)
        self.set_levels(levels)
        if delete_temporary:
            self._delete_temp = True
        if years == 'all':
            years = list(range(data_start_date.year, data_end_date.year + 1))
        else:
            for y in years:
                try:
                    assert data_start_date.year <= int(y) <= data_end_date.year
                except TypeError:
                    raise ValueError('years must be integers')
                except AssertionError:
                    raise ValueError('years must be within the available dates for ERA5 (%d-%d)' %
                                     (data_start_date.year, data_end_date.year))
        years = [str(y) for y in years]
        if months == 'all':
            months = list(range(1, 13))
        else:
            for m in months:
                try:
                    assert 1 <= int(m) <= 12
                except TypeError:
                    raise ValueError('months must be integers')
                except AssertionError:
                    raise ValueError('months must be integers from 1 to 12')
        months = ['%02d' % m for m in months]
        if days == 'all':
            days = list(range(1, 32))
        else:
            for d in days:
                try:
                    assert 1 <= int(d) <= 31
                except TypeError:
                    raise ValueError('days must be integers')
                except AssertionError:
                    raise ValueError('days must be integers from 1 to 31')
        days = ['%02d' % d for d in days]
        if hourly < 1 or hourly > 24:
            raise ValueError('hourly interval must be between 1 and 24')
        hour_daterange = pd.date_range('2000-01-01 00:00', '2000-01-01 23:00', freq='%dh' % hourly)
        hours = [d.strftime('%H:%M') for d in hour_daterange]
        if len(variables) == 0:
            print('ERA5Reanalysis.retrieve: no variables specified; will do nothing.')
            return
        if len(levels) == 0:
            print('ERA5Reanalysis.retrieve: no pressure levels specified; will do nothing.')
            return
        if int(n_proc) < 0:
            raise ValueError("'multiprocess' must be an integer >= 0")

        # Create the requests
        requests = []
        self._set_file_names()
        for variable in variables:
            for level in levels:
                request = {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': variable,
                    'pressure_level': level,
                    'year': years,
                    'month': months,
                    'day': days,
                    'time': hours
                }
                request.update(request_kwargs)
                requests.append(request)

        # Create a multi-processing tool, if necessary
        if n_proc == 0 or n_proc > 1:
            try:
                import multiprocessing
                if n_proc == 0:
                    n_proc = multiprocessing.cpu_count()
            except ImportError:
                warnings.warn("'multiprocessing' module not available; falling back to serial")
                n_proc = 1

        if n_proc == 1:
            for file, request in zip(self.raw_files, requests):
                call_fetch((self, request, file, verbose))
        else:
            pool = multiprocessing.Pool(processes=n_proc)
            pool.map(call_fetch, zip(it.repeat(self), requests, self.raw_files, it.repeat(verbose)))
            pool.close()
            pool.terminate()
            pool.join()

    def _fetch(self, request, file_name, verbose):
        # Fetch the file
        c = cdsapi.Client()
        if verbose:
            print('ERA5Reanalysis.retrieve: fetching %s at %s mb' % (request['variable'], request['pressure_level']))
        c.retrieve('reanalysis-era5-pressure-levels', request, file_name + '.tmp')

        # Add a level dimension to the file (not present by default
        if verbose:
            print('Adding level dimension')
        self._process_temp_file(file_name, float(request['pressure_level']))

    def _process_temp_file(self, file_name, level):
        ds = xr.open_dataset(file_name + '.tmp')
        ds = ds.expand_dims('level', axis=1).assign_coords(level=np.array([level], dtype=np.float32))
        ds.to_netcdf(file_name)
        if self._delete_temp:
            os.remove(file_name + '.tmp')

    def open(self, **dataset_kwargs):
        """
        Open an xarray multi-file Dataset for the processed files. Set the variables and levels with the instance
        set_variables and set_levels methods. Once opened, this Dataset is accessible by self.Dataset.

        :param dataset_kwargs: kwargs passed to xarray.open_mfdataset()
        """
        if len(self.dataset_variables) == 0:
            raise ValueError('set the variables to open with the set_variables() method')
        if len(self.dataset_levels) == 0:
            raise ValueError('set the pressure levels to open with the set_levels() method')
        self._set_file_names()
        self.Dataset = xr.open_mfdataset(self.raw_files, **dataset_kwargs)

    def close(self):
        """
        Close an opened Dataset on self.
        """
        if self.Dataset is not None:
            self.Dataset.close()
            self.Dataset = None
            self._lon_array = None
            self._lat_array = None
        else:
            raise ValueError('no Dataset to close')

    def generate_basemap(self, llcrnrlat=None, llcrnrlon=None, urcrnrlat=None, urcrnrlon=None):
        """
        Generates a Basemap object for graphical plot of ERA5 data on a 2-D plane. Bounding box parameters
        are either given, or if None, read from the extremes of the loaded lat/lon data. Other projection parameters
        are set to the default ERA5 configuration.

        :param llcrnrlat: float: lower left corner latitude
        :param llcrnrlon: float: lower left corner longitude
        :param urcrnrlat: float: upper right corner latitude
        :param urcrnrlon: float: upper right corner longitude
        :return:
        """
        from mpl_toolkits.basemap import Basemap

        try:
            default = llcrnrlat * llcrnrlon * urcrnrlat * urcrnrlon  # error if any are None
            default = False
        except TypeError:
            default = True

        if default:
            try:
                lat = self.lat
                lon = self.lon
            except (AttributeError, KeyError):
                raise ValueError('I can generate a default Basemap with None parameters, but only if I have some '
                                 'data loaded first!')
            llcrnrlon, llcrnrlat = lon[0, 0], lat[-1, -1]
            urcrnrlon, urcrnrlat = lon[-1, -1], lat[0, 0]

        basemap = Basemap(projection='cyl', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution='l')

        self.basemap = basemap
