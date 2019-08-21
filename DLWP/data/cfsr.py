#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Utilities for retrieving and processing CFS reanalysis and reforecast data using XArray.
For now, we only implement the regularly-gridded 1-degree data. Support for variables on the native Gaussian ~0.5
degree grid may come in the future.
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
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen
try:
    import pygrib
except ImportError:
    warnings.warn("module 'pygrib' not found; processing of raw CFS data unavailable.")


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


# Format strings for files to read/write
grib_dir_format = '%Y/%Y%m/%Y%m%d'
grib_file_format = 'pgb{:s}{:s}.gdas.%Y%m%d%H.grb2'
reforecast_dir_format = '{:s}/%Y%m'
reforecast_file_format = '{:s}_f.01.{:s}.{:s}.{:s}.grb2'

# Start and end dates of available data
data_start_date = datetime(1979, 1, 1)
data_end_date = datetime(2011, 3, 31)
reforecast_start_date = datetime(1999, 1, 1)
reforecast_end_date = datetime(2009, 12, 31, 18)

# Parameter tables for GRIB data. Should be included in repository.
dir_path = os.path.dirname(os.path.realpath(__file__))
grib2_table = np.genfromtxt('%s/cfsr_pgb_grib_table.csv' % dir_path, dtype='str', delimiter=',')

# netCDF fill value
fill_value = np.array(nc.default_fillvals['f4']).astype(np.float32)


# ==================================================================================================================== #
# CFSReanalysis object class
# ==================================================================================================================== #


class CFSReanalysis(object):
    """
    Class for manipulating CFS Reanalysis data with xarray. Class methods include functions to download,
    process, and export data. Currently only works with pressure-level data ('pgb').
    """

    def __init__(self, root_directory=None, resolution='l', run_type='06', fill_hourly=True, file_id=''):
        """
        Initialize an instance of the CFSReanalysis class.

        :param root_directory: str: local directory where raw files are stored. If None, defaults to ~/.cfsr
        :param resolution: str: 'h' corresponds to the high-res 0.5-degree grid; 'l' the low-res 2.5-degree grid
        :param run_type: str: one of the forecast hours or the analysis: ['01', '02', '03', '04', '05', '06', 'nl']
        :param fill_hourly: bool: if True, automatically add in 6-hourly time steps even if only 00Z dates are given
        :param file_id: str: appended to the processed file names. Useful if files for the same dates will be created
            with different parameters, i.e., hours or variables or levels.
        """
        self.raw_files = []
        self.dataset_dates = []
        self.dataset_variables = []
        if root_directory is None:
            self._root_directory = '%s/.cfsr' % os.path.expanduser('~')
        else:
            self._root_directory = root_directory
        self._resolution = resolution
        if resolution == 'h':
            self._ny = 361
            self._nx = 720
            self._root_url = 'https://nomads.ncdc.noaa.gov/modeldata/cmd_pgbh/'
        elif resolution == 'l':
            self._ny = 73
            self._nx = 144
            self._root_url = 'https://nomads.ncdc.noaa.gov/modeldata/cmd_grblow'
        else:
            raise ValueError("resolution must be 'h' or 'l'")
        if run_type not in ['01', '02', '03', '04', '05', '06', 'nl']:
            raise ValueError("run_type must be 'nl' or a 2-digit forecast hour from '01' to '06'")
        else:
            self._run_type = run_type
        self._fill_hourly = fill_hourly
        self._file_id = file_id
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

    def set_dates(self, dates):
        """
        Set the CFSReanalysis object's dataset_dates attribute, a list of datetime objects which determines which
        datetimes are retrieved and processed. This attribute is set automatically when using the method 'retrieve',
        but may be used when 'retrieve' is not desired or as an override.

        :param dates: list of datetime objects.
        :return:
        """
        self.dataset_dates = sorted(dates)
        if self._fill_hourly:
            day_set = sorted(set([datetime(d.year, d.month, d.day) for d in self.dataset_dates]))
            new_dates = []
            for day in day_set:
                new_dates.extend((day, day.replace(hour=6), day.replace(hour=12), day.replace(hour=18)))
            self.dataset_dates = new_dates

    def set_levels(self, levels):
        """
        Set the CFSReanalysis object's level_coord attribute, a list of integer height levels which determines which
        levels are processed and written to netCDF files. This attribute is set to a default, but may be overriden.
        Note that any further processing or reading of data must use the same level coordinate, i.e., choose wisely!

        :param levels: list of integer pressure height levels (mb / hPa)
        :return:
        """
        self.level_coord = sorted([l for l in levels if 0 <= int(l) <= 1000])

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
        min_dist = 2.5 if self._resolution == 'l' else 1.
        if np.min(distance) > min_dist:
            raise ValueError('no latitude/longitude points within 1 degree of requested lat/lon!')
        return np.unravel_index(np.argmin(distance, axis=None), distance.shape)

    def retrieve(self, dates, n_proc=4, verbose=False):
        """
        Retrieves CFS reanalysis data for the given datetimes, and writes them to the local directory. The same
        directory structure (%Y/%Y%m/%Y%m%d/file_name) is used locally as on the server. Creates subdirectories if
        necessary. File types retrieved are given by the object's init parameters.

        :param dates: list or tuple: date or datetime objects of of analysis times. May be 'all', in which case
            all dates in the object's 'dataset_dates' attributes are retrieved.
        :param n_proc: int: if >1, fetches files in parallel. This speeds up performance but may not scale well if
            internet I/O is the bottleneck. Set to 0 to use all available threads.
        :param verbose: bool: include progress print statements
        :return: None
        """
        # Check if any parameter is a single value
        if dates == 'all':
            dates = self.dataset_dates
        else:
            self.set_dates(dates)
            dates = self.dataset_dates

        # Determine the files to retrieve
        if verbose:
            print('CFSReanalysis.retrieve: beginning data retrieval\n')
        self.raw_files = []
        for dt in dates:
            if dt < data_start_date or dt > data_end_date:
                print('* Warning: doing nothing for date %s, out of valid data range (%s to %s)' %
                      (dt, data_start_date, data_end_date))
                continue
            if dt not in self.dataset_dates:
                self.dataset_dates.append(dt)
                # Create local directory
            grib_file_dir = datetime.strftime(dt, grib_dir_format)
            os.makedirs('%s/%s' % (self._root_directory, grib_file_dir), exist_ok=True)
            # Add GRIB file to listing
            grib_file_name = datetime.strftime(dt, grib_file_format)
            grib_file_name = '%s/%s' % (grib_file_dir, grib_file_name.format(self._resolution, self._run_type))
            if grib_file_name not in self.raw_files:
                self.raw_files.append(grib_file_name)

        if n_proc == 0 or n_proc > 1:
            try:
                import multiprocessing
                if n_proc == 0:
                    n_proc = multiprocessing.cpu_count()
            except ImportError:
                warnings.warn("'multiprocessing' module not available; falling back to serial")
                n_proc = 1

        if n_proc == 1:
            for file in self.raw_files:
                call_fetch((self, file, verbose))
        else:
            pool = multiprocessing.Pool(processes=n_proc)
            pool.map(call_fetch, zip(it.repeat(self), self.raw_files, it.repeat(verbose)))
            pool.close()
            pool.terminate()
            pool.join()

    def _fetch(self, f, verbose):
        pid = os.getpid()
        local_file = '%s/%s' % (self._root_directory, f)
        if _check_exists(local_file):
            if verbose:
                print('PID %s: local file %s exists; omitting' % (pid, local_file))
            return
        remote_file = '%s/%s' % (self._root_url, f)
        if verbose:
            print('PID %s: downloading %s' % (pid, remote_file))
        try:
            response = urlopen(remote_file)
            with open(local_file, 'wb') as fd:
                fd.write(response.read())
        except BaseException as e:
            print('warning: failed to download %s, retrying' % remote_file)
            try:
                response = urlopen(remote_file)
                with open(local_file, 'wb') as fd:
                    fd.write(response.read())
            except BaseException as e:
                print('warning: failed to download %s' % remote_file)
                print('* Reason: "%s"' % str(e))

    def write(self, variables='all', dates='all', levels='all', write_into_existing=True, omit_existing=False,
              delete_raw_files=False, n_proc=4, verbose=False):
        """
        Reads raw CFS reanalysis files for the given dates (list or tuple form) and specified variables and levels and
        writes the data to reformatted netCDF files. Processed files are saved under self._root_directory/processed;
        one file per month is created.

        :param variables: list: list of variables to retrieve from data or 'all'
        :param dates: list or tuple of datetime: date or datetime objects of model initialization; may be 'all', in
            which case, all the dates in the object's dataset_dates attribute are used (these are set when calling
            self.retrieve() or self.set_dates())
        :param levels: list or tuple of int: list of pressure levels as int (in mb); must be compatible with existing
            processed files; may be 'all', using the object's level_coord attribute
        :param write_into_existing: bool: if True, checks for existing files and appends if they exist. If False,
            overwrites any existing files.
        :param omit_existing: bool: if True, then if a processed file exists, skip it. Only useful if existing data
            are known to be complete.
        :param delete_raw_files: bool: if True, deletes the original data files from which the processed versions were
            made
        :param n_proc: int: if >1, runs write tasks in parallel, one per month of data. This speeds up performance but
            may not scale well if disk I/O is the bottleneck. Set to 0 to use all available threads.
        :param verbose: bool: include progress print statements
        :return:
        """
        # Parameter checks
        if variables == 'all':
            variables = list(grib2_table[:, 0])
        if dates == 'all':
            dates = self.dataset_dates
        else:
            self.set_dates(dates)
            dates = self.dataset_dates
        if levels == 'all':
            levels = [l for l in self.level_coord]
        else:
            self.set_levels(levels)
            levels = self.level_coord
        if len(variables) == 0:
            print('CFSReanalysis.write: no variables specified; will do nothing.')
            return
        if len(dates) == 0:
            print('CFSReanalysis.write: no dates specified; will do nothing.')
            return
        if len(levels) == 0:
            print('CFSReanalysis.write: no pressure levels specified; will do nothing.')
            return
        if int(n_proc) < 0:
            raise ValueError("'multiprocess' must be an integer >= 0")
        self.dataset_variables = list(variables)

        # Generate monthly batches of dates
        dates_index = pd.DatetimeIndex(dates).sort_values()
        months = dates_index.to_period('M')
        unique_months = months.unique()
        month_list = []
        for nm in range(len(unique_months)):
            month_list.append(list(dates_index[months == unique_months[nm]].to_pydatetime()))

        if n_proc == 0 or n_proc > 1:
            try:
                import multiprocessing
                if n_proc == 0:
                    n_proc = multiprocessing.cpu_count()
            except ImportError:
                warnings.warn("'multiprocessing' module not available; falling back to serial")
                n_proc = 1

        if n_proc == 1:
            for nm, month in enumerate(month_list):
                call_process_month((self, nm, month, unique_months, variables, levels, write_into_existing,
                                    omit_existing, delete_raw_files, verbose))
        else:
            pool = multiprocessing.Pool(processes=n_proc)
            pool.map(call_process_month, zip(it.repeat(self), range(len(month_list)), month_list,
                                             it.repeat(unique_months), it.repeat(variables), it.repeat(levels),
                                             it.repeat(write_into_existing), it.repeat(omit_existing),
                                             it.repeat(delete_raw_files), it.repeat(verbose)))
            pool.close()
            pool.terminate()
            pool.join()

    # Define a function for multi-processing
    def _process_month(self, m, month, unique_months, variables, levels, write_into_existing, omit_existing,
                       delete_raw_files, verbose):
        # Define some data reading functions that also write to the output
        def read_write_grib_lat_lon(file_name, nc_fid):
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                raise IOError('File %s not found.' % file_name)
            grib_data = pygrib.open(file_name)
            try:
                lats = np.array(grib_data[1]['latitudes'], dtype=np.float32)
                lons = np.array(grib_data[1]['longitudes'], dtype=np.float32)
                shape = grib_data[1].values.shape
                lat = lats.reshape(shape)[:, 0]
                lon = lons.reshape(shape)[0, :]
            except BaseException:
                print('* Warning: cannot get lat/lon from grib file %s' % exists_file_name)
                raise
            if verbose:
                print('PID %s: Writing latitude and longitude' % pid)
            nc_var = nc_fid.createVariable('lat', np.float32, ('lat',), zlib=True)
            nc_var.setncatts({
                'long_name': 'Latitude',
                'units': 'degrees_north'
            })
            nc_fid.variables['lat'][:] = lat
            nc_var = nc_fid.createVariable('lon', np.float32, ('lon',), zlib=True)
            nc_var.setncatts({
                'long_name': 'Longitude',
                'units': 'degrees_east'
            })
            nc_fid.variables['lon'][:] = lon
            grib_data.close()

        def read_write_grib(file_name, time_index, nc_fid):
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                print('* Warning: file %s not found' % file_name)
                return
            if verbose:
                print('PID %s: Reading %s' % (pid, exists_file_name))
            grib_data = pygrib.open(file_name)
            # Have to do this the hard way, because grib_index doesn't work on these 'multi-field' files
            grib_index = []
            grib_index_no_level = []
            for grb in grib_data:
                try:
                    grib_index.append([int(grb.discipline), int(grb.parameterCategory),
                                       int(grb.parameterNumber), grb.levelType, int(grb.level)])
                    grib_index_no_level.append([int(grb.discipline), int(grb.parameterCategory),
                                                int(grb.parameterNumber), grb.levelType])
                except RuntimeError:
                    grib_index.append([])
                    grib_index_no_level.append([])
            if verbose:
                print('PID %s: Variables to fetch: %s' % (pid, variables))
            for row in range(grib2_table.shape[0]):
                var = grib2_table[row, 0]
                if var in variables:
                    if var not in nc_fid.variables.keys():
                        if verbose:
                            print('PID %s: Creating variable %s' % (pid, var))
                        if grib2_table[row, 6] == 'pl':
                            nc_var = nc_fid.createVariable(var, np.float32, ('time', 'level', 'lat', 'lon'), zlib=True)
                        else:
                            nc_var = nc_fid.createVariable(var, np.float32, ('time', 'lat', 'lon'), zlib=True)
                        nc_var.setncatts({
                            'long_name': grib2_table[row, 4],
                            'units': grib2_table[row, 5],
                            '_FillValue': fill_value
                        })
                    if grib2_table[row, 6] == 'pl':
                        for level_index, level in enumerate(levels):
                            try:
                                if verbose:
                                    print('PID %s: Writing %s at level %d' % (pid, var, level))
                                # Match a list containing discipline, parameterCategory, parameterNumber, level.
                                # Add one because grib indexing starts at 1.
                                grib_key = grib_index.index([int(grib2_table[row, 1]), int(grib2_table[row, 2]),
                                                             int(grib2_table[row, 3]), grib2_table[row, 6],
                                                             int(level)]) + 1
                                if verbose:
                                    print('  found %s' % grib_data[grib_key])
                                data = np.array(grib_data[grib_key].values, dtype=np.float32)
                                nc_fid.variables[var][time_index, level_index, ...] = data
                            except OSError:  # missing index gives an OS read error
                                print('* Warning: grib variable %s not found in file %s' % (var, file_name))
                                pass
                            except BaseException as e:
                                print("* Warning: failed to write %s to netCDF file ('%s')" % (var, str(e)))
                    else:
                        try:
                            if verbose:
                                print('PID %s: Writing %s' % (pid, var))
                            # Match a list containing discipline, parameterCategory, parameterNumber, level.
                            # Add one because grib indexing starts at 1.
                            grib_key = grib_index_no_level.index([int(grib2_table[row, 1]), int(grib2_table[row, 2]),
                                                                  int(grib2_table[row, 3]), grib2_table[row, 6]]) + 1
                            if verbose:
                                print('  found %s' % grib_data[grib_key])
                            data = np.array(grib_data[grib_key].values, dtype=np.float32)
                            nc_fid.variables[var][time_index, ...] = data
                        except OSError:  # missing index gives an OS read error
                            print('* Warning: grib variable %s not found in file %s' % (var, file_name))
                            pass
                        except BaseException as e:
                            print("* Warning: failed to write %s to netCDF file ('%s')" % (var, str(e)))

            grib_data.close()
            return

        # We're gonna have to do this the ugly way, with the netCDF4 module.
        # Iterate over months, create a netCDF file for the month, and fill in all datetimes we want
        # Create netCDF file, or append
        pid = os.getpid()
        nc_file_dir = '%s/processed' % self._root_directory
        os.makedirs(nc_file_dir, exist_ok=True)
        nc_file_name = '%s/%s%s.nc' % (nc_file_dir, self._file_id, datetime.strftime(month[0], '%Y%m'))
        if verbose:
            print('PID %s: Writing to file %s' % (pid, nc_file_name))
        nc_file_open_type = 'w'
        init_coord = True
        if os.path.isfile(nc_file_name):
            if omit_existing:
                if verbose:
                    print('PID %s: Omitting file %s; exists' % (pid, nc_file_name))
                return
            if write_into_existing:
                nc_file_open_type = 'a'
                init_coord = False
            else:
                os.remove(nc_file_name)
        nc_file_id = nc.Dataset(nc_file_name, nc_file_open_type, format='NETCDF4')

        # Initialize coordinates, if needed
        time_axis = pd.DatetimeIndex(start=unique_months[m].start_time, end=unique_months[m].end_time,
                                     freq='6H').to_pydatetime()
        if init_coord:
            # Create dimensions
            if verbose:
                print('PID %s: Creating coordinate dimensions' % pid)
            nc_file_id.description = 'Selected variables and levels from the CFS Reanalysis'
            nc_file_id.createDimension('time', 0)
            nc_file_id.createDimension('level', len(self.level_coord))
            nc_file_id.createDimension('lat', self._ny)
            nc_file_id.createDimension('lon', self._nx)

            # Create unlimited time variable for initialization time
            nc_var = nc_file_id.createVariable('time', np.float32, 'time', zlib=True)
            time_units = 'hours since 1970-01-01 00:00:00'

            nc_var.setncatts({
                'long_name': 'Model initialization time',
                'units': time_units
            })
            nc_file_id.variables['time'][:] = nc.date2num(time_axis, time_units)

            # Create unchanging level variable
            nc_var = nc_file_id.createVariable('level', np.float32, 'level', zlib=True)
            nc_var.setncatts({
                'long_name': 'Pressure level',
                'units': 'hPa'
            })
            nc_file_id.variables['level'][:] = self.level_coord

        # Now go through the time files to add data to the netCDF file
        for dt in month:
            grib_file_dir = datetime.strftime(dt, grib_dir_format)
            grib_file_name = datetime.strftime(dt, grib_file_format.format(self._resolution, self._run_type))
            grib_file_name = '%s/%s/%s' % (self._root_directory, grib_file_dir, grib_file_name)
            # Write the latitude and longitude coordinate arrays, if needed
            if init_coord:
                try:
                    read_write_grib_lat_lon(grib_file_name, nc_file_id)
                    init_coord = False
                except (IOError, OSError):
                    print("* Warning: file %s not found for coordinates; trying the next one." % grib_file_name)
            read_write_grib(grib_file_name, list(time_axis).index(dt), nc_file_id)

            # Delete files if requested
            if delete_raw_files:
                if os.path.isfile(grib_file_name):
                    os.remove(grib_file_name)

        nc_file_id.close()

    def open(self, exact_dates=True, concat_dim='time', **dataset_kwargs):
        """
        Open an xarray multi-file Dataset for the processed files with dates set using set_dates(), retrieve(), or
        write(). Once opened, this Dataset is accessible by self.Dataset.

        :param exact_dates: bool: if True, set the Dataset to have the exact dates of this instance; otherwise,
            keep all of the monthly dates in the opened files
        :param concat_dim: passed to xarray.open_mfdataset()
        :param dataset_kwargs: kwargs passed to xarray.open_mfdataset()
        """
        nc_file_dir = '%s/processed' % self._root_directory
        if not self.dataset_dates:
            raise ValueError("use set_dates() to specify times of data to load")
        dates_index = pd.DatetimeIndex(self.dataset_dates).sort_values()
        months = dates_index.to_period('M')
        unique_months = months.unique()
        nc_files = ['%s/%s%s.nc' % (nc_file_dir, self._file_id, d.strftime('%Y%m')) for d in unique_months]
        self.Dataset = xr.open_mfdataset(nc_files, concat_dim=concat_dim, **dataset_kwargs)
        if exact_dates:
            self.Dataset = self.Dataset.sel(time=self.dataset_dates)
        self.dataset_variables = list(self.Dataset.variables.keys())

    def field(self, variable, time, level):
        """
        Shortcut method to return a 2-D numpy array from the data loaded in an CFSReanalysis.

        :param variable: str: variable to retrieve
        :param time: datetime: requested time
        :param level: int: requested pressure level
        :return: 2-d ndarray
        """
        time_index = self.dataset_dates.index(time)
        level_index = self.level_coord.index(level)
        return self.Dataset.variables[variable][time_index, level_index, ...].values

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
        Generates a Basemap object for graphical plot of CFSR data on a 2-D plane. Bounding box parameters
        are either given, or if None, read from the extremes of the loaded lat/lon data. Other projection parameters
        are set to the default CFSR configuration.

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

    def plot(self, variable, time, level, **plot_basemap_kwargs):
        """
        Wrapper to plot a specified field from an CFSReanalysis object.

        :param variable: str: variable to retrieve
        :param time: datetime: requested time
        :param level: int: requested pressure level
        :param plot_basemap_kwargs: kwargs passed to the plot.plot_functions.plot_basemap function (see the doc for
            plot_basemap for more information on options for Basemap plot)
        :return: matplotlib Figure object
        """
        from ..plot import plot_basemap
        print('CFSReanalysis.plot: plot of %s at %d mb (%s)' % (variable, level, time))
        field = self.field(variable, time, level)
        fig = plot_basemap(self.basemap, self.lon, self.lat, field, **plot_basemap_kwargs)
        return fig


# ==================================================================================================================== #
# CFSReanalysis object class
# ==================================================================================================================== #

class CFSReforecast(object):
    """
    Class for manipulating CFS Reforecast data with xarray. Class methods include functions to download,
    process, and export data. Currently only works with the 6-hourly initialization runs.
    """

    def __init__(self, root_directory=None, fill_hourly=True, file_id=''):
        """
        Initialize an instance of the CFSReanalysis class.

        :param root_directory: str: local directory where raw files are stored. If None, defaults to ~/.cfsr
        :param resolution: str: 'h' corresponds to the high-res 0.5-degree grid; 'l' the low-res 2.5-degree grid
        :param run_type: str: one of the forecast hours or the analysis: ['01', '02', '03', '04', '05', '06', 'nl']
        :param fill_hourly: bool: if True, automatically add in 6-hourly time steps even if only 00Z dates are given
        :param file_id: str: appended to the processed file names. Useful if files for the same dates will be created
            with different parameters, i.e., hours or variables or levels.
        """
        self.raw_files = []
        self.dataset_dates = []
        self.dataset_variables = []
        if root_directory is None:
            self._root_directory = '%s/.cfsr' % os.path.expanduser('~')
        else:
            self._root_directory = root_directory
        self._root_url = 'https://nomads.ncdc.noaa.gov/data/cfsr-rfl-ts45'
        self._nx = 360
        self._ny = 181
        self._dt = 6
        self._variables = ['chi200', 'dswsfc', 'lhtfl', 'prate', 'tmp2m', 'tmpsfc', 'ulwtoa', 'wind200', 'wind850',
                           'z500', 'z700', 'z1000']
        self._fill_hourly = fill_hourly
        self._file_id = file_id
        self.f_hour = []
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

    def set_dates(self, dates):
        """
        Set the CFSReanalysis object's dataset_dates attribute, a list of datetime objects which determines which
        datetimes are retrieved and processed. This attribute is set automatically when using the method 'retrieve',
        but may be used when 'retrieve' is not desired or as an override.

        :param dates: list of datetime objects.
        :return:
        """
        self.dataset_dates = sorted(dates)
        if self._fill_hourly:
            day_set = sorted(set([datetime(d.year, d.month, d.day) for d in self.dataset_dates]))
            new_dates = []
            for day in day_set:
                new_dates.extend((day, day.replace(hour=6), day.replace(hour=12), day.replace(hour=18)))
            self.dataset_dates = new_dates

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
        min_dist = 1.
        if np.min(distance) > min_dist:
            raise ValueError('no latitude/longitude points within 1 degree of requested lat/lon!')
        return np.unravel_index(np.argmin(distance, axis=None), distance.shape)

    def retrieve(self, dates, variables='all', n_proc=4, verbose=False):
        """
        Retrieves CFS reanalysis data for the given datetimes, and writes them to the local directory. The same
        directory structure (%Y/%Y%m/%Y%m%d/file_name) is used locally as on the server. Creates subdirectories if
        necessary. File types retrieved are given by the object's init parameters.

        :param dates: list or tuple: date or datetime objects of of analysis times. May be 'all', in which case
            all dates in the object's 'dataset_dates' attributes are retrieved.
        :param variables: list: list of variables to retrieve or 'all'
        :param n_proc: int: if >1, fetches files in parallel. This speeds up performance but may not scale well if
            internet I/O is the bottleneck. Set to 0 to use all available threads.
        :param verbose: bool: include progress print statements
        :return: None
        """
        # Check if any parameter is a single value
        if dates == 'all':
            dates = self.dataset_dates
        else:
            self.set_dates(dates)
            dates = self.dataset_dates
        if variables == 'all':
            variables = self._variables

        # Determine the files to retrieve
        if verbose:
            print('CFSREforecast.retrieve: beginning data retrieval\n')
        self.raw_files = []
        for var in variables:
            for dt in dates:
                if dt < reforecast_start_date or dt > reforecast_end_date:
                    print('* Warning: doing nothing for date %s, out of valid data range (%s to %s)' %
                          (dt, reforecast_start_date, reforecast_end_date))
                    continue
                if dt not in self.dataset_dates:
                    self.dataset_dates.append(dt)
                    # Create local directory
                grib_file_dir = datetime.strftime(dt, reforecast_dir_format).format(var)
                os.makedirs('%s/%s' % (self._root_directory, grib_file_dir), exist_ok=True)
                # Add GRIB file to listing. There are two dates: init date and end date.
                # The end date depends on run type.
                if dt.hour == 0:
                    # 1st of the month 4 months later
                    end_date = datetime.strftime((dt.replace(day=1) + timedelta(days=130)).replace(day=1), '%Y%m%d%H')
                else:
                    end_date = datetime.strftime(dt + timedelta(days=45), '%Y%m%d%H')
                start_date = datetime.strftime(dt, '%Y%m%d%H')
                grib_file_name = '%s/%s' % (grib_file_dir,
                                            reforecast_file_format.format(var, start_date, end_date, start_date))
                if grib_file_name not in self.raw_files:
                    self.raw_files.append(grib_file_name)

        if n_proc == 0 or n_proc > 1:
            try:
                import multiprocessing
                if n_proc == 0:
                    n_proc = multiprocessing.cpu_count()
            except ImportError:
                warnings.warn("'multiprocessing' module not available; falling back to serial")
                n_proc = 1

        if n_proc == 1:
            for file in self.raw_files:
                call_fetch((self, file, verbose))
        else:
            pool = multiprocessing.Pool(processes=n_proc)
            pool.map(call_fetch, zip(it.repeat(self), self.raw_files, it.repeat(verbose)))
            pool.close()
            pool.terminate()
            pool.join()

    def _fetch(self, f, verbose):
        pid = os.getpid()
        local_file = '%s/%s' % (self._root_directory, f)
        if _check_exists(local_file):
            if verbose:
                print('PID %s: local file %s exists; omitting' % (pid, local_file))
            return
        remote_file = '%s/%s' % (self._root_url, f)
        if verbose:
            print('PID %s: downloading %s' % (pid, remote_file))
        try:
            response = urlopen(remote_file)
            with open(local_file, 'wb') as fd:
                fd.write(response.read())
        except BaseException as e:
            print('warning: failed to download %s, retrying' % remote_file)
            try:
                response = urlopen(remote_file)
                with open(local_file, 'wb') as fd:
                    fd.write(response.read())
            except BaseException as e:
                print('warning: failed to download %s' % remote_file)
                print('* Reason: "%s"' % str(e))

    def write(self, variables='all', dates='all', forecast_hours=1080, interpolate=None, write_into_existing=True,
              omit_existing=False, delete_raw_files=False, n_proc=4, verbose=False):
        """
        Reads raw CFS reanalysis files for the given dates (list or tuple form) and specified variables and levels and
        writes the data to reformatted netCDF files. Processed files are saved under self._root_directory/processed;
        one file per month is created.

        :param variables: list: list of variables to retrieve from data or 'all'
        :param dates: list or tuple of datetime: date or datetime objects of model initialization; may be 'all', in
            which case, all the dates in the object's dataset_dates attribute are used (these are set when calling
            self.retrieve() or self.set_dates())
        :param forecast_hours: int: maximum number of forecast hours to include
        :param interpolate: tuple of (lat, lon) 1-d coordinates: if not None, interpolates from the regular grid to
            a new regular grid. If self.inverse_lat is True, expects monotonically decreasing latitude coordinate,
            otherwise must be monotonically increasing. Longitude must be 0-360.
        :param write_into_existing: bool: if True, checks for existing files and appends if they exist. If False,
            overwrites any existing files.
        :param omit_existing: bool: if True, then if a processed file exists, skip it. Only useful if existing data
            are known to be complete.
        :param delete_raw_files: bool: if True, deletes the original data files from which the processed versions were
            made
        :param n_proc: int: if >1, runs write tasks in parallel, one per month of data. This speeds up performance but
            may not scale well if disk I/O is the bottleneck. Set to 0 to use all available threads.
        :param verbose: bool: include progress print statements
        :return:
        """
        # Parameter checks
        if variables == 'all':
            variables = list(grib2_table[:, 0])
        if dates == 'all':
            dates = self.dataset_dates
        else:
            self.set_dates(dates)
            dates = self.dataset_dates
        if len(variables) == 0:
            print('CFSReanalysis.write: no variables specified; will do nothing.')
            return
        if len(dates) == 0:
            print('CFSReanalysis.write: no dates specified; will do nothing.')
            return
        self.dataset_variables = list(variables)
        if forecast_hours < self._dt:
            raise ValueError('maximum forecast_hours should be at least %d' % self._dt)
        self.f_hour = np.array(np.arange(self._dt, forecast_hours + 1, self._dt), dtype='int')
        if interpolate is not None:
            if len(interpolate) != 2:
                raise ValueError("'interpolate' must be a tuple of length 2")
            if len(interpolate[0].shape) != 1:
                raise ValueError("lat in 'interpolate' must be 1 dimensional")
            if len(interpolate[1].shape) != 1:
                raise ValueError("lon in 'interpolate' must be 1 dimensional")
            self._ny = len(interpolate[0])
            self._nx = len(interpolate[1])

        # Generate monthly batches of dates
        dates_index = pd.DatetimeIndex(dates).sort_values()
        months = dates_index.to_period('M')
        unique_months = months.unique()
        month_list = []
        for m in range(len(unique_months)):
            month_list.append(list(dates_index[months == unique_months[m]].to_pydatetime()))

        if n_proc == 0 or n_proc > 1:
            try:
                import multiprocessing
                if n_proc == 0:
                    n_proc = multiprocessing.cpu_count()
            except ImportError:
                warnings.warn("'multiprocessing' module not available; falling back to serial")
                n_proc = 1

        if n_proc == 1:
            for nm, month in enumerate(month_list):
                call_process_month((self, nm, month, unique_months, variables, interpolate, write_into_existing,
                                    omit_existing, delete_raw_files, verbose))
        else:
            pool = multiprocessing.Pool(processes=n_proc)
            pool.map(call_process_month, zip(it.repeat(self), range(len(month_list)), month_list,
                                             it.repeat(unique_months), it.repeat(variables), it.repeat(interpolate),
                                             it.repeat(write_into_existing), it.repeat(omit_existing),
                                             it.repeat(delete_raw_files), it.repeat(verbose)))
            pool.close()
            pool.terminate()
            pool.join()

    def _process_month(self, m, month, unique_months, variables, interpolate, write_into_existing, omit_existing,
                       delete_raw_files, verbose):
        if interpolate is not None:
            from scipy.interpolate import RectBivariateSpline

        def read_grib_lat_lon(file_name):
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                raise IOError('File %s not found.' % file_name)
            grib_data = pygrib.open(file_name)
            try:
                lats = np.array(grib_data[1]['latitudes'], dtype=np.float32)
                lons = np.array(grib_data[1]['longitudes'], dtype=np.float32)
                shape = grib_data[1].values.shape
                lat = lats.reshape(shape)[:, 0]
                lon = lons.reshape(shape)[0, :]
            except BaseException:
                print('* Warning: cannot get lat/lon from grib file %s' % exists_file_name)
                raise
            grib_data.close()
            return lat, lon

        def read_write_grib_lat_lon(file_name, nc_fid):
            lat, lon = read_grib_lat_lon(file_name)
            if verbose:
                print('PID %s: Writing latitude and longitude' % pid)
            nc_var = nc_fid.createVariable('lat', np.float32, ('lat',))
            nc_var.setncatts({
                'long_name': 'Latitude',
                'units': 'degrees_north'
            })
            nc_fid.variables['lat'][:] = lat
            nc_var = nc_fid.createVariable('lon', np.float32, ('lon',))
            nc_var.setncatts({
                'long_name': 'Longitude',
                'units': 'degrees_east'
            })
            nc_fid.variables['lon'][:] = lon

        def read_write_grib(file_name, time_index, variable):
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                print('* Warning: file %s not found' % file_name)
                return
            if verbose:
                print('PID %s: Reading %s' % (pid, exists_file_name))
            grib_data = pygrib.open(file_name)
            for grb in grib_data:
                if grb.forecastTime > np.max(self.f_hour):
                    break
                try:
                    f_hour_ind = list(self.f_hour).index(int(grb.forecastTime))
                except ValueError:
                    continue
                try:
                    if verbose:
                        print('PID %s: Writing forecast hour %d' % (pid, self.f_hour[f_hour_ind]))
                    data = np.array(grb.values, dtype=np.float32)
                    if interpolate is not None:
                        f_interp = RectBivariateSpline(data_lat, data_lon, data)
                        if self.inverse_lat:
                            variable[f_hour_ind, time_index, ...] = f_interp(interpolate[0][::-1], interpolate[1])
                        else:
                            variable[f_hour_ind, time_index, ...] = f_interp(interpolate[0], interpolate[1])
                    else:
                        variable[f_hour_ind, time_index, ...] = data
                except OSError:  # missing index gives an OS read error
                    print('* Warning: read error')
                    pass
                except BaseException as e:
                    print("* Warning: failed to write to netCDF file ('%s')" % str(e))
            grib_data.close()
            return

        # We're gonna have to do this the ugly way, with the netCDF4 module.
        # Iterate over months, create a netCDF file for the month, and fill in all datetimes we want
        # Create netCDF file, or append
        pid = os.getpid()
        n_fhour = len(self.f_hour)
        nc_file_dir = '%s/processed' % self._root_directory
        os.makedirs(nc_file_dir, exist_ok=True)
        nc_file_name = '%s/%sfcst_%s.nc' % (nc_file_dir, self._file_id, datetime.strftime(month[0], '%Y%m'))
        if verbose:
            print('PID %s: Writing to file %s' % (pid, nc_file_name))
        nc_file_open_type = 'w'
        init_coord = True
        if os.path.isfile(nc_file_name):
            if omit_existing:
                if verbose:
                    print('PID %s: Omitting file %s; exists' % (pid, nc_file_name))
                return
            if write_into_existing:
                nc_file_open_type = 'a'
                init_coord = False
            else:
                os.remove(nc_file_name)
        nc_file_id = nc.Dataset(nc_file_name, nc_file_open_type, format='NETCDF4')

        # Initialize coordinates
        time_axis = pd.DatetimeIndex(start=unique_months[m].start_time, end=unique_months[m].end_time,
                                     freq='6H').to_pydatetime()
        if init_coord:
            # Create dimensions
            if verbose:
                print('PID %s: Creating coordinate dimensions' % pid)
            nc_file_id.description = 'Selected variables and levels from the CFS Reanalysis'
            nc_file_id.createDimension('f_hour', n_fhour)
            nc_file_id.createDimension('time', 0)
            nc_file_id.createDimension('lat', self._ny)
            nc_file_id.createDimension('lon', self._nx)

            # Create forecast hour variable
            nc_var = nc_file_id.createVariable('f_hour', np.int, 'f_hour')
            nc_var.setncatts({
                'long_name': 'Forecast hour'
            })
            nc_file_id.variables['f_hour'][:] = self.f_hour

            # Create unlimited time variable for initialization time
            nc_var = nc_file_id.createVariable('time', np.float32, 'time')
            time_units = 'hours since 1970-01-01 00:00:00'

            nc_var.setncatts({
                'long_name': 'Model initialization time',
                'units': time_units
            })
            nc_file_id.variables['time'][:] = nc.date2num(time_axis, time_units)

        # Now go through the time files to add data to the netCDF file
        for var in variables:
            if var not in nc_file_id.variables.keys():
                if verbose:
                    print('PID %s: Creating variable %s' % (pid, var))
                var_to_write = nc_file_id.createVariable(var, np.float32, ('f_hour', 'time', 'lat', 'lon'), zlib=True)
                var_to_write.setncatts({
                    'long_name': var,
                    'units': 'N/A',
                    '_FillValue': fill_value
                })
            else:
                var_to_write = nc_file_id.variables[var]
            for dt in month:
                # File name
                grib_file_dir = datetime.strftime(dt, reforecast_dir_format).format(var)
                if dt.hour == 0:
                    # 1st of the month 4 months later
                    end_date = datetime.strftime((dt.replace(day=1) + timedelta(days=130)).replace(day=1),
                                                 '%Y%m%d%H')
                else:
                    end_date = datetime.strftime(dt + timedelta(days=45), '%Y%m%d%H')
                start_date = datetime.strftime(dt, '%Y%m%d%H')
                grib_file_name = '%s/%s/%s' % (self._root_directory, grib_file_dir,
                                               reforecast_file_format.format(var, start_date, end_date, start_date))

                # Write the latitude and longitude coordinate arrays, if needed
                if init_coord:
                    if interpolate is None:
                        try:
                            read_write_grib_lat_lon(grib_file_name, nc_file_id)
                            init_coord = False
                        except (IOError, OSError):
                            print("* Warning: file %s not found for coordinates; trying the next one."
                                  % grib_file_name)
                    else:
                        if verbose:
                            print('PID %s: Writing latitude and longitude' % pid)
                        nc_var = nc_file_id.createVariable('lat', np.float32, ('lat',))
                        nc_var.setncatts({
                            'long_name': 'Latitude',
                            'units': 'degrees_north'
                        })
                        nc_file_id.variables['lat'][:] = interpolate[0]
                        nc_var = nc_file_id.createVariable('lon', np.float32, ('lon',))
                        nc_var.setncatts({
                            'long_name': 'Longitude',
                            'units': 'degrees_east'
                        })
                        nc_file_id.variables['lon'][:] = interpolate[1]
                        init_coord = False

                # Get the data lat/lon if we need to interpolate
                if interpolate is not None:
                    try:
                        data_lat, data_lon = read_grib_lat_lon(grib_file_name)
                        if self.inverse_lat:
                            data_lat = data_lat[::-1]
                    except (IOError, OSError):
                        print("* Warning: could not get coordinates from file %s but I need coordinates to "
                              "interpolate. I'm skipping to the next one!"
                              % grib_file_name)
                        continue

                # Write the data
                read_write_grib(grib_file_name, list(time_axis).index(dt), var_to_write)

                # Delete files if requested
                if delete_raw_files:
                    if os.path.isfile(grib_file_name):
                        os.remove(grib_file_name)

        nc_file_id.close()

    def open(self, exact_dates=True, concat_dim='time', **dataset_kwargs):
        """
        Open an xarray multi-file Dataset for the processed files with dates set using set_dates(), retrieve(), or
        write(). Once opened, this Dataset is accessible by self.Dataset.

        :param exact_dates: bool: if True, set the Dataset to have the exact dates of this instance; otherwise,
            keep all of the monthly dates in the opened files
        :param concat_dim: passed to xarray.open_mfdataset()
        :param dataset_kwargs: kwargs passed to xarray.open_mfdataset()
        """
        nc_file_dir = '%s/processed' % self._root_directory
        if not self.dataset_dates:
            raise ValueError("use set_dates() to specify times of data to load")
        dates_index = pd.DatetimeIndex(self.dataset_dates).sort_values()
        months = dates_index.to_period('M')
        unique_months = months.unique()
        nc_files = ['%s/%sfcst_%s.nc' % (nc_file_dir, self._file_id, d.strftime('%Y%m')) for d in unique_months]
        self.Dataset = xr.open_mfdataset(nc_files, concat_dim=concat_dim, **dataset_kwargs)
        if exact_dates:
            self.Dataset = self.Dataset.sel(time=self.dataset_dates)
        self.dataset_variables = list(self.Dataset.variables.keys())

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
