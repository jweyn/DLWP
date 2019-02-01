#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Just read some CFS reforecast data and process it into netCDFs. This is not part of the DLWP module. That will be
written to fully download and process reforecast data in the future.
"""

import pygrib
import numpy as np
import netCDF4 as nc
import os
import pandas as pd
from datetime import datetime
from DLWP.data.cfsr import _check_exists

# Universal
verbose = True
_nx = 360
_ny = 181

# netCDF fill value
fill_value = np.array(nc.default_fillvals['f4']).astype(np.float32)


def read_write_grib_lat_lon(file_name):
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
        print('Writing latitude and longitude')
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
    grib_data.close()


def read_write_grib(file_name, time_index):
    exists, exists_file_name = _check_exists(file_name, path=True)
    if not exists:
        print('* Warning: file %s not found' % file_name)
        return
    if verbose:
        print('Loading %s' % exists_file_name)
    if verbose:
        print('  Reading')
    grib_data = pygrib.open(file_name)
    for grb in grib_data:
        try:
            f_hour_ind = list(f_hour).index(int(grb.forecastTime))
        except ValueError:
            continue
        try:
            if verbose:
                print('Writing forecast hour %d' % f_hour[f_hour_ind])
            data = np.array(grb.values, dtype=np.float32)
            z500[f_hour_ind, time_index, ...] = data
        except OSError:  # missing index gives an OS read error
            print('* Warning: read error')
            pass
        except BaseException as e:
            print("* Warning: failed to write to netCDF file ('%s')" % str(e))
    grib_data.close()
    return


# Create netCDF file, or append
root_dir = '/home/disk/vader2/njweber2/cfs_z500'
nc_file_dir = '/home/disk/wave2/jweyn/Data/DLWP'
os.makedirs(nc_file_dir, exist_ok=True)
nc_file_name = '%s/cfs_reforecast_5day_z500_2003-2010.nc' % nc_file_dir
fcst_file_format = '{}/fcst/z500.%Y%m%d%H.time.grb2'.format(root_dir)
anal_file_format = '{}/anl/z500.%Y%m%d%H.time.grb2'.format(root_dir)
if verbose:
    print('Writing to file %s' % nc_file_name)
nc_file_open_type = 'w'
init_coord = True
nc_fid = nc.Dataset(nc_file_name, nc_file_open_type, format='NETCDF4')

# Desired forecast hours (data are 6-hourly)
f_hour = np.arange(0, 6 * 24 + 1, 6, dtype='int')
n_fhour = len(f_hour)

# Desired initialization dates
all_dates = []
for year in range(2003, 2011):
    if year % 4 == 0:
        # Oops we're missing the dates after leap days...
        year_dates = list(pd.date_range('%d-01-01' % year, '%d-02-29' % year, freq='5D').to_pydatetime())
    else:
        year_dates = list(pd.date_range('%d-01-01' % year, '%d-12-31' % year, freq='5D').to_pydatetime())
    all_dates += year_dates

# Check if file exists for each date, otherwise remove them
bad_dates = []
for date in all_dates:
    fcst_file_name = datetime.strftime(date, fcst_file_format)
    if not(os.path.isfile(fcst_file_name)):
        bad_dates.append(date)
for date in bad_dates:
    all_dates.remove(date)


# Initialize coordinates
if init_coord:
    # Create dimensions
    if verbose:
        print('Creating coordinate dimensions')
    nc_fid.description = 'Selected variables and levels from the CFS Reanalysis'
    nc_fid.createDimension('f_hour', n_fhour)
    nc_fid.createDimension('time', 0)
    nc_fid.createDimension('lat', _ny)
    nc_fid.createDimension('lon', _nx)

    # Create forecast hour variable
    nc_var = nc_fid.createVariable('f_hour', np.int, 'f_hour')
    nc_var.setncatts({
        'long_name': 'Forecast hour'
    })
    nc_fid.variables['f_hour'][:] = f_hour

    # Create unlimited time variable for initialization time
    nc_var = nc_fid.createVariable('time', np.float32, 'time')
    time_units = 'hours since 1970-01-01 00:00:00'

    nc_var.setncatts({
        'long_name': 'Model initialization time',
        'units': time_units
    })
    nc_fid.variables['time'][:] = nc.date2num(all_dates, time_units)

# Create the variable we want
z500 = nc_fid.createVariable('Z', np.float32, ('f_hour', 'time', 'lat', 'lon'))
z500.setncatts({
    'long_name': 'Geopotential height',
    'units': 'm ** 2 / s ** 2',
    'level': '500 hPa',
    '_FillValue': fill_value
})

for d, date in enumerate(all_dates):
    fcst_file_name = datetime.strftime(date, fcst_file_format)
    # Write the latitude and longitude coordinate arrays, if needed
    if init_coord:
        try:
            read_write_grib_lat_lon(fcst_file_name)
            init_coord = False
        except (IOError, OSError):
            print("* Warning: file %s not found for coordinates; trying the next one." % fcst_file_name)
    read_write_grib(fcst_file_name, d)

nc_fid.close()
