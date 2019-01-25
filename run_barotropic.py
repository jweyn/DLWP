#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Run a barotropic vorticity model and save the output.
"""

from DLWP.data import CFSReanalysis
from DLWP.barotropic import BarotropicModelPsi
from datetime import datetime
import pandas as pd
import numpy as np
import xarray as xr


start_date = datetime(2007, 1, 1)
end_date = datetime(2010, 12, 31)
dates = list(pd.date_range(start_date, end_date, freq='D').to_pydatetime())
level = 500
baro_dt = 0.5
baro_step_hours = 6
baro_run_hours = 144
output_file = '/home/disk/wave2/jweyn/Data/DLWP/barotropic_2007-2010.nc'

cfs = CFSReanalysis(root_directory='/home/disk/wave2/jweyn/Data/CFSR', file_id='dlwp_')
cfs.set_dates(dates)
cfs.open()

result = np.full((baro_run_hours // baro_step_hours + 1, cfs.Dataset.dims['time'],
                  cfs.Dataset.dims['lat'], cfs.Dataset.dims['lon']), np.nan)

init_count = 0
for init_time in cfs.Dataset.time.values:
    # Create a model
    print('Initializing barotropic model at %s' % init_time)
    baro = BarotropicModelPsi(cfs.Dataset.sel(time=init_time, level=level).variables['HGT'].values,
                              72, baro_dt * 3600., pd.Timestamp(init_time).to_pydatetime())

    print('Integrating')
    out_count = 0
    for step in np.arange(0, baro_run_hours + baro_dt, baro_dt):
        if step % baro_step_hours == 0:
            result[out_count, init_count] = baro.z_grid[:]
            out_count += 1
        baro.step_forward()

    init_count += 1

result_ds = xr.Dataset({
    'Z': (['f_hour', 'time', 'lat', 'lon'], result, {
        'long_name': 'Geopotential height',
        'units': 'm',
        'level': level
    })
}, coords={
    'f_hour': ('f_hour', np.arange(0, baro_run_hours + 1, baro_step_hours)),
    'time': ('time', cfs.Dataset.time.values, {
        'long_name': 'Initialization time',
    }),
    'lat': ('lat', cfs.Dataset['lat'].values, {
        'long_name': 'Latitude',
        'units': 'degrees_north'
    }),
    'lon': ('lon', cfs.Dataset['lon'].values, {
        'long_name': 'Longitude',
        'units': 'degrees_east'
    }),
}, attrs={
    'description': 'Barotropic model prediction from CFS reanalysis'
})

result_ds.to_netcdf(output_file)
