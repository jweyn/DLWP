#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Add geopotential thickness as a variable to a predictor file.
"""

import xarray as xr
from DLWP.model.preprocessing import mean_by_batch, std_by_batch


root_directory = '/home/disk/wave2/jweyn/Data'
predictor_file = '%s/DLWP/cfs_6h_1979-2010_z500-1000_tau300-700_sfc_NH.nc' % root_directory
new_file = '%s/DLWP/cfs_6h_1979-2010_z500-1000_tau_sfc_NH.nc' % root_directory

upper_sel = {'varlev': ['HGT/500']}
lower_sel = {'varlev': ['HGT/1000']}
keep_sel = {}
new_var_coord = 'THICK/500-1000'


ds = xr.open_dataset(predictor_file)

# Predictor variable

print('Calculating predictor thickness...')

upper_mean = ds['mean'].sel(**upper_sel)
upper_std = ds['std'].sel(**upper_sel)
upper_var = ds.predictors.sel(**upper_sel) * upper_std + upper_mean
upper_var = upper_var.assign_coords(varlev=[new_var_coord])

lower_mean = ds['mean'].sel(**lower_sel)
lower_std = ds['std'].sel(**lower_sel)
lower_var = ds.predictors.sel(**lower_sel) * lower_std + lower_mean
lower_var = lower_var.assign_coords(varlev=[new_var_coord])

p_thick = upper_var - lower_var

if ds.attrs['scaling'] == 'True':
    thick_mean = mean_by_batch(p_thick, 1000, axis=0)
    thick_std = std_by_batch(p_thick, 1000, axis=0, mean=thick_mean)
    p_thick = (p_thick - thick_mean) / thick_std
else:
    thick_mean = 1.
    thick_std = 1.

ds_thick = xr.Dataset({
    'predictors': p_thick,
    'mean': (['varlev'], [thick_mean], {
        'long_name': 'Global mean of variables at levels',
        'units': 'N/A',
    }),
    'std': (['varlev'], [thick_std], {
        'long_name': 'Global std deviation of variables at levels',
        'units': 'N/A',
    })
}, attrs=ds.attrs)

# Targets

if hasattr(ds, 'targets'):
    print('Calculating target thickness...')

    upper_var = ds.targets.sel(**upper_sel) * upper_std + upper_mean
    lower_var = ds.targets.sel(**lower_sel) * lower_std + lower_mean
    upper_var = upper_var.assign_coords(varlev=[new_var_coord])
    lower_var = lower_var.assign_coords(varlev=[new_var_coord])
    t_thick = upper_var - lower_var

    if ds.attrs['scaling'] == 'True':
        t_thick = (t_thick - thick_mean) / thick_std

    ds_thick['targets'] = t_thick


# Merge the datasets

print('Concatenating datasets...')
ds_new = xr.concat([ds.sel(**keep_sel), ds_thick], dim='varlev')


# Write to output

print('Writing new file (%s)...' % new_file)
ds_new.to_netcdf(new_file)
