#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Tools for pre-processing model input data into training/validation/testing data.
"""

import numpy as np
import netCDF4 as nc
import xarray as xr
import os
import random
import warnings
from datetime import datetime

# netCDF fill value
fill_value = np.array(nc.default_fillvals['f4']).astype(np.float32)


class Preprocessor(object):

    def __init__(self, data_obj, predictor_file='.predictors.nc'):
        """
        Initialize an instance of Preprocessor for DLWP modelling. The data_obj is an instance of one of the data
        processing classes in DLWP.data, and should have data already loaded.

        :param data_obj: instance of DLWP.data class
        :param predictor_file: str: file to which to write the predictors and targets
        """
        self.raw_data = data_obj
        if self.raw_data is None:
            print('Preprocessor warning: no raw data object provided; acting as wrapper for processed data')
        else:
            if self.raw_data.Dataset is None:
                print('Preprocessor warning: opening data with default args')
                self.raw_data.open()
        self._predictor_file = predictor_file
        self.data = None

    @property
    def spatial_shape(self):
        """
        :return: the shape of the spatial component of ensemble predictors
        """
        return self.data.predictors.shape[1:]

    @property
    def n_features(self):
        """
        :return: int: the number of features in the predictor array
        """
        return int(np.prod(self.spatial_shape))

    @property
    def convolution_shape(self):
        """
        :return: the shape of the predictors expected by a convolutional layer. Note it is channels_first!
        """
        return (int(np.prod(self.data.predictors.shape[1:-2])),) + self.data.predictors.shape[-2:]

    def data_to_samples(self, time_step=1, batch_samples=100, variables='all', levels='all',
                        pairwise=False, scale_variables=False, chunk_size=32, in_memory=False, to_zarr=False,
                        overwrite=False, verbose=False):
        """
        Convert the data referenced by the data_obj in __init__ to samples ready for ingestion in a DLWP model. Write
        samples in batches of size batch_samples. The parameter scale_variables determines whether individual
        variable/level combinations are scaled and de-meaned by their spatially-averaged values.

        :param time_step: int: the number of time steps to take for the predictors and targets
        :param batch_samples: int: number of samples in the time dimension to read and process at once
        :param variables: iter: list of variables to process; may be 'all' for all variables available
        :param levels: iter: list of integer pressure levels (mb); may be 'all'
        :param pairwise: bool: if True, creates a Dataset with one less dimension and creates a variable at each
            variable-level pairing specified here. The lists of variables and levels must be the same length.
        :param scale_variables: bool: if True, apply de-mean and scaling on a variable/level basis
        :param chunk_size: int: size of the chunks in the sample (time) dimension)
        :param in_memory: bool: if True, speeds up operations by performing them in memory (may require lots of RAM)
        :param to_zarr: bool: if True, writes the resulting data structure to a zarr group in addition to the netCDF
            file. Zarr groups use efficient compression and may be significantly faster in training than netCDF files,
            and can be read just like netCDF with xarray.
        :param overwrite: bool: if True, overwrites any existing output files, otherwise, raises an error
        :param verbose: bool: print progress statements
        :return: opens Dataset on self.data
        """
        # Check time step parameter
        if int(time_step) < 1:
            raise ValueError("'time_step' must be >= 1")
        # Check chunk size
        if int(chunk_size) < 1:
            raise ValueError("'chunk_size' must be >= 1")
        # Test that data is loaded
        if self.raw_data is None:
            raise ValueError('cannot process when no data_obj was supplied at initialization')
        if self.raw_data.Dataset is None:
            raise IOError('no data loaded to data_obj')

        # Convert variables and levels to appropriate type
        vars_available = list(self.raw_data.Dataset.data_vars.keys())
        if variables == 'all':
            variables = [v for v in vars_available]
        elif not(isinstance(variables, list) or isinstance(variables, tuple)):
            variables = [variables]
        if levels == 'all':
            levels = list(self.raw_data.Dataset.level.values)
        elif not(isinstance(levels, list) or isinstance(levels, tuple)):
            levels = [levels]
        # If they're pairwise, make sure we have the same length
        if pairwise:
            if len(variables) != len(levels):
                raise ValueError('for pairwise variable/level pairs, len(variables) must equal len(levels)')
            var_lev = ['/'.join([v, str(l)]) for v, l in zip(variables, levels)]

        # Check for variables that have no level coordinate, and enforce pairwise if necessary
        var_no_lev = []
        for v in variables:
            if 'level' not in self.raw_data.Dataset[v].coords:
                var_no_lev.append(v)
        if not pairwise and len(var_no_lev) > 0:
            warnings.warn("Some variables (%s) are not on pressure levels. I'm switching to pairwise mode."
                          % var_no_lev)
            pair_var = [v for v in variables if v not in var_no_lev] * len(levels)
            new_levels = []
            for l in levels:
                new_levels = new_levels + [l] * (len(variables) - len(var_no_lev))
            variables = pair_var + var_no_lev
            levels = new_levels + [0] * len(var_no_lev)
            pairwise = True

        # Get the exact dataset we want (index times, variables, and levels)
        all_dates = self.raw_data.dataset_dates
        ds = self.raw_data.Dataset.sel(time=all_dates, level=list(set(levels)))
        if verbose:
            print('Preprocessor.data_to_samples: opening and formatting raw data')
        for v in vars_available:
            if v not in variables:
                ds = ds.drop(v)
        n_sample, n_var, n_level, n_lat, n_lon = (len(all_dates) - (2 * time_step - 1), len(variables), len(levels),
                                                  ds.dims['lat'], ds.dims['lon'])
        if n_sample < 1:
            raise ValueError('too many time steps for time dimension')

        # Arrays for scaling parameters
        if pairwise:
            means = np.zeros((n_var,), dtype=np.float32)
            stds = np.ones((n_var,), dtype=np.float32)
        else:
            means = np.zeros((n_var, n_level), dtype=np.float32)
            stds = np.ones((n_var, n_level), dtype=np.float32)

        # Sort into predictors and targets. If in_memory is false, write to netCDF.
        if not in_memory:
            if os.path.isfile(self._predictor_file) and not overwrite:
                raise IOError("predictor file '%s' already exists" % self._predictor_file)
            if verbose:
                print('Preprocessor.data_to_samples: creating output file %s' % self._predictor_file)
            nc_fid = nc.Dataset(self._predictor_file, 'w')
            nc_fid.description = 'Training data for DLWP'
            nc_fid.setncattr('scaling', 'True' if scale_variables else 'False')
            nc_fid.createDimension('sample', 0)
            nc_fid.createDimension('time_step', time_step)
            if pairwise:
                nc_fid.createDimension('varlev', n_var)
            else:
                nc_fid.createDimension('variable', n_var)
                nc_fid.createDimension('level', n_level)
            nc_fid.createDimension('lat', n_lat)
            nc_fid.createDimension('lon', n_lon)

            # Create spatial coordinates
            nc_var = nc_fid.createVariable('lat', np.float32, 'lat')
            nc_var.setncatts({
                'long_name': 'Latitude',
                'units': 'degrees_north'
            })
            nc_fid.variables['lat'][:] = ds['lat'].values

            nc_var = nc_fid.createVariable('lon', np.float32, 'lon')
            nc_var.setncatts({
                'long_name': 'Longitude',
                'units': 'degrees_east'
            })
            nc_fid.variables['lon'][:] = ds['lon'].values

            if pairwise:
                nc_var = nc_fid.createVariable('varlev', str, 'varlev')
                nc_var.setncatts({
                    'long_name': 'Variable/level pair',
                })
                nc_fid.variables['varlev'][:] = np.array(var_lev, dtype='object')
            else:
                nc_var = nc_fid.createVariable('variable', str, 'variable')
                nc_var.setncatts({
                    'long_name': 'Variable name',
                })
                nc_fid.variables['variable'][:] = np.array(variables, dtype='object')

                nc_var = nc_fid.createVariable('level', np.float32, 'level')
                nc_var.setncatts({
                    'long_name': 'Pressure level',
                    'units': 'hPa'
                })
                nc_fid.variables['level'][:] = levels

            # Create initialization time reference variable
            nc_var = nc_fid.createVariable('sample', np.float32, 'sample')
            time_units = 'hours since 1970-01-01 00:00:00'

            nc_var.setncatts({
                'long_name': 'Sample start time',
                'units': time_units
            })
            times = np.array([datetime.utcfromtimestamp(d/1e9)
                              for d in ds['time'].values[time_step-1:n_sample+time_step-1].astype(datetime)])
            nc_fid.variables['sample'][:] = nc.date2num(times, time_units)

            # Create predictors and targets variables
            if pairwise:
                dims = ('sample', 'time_step', 'varlev', 'lat', 'lon')
                chunks = (chunk_size, time_step, n_var, n_lat, n_lon)
            else:
                dims = ('sample', 'time_step', 'variable', 'level', 'lat', 'lon')
                chunks = (chunk_size, time_step, n_var, n_level, n_lat, n_lon)
            predictors = nc_fid.createVariable('predictors', np.float32, dims, chunksizes=chunks, zlib=True)
            predictors.setncatts({
                'long_name': 'Predictors',
                'units': 'N/A',
                '_FillValue': fill_value
            })
            targets = nc_fid.createVariable('targets', np.float32, dims, chunksizes=chunks, zlib=True)
            targets.setncatts({
                'long_name': 'Targets',
                'units': 'N/A',
                '_FillValue': fill_value
            })

        else:
            # Load all the data for speed... better be careful
            if verbose:
                print('Preprocessor.data_to_samples: loading data to memory')
            ds.load()
            if pairwise:
                predictors = np.full((n_sample, time_step, n_var, n_lat, n_lon), np.nan, dtype=np.float32)
            else:
                predictors = np.full((n_sample, time_step, n_var, n_level, n_lat, n_lon), np.nan, dtype=np.float32)
            targets = predictors.copy()

        # Fill in the data. Go through time steps. Iterate by variable and level for scaling.
        if pairwise:
            for vl, vl_name in enumerate(var_lev):
                sel_kw = {} if (variables[vl] in var_no_lev) else {'level': levels[vl]}
                if verbose:
                    print('Preprocessor.data_to_samples: variable/level pair %s of %s (%s)' %
                          (vl + 1, len(var_lev), vl_name))
                if scale_variables:
                    if verbose:
                        print('Preprocessor.data_to_samples: calculating mean and std')
                    v_mean = mean_by_batch(ds[variables[vl]].sel(**sel_kw), batch_samples)
                    v_std = std_by_batch(ds[variables[vl]].sel(**sel_kw), batch_samples, mean=v_mean)
                    means[vl] = 1. * v_mean
                    stds[vl] = 1. * v_std
                else:
                    v_mean = 0.0
                    v_std = 1.0
                for i, s in enumerate(list(range(0, n_sample, batch_samples))):
                    if verbose:
                        print('Preprocessor.data_to_samples: writing batch %s of %s'
                              % (i + 1, n_sample // batch_samples + 1))
                    idx = slice(s, min(s + batch_samples, n_sample))
                    for t in range(time_step):
                        idxp = slice(s + t, min(s + t + batch_samples, n_sample + t))
                        idxt = slice(s + t + time_step,
                                     min(s + t + time_step + batch_samples, n_sample + t + time_step))
                        predictors[idx, t, vl, ...] = (ds[variables[vl]].isel(time=idxp).sel(**sel_kw).values
                                                       - v_mean) / v_std
                        targets[idx, t, vl, ...] = (ds[variables[vl]].isel(time=idxt).sel(**sel_kw).values
                                                    - v_mean) / v_std
        else:
            for v, var in enumerate(variables):
                for l, lev in enumerate(levels):
                    if verbose:
                        print('Preprocessor.data_to_samples: variable %s of %s (%s); level %s of %s (%s)' %
                              (v+1, len(variables), var, l+1, len(levels), lev))
                    if scale_variables:
                        if verbose:
                            print('Preprocessor.data_to_samples: calculating mean and std')
                        v_mean = mean_by_batch(ds[var].sel(level=lev), batch_samples)
                        v_std = std_by_batch(ds[var].sel(level=lev), batch_samples, mean=v_mean)
                        means[v, l] = 1. * v_mean
                        stds[v, l] = 1. * v_std
                    else:
                        v_mean = 0.0
                        v_std = 1.0
                    for i, s in enumerate(list(range(0, n_sample, batch_samples))):
                        if verbose:
                            print('Preprocessor.data_to_samples: writing batch %s of %s'
                                  % (i+1, n_sample//batch_samples+1))
                        idx = slice(s, min(s+batch_samples, n_sample))
                        for t in range(time_step):
                            idxp = slice(s+t, min(s+t+batch_samples, n_sample+t))
                            idxt = slice(s+t+time_step, min(s+t+time_step+batch_samples, n_sample+t+time_step))
                            predictors[idx, t, v, l, ...] = (ds[var].isel(time=idxp, level=l).values - v_mean) / v_std
                            targets[idx, t, v, l, ...] = (ds[var].isel(time=idxt, level=l).values - v_mean) / v_std

        if not in_memory:
            # Create means and stds variables
            if pairwise:
                nc_var = nc_fid.createVariable('mean', np.float32, ('varlev',))
                nc_var.setncatts({
                    'long_name': 'Global mean of variables at levels',
                    'units': 'N/A',
                })
                nc_var[:] = means

                nc_var = nc_fid.createVariable('std', np.float32, ('varlev',))
                nc_var.setncatts({
                    'long_name': 'Global std deviation of variables at levels',
                    'units': 'N/A',
                })
                nc_var[:] = stds
            else:
                nc_var = nc_fid.createVariable('mean', np.float32, ('variable', 'level'))
                nc_var.setncatts({
                    'long_name': 'Global mean of variables at levels',
                    'units': 'N/A',
                })
                nc_var[:] = means

                nc_var = nc_fid.createVariable('std', np.float32, ('variable', 'level'))
                nc_var.setncatts({
                    'long_name': 'Global std deviation of variables at levels',
                    'units': 'N/A',
                })
                nc_var[:] = stds

            # Close and re-open as xarray Dataset
            nc_fid.close()
            result_ds = xr.open_dataset(self._predictor_file)
        else:
            if pairwise:
                result_ds = xr.Dataset({
                    'predictors': (['sample', 'time_step', 'varlev', 'lat', 'lon'], predictors, {
                        'long_name': 'Predictors',
                        'units': 'N/A'
                    }),
                    'targets': (['sample', 'time_step', 'varlev', 'lat', 'lon'], targets, {
                        'long_name': 'Targets',
                        'units': 'N/A'
                    }),
                    'mean': (['varlev'], means, {
                        'long_name': 'Global mean of variables at levels',
                        'units': 'N/A',
                    }),
                    'std': (['varlev'], stds, {
                        'long_name': 'Global std deviation of variables at levels',
                        'units': 'N/A',
                    })
                }, coords={
                    'sample': ('sample', ds['time'].values[:n_sample], {
                        'long_name': 'Sample start time'
                    }),
                    'varlev': ('varlev', var_lev),
                    'lat': ('lat', ds['lat'].values, {
                        'long_name': 'Latitude',
                        'units': 'degrees_north'
                    }),
                    'lon': ('lon', ds['lon'].values, {
                        'long_name': 'Longitude',
                        'units': 'degrees_east'
                    }),
                }, attrs={
                    'description': 'Training data for DLWP',
                    'scaling': 'True' if scale_variables else 'False',
                    'pairwise': 'True'
                })
            else:
                result_ds = xr.Dataset({
                    'predictors': (['sample', 'time_step', 'variable', 'level', 'lat', 'lon'], predictors, {
                        'long_name': 'Predictors',
                        'units': 'N/A'
                    }),
                    'targets': (['sample', 'time_step', 'variable', 'level', 'lat', 'lon'], targets, {
                        'long_name': 'Targets',
                        'units': 'N/A'
                    }),
                    'mean': (['variable', 'level'], means, {
                        'long_name': 'Global mean of variables at levels',
                        'units': 'N/A',
                    }),
                    'std': (['variable', 'level'], stds, {
                        'long_name': 'Global std deviation of variables at levels',
                        'units': 'N/A',
                    })
                }, coords={
                    'sample': ('sample', ds['time'].values[:n_sample], {
                        'long_name': 'Sample start time'
                    }),
                    'variable': ('variable', variables),
                    'level': ('level', levels, {
                        'long_name': 'Pressure level',
                        'units': 'hPa'
                    }),
                    'lat': ('lat', ds['lat'].values, {
                        'long_name': 'Latitude',
                        'units': 'degrees_north'
                    }),
                    'lon': ('lon', ds['lon'].values, {
                        'long_name': 'Longitude',
                        'units': 'degrees_east'
                    }),
                }, attrs={
                    'description': 'Training data for DLWP',
                    'scaling': 'True' if scale_variables else 'False',
                    'pairwise': 'False'
                })

        result_ds = result_ds.chunk({'sample': chunk_size})

        if to_zarr:
            zarr_file = '.'.join(self._predictor_file.split('.')[:-1]) + '.zarr'
            if verbose:
                print('Preprocessor.data_to_samples: writing to zarr group %s...' % zarr_file)
            try:
                result_ds.to_zarr(zarr_file, mode='w' if overwrite else 'w-')
                success = True
            except AttributeError:
                warnings.warn("xarray version must be >= 0.12.0 (got %s) to export to zarr; falling back to netCDF"
                              % xr.__version__)
                success = False
            except ValueError:
                raise IOError('zarr group path %s exists' % zarr_file)
            if success:
                self._predictor_file = zarr_file
                result_ds = xr.open_zarr(zarr_file)

        self.data = result_ds

    def open(self, **kwargs):
        """
        Open the dataset pointed to by the instance's _predictor_file attribute onto self.data

        :param kwargs: passed to xarray.open_dataset() or xarray.open_zarr()
        """
        if self._predictor_file.endswith('.zarr'):
            self.data = xr.open_zarr(self._predictor_file, **kwargs)
        else:
            self.data = xr.open_dataset(self._predictor_file, **kwargs)

    def close(self):
        """
        Close the dataset on self.data
        """
        self.data.close()
        self.data = None

    def to_file(self, predictor_file=None):
        """
        Write the data opened on self.data to the file predictor_file if not None or the instance's _predictor_file
        attribute.

        :param predictor_file: str: file path; if None, uses self._predictor_file
        """
        if self.data is None:
            raise ValueError('cannot save to file with no sample data generated or opened')
        if predictor_file is None:
            predictor_file = self._predictor_file
        if predictor_file.endswith('.zarr'):
            self.data.to_zarr(predictor_file)
        else:
            self.data.to_netcdf(predictor_file)


def delete_nan_samples(predictors, targets, large_fill_value=False, threshold=None):
    """
    Delete any samples from the predictor and target numpy arrays and return new, reduced versions.

    :param predictors: ndarray, shape [num_samples,...]: predictor data
    :param targets: ndarray, shape [num_samples,...]: target data
    :param large_fill_value: bool: if True, treats very large values (>= 1e20) as NaNs
    :param threshold: float 0-1: if not None, then removes any samples with a fraction of NaN larger than this
    :return: predictors, targets: ndarrays with samples removed
    """
    if threshold is not None and not (0 <= threshold <= 1):
        raise ValueError("'threshold' must be between 0 and 1")
    if large_fill_value:
        predictors[(predictors >= 1.e20) | (predictors <= -1.e20)] = np.nan
        targets[(targets >= 1.e20) | (targets <= -1.e20)] = np.nan
    p_shape = predictors.shape
    t_shape = targets.shape
    predictors = predictors.reshape((p_shape[0], -1))
    targets = targets.reshape((t_shape[0], -1))
    if threshold is None:
        p_ind = list(np.where(np.isnan(predictors))[0])
        t_ind = list(np.where(np.isnan(targets))[0])
    else:
        p_ind = list(np.where(np.mean(np.isnan(predictors), axis=1) >= threshold)[0])
        t_ind = list(np.where(np.mean(np.isnan(targets), axis=1) >= threshold)[0])
    bad_ind = list(set(p_ind + t_ind))
    predictors = np.delete(predictors, bad_ind, axis=0)
    targets = np.delete(targets, bad_ind, axis=0)
    new_p_shape = (predictors.shape[0],) + p_shape[1:]
    new_t_shape = (targets.shape[0],) + t_shape[1:]
    return predictors.reshape(new_p_shape), targets.reshape(new_t_shape)


def train_test_split_ind(n_sample, test_size, method='random'):
    """
    Return indices splitting n_samples into train and test index lists.

    :param n_sample: int: number of samples
    :param test_size: int: number of samples in test set
    :param method: str: 'first' ('last') to take first (last) t samples as test, or 'random'
    :return: (list, list): list of train indices, list of test indices
    """
    if method == 'first':
        test_set = list(range(0, test_size))
        train_set = list(range(test_size, n_sample))
    elif method == 'last':
        test_set = list(range(n_sample - test_size, n_sample))
        train_set = list(range(0, n_sample - test_size))
    elif method == 'random':
        train_set = list(range(n_sample))
        test_set = []
        for j in range(test_size):
            i = random.choice(train_set)
            test_set.append(i)
            train_set.remove(i)
        test_set.sort()
    else:
        raise ValueError("'method' must be 'first', 'last', or 'random'")

    return train_set, test_set


def mean_by_batch(da, batch_size, axis=0):
    """
    Loop over batches indexed in axis in an xarray DataArray to take the grand mean of the array in a memory-
    efficient way.

    :param da: xarray DataArray
    :param batch_size: int: number of samples to load and mean at a time
    :param axis: int: axis along which to index batches
    :return: float: the mean of the array
    """
    size = da.shape[axis]
    batches = list(range(0, size, batch_size))
    dim = da.dims[axis]
    total = 0.0
    for b in batches:
        total += da.isel(**{dim: slice(b, min(b+batch_size, size))}).values.sum()
    return total / da.size


def std_by_batch(da, batch_size, axis=0, mean=None):
    """
    Loop over batches indexed in axis in an xarray DataArray to take the standard deviation of the array in a memory-
    efficient way. If mean is provided, assumes the mean of the data is already known to be this value.

    :param da: xarray DataArray
    :param batch_size: int: number of samples to load and mean at a time
    :param axis: int: axis along which to index batches
    :param mean: float: the (known) mean of the array
    :return: float: the standard deviation of the array
    """
    if mean is None:
        mean = mean_by_batch(da, batch_size, axis)
    size = da.shape[axis]
    batches = list(range(0, size, batch_size))
    dim = da.dims[axis]
    total = 0.0
    for b in batches:
        total += np.sum((da.isel(**{dim: slice(b, min(b + batch_size, size))}).values - mean) ** 2.)
    return np.sqrt(total / da.size)
