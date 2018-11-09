#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
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
        self.predictor_shape = ()
        self.n_features = None

    def data_to_samples(self, batch_samples=100, variables='all', levels='all', scale_variables=False,
                        in_memory=False, overwrite=False, verbose=False):
        """
        Convert the data referenced by the data_obj in __init__ to samples ready for ingestion in a DLWP model. Write
        samples in batches of size batch_samples. The parameter scale_variables determines whether individual
        variable/level combinations are scaled and de-meaned by their spatially-averaged values.

        :param variables: iter: list of variables to process; may be 'all' for all variables available
        :param levels: iter: list of integer pressure levels (mb); may be 'all'
        :param scale_variables: bool: if True, apply de-mean and scaling on a variable/level basis
        :param in_memory: bool: if True, speeds up operations by performing them in memory (may require lots of RAM)
        :param verbose: bool: print progress statements
        :return: opens Dataset on self.data
        """
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

        # Get the exact dataset we want (index times, variables, and levels)
        all_dates = self.raw_data.dataset_dates
        ds = self.raw_data.Dataset.sel(time=all_dates, level=levels)
        if verbose:
            print('Preprocessor.data_to_samples: opening and formatting raw data')
        for v in vars_available:
            if v not in variables:
                ds = ds.drop(v)
        n_sample, n_var, n_level, n_lat, n_lon = (len(all_dates) - 1, len(variables), len(levels),
                                                  ds.dims['lat'], ds.dims['lon'])
        self.predictor_shape = (n_var, n_level, n_lat, n_lon)
        self.n_features = np.prod(self.predictor_shape)

        # Sort into predictors and targets. If in_memory is false, write to netCDF.
        if not in_memory:
            if os.path.isfile(self._predictor_file) and not overwrite:
                raise IOError("predictor file '%s' already exists" % self._predictor_file)
            if verbose:
                print('Preprocessor.data_to_samples: creating output file %s' %self._predictor_file)
            nc_fid = nc.Dataset(self._predictor_file, 'w')
            nc_fid.description = 'Training data for DLWP'
            nc_fid.setncattr('scaling', 'True' if scale_variables else 'False')
            nc_fid.createDimension('sample', 0)
            nc_fid.createDimension('variable', n_var)
            nc_fid.createDimension('level', n_level)
            nc_fid.createDimension('lat', n_lat)
            nc_fid.createDimension('lon', n_lon)

            # Create spatial coordinates
            nc_var = nc_fid.createVariable('level', np.float32, 'level')
            nc_var.setncatts({
                'long_name': 'Pressure level',
                'units': 'hPa'
            })
            nc_fid.variables['level'][:] = levels

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

            # Create predictors and targets variables
            predictors = nc_fid.createVariable('predictors', np.float32, ('sample', 'variable', 'level', 'lat', 'lon'))
            predictors.setncatts({
                'long_name': 'Predictors',
                'units': 'N/A',
                '_FillValue': fill_value
            })
            targets = nc_fid.createVariable('targets', np.float32, ('sample', 'variable', 'level', 'lat', 'lon'))
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
            predictors = np.full((n_sample, n_var, n_level, n_lat, n_lon), np.nan, dtype=np.float32)
            targets = predictors.copy()

        # Fill in the data. Each point gets filled with the target index 1 higher. Iterate by variable and level for
        # scaling.
        for v, var in enumerate(variables):
            for l, lev in enumerate(levels):
                if verbose:
                    print('Preprocessor.data_to_samples: variable %s of %s; level %s of %s' %
                          (v+1, len(variables), l+1, len(levels)))
                if scale_variables:
                    if verbose:
                        print('Preprocessor.data_to_samples: calculating mean and std')
                    v_mean = mean_by_batch(ds[var].isel(level=l), batch_samples)
                    v_std = std_by_batch(ds[var].isel(level=l), batch_samples, mean=v_mean)
                else:
                    v_mean = 0.0
                    v_std = 1.0
                for s in range(0, n_sample, batch_samples):
                    idx = slice(s, min(s+batch_samples, n_sample))
                    idxp1 = slice(s+1, min(s+1+batch_samples, n_sample+1))
                    if verbose:
                        print('Preprocessor.data_to_samples: writing batch %s of %s' % (s+1, n_sample/batch_samples+1))
                    predictors[idx, v, l, ...] = (ds[var].isel(time=idx, level=l).values - v_mean) / v_std
                    targets[idx, v, l, ...] = (ds[var].isel(time=idxp1, level=l).values - v_mean) / v_std

        if not in_memory:
            nc_fid.close()
            result_ds = xr.open_dataset(self._predictor_file)
        else:
            result_ds = xr.Dataset({
                'predictors': (['sample', 'variable', 'level', 'lat', 'lon'], predictors, {
                    'long_name': 'Predictors',
                    'units': 'N/A'
                }),
                'targets': (['sample', 'variable', 'level', 'lat', 'lon'], targets, {
                    'long_name': 'Targets',
                    'units': 'N/A'
                }),
            }, coords={
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
                'scaling': 'True' if scale_variables else 'False'
            })

        self.data = result_ds

    def open(self, **kwargs):
        """
        Open the dataset pointed to by the instance's _predictor_file attribute onto self.data

        :param kwargs: passed to xarray.open_dataset()
        """
        self.data = xr.open_dataset(self._predictor_file, **kwargs)
        self.predictor_shape = self.data.predictors.shape[1:]
        self.n_features = np.prod(self.predictor_shape)

    def close(self):
        """
        Close the dataset on self.data
        """
        self.data.close()
        self.data = None
        self.predictor_shape = ()
        self.n_features = None

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
        self.data.to_netcdf(predictor_file)


def delete_nan_samples(predictors, targets, large_fill_value=False, threshold=None):
    """
    Delete any samples from the predictor and target numpy arrays and return new, reduced versions.

    :param predictors: ndarray, shape [num_samples,...]: predictor data
    :param targets: ndarray, shape [num_samples,...]: target data
    :param large_fill_value: bool: if True, treats very large values (> 1e30) as NaNs
    :param threshold: float 0-1: if not None, then removes any samples with a fraction of NaN larger than this
    :return: predictors, targets: ndarrays with samples removed
    """
    if threshold is not None and not (0 <= threshold <= 1):
        raise ValueError("'threshold' must be between 0 and 1")
    if large_fill_value:
        predictors[(predictors > 1.e30) | (predictors < -1.e30)] = np.nan
        targets[(targets > 1.e30) | (targets < -1.e30)] = np.nan
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

