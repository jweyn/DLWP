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
import os


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


class Preprocessor(object):

    def __init__(self, data_obj, ):
        self.data = data_obj
        if self.data.Dataset is None:
            print('Preprocessor warning: opening data with default args')
            self.data.open()
        self.predictors = None
        self.targets = None
        self._predictor_shape = ()
        self._target_shape = ()

    def data_to_samples(self, variables='all', levels='all', file_name='.predictors.nc',
                        verbose=False, in_memory=False):
        # Test that data is loaded
        if self.data.Dataset is None:
            raise IOError('no data loaded to data_obj')

        # Convert variables and levels to appropriate type
        vars_available = list(self.data.Dataset.data_vars.keys())
        if variables == 'all':
            variables = [v for v in vars_available]
        elif not(isinstance(variables, list) or isinstance(variables, tuple)):
            variables = [variables]
        if levels == 'all':
            levels = list(self.data.Dataset.level.values)
        elif not(isinstance(levels, list) or isinstance(levels, tuple)):
            levels = [levels]

        # Get the exact dataset we want (index times, variables, and levels)
        all_dates = self.data.dataset_dates
        ds = self.data.Dataset.sel(time=all_dates, level=levels)
        for v in vars_available:
            if v not in variables:
                ds = ds.drop(v)

        # Sort into predictors and targets. If in_memory is false, write to netCDF.
        if not in_memory and os.path.isfile(file_name):
            raise IOError("file '%s' already exists" % file_name)

        return
