#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for validating DLWP forecasts.
"""

import numpy as np


def forecast_error(forecast, valid, method='mse', axis=None):
    """
    Calculate the error of a time series model forecast.

    :param forecast: ndarray: forecast from a DLWP model (forecast hour is first axis)
    :param valid: ndarray: validation target data for the predictors the forecast was made on
    :param method: str: 'mse' for mean squared error or 'mae' for mean absolute error
    :param axis: int, tuple, or None: take the mean of the error along this axis. Regardless of this setting, the
        forecast hour will be the first dimension.
    :return: ndarray: forecast error with forecast hour as the first dimension
    """
    if method not in ['mse', 'mae']:
        raise ValueError("'method' must be 'mse' or 'mae'")
    n_f = forecast.shape[0]
    n_val = valid.shape[0]
    me = []
    for f in range(n_f):
        if method == 'mse':
            me.append(np.mean((valid[f:] - forecast[f, :(n_val - f)]) ** 2., axis=axis))
        elif method == 'mae':
            me.append(np.mean(np.abs(valid[f:] - forecast[f, :(n_val - f)]), axis=axis))
    return np.array(me)


def persistence_error(predictors, valid, n_fhour, method='mse', axis=None):
    """
    Calculate the error of a persistence forecast out to n_fhour forecast hours.

    :param predictors: ndarray: predictor data
    :param valid: ndarray: validation target data
    :param n_fhour: int: number of steps to take forecast out to
    :param method: 'mse' for mean squared error or 'mae' for mean absolute error
    :param axis: int, tuple, or None: take the mean of the error along this axis. Regardless of this setting, the
        forecast hour will be the first dimension.
    :return: ndarray: persistence error with forecast hour as the first dimension
    """
    if method not in ['mse', 'mae']:
        raise ValueError("'method' must be 'mse' or 'mae'")
    n_f = valid.shape[0]
    me = []
    for f in range(n_fhour):
        if method == 'mse':
            me.append(np.mean((valid[f:] - predictors[:(n_f - f)]) ** 2., axis=axis))
        elif method == 'mae':
            me.append(np.mean(np.abs(valid[f:] - predictors[:(n_f - f)]), axis=axis))
    return np.array(me)


def climo_error(valid, n_fhour, method='mse', axis=None):
    """
    Calculate the error of a climatology forecast out to n_fhour forecast hours.

    :param valid: ndarray: validation target data
    :param n_fhour: int: number of steps to take forecast out to
    :param method: 'mse' for mean squared error or 'mae' for mean absolute error
    :param axis: int, tuple, or None: take the mean of the error along this axis. Regardless of this setting, the
        forecast hour will be the first dimension.
    :return: ndarray: persistence error with forecast hour as the first dimension
    """
    if method not in ['mse', 'mae']:
        raise ValueError("'method' must be 'mse' or 'mae'")
    n_f = valid.shape[0]
    me = []
    for f in range(n_fhour):
        if method == 'mse':
            me.append(np.mean((valid[:(n_f - f)] - np.mean(valid, axis=0)) ** 2., axis=axis))
        elif method == 'mae':
            me.append(np.mean(np.abs(valid[:(n_f - f)] - np.mean(valid, axis=0)), axis=axis))
    return np.array(me)


def predictors_to_time_series(predictors, time_steps, has_time_dim=True, use_first_step=False):
    """
    Reshapes predictors into a continuous time series that can be used for verification methods in this module and
    matches the reshaped output of DLWP models' 'predict_timeseries' method. This is only necessary if the data are for
    a model predicting multiple time steps. Also truncates the first (time_steps - 1) samples so that the time series
    matches the effective forecast initialization time, or the last (time_steps -1) samples if use_first_step == True.

    :param predictors: ndarray: array of predictor data
    :param time_steps: int: number of time steps in the predictor data
    :param has_time_dim: bool: if True, the time step dimension is axis=1 in the predictors, otherwise, axis 1 is
        assumed to be time_steps * num_channels_or_features
    :param use_first_step: bool: if True, keeps the first time step instead of the last (useful for validation
    :return: ndarray: reshaped predictors
    """
    idx = 0 if use_first_step else -1
    if has_time_dim:
        return predictors[:, idx]
    else:
        sample_dim = predictors.shape[0]
        feature_shape = predictors.shape[1:]
        predictors = predictors.reshape((sample_dim, time_steps, -1) + feature_shape[1:])
        return predictors[:, idx]
