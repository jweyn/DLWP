#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
High-level APIs for building data generators. These produce batches of data on-the-fly for DLWP models'
fit_generator() methods.
"""
import numpy as np
import xarray as xr
from keras.utils import Sequence

from ..util import delete_nan_samples


class DataGenerator(Sequence):
    """
    Class used to generate training data on the fly from a loaded DataSet of predictor data. Depends on the structure
    of the EnsembleSelector to do scaling and imputing of data.
    """

    def __init__(self, model, ds, batch_size=32, shuffle=False, remove_nan=True):
        """
        Initialize a DataGenerator.

        :param model: instance of a DLWP model
        :param ds: xarray Dataset: predictor dataset. Should have attributes 'predictors' and 'targets'
        :param batch_size: int: number of samples to take at a time from the dataset
        :param shuffle: bool: if True, randomly select batches
        :param remove_nan: bool: if True, remove any samples with NaNs
        """
        self.model = model
        if not hasattr(ds, 'predictors') or not hasattr(ds, 'targets'):
            raise ValueError("dataset must have 'predictors' and 'targets' variables")
        self.ds = ds
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._remove_nan = remove_nan
        self._convolution = self.model.is_convolutional
        self._keep_time_axis = self.model.is_recurrent
        self._impute_missing = self.model.impute
        self._indices = []
        self._n_sample = ds.dims['sample']
        self._has_time_step = 'time_step' in ds.dims

        self.on_epoch_end()

    @property
    def shape(self):
        """
        :return: the original shape of ensemble predictors, ([time_step,] variable, level, lat, lon)
        """
        return self.ds.predictors.shape[1:]

    @property
    def n_features(self):
        """
        :return: int: the number of features in the predictor array
        """
        return int(np.prod(self.shape))

    @property
    def dense_shape(self):
        """
        :return: the shape of flattened features. If the model is recurrent, (time_step, features); otherwise,
            (features,).
        """
        if self._keep_time_axis:
            if self._has_time_step:
                return (self.shape[0],) + (self.n_features // self.shape[0],)
            else:
                return (1,) + (self.n_features,)
        else:
            return (self.n_features,) + ()

    @property
    def convolution_shape(self):
        """
        :return: the shape of the predictors expected by a Conv2D or ConvLSTM2D layer. If the model is recurrent,
            (time_step, channels, y, x); if not, (channels, y, x).
        """
        if self._keep_time_axis:
            if self._has_time_step:
                return (self.shape[0],) + (int(np.prod(self.shape[1:-2])),) + self.shape[-2:]
            else:
                return (1,) + (int(np.prod(self.shape[:-2])),) + self.shape[-2:]
        else:
            return (int(np.prod(self.shape[:-2])),) + self.ds.predictors.shape[-2:]

    @property
    def shape_2d(self):
        """
        :return: the shape of the predictors expected by a Conv2D layer, (channels, y, x)
        """
        if self._keep_time_axis:
            self._keep_time_axis = False
            s = tuple(self.convolution_shape)
            self._keep_time_axis = True
            return s
        else:
            return self.convolution_shape

    def on_epoch_end(self):
        self._indices = np.arange(self._n_sample)
        if self._shuffle:
            np.random.shuffle(self._indices)

    def generate(self, samples, scale_and_impute=True):
        if len(samples) > 0:
            ds = self.ds.isel(sample=samples)
        else:
            ds = self.ds.isel(sample=slice(None))
        n_sample = ds.predictors.shape[0]
        p = ds.predictors.values.reshape((n_sample, -1))
        t = ds.targets.values.reshape((n_sample, -1))
        ds.close()
        ds = None

        # Remove samples with NaN; scale and impute
        if self._remove_nan:
            p, t = delete_nan_samples(p, t)
        if scale_and_impute:
            if self._impute_missing:
                p, t = self.model.imputer_transform(p, t)
            p, t = self.model.scaler_transform(p, t)

        # Format spatial shape for convolutions; also takes care of time axis
        if self._convolution:
            p = p.reshape((n_sample,) + self.convolution_shape)
            t = t.reshape((n_sample,) + self.convolution_shape)
        elif self._keep_time_axis:
            p = p.reshape((n_sample,) + self.dense_shape)
            t = t.reshape((n_sample,) + self.dense_shape)

        return p, t

    def __len__(self):
        """
        :return: the number of batches per epoch
        """
        return int(np.floor(self._n_sample / self._batch_size))

    def __getitem__(self, index):
        """
        Get one batch of data
        :param index: index of batch
        :return:
        """
        # Generate indexes of the batch
        indexes = self._indices[index * self._batch_size:(index + 1) * self._batch_size]

        # Generate data
        X, y = self.generate(indexes)

        return X, y


class SmartDataGenerator(Sequence):
    """
    Class used to generate training data on the fly from a loaded DataSet of predictor data. Depends on the structure
    of the EnsembleSelector to do scaling and imputing of data. This particular class loads the dataset efficiently by
    leveraging its knowledge of the predictor-target sequence and time_step dimension. DO NOT USE if the predictors
    and targets are not a continuous time sequence where dt between samples equals dt between time_steps.
    """

    def __init__(self, model, ds, batch_size=32, shuffle=False, remove_nan=True, load=True):
        """
        Initialize a SmartDataGenerator.

        :param model: instance of a DLWP model
        :param ds: xarray Dataset: predictor dataset. Should have attributes 'predictors' and 'targets'
        :param batch_size: int: number of samples to take at a time from the dataset
        :param shuffle: bool: if True, randomly select batches
        :param remove_nan: bool: if True, remove any samples with NaNs
        :param load: bool: if True, load the data in memory
        """
        self.model = model
        if not hasattr(ds, 'predictors'):
            raise ValueError("dataset must have 'predictors' variable")
        self.ds = ds
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._remove_nan = remove_nan
        self._convolution = self.model.is_convolutional
        self._keep_time_axis = self.model.is_recurrent
        self._impute_missing = self.model.impute
        self._indices = []
        self._n_sample = ds.dims['sample']
        if 'time_step' in ds.dims:
            self._has_time_step = True
            self.time_dim = ds.dims['time_step']
            self.da = self.ds.predictors.isel(time_step=0)
            # Add the last time steps in the series
            self.da = xr.concat((self.da, self.ds.predictors.isel(
                sample=slice(self._n_sample - self.time_dim + 1, None), time_step=-1)), dim='sample')
        else:
            self._has_time_step = False
            self.time_dim = 1
            self.da = self.ds.predictors

        if hasattr(self.ds, 'targets'):
            self.da = xr.concat((self.da, self.ds.targets.isel(sample=slice(self._n_sample - self.time_dim, None),
                                                               time_step=-1)), dim='sample')
        if load:
            self.da.load()
        self.on_epoch_end()

    @property
    def shape(self):
        """
        :return: the original shape of ensemble predictors, ([time_step,] variable, level, lat, lon)
        """
        return self.ds.predictors.shape[1:]

    @property
    def n_features(self):
        """
        :return: int: the number of features in the predictor array
        """
        return int(np.prod(self.shape))

    @property
    def dense_shape(self):
        """
        :return: the shape of flattened features. If the model is recurrent, (time_step, features); otherwise,
            (features,).
        """
        if self._keep_time_axis:
            if self._has_time_step:
                return (self.shape[0],) + (self.n_features // self.shape[0],)
            else:
                return (1,) + (self.n_features,)
        else:
            return (self.n_features,) + ()

    @property
    def convolution_shape(self):
        """
        :return: the shape of the predictors expected by a Conv2D or ConvLSTM2D layer. If the model is recurrent,
            (time_step, channels, y, x); if not, (channels, y, x).
        """
        if self._keep_time_axis:
            if self._has_time_step:
                return (self.shape[0],) + (int(np.prod(self.shape[1:-2])),) + self.shape[-2:]
            else:
                return (1,) + (int(np.prod(self.shape[:-2])),) + self.shape[-2:]
        else:
            return (int(np.prod(self.shape[:-2])),) + self.ds.predictors.shape[-2:]

    @property
    def shape_2d(self):
        """
        :return: the shape of the predictors expected by a Conv2D layer, (channels, y, x)
        """
        if self._keep_time_axis:
            self._keep_time_axis = False
            s = tuple(self.convolution_shape)
            self._keep_time_axis = True
            return s
        else:
            return self.convolution_shape

    def on_epoch_end(self):
        self._indices = np.arange(self._n_sample)
        if self._shuffle:
            np.random.shuffle(self._indices)

    def generate(self, samples, scale_and_impute=True):
        if len(samples) == 0:
            samples = np.arange(self._n_sample, dtype=np.int)
        else:
            samples = np.array(samples, dtype=np.int)
        n_sample = len(samples)
        p = np.concatenate([self.da.values[samples + n, np.newaxis] for n in range(self.time_dim)], axis=1)
        p = p.reshape((n_sample, -1))
        t = np.concatenate([self.da.values[samples + self.time_dim + n, np.newaxis] for n in range(self.time_dim)],
                           axis=1)
        t = t.reshape((n_sample, -1))

        # Remove samples with NaN; scale and impute
        if self._remove_nan:
            p, t = delete_nan_samples(p, t)
        if scale_and_impute:
            if self._impute_missing:
                p, t = self.model.imputer_transform(p, t)
            p, t = self.model.scaler_transform(p, t)

        # Format spatial shape for convolutions; also takes care of time axis
        if self._convolution:
            p = p.reshape((n_sample,) + self.convolution_shape)
            t = t.reshape((n_sample,) + self.convolution_shape)
        elif self._keep_time_axis:
            p = p.reshape((n_sample,) + self.dense_shape)
            t = t.reshape((n_sample,) + self.dense_shape)

        return p, t

    def __len__(self):
        """
        :return: the number of batches per epoch
        """
        return int(np.floor(self._n_sample / self._batch_size))

    def __getitem__(self, index):
        """
        Get one batch of data
        :param index: index of batch
        :return:
        """
        # Generate indexes of the batch
        indexes = self._indices[index * self._batch_size:(index + 1) * self._batch_size]

        # Generate data
        X, y = self.generate(indexes)

        return X, y
