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

from ..util import delete_nan_samples, insolation


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
        self._is_convolutional = self.model.is_convolutional
        self._keep_time_axis = self.model.is_recurrent
        self._impute_missing = self.model.impute
        self._indices = []
        self._n_sample = ds.dims['sample']
        self._has_time_step = 'time_step' in ds.dims

        self.on_epoch_end()

    @property
    def shape(self):
        """
        :return: the full shape of predictors, (time_step, [variable, level,] lat, lon)
        """
        if self._has_time_step:
            return self.ds.predictors.shape[1:]
        else:
            return (1,) + self.ds.predictors.shape[1:]

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
            return (self.shape[0],) + (self.n_features // self.shape[0],)
        else:
            return (self.n_features,) + ()

    @property
    def convolution_shape(self):
        """
        :return: the shape of the predictors expected by a Conv2D or ConvLSTM2D layer. If the model is recurrent,
            (time_step, channels, y, x); if not, (channels, y, x).
        """
        if self._keep_time_axis:
            return (self.shape[0],) + (int(np.prod(self.shape[1:-2])),) + self.shape[-2:]
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
        if self._is_convolutional:
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
        :return: (ndarray, ndarray): predictors, targets
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
        self._is_convolutional = self.model.is_convolutional
        self._keep_time_axis = self.model.is_recurrent
        self._impute_missing = self.model.impute
        self._indices = []
        self._n_sample = ds.dims['sample']
        if 'time_step' in ds.dims:
            self.time_dim = ds.dims['time_step']
            self.da = self.ds.predictors.isel(time_step=0)
            # Add the last time steps in the series
            self.da = xr.concat((self.da, self.ds.predictors.isel(
                sample=slice(self._n_sample - self.time_dim + 1, None), time_step=-1)), dim='sample')
        else:
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
        :return: the full shape of predictors, (time_step, [variable, level,] lat, lon)
        """
        return (self.time_dim,) + self.da.shape[1:]

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
            return (self.shape[0],) + (self.n_features // self.shape[0],)
        else:
            return (self.n_features,) + ()

    @property
    def convolution_shape(self):
        """
        :return: the shape of the predictors expected by a Conv2D or ConvLSTM2D layer. If the model is recurrent,
            (time_step, channels, y, x); if not, (channels, y, x).
        """
        if self._keep_time_axis:
            return (self.shape[0],) + (int(np.prod(self.shape[1:-2])),) + self.shape[-2:]
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
        if self._is_convolutional:
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
        :return: (ndarray, ndarray): predictors, targets
        """
        # Generate indexes of the batch
        indexes = self._indices[index * self._batch_size:(index + 1) * self._batch_size]

        # Generate data
        X, y = self.generate(indexes)

        return X, y


class SeriesDataGenerator(Sequence):
    """
    Class used to generate training data on the fly from a loaded DataSet of predictor data. Depends on the structure
    of the EnsembleSelector to do scaling and imputing of data. This class expects DataSet to contain a single variable,
    'predictors', which is a continuous time sequence of weather data. The user supplies arguments to load specific
    variables/levels and the number of time steps for the inputs/outputs. It is highly recommended to use the option
    to load the data into memory if enough memory is available as the increased I/O calls for generating the correct
    data sequences will take a toll. This class also makes it possible to add model-invariant data, such as incoming
    solar radiation, to the inputs.
    """

    def __init__(self, model, ds, input_sel=None, output_sel=None, input_time_steps=1, output_time_steps=1,
                 add_insolation=False, batch_size=32, shuffle=False, remove_nan=True, load=True):
        """
        Initialize a SeriesDataGenerator.

        :param model: instance of a DLWP model
        :param ds: xarray Dataset: predictor dataset. Should have attribute 'predictors'.
        :param input_sel: dict: variable/level selection for input features
        :param output_sel: dict: variable/level selection for output features
        :param input_time_steps: int: number of time steps in the input features
        :param output_time_steps: int: number of time steps in the output features (recommended either 1 or the same
            as input_time_steps)
        :param add_insolation: bool: if True, add insolation to the inputs. Incompatible with 3-d convolutions.
        :param batch_size: int: number of samples to take at a time from the dataset
        :param shuffle: bool: if True, randomly select batches
        :param remove_nan: bool: if True, remove any samples with NaNs
        :param load: bool: if True, load the data in memory (highly recommended if enough system memory is available)
        """
        self.model = model
        if not hasattr(ds, 'predictors'):
            raise ValueError("dataset must have 'predictors' variable")
        assert int(input_time_steps) > 0
        assert int(output_time_steps) > 0
        assert int(batch_size) > 0
        self.ds = ds
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._remove_nan = remove_nan
        self._is_convolutional = self.model.is_convolutional
        self._keep_time_axis = self.model.is_recurrent
        self._impute_missing = self.model.impute
        self._indices = []
        self._n_sample = ds.dims['sample'] - input_time_steps - output_time_steps + 1
        if 'time_step' in ds.dims:
            # Use -1 index because Preprocessor.data_to_samples (which generates a 'time_step' dim), assigns the
            # datetime 'sample' dim based on the initialization time, time_step=-1
            self.da = self.ds.predictors.isel(time_step=-1)
        else:
            self.da = self.ds.predictors

        self._input_sel = input_sel or {}
        self._output_sel = output_sel or {}
        self._input_time_steps = input_time_steps
        self._output_time_steps = output_time_steps

        self.input_da = self.da.sel(**self._input_sel)
        self.output_da = self.da.sel(**self._output_sel)
        if load:
            self.input_da.load()
            self.output_da.load()

        self.on_epoch_end()

        # Pre-generate the insolation data
        self._add_insolation = int(add_insolation)
        if add_insolation:
            sol = insolation(self.da.sample.values, self.da.lat.values, self.da.lon.values)
            self.insolation_da = xr.DataArray(sol, coords={
                'sample': self.da.sample,
                'lat': self.da.lat,
                'lon': self.da.lon
            }, dims=['sample', 'lat', 'lon'])

    @property
    def shape(self):
        """
        :return: the original shape of input data: (time_step, [variable, level,] lat, lon); excludes insolation
        """
        return (self._input_time_steps,) + self.input_da.shape[1:]

    @property
    def n_features(self):
        """
        :return: int: the number of input features; includes insolation
        """
        return int(np.prod(self.shape)) + int(np.prod(self.shape[-2:])) * self._input_time_steps * self._add_insolation

    @property
    def dense_shape(self):
        """
        :return: the shape of flattened input features. If the model is recurrent, (time_step, features); otherwise,
            (features,).
        """
        if self._keep_time_axis:
            return (self.shape[0],) + (self.n_features // self.shape[0],)
        else:
            return (self.n_features,) + ()

    @property
    def convolution_shape(self):
        """
        :return: the shape of the predictors expected by a Conv2D or ConvLSTM2D layer. If the model is recurrent,
            (time_step, channels, y, x); if not, (channels, y, x). Includes insolation.
        """
        if self._keep_time_axis:
            return (self._input_time_steps,) + (int(np.prod(self.shape[1:-2]))+self._add_insolation,) + self.shape[-2:]
        else:
            return (int(np.prod(self.shape[:-2])) +
                    self._input_time_steps * self._add_insolation,) + self.input_da.shape[-2:]

    @property
    def shape_2d(self):
        """
        :return: the shape of the predictors expected by a Conv2D layer, (channels, y, x); includes insolation
        """
        if self._keep_time_axis:
            self._keep_time_axis = False
            s = tuple(self.convolution_shape)
            self._keep_time_axis = True
            return s
        else:
            return self.convolution_shape

    @property
    def output_shape(self):
        """
        :return: the original shape of outputs: (time_step, [variable, level,] lat, lon)
        """
        return (self._output_time_steps,) + self.output_da.shape[1:]

    @property
    def output_n_features(self):
        """
        :return: int: the number of output features
        """
        return int(np.prod(self.output_shape))

    @property
    def output_dense_shape(self):
        """
        :return: the shape of flattened output features. If the model is recurrent, (time_step, features); otherwise,
            (features,).
        """
        if self._keep_time_axis:
            return (self.output_shape[0],) + (self.output_n_features // self.output_shape[0],)
        else:
            return (self.output_n_features,) + ()

    @property
    def output_convolution_shape(self):
        """
        :return: the shape of the predictors expected to be returned by a Conv2D or ConvLSTM2D layer. If the model is
            recurrent, (time_step, channels, y, x); if not, (channels, y, x).
        """
        if self._keep_time_axis:
            return (self._output_time_steps,) + (int(np.prod(self.output_shape[1:-2])),) + self.output_shape[-2:]
        else:
            return (int(np.prod(self.output_shape[:-2])),) + self.output_da.shape[-2:]

    @property
    def output_shape_2d(self):
        """
        :return: the shape of the predictors expected to be returned by a Conv2D layer, (channels, y, x)
        """
        if self._keep_time_axis:
            self._keep_time_axis = False
            s = tuple(self.output_convolution_shape)
            self._keep_time_axis = True
            return s
        else:
            return self.output_convolution_shape

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
        p = np.concatenate([self.input_da.values[samples + n, np.newaxis] for n in range(self._input_time_steps)],
                           axis=1)
        t = np.concatenate([self.output_da.values[samples + self._input_time_steps + n, np.newaxis]
                            for n in range(self._output_time_steps)], axis=1)
        if self._add_insolation:
            # Pretend like we have no insolation and keep the time axis
            self._add_insolation = False
            keep_time = bool(self._keep_time_axis)
            self._keep_time_axis = True
            s = tuple(self.convolution_shape)
            self._add_insolation = True
            self._keep_time_axis = bool(keep_time)
            sol = np.concatenate([self.insolation_da.values[samples + n, np.newaxis]
                                  for n in range(self._input_time_steps)], axis=1)
            p = p.reshape((n_sample,) + s)
            p = np.concatenate([p, sol[:, :, np.newaxis]], axis=2)
        p = p.reshape((n_sample, -1))
        t = t.reshape((n_sample, -1))

        # Remove samples with NaN; scale and impute
        if self._remove_nan:
            p, t = delete_nan_samples(p, t)
        if scale_and_impute:
            if self._impute_missing:
                p, t = self.model.imputer_transform(p, t)
            p, t = self.model.scaler_transform(p, t)

        # Format spatial shape for convolutions; also takes care of time axis
        if self._is_convolutional:
            p = p.reshape((n_sample,) + self.convolution_shape)
            t = t.reshape((n_sample,) + self.output_convolution_shape)
        elif self._keep_time_axis:
            p = p.reshape((n_sample,) + self.dense_shape)
            t = t.reshape((n_sample,) + self.output_dense_shape)

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
        :return: (ndarray, ndarray): predictors, targets
        """
        # Generate indexes of the batch
        indexes = self._indices[index * self._batch_size:(index + 1) * self._batch_size]

        # Generate data
        X, y = self.generate(indexes)

        return X, y
