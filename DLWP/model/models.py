#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
High-level APIs for building a DLWP model based on Keras and scikit-learn.
"""

import sklearn
import keras
import keras.layers
import numpy as np
import keras.models
from keras.utils import multi_gpu_model, Sequence

from .. import util
from .preprocessing import delete_nan_samples


class DLWPNeuralNet(object):
    """
    DLWP model class which uses a Keras Sequential neural network built to user specification.
    """
    def __init__(self, is_convolutional=True, is_recurrent=False, time_dim=1,
                 scaler_type='StandardScaler', scale_targets=True, apply_same_y_scaling=True, impute_missing=False):
        """
        Initialize an instance of DLWPNeuralNet.

        :param is_convolutional: bool: if True, use spatial shapes for input and output of the model
        :param is_recurrent: bool: if True, add a recurrent time axis to the model
        :param time_dim: int: the number of time steps in the input and output of the model (int >= 1)
        :param scaler_type: str: class of scikit-learn scaler to apply to the input data. If None is provided,
            disables scaling.
        :param scale_targets: bool: if True, also scale the target data. Necessary for optimizer evaluation if there
            are large magnitude differences in the output features.
        :param apply_same_y_scaling: bool: if True, if the predictors and targets are the same shape (as for time
            series prediction), apply the same scaler to predictors and targets
        :param impute_missing: bool: if True, uses scikit-learn Imputer for missing values
        """
        self.is_convolutional = is_convolutional
        self.is_recurrent = is_recurrent
        if int(time_dim) < 1:
            raise ValueError("'time_dim' must be >= 1")
        self.time_dim = time_dim
        self.scaler_type = scaler_type
        self.scale_targets = scale_targets
        self.apply_same_y_scaling = apply_same_y_scaling
        self.scaler = None
        self.scaler_y = None
        self.impute = impute_missing
        self.imputer = None
        self.imputer_y = None

        self.model = None
        self.is_parallel = False
        if scaler_type is None:
            self._is_init_fit = True
        else:
            self._is_init_fit = False

    def build_model(self, layers=(), gpus=1, **compile_kwargs):
        """
        Build a Keras Sequential model using the specified layers. Each element of layers must be a tuple consisting of
        (layer_name, layer_args, layer_kwargs); that is, each tuple is the name of the layer as defined in keras.layers,
        a tuple of arguments passed to the layer, and a dictionary of kwargs passed to the layer.

        :param layers: tuple: tuple of (layer_name, kwargs_dict) pairs added to the model
        :param gpus: int: number of GPU units on which to parallelize the Keras model
        :param compile_kwargs: kwargs passed to the 'compile' method of the Keras model
        """
        # Test the parameters
        if type(gpus) is not int:
            raise TypeError("'gpus' argument must be an int")
        if type(layers) not in [list, tuple]:
            raise TypeError("'layers' argument must be a tuple")
        layers = [l for l in layers]
        for l, layer in enumerate(layers):
            if type(layer) not in [list, tuple]:
                raise TypeError("each element of 'layers' must be a tuple")
            if len(layer) != 3:
                raise ValueError("each layer must be specified by three elements (name, args, kwargs)")
            if layer[1] is None:
                layer = [layer[0], (), layer[2]]
            if type(layer[1]) is not tuple:
                raise TypeError("the 'args' element of layer %d must be a tuple" % l)
            if layer[2] is None:
                layer = [layer[0], layer[1], {}]
            if type(layer[2]) is not dict:
                raise TypeError("the 'kwargs' element of layer %d must be a dict" % l)
            layers[l] = layer
        # Self-explanatory
        util.make_keras_picklable()
        # Build a model, either on a single GPU or on a CPU to control multiple GPUs
        self.model = keras.models.Sequential()
        for l, layer in enumerate(layers):
            try:
                layer_class = util.get_from_class('keras.layers', layer[0])
            except (ImportError, AttributeError):
                # Maybe we've defined a custom layer, which would be in DLWP.custom
                layer_class = util.get_from_class('DLWP.custom', layer[0])
            self.model.add(layer_class(*layer[1], **layer[2]))
        if gpus > 1:
            self.model = multi_gpu_model(self.model, gpus=gpus, cpu_relocation=True)
            self.is_parallel = True
        self.model.compile(**compile_kwargs)

    @staticmethod
    def _reshape(a, ret=False):
        a_shape = a.shape
        a = a.reshape((a_shape[0], -1))
        if ret:
            return a, a_shape
        return a

    def scaler_fit(self, X, y, **kwargs):
        if self.scaler_type is not None:
            scaler_class = util.get_from_class('sklearn.preprocessing', self.scaler_type)
            self.scaler = scaler_class(**kwargs)
            self.scaler_y = scaler_class(**kwargs)
            self.scaler.fit(self._reshape(X))
            if self.scale_targets:
                if self.apply_same_y_scaling:
                    self.scaler_y = self.scaler
                else:
                    self.scaler_y.fit(self._reshape(y))

    def scaler_transform(self, X, y=None):
        if self.scaler_type is None:
            if y is not None:
                return X, y
            else:
                return X
        X, X_shape = self._reshape(X, ret=True)
        X_transform = self.scaler.transform(X)
        if y is not None:
            if self.scale_targets:
                y, y_shape = self._reshape(y, ret=True)
                y_transform = self.scaler_y.transform(y)
                return X_transform.reshape(X_shape), y_transform.reshape(y_shape)
            else:
                return X_transform.reshape(X_shape), y
        else:
            return X_transform.reshape(X_shape)

    def imputer_fit(self, X, y):
        imputer_class = util.get_from_class('sklearn.preprocessing', 'Imputer')
        self.imputer = imputer_class(missing_values=np.nan, strategy="mean", axis=0, copy=False)
        self.imputer_y = imputer_class(missing_values=np.nan, strategy="mean", axis=0, copy=False)
        self.imputer.fit(self._reshape(X))
        if self.apply_same_y_scaling:
            self.imputer_y = self.imputer
        else:
            self.imputer_y.fit(self._reshape(y))

    def imputer_transform(self, X, y=None):
        X, X_shape = self._reshape(X, ret=True)
        X_transform = self.imputer.transform(X)
        if y is not None:
            y, y_shape = self._reshape(y, ret=True)
            y_transform = self.imputer_y.transform(y)
            return X_transform.reshape(X_shape), y_transform.reshape(y_shape)
        else:
            return X_transform.reshape(X_shape)

    def init_fit(self, predictors, targets):
        """
        Initialize the Imputer and Scaler of the model manually. This is useful for fitting the data pre-processors
        on a larger set of data before calls to the model 'fit' method with smaller sets of data and initialize=False.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: target data
        """
        if self.impute:
            self.imputer_fit(predictors, targets)
            predictors, targets = self.imputer_transform(predictors, y=targets)
        self.scaler_fit(predictors, targets)
        self._is_init_fit = True

    def fit(self, predictors, targets, initialize=True, **kwargs):
        """
        Fit the DLWPNeuralNet model. Also performs input feature scaling.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: target data
        :param initialize: bool: if True, initializes the Imputer and Scaler to the given predictors. 'fit' must be
            called with initialize=True the first time, or the Imputer and Scaler must be fit with 'init_fit'.
        :param kwargs: passed to the Keras 'fit' method
        """
        if initialize:
            self.init_fit(predictors, targets)
        elif not self._is_init_fit:
            raise AttributeError('DLWPNeuralNet has not been initialized for fitting with init_fit()')
        if self.impute:
            predictors, targets = self.imputer_transform(predictors, y=targets)
        predictors_scaled, targets_scaled = self.scaler_transform(predictors, targets)
        # Need to scale the validation data if it is given
        if 'validation_data' in kwargs and kwargs['validation_data'] is not None:
            if self.impute:
                predictors_test_scaled, targets_test_scaled = self.imputer_transform(*kwargs['validation_data'])
            else:
                predictors_test_scaled, targets_test_scaled = kwargs['validation_data']
            predictors_test_scaled, targets_test_scaled = self.scaler_transform(predictors_test_scaled,
                                                                                targets_test_scaled)
            kwargs['validation_data'] = (predictors_test_scaled, targets_test_scaled)
        self.model.fit(predictors_scaled, targets_scaled, **kwargs)

    def fit_generator(self, generator, **kwargs):
        """
        Fit the DLWPNeuralNet model using a generator. The generator becomes responsible for scaling and imputing
        the predictor/target data.

        :param generator: a generator for producing batches of data (see Keras docs), e.g., DataGenerator below
        :param kwargs: passed to the model's fit_generator() method
        """
        # If generator is a DataGenerator below, check that we have called init_fit
        if isinstance(generator, DataGenerator):
            if not self._is_init_fit:
                raise AttributeError('DLWPNeuralNet has not been initialized for fitting with init_fit()')
        self.model.fit_generator(generator, **kwargs)

    def predict(self, predictors, **kwargs):
        """
        Make a prediction with the DLWPNeuralNet model. Also performs input feature scaling.

        :param predictors: ndarray: predictor data
        :param kwargs: passed to Keras 'predict' method
        :return: ndarray: model prediction
        """
        if self.impute:
            predictors = self.imputer_transform(predictors)
        predictors_scaled = self.scaler_transform(predictors)
        predicted = self.model.predict(predictors_scaled, **kwargs)
        if self.scale_targets and self.scaler_type is not None:
            return self.scaler_y.inverse_transform(predicted)
        else:
            return predicted

    def predict_timeseries(self, predictors, time_steps, step_sequence=False, keep_time_dim=False, **kwargs):
        """
        Make a timeseries prediction with the DLWPNeuralNet model. Also performs input feature scaling. Forward predict
        time_steps number of time steps, intelligently using the time dimension to run the model time_steps/time_dim
        number of times and returning a time series of concatenated steps. Alternatively, using step_sequences, one can
        use only one predicted time step (the other inputs are copied from the previous input) at a time. If the model
        is not recurrent, then it is assumed that the second dimension can be reshaped to (self.time_dim, num_channels).

        :param predictors: ndarray: predictor data
        :param time_steps: int: number of time steps to predict forward
        :param step_sequence: bool: if True, takes one step at a time in a time series sequence. That is, if a model
            has a time_dim of t, the next forecast will use t-1 last steps from predictors plus the first step of the
            last prediction as inputs.
        :param keep_time_dim: if True, keep the time_step dimension in the output, otherwise integrates it into the
            forecast_hour (first) dimension
        :param kwargs: passed to Keras 'predict' method
        :return: ndarray: model prediction; first dim is time
        """
        time_steps = int(time_steps)
        if time_steps < 1:
            raise ValueError("time_steps must be an int > 0")
        if not step_sequence:
            time_steps = int(np.ceil(1. * time_steps / self.time_dim))
        time_series = np.full((time_steps,) + predictors.shape, np.nan, dtype=np.float32)
        p = predictors.copy()
        sample_dim = p.shape[0]
        if self.is_recurrent:
            feature_shape = p.shape[2:]
        else:
            feature_shape = p.shape[1:]
        for t in range(time_steps):
            if step_sequence:
                pr = self.predict(p, **kwargs)
                pr_shape = pr.shape[:]
                if not self.is_recurrent:
                    pr = pr.reshape((sample_dim, self.time_dim, -1) + feature_shape[1:])
                    p = p.reshape((sample_dim, self.time_dim, -1) + feature_shape[1:])
                p = np.concatenate((p[:, 1:], pr[:, [0]]), axis=1)
                if not self.is_recurrent:
                    p = p.reshape(predictors.shape)
                    pr = pr.reshape(pr_shape)
                time_series[t, ...] = 1. * pr  # step, sample, [time_step,] (features,)
            else:
                p = 1. * self.predict(p, **kwargs)
                time_series[t, ...] = 1. * p  # step, sample, [time_step,] (features,)
        time_series = time_series.reshape((time_steps, sample_dim, self.time_dim, -1) + feature_shape[1:])
        if not keep_time_dim:
            if step_sequence:
                time_series = time_series[:, :, 0]
            else:
                time_series = time_series.transpose((0, 2, 1) + tuple(range(3, 3 + len(feature_shape))))
                time_series = time_series.reshape((time_steps * self.time_dim, sample_dim, -1) + feature_shape[1:])
        return time_series

    def evaluate(self, predictors, targets, **kwargs):
        """
        Run the Keras model's 'evaluate' method, with input feature scaling.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: target data
        :param kwargs: passed to Keras 'evaluate' method
        :return:
        """
        if self.impute:
            predictors, targets = self.imputer_transform(predictors, targets)
        predictors_scaled, targets_scaled = self.scaler_transform(predictors, targets)
        score = self.model.evaluate(predictors_scaled, targets_scaled, **kwargs)
        return score


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
        :param batch_size: int: number of samples (days) to take at a time from the dataset
        :param shuffle: bool: if True, randomly select batches
        :param remove_nan: bool: if True, remove any samples with NaNs
        """
        self.model = model
        if not hasattr(ds, 'predictors') or not hasattr(ds, 'targets'):
            raise ValueError("dataset must have 'predictors' and 'targets' attributes")
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
        else:
            self._has_time_step = False
            self.time_dim = None

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
            s = self.convolution_shape
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
