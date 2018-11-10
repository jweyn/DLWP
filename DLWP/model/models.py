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
    def __init__(self, scaler_type='StandardScaler', scale_targets=True, apply_same_y_scaling=True,
                 impute_missing=False):
        """
        Initialize an instance of DLWPNeuralNet.

        :param scaler_type: str: class of scikit-learn scaler to apply to the input data. If None is provided,
            disables scaling.
        :param scale_targets: bool: if True, also scale the target data. Necessary for optimizer evaluation if there
            are large magnitude differences in the output features.
        :param apply_same_y_scaling: bool: if True, if the predictors and targets are the same shape (as for time
            series prediction), apply the same scaler to predictors and targets
        :param impute_missing: bool: if True, uses scikit-learn Imputer for missing values
        :param add_time_axis: bool: if True, adds a time dimension for RNNs to predictor and target data
        """
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
        for l in range(len(layers)):
            layer = layers[l]
            if type(layer) not in [list, tuple]:
                raise TypeError("each element of 'layers' must be a tuple")
            if len(layer) != 3:
                raise ValueError("each layer must be specified by three elements (name, args, kwargs)")
            if layer[1] is None:
                layer[1] = ()
            if type(layer[1]) is not tuple:
                raise TypeError("the 'args' element of layer %d must be a tuple" % l)
            if layer[2] is None:
                layer[2] = {}
            if type(layer[2]) is not dict:
                raise TypeError("the 'kwargs' element of layer %d must be a dict" % l)
        # Self-explanatory
        util.make_keras_picklable()
        # Build a model, either on a single GPU or on a CPU to control multiple GPUs
        self.model = keras.models.Sequential()
        for l in range(len(layers)):
            layer = layers[l]
            try:
                layer_class = util.get_from_class('keras.layers', layer[0])
            except (ImportError, AttributeError):
                layer_class = util.get_from_class('ensemble_net.util', layer[0])
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
        :param targets: ndarray: corresponding truth data
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
        :param targets: ndarray: corresponding truth data
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
        if 'validation_data' in kwargs:
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

    def predict_timeseries(self, predictors, time_steps, **kwargs):
        """
        Make a timeseries prediction with the DLWPNeuralNet model. Also performs input feature scaling. Forward predict
        time_steps number of times.

        :param predictors: ndarray: predictor data
        :param time_steps: int: number of time steps to predict forward
        :param kwargs: passed to Keras 'predict' method
        :return: ndarray: model prediction; first dim is time
        """
        time_steps = int(time_steps)
        if time_steps < 0:
            raise ValueError("time_steps must be an int > 0")
        time_series = np.full((time_steps,) + predictors.shape, np.nan, dtype=np.float32)
        p = predictors.copy()
        for t in range(time_steps):
            p = 1. * self.predict(p, **kwargs)
            time_series[t, ...] = 1. * p
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

    def __init__(self, model, ds, batch_size=32, shuffle=False, remove_nan=True, add_time_axis=False):
        """
        Initialize a DataGenerator.

        :param model: instance of a DLWP model
        :param ds: xarray Dataset: predictor dataset. Should have attributes 'predictors' and 'targets'
        :param batch_size: int: number of samples (days) to take at a time from the dataset
        :param shuffle: bool: if True, randomly select batches
        :param remove_nan: bool: if True, remove any samples with NaNs
        :param add_time_axis: bool: if True, add a time axis for use with Keras RNNs
        """
        self.model = model
        if not hasattr(ds, 'predictors') or not hasattr(ds, 'targets'):
            raise ValueError("dataset must have 'predictors' and 'targets' attributes")
        self.ds = ds
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._remove_nan = remove_nan
        self._add_time_axis = add_time_axis
        self._impute_missing = self.model.impute
        self._indices = []
        self._n_sample = ds.dims['sample']

        self.on_epoch_end()

    @property
    def spatial_shape(self):
        """
        :return: the shape of the spatial component of ensemble predictors
        """
        return self.ds.predictors.shape[1:]

    def on_epoch_end(self):
        self._indices = np.arange(self._n_sample)
        if self._shuffle:
            np.random.shuffle(self._indices)

    def generate(self, samples, scale_and_impute=True):
        if len(samples) > 0:
            ds = self.ds.isel(sample=samples)
        else:
            ds = self.ds.isel(sample=slice(None))
        ds.load()
        p = ds.predictors.values.reshape((ds.predictors.shape[0], -1))
        t = ds.targets.values.reshape((ds.targets.shape[0], -1))
        ds.close()
        ds = None

        # Remove samples with NaN
        if self._remove_nan:
            p, t = delete_nan_samples(p, t)
        if self._impute_missing:
            if scale_and_impute:
                p, t = self.model.imputer_transform(p, t)
                p, t = self.model.scaler_transform(p, t)
        else:
            if scale_and_impute:
                p, t = self.model.scaler_transform(p, t)

        # Add a time dimension axis for RNNs
        if self._add_time_axis:
            p = np.expand_dims(p, axis=1)
            t = np.expand_dims(t, axis=1)

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