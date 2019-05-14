#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
High-level APIs for building a DLWP model based on Keras and scikit-learn.
"""

import keras
import keras.layers
import numpy as np
import keras.models
from keras.utils import multi_gpu_model

from .generators import DataGenerator, SmartDataGenerator, SeriesDataGenerator
from .. import util


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

        self.base_model = None
        self.model = None
        self.gpus = 1
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
        self.base_model = keras.models.Sequential()
        for l, layer in enumerate(layers):
            try:
                layer_class = util.get_from_class('keras.layers', layer[0])
            except (ImportError, AttributeError):
                # Maybe we've defined a custom layer, which would be in DLWP.custom
                layer_class = util.get_from_class('DLWP.custom', layer[0])
            self.base_model.add(layer_class(*layer[1], **layer[2]))
        if gpus > 1:
            import tensorflow as tf
            with tf.device('/cpu:0'):
                self.base_model = keras.models.clone_model(self.base_model)
            self.model = multi_gpu_model(self.base_model, gpus=gpus)
            self.gpus = gpus
        else:
            self.model = self.base_model
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
        if isinstance(generator, (DataGenerator, SmartDataGenerator, SeriesDataGenerator)):
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
            if 'verbose' in kwargs and kwargs['verbose'] > 0:
                print('Time step %d/%d' % (t+1, time_steps))
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


class DLWPFunctional(object):
    """
    DLWP model class which uses model built on the Keras Functional API. This class DOES NOT support scaling or
    imputing of input/target data; this must be done separately.
    """
    def __init__(self, is_convolutional=True, is_recurrent=False, time_dim=1):
        """
        Initialize an instance of DLWPFunctional.

        :param is_convolutional: bool: if True, use spatial shapes for input and output of the model
        :param is_recurrent: bool: if True, add a recurrent time axis to the model
        :param time_dim: int: the number of time steps in the input and output of the model (int >= 1)
        """
        self.is_convolutional = is_convolutional
        self.is_recurrent = is_recurrent
        if int(time_dim) < 1:
            raise ValueError("'time_dim' must be >= 1")
        self.time_dim = time_dim

        self.scaler = None
        self.scaler_y = None
        self.impute = False
        self.imputer = None
        self.imputer_y = None
        self._n_steps = 1

        self.base_model = None
        self.model = None
        self.gpus = 1

    def build_model(self, model, gpus=1, **compile_kwargs):
        """
        Compile a Keras Functional model.

        :param model: keras.models.Model: Keras functional model
        :param gpus: int: number of GPU units on which to parallelize the Keras model
        :param compile_kwargs: kwargs passed to the 'compile' method of the Keras model
        """
        # Test the parameters
        if type(gpus) is not int:
            raise TypeError("'gpus' argument must be an int")
        # Self-explanatory
        util.make_keras_picklable()
        # Build a model, either on a single GPU or on a CPU to control multiple GPUs
        self.base_model = model
        self._n_steps = len(model.outputs)
        if gpus > 1:
            import tensorflow as tf
            with tf.device('/cpu:0'):
                self.base_model = keras.models.clone_model(self.base_model)
            self.model = multi_gpu_model(self.base_model, gpus=gpus)
            self.gpus = gpus
        else:
            self.model = self.base_model
        self.model.compile(**compile_kwargs)

    def scaler_transform(self, X, y=None):
        """
        For compatibility.
        """
        if y is not None:
            return X, y
        else:
            return X

    def fit(self, predictors, targets, **kwargs):
        """
        Fit the DLWPNeuralNet model.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: target data
        :param kwargs: passed to the Keras 'fit' method
        """
        self.model.fit(predictors, targets, **kwargs)

    def fit_generator(self, generator, **kwargs):
        """
        Fit the DLWPNeuralNet model using a generator.
        the predictor/target data.

        :param generator: a generator for producing batches of data (see Keras docs), e.g., DataGenerator below
        :param kwargs: passed to the model's fit_generator() method
        """
        self.model.fit_generator(generator, **kwargs)

    def predict(self, predictors, **kwargs):
        """
        Make a prediction with the DLWPNeuralNet model.

        :param predictors: ndarray: predictor data
        :param kwargs: passed to Keras 'predict' method
        :return: ndarray: model prediction
        """
        return self.model.predict(predictors, **kwargs)

    def predict_timeseries(self, predictors, time_steps, keep_time_dim=False, **kwargs):
        """
        Make a timeseries prediction with the DLWPNeuralNet model. Also performs input feature scaling. Forward predict
        time_steps number of time steps, intelligently using the known model outputs to run the model time_steps
        /time_dim number of times and returning a time series of concatenated steps.

        :param predictors: ndarray: predictor data
        :param time_steps: int: number of time steps to predict forward
        :param keep_time_dim: if True, keep the time_step dimension in the output, otherwise integrates it into the
            forecast_hour (first) dimension
        :param kwargs: passed to Keras 'predict' method
        :return ndarray: model prediction; first dim is time
        """
        time_steps = int(time_steps)
        if time_steps < 1:
            raise ValueError("time_steps must be an int > 0")
        steps = int(np.ceil(time_steps / self._n_steps / self.time_dim))
        out_steps = steps * self._n_steps
        time_series = np.full((out_steps,) + predictors.shape, np.nan, dtype=np.float32)
        p = predictors.copy()
        sample_dim = p.shape[0]
        if self.is_recurrent:
            feature_shape = p.shape[2:]
        else:
            feature_shape = p.shape[1:]
        for t in range(steps):
            if 'verbose' in kwargs and kwargs['verbose'] > 0:
                print('Prediction step %d/%d' % (t + 1, steps))
            result = self.predict(p, **kwargs)
            if self._n_steps == 1:
                p[:] = result[:]
            else:
                p[:] = result[-1]
            time_series[t * self._n_steps:(t + 1) * self._n_steps, ...] = np.stack(result, axis=0)
        time_series = time_series.reshape((out_steps, sample_dim, self.time_dim, -1) + feature_shape[1:])
        if not keep_time_dim:
            time_series = time_series.transpose((0, 2, 1) + tuple(range(3, 3 + len(feature_shape))))
            time_series = time_series.reshape((out_steps * self.time_dim, sample_dim, -1) + feature_shape[1:])
        return time_series

    def evaluate(self, predictors, targets, **kwargs):
        """
        Run the Keras model's 'evaluate' method, with input feature scaling.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: target data
        :param kwargs: passed to Keras 'evaluate' method
        :return:
        """
        score = self.model.evaluate(predictors, targets, **kwargs)
        return score
