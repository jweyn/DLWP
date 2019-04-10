#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
High-level APIs for building a DLWP model using PyTorch.
"""

import numpy as np
import time
import warnings
from .. import util

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

except ImportError:
    warnings.warn('DLWPTorchNN is not available because PyTorch is not installed.')


class DLWPTorchNN(object):
    """
    DLWP model class which uses a torch.nn Module built to user specification.
    """

    def __init__(self, is_convolutional=False, is_recurrent=False, time_dim=1,
                 scaler_type='StandardScaler', scale_targets=True, apply_same_y_scaling=True, impute_missing=False):
        """
        Initialize an instance of DLWPTorchNN.

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

        if scaler_type is None:
            self._is_init_fit = True
        else:
            self._is_init_fit = False

        self.model = None
        self.optimizer = None
        self.loss = None
        self.metric = None
        self.layers = []
        self.activations = []

        self.history = {}

    def build_model(self, layers, optimizer, loss, optimizer_kwargs=None, loss_kwargs=None,
                    metric='L1Loss', metric_kwargs=None):
        """
        Build a torch.nn.Module model using the specified layers. Each element of layers must be a tuple consisting of
        (layer_name, layer_args, layer_kwargs); that is, each tuple is the name of the layer as defined in torch.nn,
        a tuple of arguments passed to the layer, and a dictionary of kwargs passed to the layer. The optimizer is
        passed as a string name of a torch.optim class; the optimizer kwargs should not include any Module parameters
        as those will be added automatically. The loss is also passed as a string class name from torch.nn, as is the
        metric (which is just a loss as well).

        :param layers: tuple: tuple of (layer_name, layer_args, layer_kwargs) elements added to the model
        :param optimizer: str: name of torch.optim optimizer class to use
        :param loss: str: name of torch.nn loss class to use
        :param optimizer_kwargs: dict: kwargs passed to the optimizer class
        :param loss_kwargs: dict: kwargs passed to the loss function class
        :param metric: str: name of torch.nn loss to use as an error metric
        :param metric_kwargs: dict: kwargs passed to the metric loss function class
        """
        # Test the parameters
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
        optimizer_kwargs = optimizer_kwargs or {}
        if not isinstance(optimizer_kwargs, dict):
            raise TypeError("'optimizer_kwargs' must be a dict")
        loss_kwargs = loss_kwargs or {}
        if not isinstance(loss_kwargs, dict):
            raise TypeError("'loss_kwargs' must be a dict")
        metric_kwargs = metric_kwargs or {}
        if not isinstance(metric_kwargs, dict):
            raise TypeError("'metric_kwargs' must be a dict")

        # Build the layers
        self.model = nn.Module()
        self.layers = []
        self.activations = []
        for l, layer in enumerate(layers):
            try:
                layer_class = util.get_from_class('torch.nn', layer[0])
            except (ImportError, AttributeError):
                # Maybe we've defined a custom layer, which would be in DLWP.custom
                layer_class = util.get_from_class('DLWP.custom', layer[0])
            # Remove the activation kwarg and instead add it to activations
            if 'activation' in layer[2] and layer[2]['activation'] is not None:
                self.activations.append(util.get_from_class('torch.nn.functional', layer[2].pop('activation')))
            else:
                self.activations.append(None)
                try:
                    layer[2].pop('activation')
                except KeyError:
                    pass
            self.layers.append(layer_class(*layer[1], **layer[2]))

        # Manually compile the model's layers and forward function
        for l, layer in enumerate(self.layers):
            setattr(self.model, 'layer%d' % l, layer)
        self.model.forward = self._forward
        self.model.to(device)

        # Create the optimizer and loss
        self.loss = util.get_from_class('torch.nn', loss)(**loss_kwargs)
        self.optimizer = util.get_from_class('torch.optim', optimizer)(self.model.parameters(), **optimizer_kwargs)
        self.metric = util.get_from_class('torch.nn', metric)(**metric_kwargs)

    def _forward(self, x):
        for l, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            x = getattr(self.model, 'layer%d' % l)(x)
            if activation is not None:
                x = activation(x)
        return x

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

    def fit_generator(self, generator, epochs=1, min_epochs=None, validation_generator=None,
                      early_stop=None, lr_schedule=None, verbose=0):
        self.history['loss'] = []
        self.history['error'] = []
        if validation_generator is not None:
            self.history['val_loss'] = []
            self.history['val_error'] = []
        elif lr_schedule is not None:
            print("Warning: learning rate scheduler 'lr_sched' needs validation data; disabling")
        n_d = len(generator)
        for epoch in range(epochs):
            if verbose > 0:
                print('\nEpoch %d/%d' % (epoch + 1, epochs))
            epoch_start = time.time()
            running_loss = 0.0
            running_error = 0.0
            for b in range(len(generator)):
                # Retrieve the batch of data
                p, t = generator[b]
                p, t = torch.tensor(p).to(device), torch.tensor(t).to(device)
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                o = self.model(p)
                loss = self.loss(o, t)
                loss.backward()
                self.optimizer.step()
                # Calculate and print statistics
                running_loss = (b * running_loss + loss.item()) / (b + 1)
                running_error = (b * running_error + self._error(o, t)) / (b + 1)
                if verbose > 1:
                    print('%d/%d loss: %0.4f - error: %0.4f' %
                          (b + 1, n_d, running_loss, running_error), end='\r')
            # Calculate and print metrics
            print_line = ''
            self.history['loss'].append(running_loss)
            self.history['error'].append(running_error)
            if verbose > 0:
                print_line += ' - loss: %0.4f - error: %0.4f' % (running_loss, running_error)
            if validation_generator is not None:
                with torch.no_grad():
                    running_loss = 0.0
                    running_error = 0.0
                    for b in range(len(validation_generator)):
                        p, t = validation_generator[b]
                        p, t = torch.tensor(p).to(device), torch.tensor(t).to(device)
                        o = self.model(p)
                        running_loss = (b * running_loss + self.loss(o, t).item()) / (b + 1)
                        running_error = (b * running_error + self._error(o, t)) / (b + 1)
                self.history['val_loss'].append(running_loss)
                self.history['val_error'].append(running_error)
                if verbose > 0:
                    print_line += ' - val_loss: %0.4f – val_error: %0.4f' % (running_loss, running_error)
                if early_stop is not None:
                    if min_epochs is not None and epoch > min_epochs + early_stop:
                        if epoch - np.argmin(self.history['val_loss']) == early_stop:
                            if verbose > 0:
                                print('\nval_loss stopped improving; ending fit')
                            break
                if lr_schedule is not None:
                    lr_schedule.step(running_loss)
            if verbose > 0:
                print('%d/%d - time: %0.2f s' % (n_d, n_d, time.time() - epoch_start) + print_line, end='')
        if verbose > 0:
            print('')
        return self.history

    def predict(self, predictors):
        """
        Make a prediction with the DLWPTorchNN model. Also performs input feature scaling.

        :param predictors: ndarray: predictor data
        :return: ndarray: model prediction
        """
        if self.impute:
            predictors = self.imputer_transform(predictors)
        p = self.scaler_transform(predictors)
        all_p = []
        with torch.no_grad():
            p = torch.tensor(p).to(device)
            predicted = self.model(p).cpu().numpy()
            all_p.append(predicted)
        all_p = np.array(all_p).reshape((-1,) + predicted.shape[1:])
        if self.scale_targets and self.scaler_type is not None:
            return self.scaler_y.inverse_transform(all_p)
        else:
            return all_p

    def predict_timeseries(self, predictors, time_steps, step_sequence=False, keep_time_dim=False, verbose=0):
        """
        Make a timeseries prediction with the DLWPTorchNN model. Also performs input feature scaling. Forward predict
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
        :param verbose: bool or int: print progress
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
            if verbose:
                print('Time step %d/%d' % (t+1, time_steps))
            if step_sequence:
                pr = self.predict(p)
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
                p = 1. * self.predict(p)
                time_series[t, ...] = 1. * p  # step, sample, [time_step,] (features,)
        time_series = time_series.reshape((time_steps, sample_dim, self.time_dim, -1) + feature_shape[1:])
        if not keep_time_dim:
            if step_sequence:
                time_series = time_series[:, :, 0]
            else:
                time_series = time_series.transpose((0, 2, 1) + tuple(range(3, 3 + len(feature_shape))))
                time_series = time_series.reshape((time_steps * self.time_dim, sample_dim, -1) + feature_shape[1:])
        return time_series

    def _error(self, x, y):
        return self.metric(x, y).item()

    def evaluate(self, predictors, targets):
        """
        Return the loss and error of given predictors and targets, with input feature scaling.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: target data
        :return:
        """
        if self.impute:
            predictors, targets = self.imputer_transform(predictors, targets)
        p, t = self.scaler_transform(predictors, targets)
        with torch.no_grad():
            p, t = torch.tensor(p).to(device), torch.tensor(t).to(device)
            o = self.model(p)
            loss = self.loss(o, t).item()
            error = self._error(o, t)
        return loss, error

    def reset(self):
        """
        Reset the weights in each layer of the model.
        """
        for c in self.model.named_children():
            try:
                c[1].reset_parameters()
            except AttributeError:
                print("warning: layer '%s' cannot be reset" % c[0])
