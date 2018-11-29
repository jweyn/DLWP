#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
DLWP utilities.
"""

import pickle
import tempfile
from copy import copy
import numpy as np

import keras.models
from keras.callbacks import Callback, EarlyStopping
from keras import backend as K


# ==================================================================================================================== #
# General utility functions
# ==================================================================================================================== #

def make_keras_picklable():
    """
    Thanks to http://zachmoshe.com/2017/04/03/pickling-keras-models.html
    """

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def get_object(module_class):
    """
    Given a string with a module class name, it imports and returns the class.
    This function (c) Tom Keffer, weeWX; modified by Jonathan Weyn.
    """
    # Split the path into its parts
    parts = module_class.split('.')
    # Get the top level module
    module = parts[0]  # '.'.join(parts[:-1])
    # Import the top level module
    mod = __import__(module)
    # Recursively work down from the top level module to the class name.
    # Be prepared to catch an exception if something cannot be found.
    try:
        for part in parts[1:]:
            module = '.'.join([module, part])
            # Import each successive module
            __import__(module)
            mod = getattr(mod, part)
    except ImportError as e:
        # Can't find a recursive module. Give a more informative error message:
        raise ImportError("'%s' raised when searching for %s" % (str(e), module))
    except AttributeError:
        # Can't find the last attribute. Give a more informative error message:
        raise AttributeError("Module '%s' has no attribute '%s' when searching for '%s'" %
                             (mod.__name__, part, module_class))

    return mod


def get_from_class(module_name, class_name):
    """
    Given a module name and a class name, return an object corresponding to the class retrieved as in
    `from module_class import class_name`.

    :param module_name: str: name of module (may have . attributes)
    :param class_name: str: name of class
    :return: object pointer to class
    """
    mod = __import__(module_name, fromlist=[class_name])
    class_obj = getattr(mod, class_name)
    return class_obj


def save_model(model, file_name, history=None):
    """
    Saves a class instance with a 'model' attribute to disk. Creates two files: one pickle file containing no model
    saved as ${file_name}.pkl and one for the model saved as ${file_name}.keras. Use the `load_model()` method to load
    a model saved with this method.

    :param model: model instance (with a 'model' attribute) to save
    :param file_name: str: base name of save files
    :param history: history from Keras fitting, or None
    :return:
    """
    model.model.save('%s.keras' % file_name)
    model_copy = copy(model)
    model_copy.model = None
    with open('%s.pkl' % file_name, 'wb') as f:
        pickle.dump(model_copy, f, protocol=pickle.HIGHEST_PROTOCOL)
    if history is not None:
        with open('%s.history' % file_name, 'wb') as f:
            pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(file_name, history=False):
    """
    Loads a model saved to disk with the `save_model()` method.

    :param file_name: str: base name of save files
    :param history: bool: if True, loads the history file along with the model
    :return: model [, dict]: loaded object [, dictionary of training history]
    """
    with open('%s.pkl' % file_name, 'rb') as f:
        model = pickle.load(f)
    custom_layers = {}
    model.model = keras.models.load_model('%s.keras' % file_name, custom_objects=custom_layers, compile=True)
    if history:
        with open('%s.history' % file_name, 'rb') as f:
            h = pickle.load(f)
        return model, h
    else:
        return model


# ==================================================================================================================== #
# Custom Keras classes
# ==================================================================================================================== #

class AdamLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None, beta_1=0.9, beta_2=0.999,):
        optimizer = self.model.optimizer
        it = K.cast(optimizer.iterations, K.floatx())
        lr = K.cast(optimizer.lr, K.floatx())
        decay = K.cast(optimizer.decay, K.floatx())
        t = K.eval(it + 1.)
        new_lr = K.eval(lr * (1. / (1. + decay * it)))
        lr_t = K.eval(new_lr * (K.sqrt(1. - K.pow(beta_2, t)) / (1. - K.pow(beta_1, t))))
        print(' - LR: {:.6f}'.format(lr_t))


class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        it = K.cast(optimizer.iterations, K.floatx())
        lr = K.cast(optimizer.lr, K.floatx())
        decay = K.cast(optimizer.decay, K.floatx())
        new_lr = K.eval(lr * (1. / (1. + decay * it)))
        print(' - LR: {:.6f}'.format(new_lr))


class BatchHistory(Callback):
    def on_train_begin(self, logs=None):
        self.history = []
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.history.append({})

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history[self.epoch].setdefault(k, []).append(v)


class RNNResetStates(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()


class EarlyStoppingMin(EarlyStopping):
    """
    Extends the keras.callbacks.EarlyStopping class to provide the option to force training for a minimum number of
    epochs.
    """
    def __init__(self, min_epochs=0, **kwargs):
        """
        :param min_epochs: int: train the network for at least this number of epochs before early stopping
        :param kwargs: passed to EarlyStopping.__init__()
        """
        super(EarlyStoppingMin, self).__init__(**kwargs)
        if not isinstance(min_epochs, int) or min_epochs < 0:
            raise ValueError('min_epochs must be an integer >= 0')
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.min_epochs:
            return

        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)
