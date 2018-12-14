#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Custom Keras and PyTorch classes.
"""

from keras.callbacks import Callback, EarlyStopping
from keras import backend as K
from keras.layers.convolutional import ZeroPadding2D
import tensorflow as tf


# ==================================================================================================================== #
# Keras classes
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


class PeriodicPadding2D(ZeroPadding2D):
    """Periodic-padding layer for 2D input (e.g. image).
    Modified from keras.layers.ZeroPadding2D by @jweyn

    This layer can add periodic rows and columns
    at the top, bottom, left and right side of an image tensor.

    # Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to height and width.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`

    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, padded_rows, padded_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, padded_rows, padded_cols)`
    """

    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(PeriodicPadding2D, self).__init__(padding=padding,
                                                data_format=data_format,
                                                **kwargs)

    def call(self, inputs):
        if self.data_format == 'channels_first':
            top_slice = slice(inputs.shape[2] - self.padding[0][0], inputs.shape[2])
            bottom_slice = slice(0, self.padding[0][1])
            left_slice = slice(inputs.shape[3] - self.padding[1][0], inputs.shape[3])
            right_slice = slice(0, self.padding[1][1])
            # Pad the horizontal
            outputs = tf.concat([inputs[:, :, :, left_slice], inputs, inputs[:, :, :, right_slice]], axis=3)
            # Pad the vertical
            outputs = tf.concat([outputs[:, :, top_slice], outputs, outputs[:, :, bottom_slice]], axis=2)
        else:
            top_slice = slice(inputs.shape[1] - self.padding[0][0], inputs.shape[1])
            bottom_slice = slice(0, self.padding[0][1])
            left_slice = slice(inputs.shape[2] - self.padding[1][0], inputs.shape[2])
            right_slice = slice(0, self.padding[1][1])
            # Pad the horizontal
            outputs = tf.concat([inputs[:, :, left_slice], inputs, inputs[:, :, right_slice]], axis=2)
            # Pad the vertical
            outputs = tf.concat([outputs[:, top_slice], outputs, outputs[:, bottom_slice]], axis=1)
        return outputs


# ==================================================================================================================== #
# PyTorch classes
# ==================================================================================================================== #

class TorchReshape(object):
    def __init__(self, shape):
        if not isinstance(shape, tuple):
            raise ValueError("'shape' must be a tuple of integers")
        self.shape = shape

    def __call__(self, x):
        return x.view(*self.shape)

