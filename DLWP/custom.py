#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Custom Keras and PyTorch classes.
"""

from keras import backend as K
from keras.callbacks import Callback, EarlyStopping
from keras.layers.convolutional import ZeroPadding2D, ZeroPadding3D
from keras.layers.local import LocallyConnected2D
from keras.utils import conv_utils
from keras.engine.base_layer import InputSpec
import numpy as np

try:
    from s2cnn import S2Convolution, SO3Convolution
except ImportError:
    pass


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


class RunHistory(Callback):
    """Callback that records events into a `History` object.

    Adapted from keras.callbacks.History to include logging to Azure experiment runs.
    """

    def __init__(self, run):
        self.epoch = []
        self.history = {}
        self.run = run

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            self.run.log(k, v)


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

    This layer can add periodic rows and columns
    at the top, bottom, left and right side of an image tensor.

    Adapted from keras.layers.ZeroPadding2D by @jweyn

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
        if K.backend() == 'plaidml.keras.backend':
            shape = inputs.shape.dims
        else:
            shape = inputs.shape
        if self.data_format == 'channels_first':
            top_slice = slice(shape[2] - self.padding[0][0], shape[2])
            bottom_slice = slice(0, self.padding[0][1])
            left_slice = slice(shape[3] - self.padding[1][0], shape[3])
            right_slice = slice(0, self.padding[1][1])
            # Pad the horizontal
            outputs = K.concatenate([inputs[:, :, :, left_slice], inputs, inputs[:, :, :, right_slice]], axis=3)
            # Pad the vertical
            outputs = K.concatenate([outputs[:, :, top_slice], outputs, outputs[:, :, bottom_slice]], axis=2)
        else:
            top_slice = slice(shape[1] - self.padding[0][0], shape[1])
            bottom_slice = slice(0, self.padding[0][1])
            left_slice = slice(shape[2] - self.padding[1][0], shape[2])
            right_slice = slice(0, self.padding[1][1])
            # Pad the horizontal
            outputs = K.concatenate([inputs[:, :, left_slice], inputs, inputs[:, :, right_slice]], axis=2)
            # Pad the vertical
            outputs = K.concatenate([outputs[:, top_slice], outputs, outputs[:, bottom_slice]], axis=1)
        return outputs


class PeriodicPadding3D(ZeroPadding3D):
    """Zero-padding layer for 3D data (spatial or spatio-temporal).

    # Arguments
        padding: int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to height and width.
            - If tuple of 3 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
            - If tuple of 3 tuples of 2 ints:
                interpreted as
                `((left_dim1_pad, right_dim1_pad),
                  (left_dim2_pad, right_dim2_pad),
                  (left_dim3_pad, right_dim3_pad))`
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,
              depth)`
        - If `data_format` is `"channels_first"`:
            `(batch, depth,
              first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`

    # Output shape
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad,
              depth)`
        - If `data_format` is `"channels_first"`:
            `(batch, depth,
              first_padded_axis, second_padded_axis, third_axis_to_pad)`
    """

    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(PeriodicPadding3D, self).__init__(padding=padding,
                                                data_format=data_format,
                                                **kwargs)

    def call(self, inputs):
        if K.backend() == 'plaidml.keras.backend':
            shape = inputs.shape.dims
        else:
            shape = inputs.shape
        if self.data_format == 'channels_first':
            low_slice = slice(shape[2] - self.padding[0][0], shape[2])
            high_slice = slice(0, self.padding[0][1])
            top_slice = slice(shape[3] - self.padding[1][0], shape[3])
            bottom_slice = slice(0, self.padding[1][1])
            left_slice = slice(shape[4] - self.padding[2][0], shape[4])
            right_slice = slice(0, self.padding[2][1])
            # Pad the horizontal
            outputs = K.concatenate([inputs[:, :, :, :, left_slice], inputs, inputs[:, :, :, :, right_slice]], axis=4)
            # Pad the vertical
            outputs = K.concatenate([outputs[:, :, :, top_slice], outputs, outputs[:, :, :, bottom_slice]], axis=3)
            # Pad the depth
            outputs = K.concatenate([outputs[:, :, low_slice], outputs, outputs[:, :, high_slice]], axis=2)
        else:
            low_slice = slice(shape[1] - self.padding[0][0], shape[1])
            high_slice = slice(0, self.padding[0][1])
            top_slice = slice(shape[2] - self.padding[1][0], shape[2])
            bottom_slice = slice(0, self.padding[1][1])
            left_slice = slice(shape[3] - self.padding[2][0], shape[3])
            right_slice = slice(0, self.padding[2][1])
            # Pad the horizontal
            outputs = K.concatenate([inputs[:, :, :, left_slice], inputs, inputs[:, :, :, right_slice]], axis=3)
            # Pad the vertical
            outputs = K.concatenate([outputs[:, :, top_slice], outputs, outputs[:, :, bottom_slice]], axis=2)
            # Pad the depth
            outputs = K.concatenate([outputs[:, low_slice], outputs, outputs[:, high_slice]], axis=1)
        return outputs


class RowConnected2D(LocallyConnected2D):
    """Row-connected layer for 2D inputs.

    The `RowConnected2D` layer works similarly
    to the `Conv2D` layer, except that weights are shared only along rows,
    that is, a different set of filters is applied at each
    different row of the input.

    Adapted from keras.layers.local.LocallyConnected2D by @jweyn

    # Examples
    ```python
        # apply a 3x3 unshared weights convolution with 64 output filters
        # on a 32x32 image with `data_format="channels_last"`:
        model = Sequential()
        model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
        # now model.output_shape == (None, 30, 30, 64)
        # notice that this layer will consume (30*30)*(3*3*3*64)
        # + (30*30)*64 parameters

        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected2D(32, (3, 3)))
        # now model.output_shape == (None, 28, 28, 32)
    ```

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: Currently only support `"valid"` (case-insensitive).
            `"same"` will be supported in future.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, *args, **kwargs):
        super(RowConnected2D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_last':
            input_row, input_col = input_shape[1:-1]
            input_filter = input_shape[3]
        else:
            input_row, input_col = input_shape[2:]
            input_filter = input_shape[1]
        if input_row is None or input_col is None:
            raise ValueError('The spatial dimensions of the inputs to '
                             ' a LocallyConnected2D layer '
                             'should be fully-defined, but layer received '
                             'the inputs shape ' + str(input_shape))
        output_row = conv_utils.conv_output_length(input_row, self.kernel_size[0],
                                                   self.padding, self.strides[0])
        output_col = conv_utils.conv_output_length(input_col, self.kernel_size[1],
                                                   self.padding, self.strides[1])
        self.output_row = output_row
        self.output_col = output_col
        self.kernel_shape = (
            output_row,
            self.kernel_size[0],
            self.kernel_size[1],
            input_filter,
            self.filters)
        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(output_row, 1, self.filters),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        if self.data_format == 'channels_first':
            self.input_spec = InputSpec(ndim=4, axes={1: input_filter})
        else:
            self.input_spec = InputSpec(ndim=4, axes={-1: input_filter})
        self.built = True

    def call(self, inputs):
        output = row_conv2d(inputs,
                            self.kernel,
                            self.kernel_size,
                            self.strides,
                            (self.output_row, self.output_col),
                            self.data_format)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        output = self.activation(output)
        return output


def row_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None):
    """Apply 2D conv with weights shared only along rows.

    Adapted from K.local_conv2d by @jweyn

    # Arguments
        inputs: 4D tensor with shape:
                (batch_size, filters, new_rows, new_cols)
                if data_format='channels_first'
                or 4D tensor with shape:
                (batch_size, new_rows, new_cols, filters)
                if data_format='channels_last'.
        kernel: the row-shared weights for convolution,
                with shape (output_rows, kernel_size, input_channels, filters)
        kernel_size: a tuple of 2 integers, specifying the
                     width and height of the 2D convolution window.
        strides: a tuple of 2 integers, specifying the strides
                 of the convolution along the width and height.
        output_shape: a tuple with (output_row, output_col)
        data_format: the data format, channels_first or channels_last

    # Returns
        A 4d tensor with shape:
        (batch_size, filters, new_rows, new_cols)
        if data_format='channels_first'
        or 4D tensor with shape:
        (batch_size, new_rows, new_cols, filters)
        if data_format='channels_last'.

    # Raises
        ValueError: if `data_format` is neither
                    `channels_last` or `channels_first`.
    """
    data_format = K.normalize_data_format(data_format)

    stride_row, stride_col = strides
    output_row, output_col = output_shape

    out = []
    for i in range(output_row):
        # Slice the rows with the neighbors they need
        slice_row = slice(i * stride_row, i * stride_col + kernel_size[0])
        if data_format == 'channels_first':
            x = inputs[:, :, slice_row, :]  # batch, 16, 5, 144
        else:
            x = inputs[:, slice_row, :, :]  # batch, 5, 144, 16
        # Convolve, resulting in an array with only one row: batch, 1, 140, 6 or batch, 6, 1, 140
        x = K.conv2d(x, kernel[i], strides=strides, padding='valid', data_format=data_format)
        out.append(x)

    if data_format == 'channels_first':
        output = K.concatenate(out, axis=2)
    else:
        output = K.concatenate(out, axis=1)
    del x
    del out
    return output


class LatitudeWeightedLoss(object):
    """
    Class to create a weighted latitude-dependent loss function for a Keras model.
    """
    def __init__(self, loss_function, lats, data_format='channels_last', weighting='cosine'):
        """
        Initialize a weighted loss.

        :param loss_function: method: Keras loss function to apply after the weighting
        :param lats: ndarray: 1-dimensional array of latitude coordinates
        :param data_format: Keras data_format ('channels_first' or 'channels_last')
        :param weighting: str: type of weighting to apply. Options are:
            cosine: weight by the cosine of the latitude (default)
            midlatitude: weight by the cosine of the latitude but also apply a 25% reduction to the equator and boost
                to the mid-latitudes
        """
        self.loss_function = loss_function
        self.lats = lats
        self.data_format = K.normalize_data_format(data_format)
        if weighting not in ['cosine', 'midlatitude']:
            raise ValueError("'weighting' must be one of 'cosine' or 'midlatitude'")
        self.weighting = weighting
        lat_tensor = K.zeros(lats.shape)
        print(lats)
        lat_tensor.assign(K.cast_to_floatx(lats[:]))
        self.weights = K.cos(lat_tensor * np.pi / 180.)
        if self.weighting == 'midlatitude':
            self.weights = self.weights - 0.25 * K.sin(lat_tensor * 2 * np.pi / 180.)
        self.is_init = False

        self.__name__ = 'latitude_weighted_loss'

    def init_weights(self, shape):
        if shape[-1] is None:
            return
        # Repeat the weights tensor to match the last dimensions of the batch
        if self.data_format == 'channels_last':
            self.weights = K.expand_dims(self.weights, axis=1)
            self.weights = K.repeat_elements(self.weights, shape[-1], axis=1)
        else:
            self.weights = K.expand_dims(self.weights, axis=1)
            self.weights = K.repeat_elements(self.weights, shape[-2], axis=1)
            self.weights = K.expand_dims(self.weights, axis=2)
            self.weights = K.repeat_elements(self.weights, shape[-1], axis=2)
        self.is_init = True

    def __call__(self, y_true, y_pred):
        # Check that the weights array has been initialized to fit the dimensions
        if not self.is_init:
            self.init_weights(K.int_shape(y_true))
        if self.is_init:
            loss = self.loss_function(y_true * self.weights, y_pred * self.weights)
        else:
            loss = self.loss_function(y_true, y_pred)
        return loss


def latitude_weighted_loss(loss_function, lats, output_shape, axis=-2, weighting='cosine'):
    """
    Create a loss function that weights inputs by a function of latitude before calculating the loss.

    :param loss_function: method: Keras loss function to apply after the weighting
    :param lats: ndarray: 1-dimensional array of latitude coordinates
    :param output_shape: tuple: shape of expected model output
    :param axis: int: latitude axis in model output shape
    :param weighting: str: type of weighting to apply. Options are:
            cosine: weight by the cosine of the latitude (default)
            midlatitude: weight by the cosine of the latitude but also apply a 25% reduction to the equator and boost
                to the mid-latitudes
    :return: callable loss function
    """
    if weighting not in ['cosine', 'midlatitude']:
        raise ValueError("'weighting' must be one of 'cosine' or 'midlatitude'")
    lat_tensor = K.zeros(lats.shape)
    lat_tensor.assign(K.cast_to_floatx(lats[:]))

    weights = K.cos(lat_tensor * np.pi / 180.)
    if weighting == 'midlatitude':
        weights = weights - 0.25 * K.sin(lat_tensor * 2 * np.pi / 180.)

    weight_shape = output_shape[axis:]
    for d in weight_shape[1:]:
        weights = K.expand_dims(weights, axis=-1)
        weights = K.repeat_elements(weights, d, axis=-1)

    def loss(y_true, y_pred):
        return loss_function(y_true * weights, y_pred * weights)

    return loss


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
