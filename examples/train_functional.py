#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Example of training a DLWP model with the Keras functional API.
"""

import time
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from DLWP.model import DLWPFunctional, SeriesDataGenerator
from DLWP.util import save_model, train_test_split_ind
from keras.regularizers import l2
from keras.losses import mean_squared_error
from keras.callbacks import History, TensorBoard

from keras.layers import Input, ZeroPadding2D, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from DLWP.custom import PeriodicPadding2D, RNNResetStates, EarlyStoppingMin, slice_layer
from keras.models import Model


#%% Parameters

# File paths and names
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = os.path.join(root_directory, 'cfs_6h_1979-2010_z500-th3-7-w700-rh850-pwat_NH_T2.nc')
model_file = os.path.join(root_directory, 'dlwp_6h_z500-th3-7_NH_T2_seq6')
log_directory = os.path.join(root_directory, 'logs', 'seq')

# NN parameters. Regularization is applied to LSTM layers by default. weight_loss indicates whether to weight the
# loss function preferentially in the mid-latitudes.
model_is_convolutional = True
model_is_recurrent = False
min_epochs = 200
max_epochs = 1000
patience = 50
batch_size = 64
lambda_ = 1.e-4
weight_loss = False
loss_by_step = None
shuffle = True
skip_connections = True

# Data parameters. Specify the input/output variables/levels and input/output time steps. DLWPFunctional requires that
# the inputs and outputs match exactly (for now). Ensure that the selections use LISTS of values (even for only 1) to
# keep dimensions correct. The number of output iterations to train on is given by integration_steps. The actual number
# of forecast steps (units of model delta t) is io_time_steps * integration_steps.
io_selection = {'varlev': ['HGT/500', 'THICK/300-700']}
io_time_steps = 2
integration_steps = 6
# Option to crop the north pole. Necessary for getting an even number of latitudes for up-sampling layers.
crop_north_pole = True
# Add incoming solar radiation forcing
add_solar = False

# If system memory permits, loading the predictor data can greatly increase efficiency when training on GPUs, if the
# train computation takes less time than the data loading.
load_memory = True

# Use multiple GPUs, if available
n_gpu = 1

# Force use of the keras model.fit() method. May run faster in some instances, but uses (input_time_steps +
# output_time_steps) times more memory.
use_keras_fit = False

# Validation set to use. Either an integer (number of validation samples, taken from the end), or an iterable of
# pandas datetime objects. The train set can be set to the first <integer> samples, an iterable of dates, or None to
# simply use the remaining points. Match the type of validation_set.
validation_set = list(pd.date_range(datetime(2003, 1, 1, 0), datetime(2006, 12, 31, 18), freq='6H'))
train_set = list(pd.date_range(datetime(1979, 1, 1, 6), datetime(1982, 12, 31, 18), freq='6H'))


#%% Open data

data = xr.open_dataset(predictor_file, chunks={'sample': batch_size})

if 'time_step' in data.dims:
    time_dim = data.dims['time_step']
else:
    time_dim = 1
n_sample = data.dims['sample']

if crop_north_pole:
    data = data.isel(lat=(data.lat < 90.0))


#%% Create a model and the data generators

dlwp = DLWPFunctional(is_convolutional=model_is_convolutional, is_recurrent=model_is_recurrent, time_dim=io_time_steps)

# Find the validation set
if isinstance(validation_set, int):
    n_sample = data.dims['sample']
    ts, val_set = train_test_split_ind(n_sample, validation_set, method='last')
    if train_set is None:
        train_set = ts
    elif isinstance(train_set, int):
        train_set = list(range(train_set))
    validation_data = data.isel(sample=val_set)
    train_data = data.isel(sample=train_set)
elif validation_set is None:
    if train_set is None:
        train_set = data.sample.values
    validation_data = None
    train_data = data.sel(sample=train_set)
else:  # we must have a list of datetimes
    if train_set is None:
        train_set = np.isin(data.sample.values, np.array(validation_set, dtype='datetime64[ns]'),
                            assume_unique=True, invert=True)
    validation_data = data.sel(sample=validation_set)
    train_data = data.sel(sample=train_set)

# Build the data generators
if load_memory or use_keras_fit:
    print('Loading data to memory...')
generator = SeriesDataGenerator(dlwp, train_data, input_sel=io_selection, output_sel=io_selection,
                                input_time_steps=io_time_steps, output_time_steps=io_time_steps,
                                sequence=integration_steps, add_insolation=add_solar,
                                batch_size=batch_size, load=load_memory, shuffle=shuffle)
if use_keras_fit:
    p_train, t_train = generator.generate([])
if validation_data is not None:
    val_generator = SeriesDataGenerator(dlwp, validation_data, input_sel=io_selection, output_sel=io_selection,
                                        input_time_steps=io_time_steps, output_time_steps=io_time_steps,
                                        sequence=integration_steps, add_insolation=add_solar,
                                        batch_size=batch_size, load=load_memory)
    if use_keras_fit:
        val = val_generator.generate([])
else:
    val_generator = None
    if use_keras_fit:
        val = None


#%% Compile the model structure with some generator data information

# Up-sampling convolutional network with LSTM layer
cs = generator.convolution_shape
cso = generator.output_convolution_shape

# Convolutional NN
input_0 = Input(shape=cs, name='input_0')
periodic_padding_2 = PeriodicPadding2D(padding=(0, 2), data_format='channels_first')
zero_padding_2 = ZeroPadding2D(padding=(2, 0), data_format='channels_first')
periodic_padding_1 = PeriodicPadding2D(padding=(0, 1), data_format='channels_first')
zero_padding_1 = ZeroPadding2D(padding=(1, 0), data_format='channels_first')
max_pooling_2 = MaxPooling2D(2, data_format='channels_first')
up_sampling_2 = UpSampling2D(2, data_format='channels_first')
conv_2d_1 = Conv2D(32, 3, **{
        'dilation_rate': 2,
        'padding': 'valid',
        'activation': 'tanh',
        'data_format': 'channels_first'
    })
conv_2d_2 = Conv2D(64, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'tanh',
        'data_format': 'channels_first'
    })
conv_2d_3 = Conv2D(128, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'tanh',
        'data_format': 'channels_first'
    })
conv_2d_4 = Conv2D(32 if skip_connections else 64, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'tanh',
        'data_format': 'channels_first'
    })
conv_2d_5 = Conv2D(16 if skip_connections else 32, 3, **{
        'dilation_rate': 2,
        'padding': 'valid',
        'activation': 'tanh',
        'data_format': 'channels_first'
    })
conv_2d_6 = Conv2D(cso[0], 5, **{
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
split_1_1 = slice_layer(0, 16, axis=1)
split_1_2 = slice_layer(16, 32, axis=1)
split_2_1 = slice_layer(0, 32, axis=1)
split_2_2 = slice_layer(32, 64, axis=1)


def basic_model(x):
    x = periodic_padding_2(zero_padding_2(x))
    x = conv_2d_1(x)
    x = max_pooling_2(x)
    x = periodic_padding_1(zero_padding_1(x))
    x = conv_2d_2(x)
    x = max_pooling_2(x)
    x = periodic_padding_1(zero_padding_1(x))
    x = conv_2d_3(x)
    x = up_sampling_2(x)
    x = periodic_padding_1(zero_padding_1(x))
    x = conv_2d_4(x)
    x = up_sampling_2(x)
    x = periodic_padding_2(zero_padding_2(x))
    x = conv_2d_5(x)
    x = periodic_padding_2(zero_padding_2(x))
    x = conv_2d_6(x)
    return x


def skip_model(x):
    x = periodic_padding_2(zero_padding_2(x))
    x = conv_2d_1(x)
    x, x1 = split_1_1(x), split_1_2(x)
    x = max_pooling_2(x)
    x = periodic_padding_1(zero_padding_1(x))
    x = conv_2d_2(x)
    x, x2 = split_2_1(x), split_2_2(x)
    x = max_pooling_2(x)
    x = periodic_padding_1(zero_padding_1(x))
    x = conv_2d_3(x)
    x = up_sampling_2(x)
    x = periodic_padding_1(zero_padding_1(x))
    x = conv_2d_4(x)
    x = concatenate([x, x2], axis=1)
    x = up_sampling_2(x)
    x = periodic_padding_2(zero_padding_2(x))
    x = conv_2d_5(x)
    x = concatenate([x, x1], axis=1)
    x = periodic_padding_2(zero_padding_2(x))
    x = conv_2d_6(x)
    return x


model_function = skip_model if skip_connections else basic_model
outputs = [model_function(input_0)]
for o in range(1, integration_steps):
    outputs.append(model_function(outputs[o-1]))

if loss_by_step is None:
    loss_by_step = [1./integration_steps] * integration_steps
model = Model(inputs=input_0, outputs=outputs)

# Build the DLWP model
dlwp.build_model(model, loss='mse', loss_weights=loss_by_step, optimizer='adam', metrics=['mae'], gpus=n_gpu)
print(dlwp.base_model.summary())


#%% Train, evaluate, and save the model

# Train and evaluate the model
start_time = time.time()
print('Begin training...')
# run = Run.get_context()
history = History()
early = EarlyStoppingMin(min_epochs=min_epochs, monitor='val_loss' if val_generator is not None else 'loss',
                         min_delta=0., patience=patience, restore_best_weights=True, verbose=1)
tensorboard = TensorBoard(log_dir=log_directory, batch_size=batch_size, update_freq='epoch')

if use_keras_fit:
    dlwp.fit(p_train, t_train, batch_size=batch_size, epochs=max_epochs, verbose=1, validation_data=val,
             callbacks=[history, RNNResetStates(), early])
else:
    dlwp.fit_generator(generator, epochs=max_epochs, verbose=1, validation_data=val_generator,
                       use_multiprocessing=True, callbacks=[history, RNNResetStates(), early])
end_time = time.time()

# Save the model
if model_file is not None:
    save_model(dlwp, model_file, history=history)

# Evaluate the model
print("\nTrain time -- %s seconds --" % (end_time - start_time))
if validation_data is not None:
    score = dlwp.evaluate(*val_generator.generate([]), verbose=0)
    print('Validation loss:', score[0])
    print('Validation mean absolute error:', score[1])
