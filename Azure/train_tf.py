#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Example of training a DLWP model using a dataset of predictors generated with DLWP.model.Preprocessor.

Uses Microsoft Azure resources. Launch this script as an Azure experiment using 'Train on Azure.ipynb'
"""

import argparse
import os
import shutil
import time
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from DLWP.model import DLWPNeuralNet, SeriesDataGenerator
from DLWP.util import save_model, train_test_split_ind
from DLWP.custom import RNNResetStates, EarlyStoppingMin, latitude_weighted_loss, RunHistory
from keras.regularizers import l2
from keras.losses import mean_squared_error
from keras.callbacks import TensorBoard
from azureml.core import Run


#%% Parse user arguments

parser = argparse.ArgumentParser()
parser.add_argument('--root-directory', type=str, dest='root_directory', default='.',
                    help='Destination root data directory on Azure Blob storage')
parser.add_argument('--predictor-file', type=str, dest='predictor_file',
                    help='Path and name of data file in root-directory')
parser.add_argument('--model-file', type=str, dest='model_file',
                    help='Path and name of model save file in root-directory')
parser.add_argument('--log-directory', type=str, dest='log_directory', default='./logs',
                    help='Destination for log files in root-directory')
parser.add_argument('--temp-dir', type=str, dest='temp_dir', default='None',
                    help='If specified, copies the predictor file here for use during training (e.g., fast SSD)')

args = parser.parse_args()
if args.temp_dir != 'None':
    os.makedirs(args.temp_dir, exist_ok=True)


#%% Parameters

root_directory = args.root_directory
predictor_file = os.path.join(root_directory, args.predictor_file)
model_file = os.path.join(root_directory, args.model_file)
log_directory = os.path.join(root_directory, args.log_directory)

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

# Data parameters. Specify the input variables/levels, output variables/levels, and time steps in/out. Note that for
# LSTM layers, the model can only predict effectively if the output time steps is 1 or equal to the input time steps.
# Ensure that the selections use LISTS of values (even for only 1) to keep dimensions correct.
input_selection = {'varlev': ['HGT/500', 'THICK/300-700']}
output_selection = {'varlev': ['HGT/500', 'THICK/300-700']}
input_time_steps = 2
output_time_steps = 2
# Option to crop the north pole. Necessary for getting an even number of latitudes for up-sampling layers.
crop_north_pole = True
# Add incoming solar radiation forcing
add_solar = False

# If system memory permits, loading the predictor data can greatly increase efficiency when training on GPUs, if the
# train computation takes less time than the data loading.
load_memory = False

# Use multiple GPUs, if available
n_gpu = 2

# Force use of the keras model.fit() method. Currently investigating why model.fit() produces better weights than
# model.fit_generator(). Note that this uses 2 * input_time_steps * output_time_steps times more memory.
use_keras_fit = True

# Validation set to use. Either an integer (number of validation samples, taken from the end), or an iterable of
# pandas datetime objects. The train set can be set to the first <integer> samples, an iterable of dates, or None to
# simply use the remaining points. Match the type of validation_set.
validation_set = list(pd.date_range(datetime(2003, 1, 1, 0), datetime(2006, 12, 31, 18), freq='6H'))
train_set = list(pd.date_range(datetime(1979, 1, 1, 6), datetime(2002, 12, 31, 18), freq='6H'))


#%% Open data. If temporary file is specified, copy it there.

if args.temp_dir != 'None':
    new_predictor_file = os.path.join(args.temp_dir, args.predictor_file)
    print('Copying predictor file to %s...' % new_predictor_file)
    if os.path.isfile(new_predictor_file):
        print('File already exists!')
    else:
        shutil.copy(predictor_file, new_predictor_file, follow_symlinks=True)
    data = xr.open_dataset(new_predictor_file, chunks={'sample': batch_size})
else:
    data = xr.open_dataset(predictor_file, chunks={'sample': batch_size})

if 'time_step' in data.dims:
    time_dim = data.dims['time_step']
else:
    time_dim = 1
n_sample = data.dims['sample']

if crop_north_pole:
    data = data.isel(lat=(data.lat < 90.0))


#%% Build a model and the data generators

dlwp = DLWPNeuralNet(is_convolutional=model_is_convolutional, is_recurrent=model_is_recurrent, time_dim=time_dim,
                     scaler_type=None, scale_targets=False)

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

# For multiple GPUs, increase the batch size
batch_size = n_gpu * batch_size

# Build the data generators
if load_memory:
    print('Loading data to memory...')
generator = SeriesDataGenerator(dlwp, train_data, input_sel=input_selection, output_sel=output_selection,
                                input_time_steps=input_time_steps, output_time_steps=output_time_steps,
                                batch_size=batch_size, add_insolation=add_solar, load=load_memory)
p_train, t_train = generator.generate([])
if validation_data is not None:
    val_generator = SeriesDataGenerator(dlwp, validation_data, input_sel=input_selection, output_sel=output_selection,
                                        input_time_steps=input_time_steps, output_time_steps=output_time_steps,
                                        batch_size=batch_size, add_insolation=add_solar, load=load_memory)
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
layers = (
    # --- These layers add a convolutional LSTM at the beginning --- #
    # ('Reshape', (generator.shape_2d,), {'input_shape': cs}),
    # ('PeriodicPadding2D', ((0, 2),), {'data_format': 'channels_first'}),
    # ('ZeroPadding2D', ((2, 0),), {'data_format': 'channels_first'}),
    # ('Reshape', ((cs[0], cs[1], cs[2] + 4, cs[3] + 4),), None),
    # ('ConvLSTM2D', (4 * cs[1], 3), {  # 4 * cs[1]
    #     'dilation_rate': 2,
    #     'padding': 'valid',
    #     'data_format': 'channels_first',
    #     'activation': 'tanh',
    #     'return_sequences': True,
    #     'kernel_regularizer': l2(lambda_)
    # }),
    # ('Reshape', ((4 * cs[1] * cs[0], cs[2], cs[3]),), None),  # 4 * cs[0] * cs[1]
    # -------------------------------------------------------------- #
    ('PeriodicPadding2D', ((0, 2),), {
        'data_format': 'channels_first',
        'input_shape': cs
    }),
    ('ZeroPadding2D', ((2, 0),), {'data_format': 'channels_first'}),
    ('Conv2D', (32, 3), {
        'dilation_rate': 2,
        'padding': 'valid',
        'activation': 'tanh',
        'data_format': 'channels_first'
    }),
    # ('BatchNormalization', None, {'axis': 1}),
    ('MaxPooling2D', (2,), {'data_format': 'channels_first'}),
    ('PeriodicPadding2D', ((0, 1),), {'data_format': 'channels_first'}),
    ('ZeroPadding2D', ((1, 0),), {'data_format': 'channels_first'}),
    ('Conv2D', (64, 3), {
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'tanh',
        'data_format': 'channels_first'
    }),
    # ('BatchNormalization', None, {'axis': 1}),
    ('MaxPooling2D', (2,), {'data_format': 'channels_first'}),
    ('PeriodicPadding2D', ((0, 1),), {'data_format': 'channels_first'}),
    ('ZeroPadding2D', ((1, 0),), {'data_format': 'channels_first'}),
    ('Conv2D', (128, 3), {
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'tanh',
        'data_format': 'channels_first'
    }),
    # ('BatchNormalization', None, {'axis': 1}),
    ('UpSampling2D', (2,), {'data_format': 'channels_first'}),
    ('PeriodicPadding2D', ((0, 1),), {'data_format': 'channels_first'}),
    ('ZeroPadding2D', ((1, 0),), {'data_format': 'channels_first'}),
    ('Conv2D', (64, 3), {
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'tanh',
        'data_format': 'channels_first'
    }),
    # ('BatchNormalization', None, {'axis': 1}),
    ('UpSampling2D', (2,), {'data_format': 'channels_first'}),
    ('PeriodicPadding2D', ((0, 2),), {'data_format': 'channels_first'}),
    ('ZeroPadding2D', ((2, 0),), {'data_format': 'channels_first'}),
    ('Conv2D', (32, 3), {
        'dilation_rate': 2,
        'padding': 'valid',
        'activation': 'tanh',
        'data_format': 'channels_first'
    }),
    # ('BatchNormalization', None, {'axis': 1}),
    ('PeriodicPadding2D', ((0, 2),), {'data_format': 'channels_first'}),
    ('ZeroPadding2D', ((2, 0),), {'data_format': 'channels_first'}),
    # --- Change the number of filters to cso[0] * cso[1] for LSTM model --- #
    ('Conv2D', (cso[0], 5), {
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    }),
    # ('Reshape', (cso,), None)
)

# Example custom loss function: pass to loss= in build_model()
if weight_loss:
    loss_function = latitude_weighted_loss(mean_squared_error, generator.ds.lat.values, generator.convolution_shape,
                                           axis=-2, weighting='midlatitude')
else:
    loss_function = 'mse'

# Build the model
dlwp.build_model(layers, loss=loss_function, optimizer='adam', metrics=['mae'], gpus=n_gpu)
print(dlwp.model.summary())


#%% Initialize the scaler/imputer if necessary

# Generate the data to fit the scaler and imputer. The generator will by default apply scaling because it is necessary
# to automate its use in the Keras fit_generator method, so disable it when dealing with data to fit the scaler and
# imputer. If using pre-scaled data (i.e., dlwp was initialized with scaler_type=None and the Preprocessor data was
# generated with scale_variables=True), this step should be omitted.
fit_set = list(range(2))  # Use a much larger value when it matters
p_fit, t_fit = generator.generate(fit_set, scale_and_impute=False)
dlwp.init_fit(p_fit, t_fit)
p_fit, t_fit = (None, None)


#%% Train, evaluate, and save the model

# Train and evaluate the model
start_time = time.time()
print('Begin training...')
run = Run.get_context()
history = RunHistory(run)
early = EarlyStoppingMin(min_epochs=min_epochs, monitor='val_loss' if val_generator is not None else 'loss',
                         min_delta=0., patience=patience, restore_best_weights=True, verbose=1)
tensorboard = TensorBoard(log_dir=log_directory, batch_size=batch_size, update_freq='epoch')

if use_keras_fit:
    dlwp.fit(p_train, t_train, batch_size=batch_size, epochs=max_epochs, verbose=2, validation_data=val,
             callbacks=[history, RNNResetStates(), early])
else:
    dlwp.fit_generator(generator, epochs=max_epochs, verbose=2, validation_data=val_generator,
                       use_multiprocessing=True, callbacks=[history, RNNResetStates(), early])
end_time = time.time()

# Evaluate the model
print("\nTrain time -- %s seconds --" % (end_time - start_time))
print('Train loss:', history.history['loss'][-patience - 1])
print('Train mean absolute error:', history.history['mean_absolute_error'][-patience - 1])
run.log('TRAIN_LOSS', history.history['loss'][-patience - 1])
if validation_data is not None:
    score = dlwp.evaluate(*val_generator.generate([]), verbose=0)
    print('Validation loss:', score[0])
    print('Validation mean absolute error:', score[1])
    run.log('VAL_LOSS', score[0])

# Save the model
if model_file is not None:
    save_model(dlwp, model_file, history=history)
