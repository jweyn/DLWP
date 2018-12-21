#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Example of training a DLWP model using a dataset of predictors generated with DLWP.model.Preprocessor.
"""

import time
import numpy as np
from DLWP.model import DLWPNeuralNet, DataGenerator, Preprocessor
from DLWP.model.preprocessing import train_test_split_ind
from DLWP.util import save_model
from DLWP.custom import RNNResetStates, EarlyStoppingMin
from keras.regularizers import l2
from keras.callbacks import History


#%% Open some data using the Preprocessor wrapper

root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_1979-2010_hgt_500_NH_T2.nc' % root_directory
model_file = '%s/dlwp_1979-2010_hgt_500_NH_T2F_CLSTM_16_5_2_PBC' % root_directory
model_is_convolutional = True
model_is_recurrent = True
min_epochs = 75
max_epochs = 200
batch_size = 216
lambda_ = 1.e-3

processor = Preprocessor(None, predictor_file=predictor_file)
processor.open()
if 'time_step' in processor.data.dims:
    time_dim = processor.data.dims['time_step']
else:
    time_dim = 1

n_sample = processor.data.dims['sample']
# If validation set must be a multiple of batch_size (i.e., stateful RNN):
# n_val = int(np.floor(n_sample * 0.2)) // batch_size * batch_size
# Fixed last 8 years, 4x daily
n_val = 4 * (365 * 8 + 2)
train_set, val_set = train_test_split_ind(n_sample, n_val, method='last')


#%% Build a model and the data generators

dlwp = DLWPNeuralNet(is_convolutional=model_is_convolutional, is_recurrent=model_is_recurrent, time_dim=time_dim,
                     scaler_type=None, scale_targets=False)

# Build the data generators
generator = DataGenerator(dlwp, processor.data.isel(sample=train_set), batch_size=batch_size)
val_generator = DataGenerator(dlwp, processor.data.isel(sample=val_set), batch_size=batch_size)


#%% Compile the model structure with some generator data information

# Convolutional LSTM
layers = (
    ('Reshape', (generator.shape_2d,), {'input_shape': generator.convolution_shape}),
    ('PeriodicPadding2D', ((0, 2),), {'data_format': 'channels_first'}),
    ('ZeroPadding2D', ((2, 0),), {'data_format': 'channels_first'}),
    ('Reshape', ((2, 1, 41, 148),), None),
    ('ConvLSTM2D', (16, 5), {
        'strides': 2,
        'padding': 'valid',
        'data_format': 'channels_first',
        'activation': 'tanh',
        'kernel_regularizer': l2(lambda_),
        'return_sequences': False,
        'input_shape': generator.convolution_shape
        # 'stateful': True,
        # 'batch_input_shape': (batch_size, 1,) + generator.convolution_shape,
    }),
    # ('Reshape', ((32, 37, 144),), None),
    # ('PeriodicPadding2D', ((0, 2),), {'data_format': 'channels_first'}),
    # ('ZeroPadding2D', ((2, 0),), {'data_format': 'channels_first'}),
    # ('RowConnected2D', (2, 5), {
    #     'strides': 1,
    #     'padding': 'valid',
    #     'data_format': 'channels_first',
    #     'activation': 'linear',
    # }),
    ('Flatten', None, None),
    ('Dense', (generator.n_features,), {'activation': 'linear'}),
    ('Reshape', (generator.convolution_shape,), None)
)

# # Convolutional feed-forward (set convolution=True for generators)
# layers = (
#     ('PeriodicPadding2D', ((0, 5),), {
#             'data_format': 'channels_first',
#             'input_shape': generator.convolution_shape,
#         }),
#     ('ZeroPadding2D', ((2, 0),), {'data_format': 'channels_first'}),
#     ('Conv2D', (64, 5), {
#         'strides': 1,
#         'padding': 'valid',
#         'activation': 'tanh',
#         'kernel_regularizer': l2(lambda_),
#         'data_format': 'channels_first',
#         # 'input_shape': generator.convolution_shape
#     }),
#     ('ZeroPadding2D', ((1, 0),), {'data_format': 'channels_first'}),
#     ('Conv2D', (128, 3), {
#         'strides': 1,
#         'padding': 'valid',
#         'activation': 'tanh',
#         'kernel_regularizer': l2(lambda_),
#         'data_format': 'channels_first',
#     }),
#     ('ZeroPadding2D', ((2, 0),), {'data_format': 'channels_first'}),
#     ('Conv2D', (6, 5), {
#         'strides': 1,
#         'padding': 'valid',
#         'activation': 'tanh',
#         'kernel_regularizer': l2(lambda_),
#         'data_format': 'channels_first',
#     }),
# )

# # Feed-forward dense neural network
# layers = (
#     ('Dense', (generator.n_features // 2,), {
#         'activation': 'tanh',
#         'kernel_regularizer': l2(lambda_),
#         'input_shape': (generator.n_features,)
#     }),
#     ('Dropout', (0.25,), None),
#     ('Dense', (generator.n_features,), {'activation': 'linear'})
# )

dlwp.build_model(layers, loss='mse', optimizer='adam', metrics=['mae'])
print(dlwp.model.summary())


#%% Load the scaling and validating data

# Generate the data to fit the scaler and imputer. The generator will by default apply scaling because it is necessary
# to automate its use in the Keras fit_generator method, so disable it when dealing with data to fit the scaler and
# imputer. If using pre-scaled data (i.e., dlwp was initialized with scaler_type=None and the Preprocessor data was
# generated with scale_variables=True), this step should be omitted.
fit_set = train_set[:2]  # Use a much larger value when it matters
p_fit, t_fit = generator.generate(fit_set, scale_and_impute=False)
dlwp.init_fit(p_fit, t_fit)
p_fit, t_fit = (None, None)

# If system memory permits, loading the predictor data can greatly increases efficiency when training on GPUs, if the
# train computation takes less time than the data loading.
# generator.ds.load()

# Load the validation data. Better to load in memory to avoid file read errors while training.
p_val, t_val = val_generator.generate([], scale_and_impute=False)


#%% Train, evaluate, and save the model

# Train and evaluate the model
start_time = time.time()
history = History()
early = EarlyStoppingMin(min_epochs=min_epochs, monitor='val_mean_absolute_error', min_delta=0., patience=20,
                         restore_best_weights=True, verbose=1)
dlwp.fit_generator(generator, epochs=max_epochs, verbose=1, validation_data=(p_val, t_val), use_multiprocessing=True,
                   callbacks=[history, RNNResetStates(), early])
end_time = time.time()

# Evaluate the model
score = dlwp.evaluate(p_val, t_val, verbose=0)
print("\nTrain time -- %s seconds --" % (end_time - start_time))
print('Test loss:', score[0])
print('Test mean absolute error:', score[1])

# Save the model
if model_file is not None:
    save_model(dlwp, model_file, history=history)
