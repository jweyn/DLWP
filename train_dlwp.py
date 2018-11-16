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
from DLWP.util import RNNResetStates, save_model
from keras.regularizers import l2
from keras.callbacks import History


#%% Open some data using the Preprocessor wrapper

root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_2000-2009_hgt_500_NH.nc' % root_directory
model_file = '%s/dlwp_2000-2009_hgt_500_NH_CLSTM_16_5_2' % root_directory
epochs = 50
lambda_ = 1.e-3

processor = Preprocessor(None, predictor_file=predictor_file)
processor.open()

n_sample = processor.data.dims['sample']
n_val = int(np.floor(n_sample * 0.2))
train_set, val_set = train_test_split_ind(n_sample, n_val, method='last')


#%%  Build a model and the data generators

dlwp = DLWPNeuralNet(scaler_type=None, scale_targets=False)

# Build the data generators
generator = DataGenerator(dlwp, processor.data.isel(sample=train_set), batch_size=216,
                          convolution=True, add_time_axis=True)
val_generator = DataGenerator(dlwp, processor.data.isel(sample=val_set), batch_size=216,
                              convolution=True, add_time_axis=True)


#%% Compile the model structure with some generator data information

# # Recurrent neural network (set add_time_axis=True for generators)
# layers = (
#     ('SimpleRNN', (int(generator.n_features) // 2,), {
#         'activation': 'tanh',
#         'kernel_regularizer': l2(lambda_),
#         'return_sequences': True,
#         'input_shape': (1, generator.n_features)
#     }),
#     ('Flatten', None, None),
#     ('Dense', (generator.n_features,), {'activation': 'linear'}),
#     ('Reshape', ((1, generator.n_features),), None)
# )

# Convolutional LSTM (set add_time_axis=True and convolution=True for generators)
layers = (
    ('ConvLSTM2D', (16, 5), {
        'strides': 2,
        'activation': 'tanh',
        'kernel_regularizer': l2(lambda_),
        'return_sequences': True,
        'data_format': 'channels_first',
        'input_shape': (1,) + generator.convolution_shape
    }),
    ('Flatten', None, None),
    ('Dense', (generator.n_features,), None),
    ('Reshape', ((1,) + generator.convolution_shape,), None)
)

# # Feed-forward dense neural network
# layers = (
#     ('Dense', (1024,), {
#         'activation': 'tanh',
#         # 'kernel_regularizer': l2(lambda_),
#         'input_shape': (generator.n_features,)
#     }),
#     ('Dense', (1024,), {
#         'activation': 'tanh',
#         # 'kernel_regularizer': l2(lambda_)
#     }),
#     ('Dense', (1024,), {
#         'activation': 'tanh',
#         # 'kernel_regularizer': l2(lambda_)
#     }),
#     ('Dense', (generator.n_features,), {
#         'activation': 'linear',
#         # 'kernel_regularizer': l2(lambda_)
#     }),
# )

dlwp.build_model(layers, loss='mse', optimizer='adam', metrics=['mae'])

#%% Load the scaling and validating data

# Generate the data to fit the scaler and imputer. The generator will by default apply scaling because it is necessary
# to automate its use in the Keras fit_generator method, so disable it when dealing with data to fit the scaler and
# imputer. If using pre-scaled data (i.e., dlwp was initialized with scaler_type=None and the Preprocessor data was
# generated with scale_variables=True), this step becomes academic.
fit_set = train_set[:2]  # Use a much larger value when it matters
p_fit, t_fit = generator.generate(fit_set, scale_and_impute=False)
dlwp.init_fit(p_fit, t_fit)
p_fit, t_fit = (None, None)

# If system memory permits, loading the predictor data greatly increases efficiency when training on GPUs.
# generator.ds.load()

# Load the validation data. Better to load in memory to avoid file read errors while training.
p_val, t_val = val_generator.generate([], scale_and_impute=False)


#%% Train, evaluate, and save the model

# Train and evaluate the model
start_time = time.time()
history = History()
dlwp.fit_generator(generator, epochs=epochs, verbose=1, validation_data=(p_val, t_val),
                   use_multiprocessing=True, callbacks=[history, RNNResetStates()])
end_time = time.time()

# Evaluate the model
score = dlwp.evaluate(p_val, t_val, verbose=0)
print("\nTrain time -- %s seconds --" % (end_time - start_time))
print('Test loss:', score[0])
print('Test mean absolute error:', score[1])

# Save the model
if model_file is not None:
    save_model(dlwp, model_file, history=history)
