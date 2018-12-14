#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Example of training a DLWP model using a dataset of predictors generated with DLWP.model.Preprocessor. Uses the torch
implementation of DLWPTorchNN.
"""

import time
import numpy as np
from DLWP.model import DLWPTorchNN, DataGenerator, Preprocessor
from DLWP.model.preprocessing import train_test_split_ind
from DLWP.util import save_torch_model


#%% Open some data using the Preprocessor wrapper

root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_1979-2010_hgt-tmp_250-500-1000_NH.nc' % root_directory
model_file = '%s/dlwp_1979-2010_hgt-tmp_250-500-1000_NH_CONV_64_torch' % root_directory
model_is_recurrent = False
min_epochs = 75
max_epochs = 20
batch_size = 216
lambda_ = 1.e-3

processor = Preprocessor(None, predictor_file=predictor_file)
processor.open()

n_sample = processor.data.dims['sample']
# Fixed last 8 years
n_val = 2 * 4 * (365 * 3 + 366)
train_set, val_set = train_test_split_ind(n_sample, n_val, method='last')


#%% Build a model and the data generators

dlwp = DLWPTorchNN(scaler_type=None, scale_targets=False, is_recurrent=model_is_recurrent)

# Build the data generators
generator = DataGenerator(dlwp, processor.data.isel(sample=train_set), batch_size=batch_size, convolution=True)
val_generator = DataGenerator(dlwp, processor.data.isel(sample=val_set), batch_size=batch_size, convolution=True)


#%% Compile the model structure with some generator data information

# Convolutional feed-forward network
layers = (
    ('Conv2d', (6, 16, 5), {
        'stride': 1,
        'padding': 2,
        'activation': 'tanh',
    }),
    ('Conv2d', (16, 6, 5), {
        'stride': 1,
        'padding': 2,
        'activation': None,
    }),
)

dlwp.build_model(layers, loss='MSELoss', optimizer='Adam')


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
val_generator.ds.load()
p_val, t_val = val_generator.generate([], scale_and_impute=False)


#%% Train, evaluate, and save the model

# Train and evaluate the model
start_time = time.time()
history = dlwp.fit_generator(generator, epochs=max_epochs, verbose=2, validation_generator=val_generator)
end_time = time.time()

# Evaluate the model
score = dlwp.evaluate(p_val[:10], t_val[:10])
print("\nTrain time -- %s seconds --" % (end_time - start_time))
print('Test loss:', score[0])
print('Test mean absolute error:', score[1])

# Save the model
if model_file is not None:
    save_torch_model(dlwp, model_file, history=history)
