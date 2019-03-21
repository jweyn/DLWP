#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Example of training a DLWP model using a dataset of predictors generated with DLWP.model.Preprocessor. Uses the PyTorch
deep learning library in DLWPTorchNN. PyTorch has spherical convolutions available through the s2cnn module, an
optional installation (https://github.com/jonas-koehler/s2cnn).
"""

import time
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from DLWP.model import DLWPTorchNN, DataGenerator
from DLWP.model.preprocessing import train_test_split_ind
from DLWP.util import save_torch_model
from s2cnn import s2_near_identity_grid


#%% Open some data using the Preprocessor wrapper

root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_analysis_1979-2010_all_20-30-50-70-85-100_G_T2.zarr' % root_directory
model_file = '%s/dlwp_1979-2010_all_20-30-50-70-85-100_G_T2_FINAL' % root_directory
log_directory = '%s/logs/all' % root_directory
model_is_convolutional = True
model_is_recurrent = False
min_epochs = 200
max_epochs = 10
patience = 50
batch_size = 64
lambda_ = 1.e-4

data = xr.open_zarr(predictor_file)
data = data.chunk({'sample': batch_size})
if 'time_step' in data.dims:
    time_dim = data.dims['time_step']
else:
    time_dim = 1
n_sample = data.dims['sample']

# If system memory permits, loading the predictor data can greatly increase efficiency when training on GPUs, if the
# train computation takes less time than the data loading.
load_memory = False

# Re-grid to remove extra lat dimension and downscale in lon (needs to fit square for spherical convolutions)
data = data.isel(lat=slice(0, -1), lon=slice(None, None, 2))

# Validation set to use. Either an integer (number of validation samples, taken from the end), or an iterable of
# pandas datetime objects. The train set can be set to the first <integer> samples, an iterable of dates, or None to
# simply use the remaining points. Match the type of validation_set.
validation_set = list(pd.date_range(datetime(2003, 1, 1, 6), datetime(2006, 12, 31, 6), freq='6H'))
train_set = list(pd.date_range(datetime(1979, 1, 1, 6), datetime(2003, 1, 1, 0), freq='6H'))

# For upsampling, we need an even number of lat/lon points. We'll crop out the north pole.
data = data.isel(lat=(data.lat < 90.0))


#%% Build a model and the data generators

dlwp = DLWPTorchNN(is_convolutional=model_is_convolutional, is_recurrent=model_is_recurrent, time_dim=time_dim,
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

# Build the data generators
generator = DataGenerator(dlwp, train_data, batch_size=batch_size)
if validation_data is not None:
    val_generator = DataGenerator(dlwp, validation_data, batch_size=batch_size)
else:
    val_generator = None


#%% Compile the model structure with some generator data information

# Convolutional feed-forward network
s2_grid = s2_near_identity_grid(max_beta=0.2, n_alpha=12, n_beta=1)
truncation = 12
layers = (
    ('S2Convolution', (3, 16, 36, truncation, s2_grid), {
        'mean_gamma': True,
        'activation': 'tanh'
    }),
    ('S2Convolution', (16, 16, truncation, truncation, s2_grid), {
        'mean_gamma': True,
        'activation': 'tanh'
    }),
    ('TorchReshape', ((-1, 16 * (2 * truncation) ** 2),), None),
    ('Linear', (16 * (2 * truncation) ** 2, generator.n_features), None),
    ('TorchReshape', ((-1,) + generator.convolution_shape,), None)
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
if load_memory:
    generator.ds.load()

# Load the validation data. Better to load in memory to avoid file read errors while training.
if val_generator is not None:
    val_generator.ds.load()
    p_val, t_val = val_generator.generate([], scale_and_impute=False)


#%% Train, evaluate, and save the model

# Train and evaluate the model
start_time = time.time()
history = dlwp.fit_generator(generator, epochs=max_epochs, verbose=2, validation_generator=val_generator)
end_time = time.time()

# Evaluate the model
print("\nTrain time -- %s seconds --" % (end_time - start_time))
if validation_data is not None:
    score = dlwp.evaluate(p_val, t_val)
    print('Validation loss:', score[0])
    print('Validation mean absolute error:', score[1])

# Save the model
if model_file is not None:
    save_torch_model(dlwp, model_file, history=history)
