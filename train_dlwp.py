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


root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_2000-2009_hgt_300-700.nc' % root_directory
epochs = 3


processor = Preprocessor(None, predictor_file=predictor_file)
processor.open()

n_sample = processor.data.dims['sample']
n_val = int(np.floor(n_sample * 0.2))
train_set, val_set = train_test_split_ind(n_sample, n_val, method='last')

# Build a model
dlwp = DLWPNeuralNet()
layers = (
    ('Dense', (2048,), {
        'activation': 'tanh',
        'input_shape': (processor.n_features,)
    }),
    ('Dense', (processor.n_features,), {
        'activation': 'linear'
    })
)
dlwp.build_model(layers, loss='mse', optimizer='adam', metrics=['mae'])

# Build the data generators
generator = DataGenerator(dlwp, processor.data.isel(sample=train_set), batch_size=128)
val_generator = DataGenerator(dlwp, processor.data.isel(sample=val_set), batch_size=128)

# Generate the data to fit the scaler and imputer. The generator will by default apply scaling because it is necessary
# to automate its use in the Keras fit_generator method, so disable it when dealing with data locally.
fit_set = train_set[:2000]
p_fit, t_fit = generator.generate(fit_set, scale_and_impute=False)
dlwp.init_fit(p_fit, t_fit)
p_fit, t_fit = (None, None)

# Train and evaluate the model
start_time = time.time()
history = dlwp.fit_generator(generator, epochs=epochs, verbose=1, validation_data=val_generator,
                             use_multiprocessing=True)

end_time = time.time()

# Evaluate the model
score = dlwp.evaluate(*val_generator.generate([], scale_and_impute=False), verbose=0)
print("\nTrain time -- %s seconds --" % (end_time - start_time))
print('Test loss:', score[0])
print('Test mean absolute error:', score[1])
