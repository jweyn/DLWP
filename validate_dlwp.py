#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Simple routines for evaluating the performance of a DLWP model.
"""

from DLWP.model import DataGenerator, Preprocessor
from DLWP.model import verify
from DLWP.util import load_model
from DLWP.model.preprocessing import train_test_split_ind
import keras.backend as K
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


#%% User parameters

# Open the data file
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_1979-2010_hgt-tmp_250-500-1000_NH.nc' % root_directory

# Names of model files, located in the root_directory
models = ['dlwp_1979-2010_hgt-tmp_250-500-1000_NH_CONV_64x5x1_16x5x2_MP',
          'dlwp_1979-2010_hgt-tmp_250-500-1000_NH_CONV_16x5x1_16x5x2_16x3x1_MP']
model_labels = ['Conv_64-16', 'Conv_16-16-16x3']

# Number of validation samples. For example, 2 years of 6-hourly data is 4 * (365 + 366) if one is a leap year.
n_val = 4 * (365 + 365)

# Number of forward integration weather forecast time steps
num_forecast_steps = 24

# Calculate for a specific variable index. If None, then averages all variables.
variable_index = 1

# Do specific plots
plot_directory = './Plots'
plot_example = None  # None to disable or the index of the sample
plot_example_date = datetime(2009, 1, 1)
plot_example_f_hour = 48  # Forecast hour index of the sample
plot_history = True
plot_mse = True
mse_title = 'Forecast MSE 2009-10, $\hat{Z}$ 500 NH'
mse_file_name = 'mse_Conv_64_Conv_16_250-500-1000.pdf'


#%% Define some plotting functions

def history_plot(train_hist, val_hist, model_name):
    fig = plt.figure()
    fig.set_size_inches(6, 4)
    plt.plot(train_hist, label='train MAE', linewidth=2)
    plt.plot(val_hist, label='val MAE', linewidth=2)
    plt.grid(True, color='lightgray', zorder=-100)
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.legend(loc='best')
    plt.title('%s training history' % model_name)
    plt.savefig('%s/%s_history.pdf' % (plot_directory, model_name), bbox_inches='tight')
    plt.show()


def example_plot(base, verif, forecast, model_name):
    # Plot an example forecast
    lons, lats = np.meshgrid(processor.data.lon, processor.data.lat)
    fig = plt.figure()
    fig.set_size_inches(6, 6)
    m = Basemap(llcrnrlon=0., llcrnrlat=0., urcrnrlon=360., urcrnrlat=90.,
                resolution='l', projection='cyl', lat_0=40., lon_0=0.)
    x, y = m(lons, lats)
    ax = plt.subplot(311)
    m.pcolormesh(x, y, base, vmin=-2.5, vmax=1.5, cmap='YlGnBu_r')
    m.drawcoastlines()
    m.drawparallels(np.arange(0., 91., 45.))
    m.drawmeridians(np.arange(0., 361., 90.))
    ax.set_title('$t=%d$ predictors (%s)' % (plot_example_f_hour, plot_example_date))
    ax = plt.subplot(312)
    m.pcolormesh(x, y, verif, vmin=-2.5, vmax=1.5, cmap='YlGnBu_r')
    m.drawcoastlines()
    m.drawparallels(np.arange(0., 91., 45.))
    m.drawmeridians(np.arange(0., 361., 90.))
    forecast_time = plot_example_date + timedelta(hours=plot_example_f_hour)
    ax.set_title('$t=%d$ verification (%s)' % (plot_example_f_hour, forecast_time))
    ax = plt.subplot(313)
    m.pcolormesh(x, y, forecast, vmin=-2.5, vmax=1.5, cmap='YlGnBu_r')
    # plt.colorbar(orientation='horizontal')
    m.drawcoastlines()
    m.drawparallels(np.arange(0., 91., 45.))
    m.drawmeridians(np.arange(0., 361., 90.))
    ax.set_title('$t=%d$ forecast (%s)' % (plot_example_f_hour, forecast_time))
    plt.savefig('%s/%s_example_%d.pdf' % (plot_directory, model_name, plot_example_f_hour), bbox_inches='tight')
    plt.show()


def zonal_mean_plot(obs_mean, obs_std, pred_mean, pred_std, model_name):
    fig = plt.figure()
    fig.set_size_inches(4, 6)
    plt.fill_betweenx(processor.data.lat, obs_mean - obs_std, obs_mean + obs_std,
                      facecolor='lightgray', zorder=-100)
    plt.plot(obs_mean, processor.data.lat, label='observed')
    plt.plot(pred_mean, processor.data.lat, label='144-hour LSTM prediction')
    plt.legend(loc='best')
    plt.grid(True, color='lightgray', zorder=-100)
    plt.plot(pred_mean - pred_std, processor.data.lat, 'k:', linewidth=0.7)
    plt.plot(pred_mean + pred_std, processor.data.lat, 'k:', linewidth=0.7)
    plt.xlabel('zonal mean height')
    plt.ylabel('latitude')
    plt.ylim([0., 90.])
    plt.savefig('%s/%s_zonal_climo.pdf' % (plot_directory, model_name), bbox_inches='tight')
    plt.show()


#%% Iterate through the models and calculate their stats

# Use the predictor file as a wrapper
processor = Preprocessor(None, predictor_file=predictor_file)
processor.open()
n_sample = processor.data.dims['sample']
train_set, val_set = train_test_split_ind(n_sample, n_val, method='last')

# Lists to populate
mse = []
train_history = []
val_history = []
f_hour = np.arange(6., num_forecast_steps*6.+1., 6.)

for m, model in enumerate(models):
    print('Loading model %s...' % model)
    dlwp, history = load_model('%s/%s' % (root_directory, model), True)
    if not hasattr(dlwp, 'is_recurrent'):
        dlwp.is_recurrent = False
        for layer in dlwp.model.layers:
            if 'LSTM' in layer.name.upper():
                dlwp.is_recurrent = True
    recurrent_axis = slice(None) if dlwp.is_recurrent else None

    # Create data generators
    val_generator = DataGenerator(dlwp, processor.data.isel(sample=val_set), batch_size=216, convolution=True)
    p_val, t_val = val_generator.generate([], scale_and_impute=False)

    # Make a time series prediction
    print('Predicting with model %s...' % model_labels[m])
    time_series = dlwp.predict_timeseries(p_val, num_forecast_steps)

    # Calculate the MSE for each forecast hour relative to observations
    variable_index = variable_index or slice(None)
    mse.append(verify.forecast_error(time_series[:, :, recurrent_axis, variable_index],
                                     t_val[:, recurrent_axis, variable_index]))

    train_history.append(history['mean_absolute_error'])
    val_history.append(history['val_mean_absolute_error'])

    # Plot learning curves
    if plot_history:
        history_plot(history['mean_absolute_error'], history['val_mean_absolute_error'], model_labels[m])

    # Plot an example
    if plot_example is not None:
        example_plot(p_val[plot_example, recurrent_axis, variable_index],
                     p_val[plot_example + plot_example_f_hour // 6, recurrent_axis, variable_index],
                     time_series[plot_example_f_hour // 6, plot_example, recurrent_axis, variable_index],
                     model_labels[m])

    # Plot the zonal climatology of the last forecast hour
    obs_zonal_mean = np.mean(p_val[num_forecast_steps:, recurrent_axis, variable_index], axis=(0, -1)).squeeze()
    obs_zonal_std = np.mean(np.std(p_val[num_forecast_steps:, recurrent_axis, variable_index], axis=-1),
                            axis=0).squeeze()
    pred_zonal_mean = np.mean(time_series[-1, :-num_forecast_steps, recurrent_axis, variable_index],
                              axis=(0, -1)).squeeze()
    pred_zonal_std = np.mean(np.std(time_series[-1, :-num_forecast_steps, recurrent_axis, variable_index], axis=-1),
                             axis=0).squeeze()
    zonal_mean_plot(obs_zonal_mean, obs_zonal_std, pred_zonal_mean, pred_zonal_std, model_labels[m])

    # Clear the model
    dlwp = None
    K.clear_session()


# Plot the combined MSE as a function of forecast hour for all models
fig = plt.figure()
fig.set_size_inches(6, 4)
for m, model in enumerate(model_labels):
    plt.plot(f_hour, mse[m], label=model, linewidth=2.)
plt.legend(loc='best')
plt.grid(True, color='lightgray', zorder=-100)
plt.xlabel('forecast hour')
plt.ylabel('MSE')
plt.title(mse_title)
plt.savefig('%s/%s' % (plot_directory, mse_file_name), bbox_inches='tight')
plt.show()
