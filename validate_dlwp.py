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
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


#%% User parameters

# Open the data file
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_1979-2010_hgt_500_NH_T2.nc' % root_directory

# Names of model files, located in the root_directory
models = ['dlwp_1979-2010_hgt_500_NH_T2F_CLSTM_16_5_2',
          'dlwp_1979-2010_hgt_500_NH_T2F_CLSTM_16_5_2_PBC',
          'dlwp_1979-2010_hgt_500_NH_T2F_CLSTM_PBC_ROW_test']
model_labels = ['LSTM', 'LSTM-PBC', 'LSTM-PBC-local']

# Load the result of a barotropic model for comparison
baro_model_file = '%s/barotropic_2007-2010.nc' % root_directory
baro_ds = xr.open_dataset(baro_model_file)
baro_ds = baro_ds.isel(lat=(baro_ds.lat >= 0.0))  # Northern hemisphere only

# Number of validation samples. For example, 2 years of 6-hourly data is 4 * (365 + 366) if one is a leap year.
n_val = 4 * (365 * 4 + 1)

# Number of forward integration weather forecast time steps
num_forecast_steps = 24

# Calculate statistics for a specific variable index. If None, then averages all variables. Cannot be None if using a
# barotropic model for comparison (specify the index of Z500).
variable_index = 0

# Do specific plots
plot_directory = './Plots'
plot_example = None  # None to disable or the index of the sample
plot_example_date = datetime(2007, 1, 1)
plot_example_f_hour = 48  # Forecast hour index of the sample
plot_history = False
plot_zonal = False
plot_mse = True
mse_title = 'Forecast MSE 2007-10, $\hat{Z}_{500}$ NH'
mse_file_name = 'mse_CLSTM_T2.pdf'


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


def example_plot(base, verif, forecast, model_name, plot_diff=True):
    # Plot an example forecast
    lons, lats = np.meshgrid(processor.data.lon, processor.data.lat)
    fig = plt.figure()
    fig.set_size_inches(9, 9)
    m = Basemap(llcrnrlon=0., llcrnrlat=0., urcrnrlon=360., urcrnrlat=90.,
                resolution='l', projection='cyl', lat_0=40., lon_0=0.)
    x, y = m(lons, lats)
    ax = plt.subplot(311)
    if plot_diff:
        m.contour(x, y, base, np.arange(-2.5, 1.6, 0.5), cmap='jet')
    else:
        m.pcolormesh(x, y, base, vmin=-2.5, vmax=1.5, cmap='YlGnBu_r')
    m.drawcoastlines()
    m.drawparallels(np.arange(0., 91., 45.))
    m.drawmeridians(np.arange(0., 361., 90.))
    ax.set_title('$t=0$ predictors (%s)' % plot_example_date)
    ax = plt.subplot(312)
    if plot_diff:
        m.contour(x, y, verif, np.arange(-2.5, 1.6, 0.5), cmap='jet')
    else:
        m.pcolormesh(x, y, verif, vmin=-2.5, vmax=1.5, cmap='YlGnBu_r')
    m.drawcoastlines()
    m.drawparallels(np.arange(0., 91., 45.))
    m.drawmeridians(np.arange(0., 361., 90.))
    forecast_time = plot_example_date + timedelta(hours=plot_example_f_hour)
    ax.set_title('$t=%d$ verification (%s)' % (plot_example_f_hour, forecast_time))
    ax = plt.subplot(313)
    if plot_diff:
        m.contour(x, y, forecast, np.arange(-2.5, 1.6, 0.5), cmap='jet')
        m.pcolormesh(x, y, forecast - verif, vmin=-1, vmax=1, cmap='seismic')
    else:
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
    plt.plot(pred_mean, processor.data.lat, label='%d-hour prediction' % 6 * num_forecast_steps)
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
f_hour = np.arange(6., num_forecast_steps*6.+1., 6.)

for m, model in enumerate(models):
    print('Loading model %s...' % model)
    dlwp, history = load_model('%s/%s' % (root_directory, model), True)

    # Build in some tolerance for old models trained with former APIs missing the is_convolutional and is_recurrent
    # attributes. This may not always work!
    if not hasattr(dlwp, 'is_recurrent'):
        dlwp.is_recurrent = False
        for layer in dlwp.model.layers:
            if 'LSTM' in layer.name.upper() or 'LST_M' in layer.name.upper():
                dlwp.is_recurrent = True
    if not hasattr(dlwp, 'is_convolutional'):
        dlwp.is_convolutional = False
        for layer in dlwp.model.layers:
            if 'CONV' in layer.name.upper():
                dlwp.is_convolutional = True
    if not hasattr(dlwp, 'time_dim'):
        dlwp.time_dim = 1

    # Recurrent time axis. This is probably unnecessary with the new predict_timeseries in 0.1.0
    recurrent_axis = slice(-1) if dlwp.is_recurrent else None
    time_dim = 1 * dlwp.time_dim

    # Create data generators
    val_generator = DataGenerator(dlwp, processor.data.isel(sample=val_set), batch_size=216)
    p_val, t_val = val_generator.generate([], scale_and_impute=False)

    # Make a time series prediction and convert the predictors for comparison
    print('Predicting with model %s...' % model_labels[m])
    time_series = dlwp.predict_timeseries(p_val, num_forecast_steps)
    p_series = verify.predictors_to_time_series(p_val, dlwp.time_dim, has_time_dim=dlwp.is_recurrent)
    t_series = verify.predictors_to_time_series(t_val, dlwp.time_dim, has_time_dim=dlwp.is_recurrent,
                                                use_first_step=True)

    # Calculate the MSE for each forecast hour relative to observations
    if variable_index is None:
        variable_index = slice(None)
    mse.append(verify.forecast_error(time_series[:, :, variable_index],
                                     t_series[:, variable_index]))

    # Plot learning curves
    if plot_history:
        history_plot(history['mean_absolute_error'], history['val_mean_absolute_error'], model_labels[m])

    # Plot an example
    if plot_example is not None:
        example_plot(p_series[plot_example, variable_index].squeeze(),
                     p_series[plot_example + plot_example_f_hour // 6 - 1, variable_index].squeeze(),
                     time_series[plot_example_f_hour // 6 - 1, plot_example, variable_index].squeeze(),
                     model_labels[m])

    # Plot the zonal climatology of the last forecast hour
    if plot_zonal:
        obs_zonal_mean = np.mean(p_series[num_forecast_steps:, variable_index], axis=(0, -1)).squeeze()
        obs_zonal_std = np.mean(np.std(p_series[num_forecast_steps:, variable_index], axis=-1), axis=0).squeeze()
        pred_zonal_mean = np.mean(time_series[-1, :-num_forecast_steps, variable_index], axis=(0, -1)).squeeze()
        pred_zonal_std = np.mean(np.std(time_series[-1, :-num_forecast_steps, variable_index], axis=-1),
                                 axis=0).squeeze()
        zonal_mean_plot(obs_zonal_mean, obs_zonal_std, pred_zonal_mean, pred_zonal_std, model_labels[m])

    # Clear the model
    dlwp = None
    time_series = None
    K.clear_session()


#%% Add Barotropic model, persistence, and climatology

if baro_ds is not None:
    print('Loading barotropic model data from %s...' % baro_model_file)
    print('I hope you selected only the variable corresponding to Z500!')
    baro_values = baro_ds.variables['Z'].values

    # Normalize by the same std and mean as the predictor dataset
    var_idx = variable_index // processor.data.dims['level']
    lev_idx = variable_index % processor.data.dims['level']
    z500_mean = processor.data.isel(variable=var_idx, level=lev_idx).variables['mean'].values
    z500_std = processor.data.isel(variable=var_idx, level=lev_idx).variables['std'].values
    baro_values = (baro_values - z500_mean) / z500_std

    # The barotropic model is initialized at exactly all 6-hourly dates within the last 4 years, but the predictor
    # file samples must exclude the last (2010-12-31 18) forecast init date, because it doesn't have a target sample.
    # Hence we shift the init_dates to match.
    mse.append(verify.forecast_error(baro_values[:, (time_dim-1):(-time_dim)],
                                     t_series[:, variable_index].squeeze()))
    model_labels.append('Barotropic')

print('Calculating persistence forecasts...')
mse.append(verify.persistence_error(p_series[:, variable_index], t_series[:, variable_index], num_forecast_steps))
model_labels.append('Persistence')

print('Calculating climatology forecasts...')
mse.append(verify.climo_error(t_series[:, variable_index], num_forecast_steps))
model_labels.append('Climatology')


#%% Plot the combined MSE as a function of forecast hour for all models

fig = plt.figure()
fig.set_size_inches(6, 4)
for m, model in enumerate(model_labels):
    if model != 'Barotropic':
        plt.plot(f_hour, mse[m], label=model, linewidth=2.)
    else:
        plt.plot(baro_ds.f_hour, mse[m], label=model, linewidth=2.)
plt.legend(loc='best')
plt.grid(True, color='lightgray', zorder=-100)
plt.xlabel('forecast hour')
plt.ylabel('MSE')
plt.title(mse_title)
plt.savefig('%s/%s' % (plot_directory, mse_file_name), bbox_inches='tight')
plt.show()
