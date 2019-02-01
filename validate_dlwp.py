#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Simple routines for evaluating the performance of a DLWP model.
"""

from DLWP.model import DataGenerator, Preprocessor
from DLWP.model import verify
from DLWP.util import load_model
from DLWP.custom import latitude_weighted_loss
from DLWP.model.preprocessing import train_test_split_ind
import keras.backend as K
from keras.losses import mean_squared_error
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


#%% User parameters

# Open the data file
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_1979-2010_hgt_250-500_NH_T2.nc' % root_directory

# Names of model files, located in the root_directory
models = [
    'dlwp_1979-2010_hgt_250-500_NH_T2F_CONVx5-x2-upsample-dilate',
    'dlwp_1979-2010_hgt_250-500_NH_T2F_CONVx5-upsample-dilate',
    'dlwp_1979-2010_hgt_500_NH_T2F_CONVx5-upsample-dilate'
]
model_labels = [
    'CNN 2-level 2x',
    'CNN 2-level',
    'CNN 1-level'
]

# Optional list of selections to make from the predictor dataset for each model. This is useful if, for example,
# you want to examine models that have different numbers of vertical levels but one predictor dataset contains
# the data that all models need.
predictor_sel = [
    None,
    None,
    {'level': [500]}
]

# Validation set to use. Either an integer (number of validation samples, taken from the end), or an iterable of
# pandas datetime objects.
# validation_set = 4 * (365 * 4 + 1)
start_date = datetime(2003, 1, 1, 0)
end_date = datetime(2006, 12, 31, 6)
validation_set = np.array(pd.date_range(start_date, end_date, freq='6H'), dtype='datetime64')
# validation_set = [d for d in validation_set if d.month in [1, 2, 12]]

# Generate a verification array in memory. May use significant memory but is required if the validation set is not a
# continuous selection from predictor dataset.
generate_verification = True

# Load a CFS Reforecast model for comparison
cfs_model_file = '%s/cfs_reforecast_5day_z500_2003-2010_interp.nc' % root_directory
cfs_ds = xr.open_dataset(cfs_model_file)
cfs_ds = cfs_ds.isel(lat=(cfs_ds.lat >= 0.0))  # Northern hemisphere only
# Actually let's change the validation dates to those in the CFS dataset
validation_set = [d for d in cfs_ds.time.values if d in validation_set]

# Load a barotropic model for comparison
baro_model_file = '%s/barotropic_%s-%s.nc' % (root_directory, start_date.year, end_date.year)
baro_ds = xr.open_dataset(baro_model_file)
baro_ds = baro_ds.isel(lat=(baro_ds.lat >= 0.0))  # Northern hemisphere only

# Number of forward integration weather forecast time steps
num_forecast_steps = 24

# Latitude bounds for MSE calculation
lat_range = [20., 70.]

# Calculate statistics for a specific variable and level. If None, then averages all variables. Cannot be None if using
# a barotropic model for comparison (specify Z500).
variable = 'HGT'
level = 500

# Do specific plots
plot_directory = './Plots'
plot_example = None  # None to disable or the date index of the sample
plot_example_f_hour = 48  # Forecast hour index of the sample
plot_history = False
plot_zonal = False
plot_mse = True
mse_title = r'Forecast error: 2003-06; $\hat{Z}_{500}$; 20-70$^{\circ}$N'
mse_file_name = 'mse_hgt_250-500_T2_20-70_CFS.pdf'


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
    lons, lats = np.meshgrid(base.lon, base.lat)
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
    ax.set_title('$t=0$ predictors (%s)' % plot_example)
    ax = plt.subplot(312)
    if plot_diff:
        m.contour(x, y, verif, np.arange(-2.5, 1.6, 0.5), cmap='jet')
    else:
        m.pcolormesh(x, y, verif, vmin=-2.5, vmax=1.5, cmap='YlGnBu_r')
    m.drawcoastlines()
    m.drawparallels(np.arange(0., 91., 45.))
    m.drawmeridians(np.arange(0., 361., 90.))
    forecast_time = plot_example + timedelta(hours=plot_example_f_hour)
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
    plt.fill_betweenx(obs_mean.lat, obs_mean - obs_std, obs_mean + obs_std,
                      facecolor='lightgray', zorder=-100)
    plt.plot(obs_mean, obs_mean.lat, label='observed')
    plt.plot(pred_mean, pred_mean.lat, label='%d-hour prediction' % (6 * num_forecast_steps))
    plt.legend(loc='best')
    plt.grid(True, color='lightgray', zorder=-100)
    plt.plot(pred_mean - pred_std, pred_mean.lat, 'k:', linewidth=0.7)
    plt.plot(pred_mean + pred_std, pred_mean.lat, 'k:', linewidth=0.7)
    plt.xlabel('zonal mean height')
    plt.ylabel('latitude')
    plt.ylim([0., 90.])
    plt.savefig('%s/%s_zonal_climo.pdf' % (plot_directory, model_name), bbox_inches='tight')
    plt.show()


#%% Iterate through the models and calculate their stats

# Use the predictor file as a wrapper
processor = Preprocessor(None, predictor_file=predictor_file)
processor.open()

# Find the validation set
if isinstance(validation_set, int):
    n_sample = processor.data.dims['sample']
    train_set, val_set = train_test_split_ind(n_sample, validation_set, method='last')
    validation_data = processor.data.isel(sample=val_set)
else:  # we must have a list of datetimes
    validation_data = processor.data.sel(sample=validation_set)

# Shortcuts for latitude range
lat_min = np.min(lat_range)
lat_max = np.max(lat_range)

# Format the predictor indexer and variable index in reshaped array
predictor_sel = predictor_sel or [None] * len(models)
variable = variable or processor.data.variables['variable'][:]
level = level or processor.data.variables['level'][:]

# Lists to populate
mse = []
f_hour = np.arange(6., num_forecast_steps * 6. + 1., 6.)

# Generate verification
if generate_verification:
    print('Generating validation set...')
    verification = xr.DataArray(
        np.zeros((num_forecast_steps, validation_data.dims['sample'], validation_data.dims['variable'],
                  validation_data.dims['level'], validation_data.dims['lat'], validation_data.dims['lon'])),
        coords=[f_hour, validation_data.sample, validation_data.variable, validation_data.level,
                validation_data.lat, validation_data.lon],
        dims=['f_hour', 'sample', 'variable', 'level', 'lat', 'lon']
    )
    valid_da = processor.data.targets.isel(time_step=0)
    for d, date in enumerate(validation_set):
        verification[:, d] = valid_da.sel(
            sample=pd.date_range(date, date + np.timedelta64(timedelta(hours=6 * (num_forecast_steps - 1))),
                                 freq='6H')).values

for m, model in enumerate(models):
    print('Loading model %s...' % model)

    # Some tolerance for using a weighted loss function. Unreliable but doesn't hurt.
    if 'weight' in model.lower():
        lats = validation_data.lat.values
        output_shape = (validation_data.dims['lat'], validation_data.dims['lon'])
        if 'upsample' in model.lower():
            lats = lats[1:]
        customs = {'loss': latitude_weighted_loss(mean_squared_error, lats, output_shape, axis=-2,
                                                  weighting='midlatitude')}
    else:
        customs = None

    # Load the model
    dlwp, history = load_model('%s/%s' % (root_directory, model), True, custom_objects=customs)

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
    time_dim = 1 * dlwp.time_dim

    # Create data generators
    if predictor_sel[m] is not None:
        val_ds = validation_data.sel(**predictor_sel[m])
    else:
        val_ds = validation_data.copy()
    if 'upsample' in model.lower():
        val_ds = val_ds.isel(lat=(val_ds.lat < 90.0))
    val_generator = DataGenerator(dlwp, val_ds, batch_size=216)
    p_val, t_val = val_generator.generate([], scale_and_impute=False)

    # Make a time series prediction and convert the predictors for comparison
    print('Predicting with model %s...' % model_labels[m])
    time_series = dlwp.predict_timeseries(p_val, num_forecast_steps)
    time_series = verify.add_metadata_to_forecast(time_series, f_hour, val_ds)
    p_series = verify.predictors_to_time_series(p_val, time_dim, has_time_dim=dlwp.is_recurrent, meta_ds=val_ds)

    # Generate the validation, either for discrete times or continuous
    if generate_verification:
        if predictor_sel[m] is not None:
            verif = verification.sel(**predictor_sel[m])
        else:
            verif = verification.copy()
        if 'upsample' in model.lower():
            verif = verif.isel(lat=(verif.lat < 90.0))
    else:
        verif = verify.predictors_to_time_series(t_val, time_dim, has_time_dim=dlwp.is_recurrent,
                                                    use_first_step=True, meta_ds=val_ds)

    # Slice the arrays as we want
    time_series = time_series.sel(variable=(slice(None) if variable is None else variable),
                                  level=(slice(None) if level is None else level),
                                  lat=((time_series.lat >= lat_min) & (time_series.lat <= lat_max)))
    p_series = p_series.sel(variable=(slice(None) if variable is None else variable),
                            level=(slice(None) if level is None else level),
                            lat=((p_series.lat >= lat_min) & (p_series.lat <= lat_max)))
    verif = verif.sel(variable=(slice(None) if variable is None else variable),
                      level=(slice(None) if level is None else level),
                      lat=((verif.lat >= lat_min) & (verif.lat <= lat_max)))

    # Calculate the MSE for each forecast hour relative to observations
    mse.append(verify.forecast_error(time_series.values, verif.values))

    # Plot learning curves
    if plot_history:
        history_plot(history['mean_absolute_error'], history['val_mean_absolute_error'], model_labels[m])

    # Plot an example
    if plot_example is not None:
        plot_dt = np.datetime64(plot_example)
        example_plot(p_series.sel(time=plot_dt),
                     p_series.sel(time=plot_dt + np.timedelta64(timedelta(hours=plot_example_f_hour))),
                     time_series.sel(f_hour=plot_example_f_hour, time=plot_dt), model_labels[m])

    # Plot the zonal climatology of the last forecast hour
    if plot_zonal:
        obs_zonal_mean = p_series[num_forecast_steps:].mean(axis=(0, -1))
        obs_zonal_std = p_series[num_forecast_steps:].std(axis=-1).mean(axis=0)
        pred_zonal_mean = time_series[-1, :-num_forecast_steps].mean(axis=(0, -1))
        pred_zonal_std = time_series[-1, :-num_forecast_steps].std(axis=-1).mean(axis=0)
        zonal_mean_plot(obs_zonal_mean, obs_zonal_std, pred_zonal_mean, pred_zonal_std, model_labels[m])

    # Clear the model
    dlwp = None
    time_series = None
    K.clear_session()


#%% Add Barotropic model

if baro_ds is not None:
    print('Loading barotropic model data from %s...' % baro_model_file)
    if variable is None or level is None:
        raise ValueError("specific 'variable' and 'level' for Z500 must be specified to use barotropic model")
    baro_ds = baro_ds.isel(lat=((baro_ds.lat >= lat_min) & (baro_ds.lat <= lat_max)))
    if isinstance(validation_set, int):
        baro_ds = baro_ds.isel(time=slice(0, baro_ds.dims['time']-time_dim+1))
    else:
        baro_ds = baro_ds.sel(time=validation_set)

    # Verify against the included forecast hour 0
    baro_forecast = baro_ds.isel(f_hour=(baro_ds.f_hour > 0))
    baro_f = baro_forecast.variables['Z'].values

    # Normalize by the same std and mean as the predictor dataset
    z500_mean = processor.data.sel(variable=variable, level=level).variables['mean'].values
    z500_std = processor.data.sel(variable=variable, level=level).variables['std'].values
    baro_f = (baro_f - z500_mean) / z500_std

    if generate_verification:
        try:
            verif = verif.sel(variable=variable, level=level)
        except ValueError:
            pass
        mse.append(verify.forecast_error(baro_f, verif.values))
    else:
        baro_verif = baro_ds.isel(f_hour=0)
        baro_v = baro_verif.variables['Z'].values.squeeze()
        baro_v = (baro_v - z500_mean) / z500_std
        mse.append(verify.forecast_error(baro_f, baro_v))
    model_labels.append('Barotropic')


#%% Add the CFS model

if cfs_ds is not None:
    print('Loading CFS model data from %s...' % cfs_model_file)
    if variable is None or level is None:
        raise ValueError("specific 'variable' and 'level' for Z500 must be specified to use CFS model model")
    cfs_ds = cfs_ds.isel(lat=((cfs_ds.lat >= lat_min) & (cfs_ds.lat <= lat_max)))
    if isinstance(validation_set, int):
        raise ValueError("I can only compare to a CFS Reforecast with datetime validation set")
    else:
        cfs_ds = cfs_ds.sel(time=validation_set)

    # Verify against the included forecast hour 0 # 6 temporarily
    cfs_forecast = cfs_ds.isel(f_hour=(cfs_ds.f_hour > 0))
    cfs_f = cfs_forecast.variables['Z'].values

    # Normalize by the same std and mean as the predictor dataset
    z500_mean = processor.data.sel(variable=variable, level=level).variables['mean'].values
    z500_std = processor.data.sel(variable=variable, level=level).variables['std'].values
    cfs_f = (cfs_f - z500_mean) / z500_std

    if generate_verification:
        try:
            verif = verif.sel(variable=variable, level=level)
        except ValueError:
            pass
        mse.append(verify.forecast_error(cfs_f, verif.values))
    else:
        cfs_verif = cfs_ds.isel(f_hour=6)
        cfs_v = cfs_verif.variables['Z'].values.squeeze()
        cfs_v = (cfs_v - z500_mean) / z500_std
        mse.append(verify.forecast_error(cfs_f, cfs_v))
    model_labels.append('CFS')


#%% Add persistence and climatology

print('Calculating persistence forecasts...')
if generate_verification:
    mse.append(verify.forecast_error(np.repeat(p_series.values[None, ...], num_forecast_steps, axis=0), verif.values))
else:
    mse.append(verify.persistence_error(p_series.values, verif.values, num_forecast_steps))
model_labels.append('Persistence')

print('Calculating climatology forecasts...')
mse.append(verify.monthly_climo_error(processor.data['predictors'].isel(time_step=-1).sel(
    variable=(slice(None) if variable is None else variable),
    level=(slice(None) if level is None else level),
    lat=((processor.data.lat >= lat_min) & (processor.data.lat <= lat_max))),
    validation_set, n_fhour=num_forecast_steps))
model_labels.append('Climatology')


#%% Plot the combined MSE as a function of forecast hour for all models

fig = plt.figure()
fig.set_size_inches(6, 4)
for m, model in enumerate(model_labels):
    if model != 'Barotropic':
        plt.plot(f_hour, mse[m], label=model, linewidth=2.)
    else:
        plt.plot(baro_forecast.f_hour, mse[m], label=model, linewidth=2.)
plt.legend(loc='best')
plt.grid(True, color='lightgray', zorder=-100)
plt.xlabel('forecast hour')
plt.ylabel('MSE')
plt.title(mse_title)
plt.savefig('%s/%s' % (plot_directory, mse_file_name), bbox_inches='tight')
plt.show()
