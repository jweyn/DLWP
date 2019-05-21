#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Make an ensemble forecast of specific variables. Includes spaghetti map plot at a specific forecast hour and time-
series plot at a specific location.
"""

import keras.backend as K
from keras.losses import mean_squared_error
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from DLWP.model import SeriesDataGenerator, TimeSeriesEstimator, DLWPFunctional
from DLWP.model import verify
from DLWP.plot import history_plot, forecast_example_plot, zonal_mean_plot
from DLWP.util import load_model, train_test_split_ind
from DLWP.custom import latitude_weighted_loss


#%% User parameters

# Open the data file
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_6h_1979-2010_z500-th3-7-w700-rh850-pwat_NH_T2.nc' % root_directory

# Names of model files, located in the root_directory, and labels for those models
tau_models = [1, 3, 6, 7, 9, 10, 11]
tau_lstm_models = [2, 4, 6, 7, 8, 10, 11]
models = (['dlwp_6h_tau_z-tau-out_fit'] + ['tau-random/tau-%02d' % m for m in tau_models] +
          ['dlwp_1979-2010_hgt-thick_300-500-700_NH_T2F_FINAL-lstm'] +
          ['tau-random/tau-lstm-%02d' % m for m in tau_lstm_models])
model_labels = ['tau-%02d' % m for m in list([0] + tau_models)] + \
               ['tau-lstm-%02d' % m for m in list([0] + tau_lstm_models)]

# Optional list of selections to make from the predictor dataset for each model. This is useful if, for example,
# you want to examine models that have different numbers of vertical levels but one predictor dataset contains
# the data that all models need. Separate input and output selections are available for models using different inputs
# and outputs. Also specify the number of input/output time steps in each model.
input_selection = [
    {'varlev': ['HGT/500', 'THICK/300-700']},
] * len(models)
output_selection = [
    {'varlev': ['HGT/500', 'THICK/300-700']},
] * len(models)
add_insolation = [False] * len(models)
input_time_steps = [2] * len(models)
output_time_steps = [2] * len(models)

# Models which use up-sampling need to have an even number of latitudes. This is usually done by cropping out the
# north pole. Set this option to do that.
crop_north_pole = True

# Validation set to use. Either an integer (number of validation samples, taken from the end), or an iterable of
# pandas datetime objects.
# validation_set = 4 * (365 * 4 + 1)
start_date = datetime(2007, 4, 11, 18)
end_date = datetime(2007, 4, 30, 18)
validation_set = np.array(pd.date_range(start_date, end_date, freq='6H'), dtype='datetime64')

# Make plots for these initialization times
plot_dates = list(pd.date_range('2007-04-12', '2007-04-17', freq='D').to_pydatetime())

# Number of forward integration weather forecast time steps
num_forecast_steps = 12
dt = 6

# Latitude / Longitude limits
latitude_range = [20., 80.]
longitude_range = [220., 300.]

# Calculate forecasts for a selected variable and level, or varlev if the predictor data was produced pairwise.
# Provide as a dictionary to extract to kwargs. Also provide the desired contour level for the spaghetti plot (in
# original, not scaled, units).
selection = {
    'varlev': 'HGT/500'
}
contour = 5400
var_label = 'Z500 (m)'

# Do specific plots
plot_directory = './Plots'
plot_map_hour = 72
plot_map = 'MAP_tau-ensemble'
plot_point = [40., 285.]
plot_line = 'tau-ensemble_nyc'


#%% Pre-processing

data = xr.open_dataset(predictor_file)
if crop_north_pole:
    data = data.isel(lat=(data.lat < 90.0))

# Find the validation set
if isinstance(validation_set, int):
    n_sample = data.dims['sample']
    train_set, val_set = train_test_split_ind(n_sample, validation_set, method='last')
    validation_data = data.isel(sample=val_set)
else:  # we must have a list of datetimes
    validation_data = data.sel(sample=validation_set)

# Shortcuts for latitude range
lat_min = np.min(latitude_range)
lat_max = np.max(latitude_range)
lon_min = np.min(longitude_range)
lon_max = np.max(longitude_range)

# Format the predictor indexer and variable index in reshaped array
input_selection = input_selection or [None] * len(models)
output_selection = output_selection or [None] * len(models)
selection = selection or {}

# Lists to populate
model_forecasts = []
f_hours = []

# Generate verification
print('Generating verification...')
validation_data.load()
verification = verify.verification_from_samples(validation_data.sel(**selection),
                                                forecast_steps=num_forecast_steps, dt=dt)
verification = verification.sel(lat=((verification.lat >= lat_min) & (verification.lat <= lat_max)),
                                lon=((verification.lon >= lon_min) & (verification.lon <= lon_max)))

# Get the mean and std of the data
variable_mean = data.sel(**selection).variables['mean'].values
variable_std = data.sel(**selection).variables['std'].values

# Scale the verification
verification = verification * variable_std + variable_mean


#%% Iterate through the models and calculate their stats

for m, model in enumerate(models):
    print('Loading model %s...' % model)

    # Some tolerance for using a weighted loss function. Unreliable but doesn't hurt.
    if 'weight' in model.lower():
        lats = validation_data.lat.values
        output_shape = (validation_data.dims['lat'], validation_data.dims['lon'])
        if crop_north_pole:
            lats = lats[1:]
        customs = {'loss': latitude_weighted_loss(mean_squared_error, lats, output_shape, axis=-2,
                                                  weighting='midlatitude')}
    else:
        customs = None

    # Load the model
    dlwp, history = load_model('%s/%s' % (root_directory, model), True, custom_objects=customs)

    # Assign forecast hour coordinate
    if isinstance(dlwp, DLWPFunctional):
        if not hasattr(dlwp, '_n_steps'):
            dlwp._n_steps = 6
        if not hasattr(dlwp, 'time_dim'):
            dlwp.time_dim = 2
        sequence = dlwp._n_steps
    else:
        sequence = None
    f_hours.append(np.arange(dt, num_forecast_steps * dt + 1., dt))

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

    # Create data generator
    val_generator = SeriesDataGenerator(dlwp, validation_data, add_insolation=add_insolation[m],
                                        input_sel=input_selection[m], output_sel=output_selection[m],
                                        input_time_steps=input_time_steps[m], output_time_steps=output_time_steps[m],
                                        batch_size=64)

    # Create TimeSeriesEstimator
    estimator = TimeSeriesEstimator(dlwp, val_generator)

    # Very crude but for this test I want to exclude the predicted thickness from being added back
    if model_labels[m] == r'$\tau$ LSTM16':
        estimator._outputs_in_inputs = {'varlev': np.array(['HGT/500'])}

    # Make a time series prediction
    print('Predicting with model %s...' % model_labels[m])
    time_series = estimator.predict(num_forecast_steps, verbose=1)

    # Slice and scale the forecast
    time_series = time_series.sel(lat=((time_series.lat >= lat_min) & (time_series.lat <= lat_max)),
                                  lon=((time_series.lon >= lon_min) & (time_series.lon <= lon_max)),
                                  **selection)
    time_series = time_series * variable_std + variable_mean

    model_forecasts.append(time_series.copy())

    # Clear the model
    dlwp = None
    time_series = None
    K.clear_session()


#%% Define plotting functions

def plot_spaghetti(m, verif, forecasts, file_name=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    plot_fn = m.contour
    contours = [contour]
    cm = plt.get_cmap('gist_rainbow')

    lons, lats = np.meshgrid(verif.lon, verif.lat)
    x, y = m(lons, lats)
    m.drawcoastlines(color=(0.7, 0.7, 0.7))
    m.drawparallels(np.arange(0., 91., 30.), linewidth=0.5, color=(0.5, 0.5, 0.5))
    m.drawmeridians(np.arange(0., 361., 30.), linewidth=0.5, color=(0.5, 0.5, 0.5))

    legend_lines = []

    # Plot ensemble member forecasts
    for f, forecast in enumerate(forecasts):
        c = cm(1. * f / len(forecasts))
        cs = plot_fn(x, y, forecast.values, contours, colors=[c], linewidths=0.8)
        legend_lines.append(Line2D([0], [0], color=c, linewidth=0.8))

    # Plot ensemble mean
    mean = np.mean(np.array([f.values for f in forecasts]), axis=0)
    cs = plot_fn(x, y, mean, contours, colors='k', linewidths=2)
    legend_lines.append(Line2D([0], [0], color='k', linewidth=2))

    # Plot verification
    cs = plot_fn(x, y, verif.values, contours, colors='k', linewidths=2, linestyles='--')
    legend_lines.append(Line2D([0], [0], color='k', linewidth=2, linestyle='--'))

    # Legend and title
    plt.title('%s – %s\nInitialized %s; %d-hour forecast' %
              (var_label, contour, pd.Timestamp(verif.time.values).strftime('%Y-%m-%d %HZ'),
               forecast.f_hour.values.astype('timedelta64[h]').astype('int')))
    plt.legend(legend_lines, model_labels + ['Mean', 'Truth'], loc='best', ncol=4, fontsize=8)

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')


def plot_series(verif, forecasts, title=None, file_name=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    cm = plt.get_cmap('gist_rainbow')

    for f, forecast in enumerate(forecasts):
        plt.plot(forecast.f_hour.values.astype('timedelta64[h]'), forecast.values,
                 color=cm(1. * f / len(forecasts)), linewidth=0.8, label=model_labels[f])

    mean = np.mean(np.array([f.values for f in forecasts]), axis=0)
    plt.plot(forecasts[0].f_hour.values.astype('timedelta64[h]'), mean, 'k-', linewidth=2, label='Mean')
    plt.plot(verif.f_hour.values.astype('timedelta64[h]'), verif.values, 'k--', linewidth=2, label='Truth')

    # Legend and title
    if title is not None:
        plt.title(title)
    plt.legend(loc='best', ncol=4, fontsize=8)
    plt.xlim([0, dt * num_forecast_steps])
    plt.xticks(np.arange(0, num_forecast_steps * dt + 1, 2 * dt))
    plt.xlabel('forecast hour')
    plt.ylabel(var_label)
    ax.grid(True, linestyle='-', color=(0.8, 0.8, 0.8), zorder=-100)

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')


#%% Iterate through each plot to make

from mpl_toolkits.basemap import Basemap

basemap = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,
                  resolution='l', projection='cyl', lat_0=40., lon_0=0.)

for date in plot_dates:
    print('Plotting for %s...' % date)

    date64 = np.datetime64(date)
    verif_date = date + timedelta(hours=plot_map_hour)
    verif_date64 = date64 + np.timedelta64(timedelta(hours=plot_map_hour))

    if plot_map is not None:
        plot_forecasts = [f.sel(f_hour=np.array(plot_map_hour, dtype='timedelta64[h]'), time=date64)
                          for f in model_forecasts]
        plot_verif = verification.sel(f_hour=np.array(plot_map_hour, dtype='timedelta64[h]'), time=date64)

        plot_spaghetti(basemap, plot_verif, plot_forecasts,
                       file_name='%s/%s_%s-%03dh.pdf' % (plot_directory, plot_map,
                                                         datetime.strftime(verif_date, '%Y%m%d%H'), plot_map_hour))

    if plot_line is not None:
        plot_forecasts = [f.sel(lat=plot_point[0], lon=plot_point[1], time=date64) for f in model_forecasts]
        plot_verif = verification.sel(lat=plot_point[0], lon=plot_point[1], time=date64)

        plot_series(plot_verif, plot_forecasts, title='HGT/500; latitude 40 longitude -75',
                    file_name='%s/%s_%s.pdf' % (plot_directory, plot_line, datetime.strftime(date, '%Y%m%d%H')))
