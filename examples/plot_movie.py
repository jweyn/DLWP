#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Plot a sequence of forecasts from a DLWP model.
"""

from DLWP.model import SeriesDataGenerator, verify, TimeSeriesEstimator, DLWPFunctional
from DLWP.util import load_model
from DLWP.plot.util import blue_red_colormap
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.basemap import Basemap


#%% User parameters

# Data file
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_6h_1979-2010_z500_tau300-700.nc' % root_directory

# Model name
model = '%s/dlwp_6h_global_tau-lstm_z-tau-out' % root_directory
model_label = r'$\tau$ LSTM global'

# Selection from the predictor file
input_selection = {'varlev': ['HGT/500', 'THICK/300-700']}
output_selection = {'varlev': ['HGT/500', 'THICK/300-700']}
input_time_steps = 2
output_time_steps = 2
add_insolation = False

# Date(s) of plots: the initialization times
plot_dates = list(pd.date_range('2007-04-15', '2007-04-15', freq='D').to_pydatetime())
num_plot_steps = 336 // 6
model_dt = 6

# Variable and level index to plot; scaling option
selection = {'varlev': 'HGT/500'}
scale_variables = True
scale_factor = 0.1

# Models which use up-sampling need to have an even number of latitudes. This is usually done by cropping out the
# north pole. Set this option to do that.
crop_north_pole = True

# Plot options
plot_type = 'contour'
plot_errors = True
plot_colormap = 'winter'
plot_colorbars = False
contour_range = [480, 600]
contour_step = 6

# Add a Laplacian to the forecast maps (e.g., vorticity)
plot_laplace = True
laplace_colormap = blue_red_colormap(256, white_padding=64)
laplace_range = [-2., 2.]
laplace_scale = 1.e4 * 9.81 / (2 * 7.29e-5)

# Output file and other small details
plot_directory = './Plots/tau-lstm-global'
plot_file_name = 'MAP_tau-lstm-global'
plot_file_type = 'png'


#%% Plot functions

def make_plot(m, time, verif, forecast, fill=None, file_name=None):

    fig = plt.figure()
    fig.set_size_inches(9, 6)
    gs1 = gs.GridSpec(2, 1)
    gs1.update(wspace=0.04, hspace=0.04)

    plot_fn = getattr(m, plot_type)
    contours = np.arange(np.min(contour_range), np.max(contour_range), contour_step)
    if fill is None:
        fill = [None] * 2

    def plot_panel(n, da, title, filler):
        ax = plt.subplot(gs1[n])
        lons, lats = np.meshgrid(da.lon, da.lat)
        x, y = m(lons, lats)
        m.drawcoastlines(color=(0.7, 0.7, 0.7))
        m.drawparallels(np.arange(0., 91., 30.), color=(0.5, 0.5, 0.5))
        m.drawmeridians(np.arange(0., 361., 60.), color=(0.5, 0.5, 0.5))
        if filler is not None:
            m.pcolormesh(x, y, filler.values, vmin=np.min(laplace_range), vmax=np.max(laplace_range),
                         cmap=laplace_colormap)
        cs = plot_fn(x, y, da.values, contours, cmap=plot_colormap)
        plt.clabel(cs, fmt='%1.0f')
        ax.text(0.01, 0.01, title, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    plot_panel(0, verif, 'Observed (%s)' % datetime.strftime(time, '%HZ %e %b %Y'), fill[0])
    plot_panel(1, forecast, 'DLWP (%s)' % datetime.strftime(time, '%HZ %e %b %Y'), fill[1])

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', dpi=200)
    plt.close()


def add_pole(da):
    pole = da.sel(lat=da.lat.max()).mean('lon').drop('lat')
    pole = pole.expand_dims(dim='lat', axis=-1).assign_coords(lat=[90.])
    pole = xr.concat([pole.expand_dims(dim='lon', axis=-1).assign_coords(lon=[l]) for l in da.lon], dim='lon')
    result = xr.concat([pole, da], dim='lat')
    return result


def add_southern_hemisphere(da):
    if da.lat.min() < 0.:
        print('Latitudes below 0 exist; not adding southern hemisphere.')
        return da
    da_s = da.assign_coords(lat=(-1. * da.lat.values))
    result = xr.concat([da, da_s.sel(lat=(da_s.lat < 0.0)).isel(lat=slice(None, None, -1))], dim='lat')
    return result


def laplacian(da, engine):
    a = da.values.reshape(-1, da.sizes['lat'], da.sizes['lon'])
    a = a.transpose((1, 2, 0))
    n = engine.wavenumbers[1] + 1.
    factor = -1 * n * (n + 1) / (engine.radius ** 2.)
    result = engine.spec_to_grid(factor[:, None] * engine.grid_to_spec(a))
    result = result.transpose((2, 0, 1))
    return result.reshape(da.shape)


#%% Load the data

if not isinstance(plot_dates, list):
    plot_dates = [plot_dates]

plot_f_hour = np.arange(model_dt, num_plot_steps * model_dt + 1, model_dt)

if plot_laplace:
    from DLWP.barotropic.pyspharm_transforms import TransformsEngine

# Add dates needed for inputs to the model
init_sel = list(plot_dates)
for date in plot_dates:
    for f in range(1, input_time_steps):
        verif_date = date - timedelta(hours=f * model_dt)
        if verif_date not in init_sel:
            init_sel.append(verif_date)
init_sel.sort()

# Add dates needed for the verification
verif_sel = list(init_sel)
for date in plot_dates:
    for f in plot_f_hour:
        verif_date = date + timedelta(hours=int(f))
        if verif_date not in verif_sel:
            verif_sel.append(verif_date)

verif_sel.sort()

data = xr.open_dataset(predictor_file)
data = data.sel(sample=np.array(verif_sel, dtype='datetime64'))
data.load()

if crop_north_pole:
    data = data.isel(lat=(data.lat < 90.0))

# Get the mean and std of the data
variable_mean = data.sel(**selection).variables['mean'].values
variable_std = data.sel(**selection).variables['std'].values

os.makedirs(plot_directory, exist_ok=True)


#%% Make the verification and forecast

verification = verify.verification_from_samples(data.sel(**selection), forecast_steps=num_plot_steps, dt=model_dt)

# Scale the verification
if scale_variables:
    verification = verification * variable_std + variable_mean

print('Loading model %s...' % model)
dlwp = load_model(model)

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
if isinstance(dlwp, DLWPFunctional):
    if not hasattr(dlwp, '_n_steps'):
        dlwp._n_steps = 6
    if not hasattr(dlwp, 'time_dim'):
        dlwp.time_dim = 2
    sequence = dlwp._n_steps
else:
    sequence = None

# Create data generator
generator = SeriesDataGenerator(dlwp, data, batch_size=216,
                                input_sel=input_selection, output_sel=output_selection,
                                input_time_steps=input_time_steps, output_time_steps=output_time_steps,
                                add_insolation=add_insolation, sequence=sequence)
p_val, t_val = generator.generate([], scale_and_impute=False)

# Create TimeSeriesEstimator
estimator = TimeSeriesEstimator(dlwp, generator)

# Make a time series prediction and convert the predictors for comparison
print('Predicting with model %s...' % model_label)
forecast = estimator.predict(num_plot_steps, verbose=1)
forecast = forecast.sel(**selection)

# Scale the forecast
if scale_variables:
    forecast = forecast * variable_std + variable_mean

# Take the Laplacian if vorticity is desired
if plot_laplace:
    if not scale_variables:
        forecast_lap = forecast * variable_std + variable_mean
        verification_lap = verification * variable_std + variable_mean
    else:
        forecast_lap = forecast.copy()
        verification_lap = verification.copy()

    # Fix missing poles and flip over the equator to add the southern hemisphere back in
    if crop_north_pole:
        forecast_lap = add_pole(forecast_lap)
        verification_lap = add_pole(verification_lap)
    forecast_lap = add_southern_hemisphere(forecast_lap)
    verification_lap = add_southern_hemisphere(verification_lap)

    # Transform engine
    transform = TransformsEngine(forecast_lap.sizes['lon'], forecast_lap.sizes['lat'],
                                 2 * (forecast_lap.sizes['lat'] // 2))

    # Transform and crop back the forecast Laplacian
    forecast_lap[:] = laplacian(forecast_lap, transform)
    forecast_lap = forecast_lap.sel(lat=forecast.lat, lon=forecast.lon)
    forecast_lap = laplace_scale * forecast_lap

    # Transform and crop back the verification Laplacian
    verification_lap[:] = laplacian(verification_lap, transform)
    verification_lap = verification_lap.sel(lat=verification.lat, lon=verification.lon)
    verification_lap = laplace_scale * verification_lap


#%% Plot the forecasts

basemap = Basemap(llcrnrlon=0, llcrnrlat=np.min(verification.lat), urcrnrlon=360, urcrnrlat=90,
                  resolution='l', projection='cyl', lat_0=40., lon_0=0.)

for date in plot_dates:
    print('Plotting for %s...' % date)
    date64 = np.datetime64(date)

    for step in range(num_plot_steps):
        print('  image %d of %d...' % (step + 1, num_plot_steps))
        f_hour = model_dt * (step + 1)
        plot_time = date + timedelta(hours=f_hour)

        file_name_complete = '%s/%s_%s_f%03d.%s' % (plot_directory, plot_file_name,
                                                    datetime.strftime(date, '%Y%m%d%H'), step + 1, plot_file_type)

        make_plot(basemap, plot_time, scale_factor * verification.sel(time=date64, f_hour=np.timedelta64(f_hour, 'h')),
                  forecast=scale_factor * forecast.sel(time=date64, f_hour=np.timedelta64(f_hour, 'h')),
                  fill=None if not plot_laplace else (verification_lap.sel(time=date64,
                                                                           f_hour=np.timedelta64(f_hour, 'h')),
                                                      forecast_lap.sel(time=date64,
                                                                       f_hour=np.timedelta64(f_hour, 'h'))),
                  file_name=file_name_complete)
