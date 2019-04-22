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
from DLWP.data import CFSReforecast
import keras.backend as K
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import string
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.basemap import Basemap


#%% User parameters

# Open the data file
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_analysis_2007-2009_hgt-thick_300-500-700_NH_T2.nc' % root_directory

# Names of model files, located in the root_directory, and labels for those models (don't use /)
models = [
    'dlwp_1979-2010_hgt_500_NH_T2F_FINAL',
    'dlwp_1979-2010_hgt-thick_300-500-700_NH_T2F_FINAL',
    'dlwp_1979-2010_hgt_500_NH_T2F_FINAL-lstm',
    'dlwp_1979-2010_hgt-thick_300-500-700_NH_T2F_FINAL-lstm'
]
model_labels = [
    '$Z$',
    r'$\tau$',
    '$Z$ LSTM',
    r'$\tau$ LSTM'
]

# Optional list of selections to make from the predictor dataset for each model. This is useful if, for example,
# you want to examine models that have different numbers of vertical levels but one predictor dataset contains
# the data that all models need.
predictor_sel = [
    {'variable': ['HGT']},
    None,
    {'variable': ['HGT']},
    None
]

# Load a barotropic model
baro_model_file = '%s/barotropic_anal_2007-2009.nc' % root_directory
baro_ds = xr.open_dataset(baro_model_file, cache=False)

# Load the CFS model
cfs_model_dir = '%s/../CFSR/reforecast' % root_directory
cfs = CFSReforecast(root_directory=cfs_model_dir, file_id='dlwp_', fill_hourly=False)

# Date(s) of plots: the initialization time
plot_dates = list(pd.date_range('2007-04-12', '2007-04-17', freq='D').to_pydatetime())
plot_forecast_hour = 24
model_dt = 6

# Variable and level index to plot; scaling option
variable_sel = {
    'variable': 'HGT',
    'level': 500
}
scale_variables = True
scale_factor = 0.1

# Models which use up-sampling need to have an even number of latitudes. This is usually done by cropping out the
# north pole. Set this option to do that.
crop_north_pole = True

# Latitude / Longitude limits
latitude_range = [20., 80.]
longitude_range = [220., 300.]

# Plot options
plot_type = 'contour'
plot_errors = True
plot_colormap = 'winter'
plot_colorbars = False
contour_range = [480, 600]
contour_step = 6
error_maxmin = 20

# Add a Laplacian to the forecast maps (e.g., vorticity)
plot_laplace = True
laplace_colormap = 'seismic'
laplace_range = [-3., 3.]
laplace_scale = 1.e4 * 9.81 / (2 * 7.29e-5)

# Output file and other small details
plot_directory = './Plots'
plot_file_name = 'MAP_Z_VORT_ANAL_24'
plot_file_type = 'pdf'


#%% Plot function

def make_plot(m, time, init, verif, forecasts, model_names, fill=None, skip_plots=(), file_name=None):
    num_panels = len(forecasts) + 2
    num_cols = int(np.ceil(num_panels / 2))
    verif_time = time + timedelta(hours=plot_forecast_hour)

    fig = plt.figure()
    fig.set_size_inches(4 * num_cols, 6)
    gs1 = gs.GridSpec(2, num_cols)
    gs1.update(wspace=0.04, hspace=0.04)

    plot_fn = getattr(m, plot_type)
    contours = np.arange(np.min(contour_range), np.max(contour_range), contour_step)
    diff = None
    if fill is None:
        fill = [None] * (len(forecasts) + 2)

    def plot_panel(n, da, title, filler):
        ax = plt.subplot(gs1[n])
        lons, lats = np.meshgrid(da.lon, da.lat)
        x, y = m(lons, lats)
        m.drawcoastlines(color=(0.7, 0.7, 0.7))
        m.drawparallels(np.arange(0., 91., 30.))
        m.drawmeridians(np.arange(0., 361., 60.))
        if filler is not None:
            m.pcolormesh(x, y, filler.values, vmin=np.min(laplace_range), vmax=np.max(laplace_range),
                         cmap=laplace_colormap)
            # plt.colorbar()
        if diff is not None:
            m.pcolormesh(x, y, da.values - diff.values, vmin=-error_maxmin, vmax=error_maxmin, cmap='seismic',
                         alpha=0.4)
            # plt.colorbar()
        cs = plot_fn(x, y, da.values, contours, cmap=plot_colormap)
        plt.clabel(cs, fmt='%1.0f')
        ax.text(0.01, 0.01, title, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    plot_panel(0, init, 'a) Initial (%s)' % datetime.strftime(time, '%HZ %e %b %Y'), fill[0])
    plot_panel(1, verif, 'b) Verification (%s)' % datetime.strftime(verif_time, '%HZ %e %b %Y'), fill[1])
    plot_num = 2
    if plot_errors and not plot_laplace:
        diff = verif
    for f, forecast in enumerate(forecasts):
        if f + 2 in skip_plots:
            plot_num += 1
        plot_panel(plot_num, forecast, '%s) %s' % (string.ascii_lowercase[f+2], model_names[f]), fill[f+2])
        plot_num += 1

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()


def add_pole(da):
    pole = da.sel(lat=da.lat.max()).mean('lon').drop('lat')
    pole = pole.expand_dims(dim='lat', axis=-1).assign_coords(lat=[90.])
    pole = xr.concat([pole.expand_dims(dim='lon', axis=-1).assign_coords(lon=[l]) for l in da.lon], dim='lon')
    result = xr.concat([pole, da], dim='lat')
    return result


def add_southern_hemisphere(da):
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

if plot_laplace:
    from DLWP.barotropic.pyspharm_transforms import TransformsEngine

# Add verification dates to the dataset
sel_dates = list(plot_dates)
for date in plot_dates:
    verif_date = date + timedelta(hours=plot_forecast_hour)
    if verif_date not in sel_dates:
        sel_dates.append(verif_date)
sel_dates.sort()

data = xr.open_dataset(predictor_file)
data = data.sel(sample=np.array(sel_dates, dtype='datetime64'))

lat_min = np.min(latitude_range)
lat_max = np.max(latitude_range)
lon_min = np.min(longitude_range)
lon_max = np.max(longitude_range)

# Get the mean and std of the data
variable_mean = data.sel(**variable_sel).variables['mean'].values
variable_std = data.sel(**variable_sel).variables['std'].values


#%% Make forecasts

model_forecasts = []
if plot_laplace:
    laplace_forecasts = []
else:
    laplace_fill = None
num_forecast_steps = int(np.ceil(plot_forecast_hour / model_dt))
f_hour = np.arange(model_dt, num_forecast_steps * model_dt + 1, model_dt)
dlwp, p_val, t_val = None, None, None

for mod, model in enumerate(models):
    print('Loading model %s...' % model)
    dlwp, history = load_model('%s/%s' % (root_directory, model), True)

    # Create data generator
    if predictor_sel[mod] is not None:
        val_ds = data.sel(**predictor_sel[mod])
    else:
        val_ds = data.copy()
    if crop_north_pole:
        val_ds = val_ds.isel(lat=(val_ds.lat < 90.0))
    val_generator = DataGenerator(dlwp, val_ds, batch_size=216)
    p_val, t_val = val_generator.generate([], scale_and_impute=False)

    # Make a time series prediction and convert the predictors for comparison
    print('Predicting with model %s...' % model_labels[mod])
    time_series = dlwp.predict_timeseries(p_val, num_forecast_steps)
    if scale_variables:
        time_series = time_series * variable_std + variable_mean
    time_series = verify.add_metadata_to_forecast(time_series, f_hour, val_ds)

    # Take the Laplacian if vorticity is desired
    time_series = time_series.sel(**variable_sel)
    if plot_laplace:
        if not scale_variables:
            lap_series = time_series * variable_std + variable_mean
        else:
            lap_series = time_series.copy()

        # Fix missing poles and flip over the equator to add the southern hemisphere back in
        if crop_north_pole:
            lap_series = add_pole(lap_series)
        lap_series = add_southern_hemisphere(lap_series)

        # Transform engine
        transform = TransformsEngine(lap_series.sizes['lon'], lap_series.sizes['lat'],
                                     2 * (lap_series.sizes['lat'] // 2))
        lap_series[:] = laplacian(lap_series, transform)
        lap_series = lap_series.sel(lat=((lap_series.lat >= lat_min) & (lap_series.lat <= lat_max)),
                                    lon=((lap_series.lon >= lon_min) & (lap_series.lon <= lon_max)))

        laplace_forecasts.append(laplace_scale * lap_series)

    # Slice the array as we want it
    time_series = time_series.sel(lat=((time_series.lat >= lat_min) & (time_series.lat <= lat_max)),
                                  lon=((time_series.lon >= lon_min) & (time_series.lon <= lon_max)))

    model_forecasts.append(scale_factor * time_series)

    # Clear the model
    dlwp, time_series, p_val, t_val = None, None, None, None
    K.clear_session()


#%% Add the barotropic model

if baro_ds is not None:
    baro = baro_ds['Z'].sel(f_hour=(baro_ds.f_hour <= plot_forecast_hour), time=plot_dates)
    baro.load()
    if plot_laplace:
        baro_lap = xr.DataArray(laplacian(baro, transform), coords=baro.coords)
        baro_lap = baro_lap.sel(lat=((baro_ds.lat >= lat_min) & (baro_ds.lat <= lat_max)),
                                lon=((baro_ds.lon >= lon_min) & (baro_ds.lon <= lon_max)))
        laplace_forecasts.append(laplace_scale * baro_lap)
    baro = baro.sel(lat=((baro_ds.lat >= lat_min) & (baro_ds.lat <= lat_max)),
                    lon=((baro_ds.lon >= lon_min) & (baro_ds.lon <= lon_max)))
    if not scale_variables:
        baro = (baro - variable_mean) / variable_std
    model_forecasts.append(scale_factor * baro)
    model_labels.append('Barotropic')


#%% Add the CFS model

if cfs is not None:
    cfs.set_dates(sel_dates)
    cfs.open()
    cfs_da = cfs.Dataset['z500'].sel(f_hour=(cfs.Dataset.f_hour <= plot_forecast_hour), time=plot_dates)
    cfs_da.load()
    if plot_laplace:
        cfs_lap = xr.DataArray(laplacian(cfs_da, transform), coords=cfs_da.coords)
        cfs_lap = cfs_lap.sel(lat=((cfs_lap.lat >= lat_min) & (cfs_lap.lat <= lat_max)),
                              lon=((cfs_lap.lon >= lon_min) & (cfs_lap.lon <= lon_max)))
        laplace_forecasts.append(laplace_scale * cfs_lap)
    cfs_da = cfs_da.sel(lat=((cfs_da.lat >= lat_min) & (cfs_da.lat <= lat_max)),
                        lon=((cfs_da.lon >= lon_min) & (cfs_da.lon <= lon_max)))
    if not scale_variables:
        cfs_da = (cfs_da - variable_mean) / variable_std
    model_forecasts.append(scale_factor * cfs_da)
    model_labels.append('CFS')


#%% Run the plots

basemap = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,
                  resolution='l', projection='cyl', lat_0=40., lon_0=0.)

# Rearrange so that Baro/CFS are at the beginning
model_labels = model_labels[mod+1:] + model_labels[:mod+1]
model_forecasts = model_forecasts[mod+1:] + model_forecasts[:mod+1]
if plot_laplace:
    laplace_forecasts = laplace_forecasts[mod + 1:] + laplace_forecasts[:mod + 1]

for date in plot_dates:
    print('Plotting for %s...' % date)
    date64 = np.datetime64(date)
    verif_date64 = date64 + np.timedelta64(timedelta(hours=plot_forecast_hour))

    plot_fields = [f.sel(f_hour=plot_forecast_hour, time=date64) for f in model_forecasts]

    init_data = data['predictors'].isel(time_step=-1).sel(sample=date64, **variable_sel)
    verif_data = data['predictors'].isel(time_step=-1).sel(sample=verif_date64, **variable_sel)
    if scale_variables:
        init_data = init_data * variable_std + variable_mean
        verif_data = verif_data * variable_std + variable_mean
    if plot_laplace:
        if not scale_variables:
            init_lap = init_data * variable_std + variable_mean
            verif_lap = verif_data * variable_std + variable_mean
        else:
            init_lap = init_data.copy()
            verif_lap = verif_data.copy()
        init_lap = add_southern_hemisphere(init_lap)
        verif_lap = add_southern_hemisphere(verif_lap)
        init_lap[:] = laplace_scale * laplacian(init_lap, transform)
        verif_lap[:] = laplace_scale * laplacian(verif_lap, transform)
        init_lap = init_lap.sel(lat=((init_lap.lat >= lat_min) & (init_lap.lat <= lat_max)),
                                lon=((init_lap.lon >= lon_min) & (init_lap.lon <= lon_max)))
        verif_lap = verif_lap.sel(lat=((verif_lap.lat >= lat_min) & (verif_lap.lat <= lat_max)),
                                  lon=((verif_lap.lon >= lon_min) & (verif_lap.lon <= lon_max)))
        laplace_fill = [f.sel(f_hour=plot_forecast_hour, time=date64) for f in laplace_forecasts]
        laplace_fill = [init_lap, verif_lap] + laplace_fill

    init_data = init_data.sel(lat=((init_data.lat >= lat_min) & (init_data.lat <= lat_max)),
                              lon=((init_data.lon >= lon_min) & (init_data.lon <= lon_max)))
    verif_data = verif_data.sel(lat=((verif_data.lat >= lat_min) & (verif_data.lat <= lat_max)),
                                lon=((verif_data.lon >= lon_min) & (verif_data.lon <= lon_max)))

    file_name_complete = '%s/%s_%s.%s' % (plot_directory, plot_file_name, datetime.strftime(date, '%Y%m%d%H'),
                                          plot_file_type)

    make_plot(basemap, date, scale_factor * init_data, scale_factor * verif_data,
              plot_fields, model_labels, fill=laplace_fill, file_name=file_name_complete)
