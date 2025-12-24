# Re-writing  of S3M with python language. Project related to the doctoral
# thesis:Data assimilation and deep learning. I will try to re-write
# S3M in python and develop an EnKF data assimilation procedure to
# replace the current nudging procedure. Then I will use data assimilation
# data to train a neural network (LSTM), to exploit the computational
# capacity of DL to make this procedure  comparable with operative times.
# I will perform a point data assimilation, and then I will interpolate
# the corrections to the whole  domain using gaussian process regression.
# ----------------------------------------------------------------------------------------------------------------------
"""
S3M 1D main
__date__ = '2025'
__version__ = '2.0.0'
__author__ =
        'Giulia Blandini'(giulia.blandini@cimafoundation.org',

__references__=
        'Francesco Avanzi' (francesco.avanzi@cimafoundation.org'),
        'Fabio Delogu' (fabio.delogu@cimafoundation.org'),
        'Simone Gabellani' (simone.gabellani@cimafoundation.org'),

__library__ = 's3m libraries on fortran  and new : lib_utilis_flux , lib_utilis_pre_proc , diagnostic'

General command line:
python S3M_2D_run.py -settings_file "configuration_s3m_1D_package.json" -time_start "{ fill in }"
 -time_end {fill in -domain "{ fill in }"

Calibration approach :
@Misc{,
    author = {Fernando Nogueira},
    title = {{Bayesian Optimization}: Open source constrained global optimization tool for {Python}},
    year = {2014--},
    url = " https://github.com/bayesian-optimization/BayesianOptimization"
}
downscaling : TopoPyScale
@article{Filhol2023, doi = {10.21105/joss.05059},
url = {https://doi.org/10.21105/joss.05059},
year = {2023}, publisher = {The Open Journal},
volume = {8}, number = {86}, pages = {5059},
 author = {Simon Filhol and Joel Fiddes and Kristoffer Aalstad},
 title = {TopoPyScale: A Python Package for Hillslope Climate Downscaling},
 journal = {Journal of Open Source Software} }
}
"""
import logging
import numpy as np
import pandas
import os
import pickle
import xarray as xr
import time
import gc
import rioxarray as rxr
from datetime import datetime
from lib_utils_system import fill_tags2string
from lib_utils_logging import set_logging_file
from lib_data_io_json import read_file_settings
from S3M_2D_physics import S3M_2D_physics
from lib_utilis_data_proc import get_args, rmse
from bayes_opt import BayesianOptimization
from PLOTS_S3M import process_and_plot_snow_data, plot_ensemble, plot_meteo_ensemble
from S3M_2D_assimilation import perturb_point,assimilation_point
from S3M1DTEST import plot_ensemble, plot_assimilation
from joblib import Parallel, delayed
from S3M_1D_physics import S3M_1D_physics_points
import matplotlib.pyplot as plt




def compute_relative_humidity(temp , qair, pressure):
    """
    Compute relative humidity  using specific humidity, temperature, and pressure.
    qair: specific humidity (kg/kg)
    temp: temperature (K)
    pressure: pressure (Pa)
    Returns relative humidity in percentage (0-100%).
    """
    mr = qair / (1 - qair)
    e = mr * pressure / (0.62197 + mr)
    es = 611.2 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))
    rh = e / es
    rh = np.clip(rh, 0, 1)
    rh_percentage = rh * 100
    return rh_percentage

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def log_and_return(cal, log_stream, start_time, rmse_hs,st_dev_hs):
    """
    Print the execution time and return the desired
    metric based on calibration flag
    """
    end_time = datetime.now()
    log_stream.info(f"Execution time: {end_time - start_time}")

    if cal == 1:
        if rmse_hs is not None and st_dev_hs is not None:
                log_stream.info(f'RMSE HS: {rmse_hs}\n')
                return rmse_hs / st_dev_hs
        else:
            return 1000
    else:
        if rmse_hs is not None:
            log_stream.info(f'RMSE HS: {rmse_hs}\n')
            return rmse_hs
        else:
            return np.nan


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def process_sample(mrad, mr, window_melting, lat, lon, values, start, end, state_input, output_input):

    """ Return the negative of the cost function for minimization """

    f_cost = S3M_2D(mrad, mr, window_melting, [lat], [lon], values, start, end, state_input, output_input)

    return -f_cost


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def optimize_for_key(key,values,start,end):

    """ Perform Bayesian optimization for a specific (lat, lon) key """


    lon,lat = key
    log_stream.info(f"Latitude: {lat}, Longitude: {lon}")

    if all(np.isnan(values['snd'])):
        log_stream.info(f"Skipping optimization for {lat}, {lon}")
        return {'Latitude': lat, 'Longitude': lon, 'SAMPLE': None}

    pbounds = {'mrad': (0.8, 2.5), 'mr': (0.5, 3), 'window_melting': (2, 30) ,'alpha': (0.1, 0.9)}

    optimizer = BayesianOptimization(
        f=lambda mrad, mr, window_melting, alpha: process_sample(mrad, mr, window_melting, lat, lon, values, start, end, state_input, output_input),
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.probe(
        params={'mrad': 1.125, 'mr': 1.1, 'window_melting': 10},
        lazy=True,
    )
    optimizer.maximize(init_points=3, n_iter=10)

    return {'Latitude': lat, 'Longitude': lon, 'SAMPLE': optimizer.max['params'],'F_COST': optimizer.max['target']}


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------



def S3M_2D(mrad, mr, window_melting,alpha ,lat, lon, values, start, end,  state_input, output_input):
    #  ------------------------------------------------------------------------------------------------------------------
    # trace time of the algorithm
    t_start = time.time()
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    # ------------------------------------------------------------------------------------------------------------------
    # Set algorithm settings
    alg_settings, alg_time_start, alg_time_end, alg_domain = get_args()
    data_settings = read_file_settings(alg_settings)
    # ------------------------------------------------------------------------------------------------------------------
    # Set logging
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger_name = "S3M_2D_logger"
    log_stream = logging.getLogger(logger_name)
    # ------------------------------------------------------------------------------------------------------------------
    try :
        start_datetime = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        start_datetime = datetime.strptime(alg_time_start, "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(alg_time_end, "%Y-%m-%d %H:%M:%S")

    start_year = start_datetime.year
    end_year = end_datetime.year
    lenght_time = start+"_" + end
    tag = {'source_file_datetime' : lenght_time}
    # -------------------------------------- #FLAGS# --------------------------------------------------------------------

    ncdf = data_settings['data']['info_file']['ncdf']
    data_assimilation= data_settings['data']['info_file']["perturbations"]["per_Obs"]
    cal = data_settings['calibration']['calibrate']
    change_part = data_settings['data']['info_file']["change_part"]
    Ice_flag = data_settings['data']['info_file']["Ice_flag"]
    comparison_ol = data_settings['data']['info_file']['comparison_ol']
    downscaled = data_settings['data']['info_file']['downscaled']
    # ------------------------------------------------------------------------------------------------------------------
    # upload files
    comparison_ol_meteo = data_settings['data']['info_file']['comparison_ol_meteo']
    obs = data_settings['data']['info_file']["time_series_obs"]
    obs_mask = data_settings['data']['info_file']["obs_mask"]
    dem = data_settings['data']["info_file"]["dem"]
    input_path = data_settings['data']['info_file']['input_path']
    w = pandas.read_pickle(data_settings['data']['info_file']['weights'])
    svf = xr.open_dataset(data_settings['data']['info_file']['svf'], engine='netcdf4')
    slope = rxr.open_rasterio(data_settings['data']['info_file']['slope'], engine='rasterio').squeeze()
    gc.collect()
    # ------------------------------------------------------------------------------------------------------------------
    parameters = data_settings['data']['info_file']['parameters']
    out = data_settings['data']['info_file']['out']
    state = data_settings['data']['info_file']['state']
    state_vector = [val for val in state.values()]
    output_vector = [val for val in out.values()]
    method = data_settings['data']['info_file']['method']
    window_melting = int(parameters["window_melting"])
    # ------------------------------------------------------------------------------------------------------------------
    # create a figure with two subplots
    if 1:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        values = {
            'val_x': [0],
            'val_swe_a': [0],
            'val_swe_b': [0],
            'val_swe_b1': [0],
            'val_swe_open': [0],
            'val_hs_a': [0],
            'val_hs_b': [0],
            'val_hs_b1': [0],
            'val_hs_open': [0],
            'val_hs_obs': [0],
        }

    if cal != 1:
        trial = {'source_file_datetime_generic': start_time_str}
        log_folder_name = data_settings["log"]["folder_name"]
        logger_name = logger_name
        file_name = fill_tags2string(data_settings['log']['file_name'], data_settings['template'], trial)
        set_logging_file(logger_name=logger_name, logger_file=os.path.join(log_folder_name, file_name))
        dem_da = rxr.open_rasterio(dem, engine='rasterio').sel(band=1).drop_vars('band').chunk({"x": 100, "y": 100})
        lon_dem = dem_da.x.values
        lat_dem = dem_da.y.values
        ny, nx = dem_da.shape
        y_dem =dem_da["y"].values
        x_dem =dem_da["x"].values
        try:
            interpolated_par = xr.open_dataset(parameters["interpolated_parameters"])
            parameters["mrad0"] = interpolated_par["mrad"].values
            parameters["mr0"] = interpolated_par["mr"].values
            parameters["window_melting"] = interpolated_par["window_melting"]
            #parameters["alpha"] = interpolated_par["alpha"]
        except:
            print("single parameters")
        snow_depth_series = []
        swe_series = []
        density_series = []
        gc.collect()
    # ------------------------------------------------------------------------------------------------------------------
    if cal == 1:
        calibration = os.path.join(data_settings['calibration']['folder_name'], str(start_year) + "_" + str(end_year))
        # make dir
        os.makedirs(calibration, exist_ok=True)
        parameters["mrad0"] = np.round(mrad, 3)
        parameters["mr0"] = np.round(mr, 3)
        parameters["window_melting"] = int(window_melting)
        #parameters["alpha"] = np.round(alpha, 3)
        snow_depth_series = []
        swe_series = []
        density_series = []
        lat = lat
        lon = lon
    # ------------------------------------------------------------------------------------------------------------------
    elif data_assimilation == 1:
        project_name = 'PhD_project  S3M 2D -DATA ASSIMILATION'
        N = data_settings['data']['info_file']["perturbations"]['N']
        alg_type = 'PYTHON VERSION'
        log_stream.info(' ============================================================================')
        log_stream.info('[' + project_name + ' ' + alg_type + ' - ')
        log_stream.info(' ============================================================================')
        log_stream.info('[' + str(N) + ' ' + 'ENSEMBLE MEMBERS')
        log_stream.info(' ============================================================================')
        # --------------------------------------------------------------------------------------------------------------
        perturbations_data_file = fill_tags2string(data_settings['data']['info_file']["perturbations"]['file_pkl'], data_settings['template'], tag)
        perturbations_data = pandas.read_pickle(os.path.join(data_settings['data']['info_file']["perturbations"]['folder_pkl'], perturbations_data_file))
        R_dict= pandas.read_pickle(fill_tags2string(data_settings['data']['info_file']["perturbations"]['R'], data_settings['template'], tag))
        statistics = pandas.read_pickle(fill_tags2string(data_settings['data']['info_file']["perturbations"]['statistics'], data_settings['template'], tag))
        R_state = pandas.read_pickle(fill_tags2string(data_settings['data']['info_file']["perturbations"]['R_state'], data_settings['template'], tag))
        statistics_state = pandas.read_pickle(fill_tags2string(data_settings['data']['info_file']["perturbations"]['statistics_state'], data_settings['template'], tag))
        state_limits = pandas.read_pickle(fill_tags2string(data_settings['data']['info_file']["perturbations"]['state_limits'], data_settings['template'], tag))
        inflation_deflation = [f for f in data_settings['data']['info_file']["perturbations"]['inflation_deflation'].values()]
        # --------------------------------------------------------------------------------------------------------------
        scale_mean_prec = data_settings['data']['info_file']["perturbations"]["scale_mean_prec"]
        pert_asymm_prec = data_settings['data']['info_file']["perturbations"]["pert_asymm_prec"]
        c_asymm_prec = data_settings['data']['info_file']["perturbations"]["c_asymm_prec"]
        pert_prec = data_settings['data']['info_file']["perturbations"]["pert_prec"]
        pert_rad = data_settings['data']['info_file']["perturbations"]["pert_rad"]
        pert_temp = data_settings['data']['info_file']["perturbations"]["pert_temp"]
        pert_rh = data_settings['data']['info_file']["perturbations"]["pert_rh"]
        # --------------------------------------------------------------------------------------------------------------
        # Define error covariance matrices
        sigma_err_swe = data_settings['data']['info_file']["perturbations"]["error_sigma_swe"]
        sigma_err_hs = data_settings['data']['info_file']["perturbations"]["error_sigma_snow_depth"]
        R_measures = np.array([[sigma_err_swe ** 2, 0], [0, sigma_err_hs ** 2]])
        L_tilde = pickle.load(open(data_settings['data']['info_file']["perturbations"]['L_tilde'], "rb"))
        L0 = pickle.load(open(data_settings['data']['info_file']["perturbations"]['L0'], "rb"))
        gc.collect()
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    if downscaled == 1:
        air_temp_tag = data_settings['data']["info_file"]['tags']['temperature_tag']
        precip_tag = data_settings['data']["info_file"]['tags']['prc_tag']
        rel_hum_tag = data_settings['data']["info_file"]['tags']['rh_tag']
        solar_rad_tag = data_settings['data']["info_file"]['tags']['rad_tag']
        specific_humidity_tag = data_settings['data']["info_file"]['tags']['specific_humidity_tag']
        pressure_tag = data_settings['data']["info_file"]['tags']['pressure_tag']
        var_list = [air_temp_tag, precip_tag,solar_rad_tag, specific_humidity_tag, "t1d" ,"t10d", pressure_tag]

        yy_coarse, xx_coarse = np.meshgrid(y_dem, x_dem, indexing='ij')  # 'ij' keeps shape as (ny, nx)
        multi_index_coarse = pandas.MultiIndex.from_arrays(
            [yy_coarse.ravel(), xx_coarse.ravel()],
            names=("y", "x"))
        lookup_table = {(y_val, x_val): idx
            for idx, (y_val, x_val) in enumerate(multi_index_coarse)}
        print(f"MultiIndex created in {time.time() - t_start:.2f} seconds")

        gc.collect()

        downscaled_ds = xr.open_dataset(input_path, chunks={"time": 100})  # Adjust chunk size as needed
        downscaled_ds = downscaled_ds.sel(time=slice(start_datetime, end_datetime))
        vars_in_ds = [v for v in var_list if v in downscaled_ds.data_vars]
        downscaled_ds = downscaled_ds[vars_in_ds]
        gc.collect()
        # ------------------------------------------------------------------------------------------------------------------
        Time = pandas.to_datetime(downscaled_ds.time.values)
        t1d = downscaled_ds[air_temp_tag].rolling(time=24, min_periods=1).mean()
        t10d = downscaled_ds[air_temp_tag].rolling(time=24, min_periods=1).mean()
        # convert into DataArray
        downscaled_ds["t1d"] = xr.DataArray(t1d, dims=downscaled_ds[air_temp_tag].dims,
                                            coords=downscaled_ds[air_temp_tag].coords)
        downscaled_ds["t10d"] = xr.DataArray(t10d, dims=downscaled_ds[air_temp_tag].dims,
                                             coords=downscaled_ds[air_temp_tag].coords)
        nt = len(Time)
        ng = ny * nx
        state_matrix = np.zeros((nt, ng, 4), dtype=np.float32)
        output_matrix = np.zeros((nt, ng, 17), dtype=np.float32)
        meteo = np.zeros((nt, ng, 6), dtype=np.float32)
        state_backup = np.zeros((4, ng,), dtype=np.float32)
        output_backup = np.zeros((17, ng,), dtype=np.float32)

        if state_input and output_input is not None:
            state_input = state_input.to_array()
            output_input = output_input.to_array()

            for q in range(0, 4):
                state_backup[:, :, q] = state_input[q, :, :].values

            for p in range(0, 6):
                output_backup[:, :, p] = output_input[p, :, :].values

            output_backup[:, :, 10] = output_input[6, :, :].values
            output_backup[:, :, 11] = output_input[7, :, :].values
            output_backup[:, :, 12] = ((state_input[1, :, :].values / 1000) * parameters["RhoW"]) / state_input[2, :,
                                                                                                    :].values
            output_backup[:, :, 13] = output_input[8, :, :].values
            output_backup[:, :, 14] = output_input[9, :, :].values
            output_backup[:, :, 15] = output_input[10, :, :].values
        else:
            state_matrix[0, :, :, :] = state_vector
            output_matrix[0, :, :, :] = output_vector
        print(f"Open data in {time.time() - t_start:.2f} seconds")
        gc.collect()

        n_clusters = 1500
        result = {}
        # inserisci un decoratore per velocizzare il codice
        for var in var_list:
            data = downscaled_ds[var].values  # shape: (n_clusters, n_time)
            n_time = data.shape[1]
            mapped = np.full((ny, nx, n_time), np.nan)
            for i in range(ny):  # x-direction
                for j in range(nx):  # y-direction
                    if (i, j) in w:
                        wij = w[i, j]  # dict {cluster_id: freq}
                        keys = list(wij.keys())
                        # Filter only valid cluster IDs
                        valid_keys = [k for k in keys if 0 <= k < n_clusters]
                        if len(valid_keys) > 0:
                            weights = np.array([wij[k] for k in valid_keys])
                            values = np.array([data[k, :] for k in valid_keys])  # shape: (n_valid, n_time)
                            # Weighted average across clusters → shape: (n_time,)
                            mapped[i,j, :] = np.average(values, axis=0, weights=weights)
            # Flatten to (point, time)
            mapped_flat = mapped.reshape(ny * nx, n_time)
            result[var] = mapped_flat

        # Wrap into Dataset
        era5_mapped = xr.Dataset({
            var: xr.DataArray(data=vals, coords={'point': multi_index_coarse, 'time': Time}, dims=('point', 'time'))
            for var, vals in result.items()
        })
        print(f"Mapped data in {time.time() - t_start:.2f} seconds")
        downscaled_ds.close()
        gc.collect()
        meteo[:, :, 0] = era5_mapped[air_temp_tag] - 273.15  # Directly assign air temperature in Celsius
        meteo[:, :, 1] = era5_mapped[precip_tag]  # Precipitation
        meteo[:, :, 2] = compute_relative_humidity(
            era5_mapped[air_temp_tag],
            era5_mapped[specific_humidity_tag],
            era5_mapped[pressure_tag])  # Relative humidity
        meteo[:, :, 3] = era5_mapped[solar_rad_tag]  # Solar radiation
        meteo[:, :, 4] = era5_mapped["t1d"] - 273.15
        meteo[:, :, 5] = era5_mapped["t10d"] - 273.15  # Average temperature over 10 days
        print(f"Meteo matrix built in {time.time() - t_start:.2f} seconds")
        era5_mapped.close()
        gc.collect()
        # Build meteo matrix
        if cal == 1:
            era5 = era5_mapped.unstack("point").compute()  # Compute only when necessary
            era5_mapped = era5.sel(y=lat, x=lon,
                                   method='nearest')

        # --------------------------------------------------------------------------------------------------------------------------
        if data_assimilation == 1 and cal == 0:
            obs_mask = pandas.read_pickle(obs_mask)
            ys, xs = zip(*obs_mask)

        # --------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------
    elif downscaled == 0:
        ds = xr.open_dataset(comparison_ol_meteo)
        meteo_ds = ds[["AirTemperature", "IncRadiation", "Rain", "RelHumidity"]]
        Time = pandas.date_range(start=start_datetime, end= end_datetime , freq='h')
        meteo_ds = meteo_ds.sel(time=slice(start_datetime, end_datetime))
        #start_datetime = "2019-01-15 00:00:00"
        #end_datetime = "2019-01-31  00:00:00"
        #Time = pandas.date_range(start=start_datetime, end= end_datetime , freq='h')
        #meteo_ds = meteo_ds.sel(time=slice(start_datetime, end_datetime))
        nx_meteo, ny_meteo = len(meteo_ds.lon), len(meteo_ds.lat)
        if nx_meteo != nx or ny_meteo != ny:
            Latitudes_1d = np.unique(meteo_ds.lat)
            Longitudes_1d = np.unique(meteo_ds.lon)
            AirTemperature_3D = np.repeat(
                meteo_ds['AirTemperature'].values[:, :, np.newaxis],  # shape (time, 43, 1)
                43, axis=2
            )  # now shape is (8760, 43, 43)

            Rain_3D = np.repeat(
                meteo_ds['Rain'].values[:, :, np.newaxis],  # shape (time, 43, 1)
                43, axis=2
            )  # now shape is (8760, 43, 43)

            IncRadiation_3D = np.repeat(
                meteo_ds['IncRadiation'].values[:, :, np.newaxis],  # shape (time, 43, 1)
                43, axis=2
            )  # now shape is (8760, 43, 43)
            RelHumidity_3D = np.repeat(
                meteo_ds['RelHumidity'].values[:, :, np.newaxis],  # shape (time, 43, 1)
                43, axis=2
            )  # now shape is (8760, 43, 43)

            # Build the new dataset
            meteo_ds = xr.Dataset(
                coords={
                    "time": meteo_ds.time,
                    "Latitude": Latitudes_1d,
                    "Longitude": Longitudes_1d
                },
                data_vars={
                    "AirTemperature": (("time", "Latitude", "Longitude"), AirTemperature_3D),
                    "Rain": (("time", "Latitude", "Longitude"), Rain_3D),
                    "IncRadiation": (("time", "Latitude", "Longitude"), IncRadiation_3D),
                    "RelHumidity": (("time", "Latitude", "Longitude"), RelHumidity_3D)
                })
            nx = nx_meteo
            ny = ny_meteo
            lat_dem = Latitudes_1d
            lon_dem = Longitudes_1d
        ds.close()
        gc.collect()
        # --------------------------------------------------------------------------------------------------
        if cal == 1:
            meteo_ds = meteo_ds.sel(y=lat, x=lon,method='nearest')
        # ------------------------------------------------------------------------------------------------------------------
        nt = len(Time)  # +1 to include the initial condition
        m = 6
        meteo = np.zeros((nt,ny,nx, m), dtype=np.float32)
        meteo[:, :, :, 0] = meteo_ds["AirTemperature"]
        meteo[:, :, :, 1] = meteo_ds["Rain"]
        # set as limit 0 the negative precipitation values
        meteo[:, :, :, 1] = np.where(meteo[:, :, :, 1] < 0, 0, meteo[:, :, :, 1])
        meteo[:, :, :, 2] = meteo_ds["RelHumidity"]
        # BOUND  BETWEEN 0 AND 100 %
        meteo[:, :, :, 2] = np.where(meteo[:, :, :, 2] < 0, 0, meteo[:, :, :, 2])
        # #remove rel humidity values greater than 100%
        meteo[:, :, :, 2] = np.where(meteo[:, :, :, 2] > 100, 100, meteo[:, :, :, 2])
        meteo[:, :, :, 3] = meteo_ds["IncRadiation"]
        meteo[:, :, :, 4] = meteo_ds["AirTemperature"].rolling(time=24, min_periods=1).mean()
        meteo[:, :, :, 5] = meteo_ds["AirTemperature"].rolling(time=48, min_periods=1).mean()

        gc.collect()
        print(f"Open data in {time.time() - t_start:.2f} seconds")
        #------------------------------------------------------------------------------------------------------------------

        if data_assimilation == 0:

            state_backup = np.zeros((ny, nx, 4), dtype=np.float32)
            output_backup = np.zeros((ny, nx, 17), dtype=np.float32)
            state_matrix= np.zeros((nt, ny, nx, 4), dtype=np.float32)
            output_matrix= np.zeros((nt, ny, nx, 17), dtype=np.float32)
            # save meteo as a netcdf file for checking
            meteo_output = xr.Dataset(
                    {
                        "AirTemperature": xr.DataArray(
                            meteo[:, :, :, 0],
                            dims=["time", "lat", "lon"],
                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem},
                        ),
                        "Rain": xr.DataArray(
                            meteo[:, :, :, 1],
                            dims=["time", "lat", "lon"],
                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem},
                        ),
                        "RelHumidity": xr.DataArray(
                            meteo[:, :, :, 2],
                            dims=["time", "lat", "lon"],
                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem},
                        ),
                        "IncRadiation": xr.DataArray(
                            meteo[:, :, :, 3],
                            dims=["time", "lat", "lon"],
                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem},
                        ),
                        "T1D_AvgTemp": xr.DataArray(
                            meteo[:, :, :, 4],
                            dims=["time", "lat", "lon"],
                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem},
                        ),
                        "T10D_AvgTemp": xr.DataArray(
                            meteo[:, :, :, 5],
                            dims=["time", "lat", "lon"],
                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem},
                        ),
                    }
                )

            # Define the output path for the NetCDF file
            meteo_path = os.path.join(data_settings['data']['output_file']['folder_name'],
                                          f"meteo_data_{start}_{end}.nc")

            # Save the dataset to a NetCDF file
            meteo_output.to_netcdf(meteo_path)
            print(f"Meteo matrix built in {time.time() - t_start:.2f} seconds")
        # --------------------------------------------------------------------------------------------------------------------------
        if data_assimilation == 1 and cal == 0:
            state_matrix = np.zeros((nt, ny, nx, 4), dtype=np.float32)
            output_matrix = np.zeros((nt, ny, nx, 17), dtype=np.float32)

            # --------------------------------------------------------------------------------------------------------------------------
            # CREATE OBSERVATION VECTOR
            obs_mask = pandas.read_pickle(obs_mask)
            ys, xs = zip(*obs_mask)
            lat = list(ys)
            lon = list(xs)
            lat_point = lat_dem[list(ys)]
            lon_point = lon_dem[list(xs)]

            n_obs = len(lat_point)
            obs = pandas.read_pickle(obs)
            meteo_input = np.zeros((nt, 6, n_obs), dtype=np.float32)

            y_p = 0
            y = np.zeros((nt, n_obs))
            s_dic = {}
            while y_p < n_obs:
                for (lat, lon), df in obs.items():
                    s = []
                    df = df.set_index('time')
                    df = df[~df.index.duplicated(keep='first')]
                    df_res = df.resample('h').asfreq()
                    # slice the dataframe to the start and end datetime
                    df_1 = df_res.loc[start_datetime:end_datetime]
                    q = perturbations_data[(y_p, y_p)]

                    for var in ["air_temp_degC", "prc_mm", "swin_wm", "rel_hum_perc", "T_albedo", "T_melting"]:
                        df = q[var].sort_values("emp_rip")
                        s.append(df.values)  # y, x for np.interp

                    s_dic[(y_p, y_p)] = s
                    meteo_input[:, :, y_p ] = meteo[:, y_p, y_p, :]

                    # if the df is empty fill with nan
                    if df_1.empty:
                        y[:, y_p] = np.ones_like(y[:, y_p]) * np.nan

                    else:
                        y[:, y_p] = df_1['Snow_depth_cm'] / 100

                    y_p += 1

            m = meteo_input.shape[1]
            hs = y[:, :]
            swe = np.ones_like(hs) * np.nan
            y = np.array([swe, hs])
            # ----------------------------------------------------------------------------------------------------------

            state_matrix_ensemble = np.zeros((nt, N, 4, n_obs), dtype=np.float32)
            output_matrix_ensemble = np.zeros((nt, N, 17,n_obs), dtype=np.float32)
            meteo_matrix = np.zeros((nt, N, 6, n_obs), dtype=np.float32)
            state_vector_point = np.zeros((nt, 4, n_obs), dtype=np.float32)
            output_vector_point = np.zeros((nt, 17, n_obs), dtype=np.float32)

            for point in range(0,n_obs):
                 state_vector_point[-1, :, point] = state_vector
                 output_vector_point[-1, :, point] = output_vector

            for i in range(0,N):
                 state_matrix_ensemble[-1, i, :, :] = np.array(state_vector)[:, None]
                 output_matrix_ensemble[-1, i, :, :] = np.array(output_vector)[:, None]

            temporary_val = np.zeros((nt, N, m,n_obs))
            output_matrix_old = np.zeros((nt, N, y.shape[0],n_obs))
            output_a_mean = np.zeros((nt, y.shape[0],n_obs))
            Xb_old = np.zeros((nt, N, len(state_vector),n_obs))
            Xa_mean = np.zeros((nt, len(state_vector),n_obs))
            Pa = np.zeros((nt, len(state_vector), len(state_vector),n_obs))
            keys = list(statistics[(18, 18)]['statistics'].keys())
            N = int(N)

            # drop "key" if present
            if "key" in keys:
                keys.remove("key")
            inflat_deflat = np.zeros((nt, N, m))
            for p in range(m):
                inflat_deflat[:, :, p] = inflation_deflation[p]
            gc.collect()
        # --------------------------------------------------------------------------------------------------------------------------

    print(f"starting run")
    for j in range(0, len(Time)):

        t0 = time.time()

        if data_assimilation == 0 and cal == 0:
            meteo[j, :, :, :], state_matrix[j, :, :, :], output_matrix[j, :, :, :], mass_balance = S3M_2D_physics(
                meteo[j, :, :, :], parameters, state_backup[:, :, :], output_backup[:, :, :], Time[j], change_part,
                Ice_flag, lat_dem, lon_dem, slope, svf)

            state_backup[:, :, :] = state_matrix[j, :, :, :]
            output_backup[:, :, :] = output_matrix[j, :, :, :]
            print(f"Run for time step {j} done in {time.time() - t0:.2f} seconds")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif cal == 1:
            meteo[j, :, :, :], state_matrix[(j + 1), :, :, :], output_matrix[(j + 1), :, :, :], mass_balance = S3M_2D_physics(meteo[j, :, :, :],parameters, state_matrix[j, :, :, :],output_matrix[j, :, :, :], Time[j], change_part, Ice_flag, lat, lon, slope, svf)

            print(f"Run for time step {i} done in {time.time() - t0:.2f} seconds")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif data_assimilation == 1 and cal == 0 :
            results = Parallel(n_jobs=25)(
                    delayed(perturb_point)(
                    meteo_input[j, :, point],
                    state_matrix_ensemble[(j-1), :, :, point],
                    temporary_val[(j-1), :, :, point],
                    meteo_matrix[j, :, :, point],
                    output_matrix_ensemble[(j-1), :, :, point],
                    s_dic[(point, point)],
                    statistics[(point,point)],
                    keys,
                    inflat_deflat[j, :, :],
                    R_dict[(point, point)]["R"].values,
                    R_state[(point, point)]["R"].values,
                    state_limits,
                    state_vector,
                    pert_prec,
                    pert_rad,
                    pert_temp,
                    pert_rh,
                    scale_mean_prec,
                    pert_asymm_prec,
                    c_asymm_prec,
                    L0,
                    L_tilde,
                statistics_state[(point,point)]
                )
                for point  in range(0,n_obs)
            )

            for loc, (meteo_pt, state_pt, output_pt,temporary_val_pt) in enumerate(results):
                meteo_matrix[j, :, :, loc] = meteo_pt
                state_matrix_ensemble[(j-1), :, :, loc] = state_pt
                output_matrix_ensemble[(j-1), :, :, loc] = output_pt
                temporary_val[(j-1), :, :, loc] = temporary_val_pt
            # ------------------------------------------------------------------------------------------------------------------
            # FORWARD STEP FOR THE DETERMINISTIC CASE
            meteo_input[j,:,:], state_vector_point[j,:,:], output_vector_point[j,:,:]= S3M_1D_physics_points(meteo_input[j,:,:],
                                                                           parameters, state_vector_point[j-1,:,:],
                                                                           output_vector_point[j-1,:,:], Time[j], change_part,Ice_flag,lat_point,lon_point)

            # -----------------------------------------------------------------------------------------------------
            # FORECAST STEP FOR THE ENSEMBLE

            result = Parallel(n_jobs=25)(
                delayed(S3M_1D_physics_points)(
                    meteo_matrix[j, i, :, :],
                    parameters,
                    state_matrix_ensemble[(j - 1), i, :, :],
                    output_matrix_ensemble[(j - 1), i, :, :],
                    Time[j],
                    change_part,
                    Ice_flag,
                    lat_point,
                    lon_point)
                for i in range(0,N)
            )
            # Unpack results
            for i, (meteo_i, state_i, output_i) in enumerate(result):
                meteo_matrix[j, i, :, :] = meteo_i
                state_matrix_ensemble[j, i, :, :] = state_i
                output_matrix_ensemble[j, i, :, :] = output_i


            output_matrix_old[j, :, 0, :] = output_matrix_ensemble[j, :, 10, :]
            output_matrix_old[j, :, 1, :] = output_matrix_ensemble[j, :, 14, :]
            # ------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------
            # ASSIMILATION STEP
            result = Parallel(n_jobs=25)(delayed(assimilation_point)(meteo_matrix[j, :, :, p],  state_matrix_ensemble[j, :, :, p],  output_matrix_ensemble[j, :, :, p], y[:, j, p], parameters,
                                   state_limits, R_measures, Xb_old[j, :, :, p],  Xa_mean[j, :, p], Pa[j, :, :, p], output_a_mean[j, :, p], N,statistics_state[(p,p)])
                                             for p in range(0,n_obs))
            # Unpack results
            for p, (Xa_mean_p, Pa_p, output_a_mean_p, Xb_old_p) in enumerate(result):
                Xa_mean[j, :, p] = Xa_mean_p
                Pa[j, :, :, p] = Pa_p
                output_a_mean[j, :, p] = output_a_mean_p
                Xb_old[j, :, :, p] = Xb_old_p
            # -----------------------------------------------------------------------------------------------------------------
             # update state
            for i in range(N):
                state_matrix_ensemble[j, i, :, :] = Xa_mean[j, :, :]
                output_matrix_ensemble[j, i, 10, :] = output_a_mean[j, 0, :]
                output_matrix_ensemble[j, i, 14, :] = output_a_mean[j, 1, :]

        if 1:
            val_x = values['val_x']
            val_x.append(val_x[-1] + 1)

            new_values = {
                    'val_swe_a': output_a_mean[j, 0, 7] ,
                    'val_swe_b': output_matrix_old[j, 0, 0, 7],
                    'val_swe_b1' : output_matrix_old[j, 1, 0, 7],
                    'val_swe_open': output_vector_point[j, 10, 7],
                    'val_hs_a': output_a_mean[j, 1, 7],
                    'val_hs_b': output_matrix_old[j, 0, 1, 7],
                    'val_hs_b1' : output_matrix_old[j, 1, 1, 7],
                    'val_hs_open': output_vector_point[j, 14, 7],
                    'val_hs_obs': y[1, j, 7]
                }


            for key, value in new_values.items():
                    values[key].append(value)

            axs[0].clear()
            axs[1].clear()
            # -----------------
            # SWE plot
            # -----------------
            axs[0].plot(val_x, values['val_swe_a'], label='analysis', color='cyan')
            axs[0].plot(val_x, values['val_swe_b'], label='background', color='lightgrey')
            axs[0].plot(val_x, values['val_swe_b1'],label='background', color='lightgreen')
            axs[0].plot(val_x, values['val_swe_open'], label='open loop', color='k')
            axs[0].legend()

            # -----------------
            # HS plot
            # -----------------
            axs[1].plot(val_x, values['val_hs_a'], label='analysis', color='cyan')
            axs[1].plot(val_x, values['val_hs_b'], label='background', color='lightgrey')
            axs[1].plot(val_x, values['val_hs_b1'],label='background', color='lightgreen')
            axs[1].plot(val_x, values['val_hs_open'], label='open loop', color='k')
            axs[1].plot(val_x, values['val_hs_obs'],label='obs', color='red',marker='o', linestyle='None', markersize=0.5)
            axs[1].legend()

            folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plot_update/"
            plt.savefig(os.path.join(folder, f'assimilation_point_{j}.png'))

            #Step 6 interpolate back the correction to full grid with Gausssian processes
            # to be done

        print(f"Run for time step {j} done in {time.time() - t0:.2f} seconds")
        # --------------------------------------------------------------------------------------------------------------------------
    # TRACE TIME
    print(f"Total run done in {time.time() - t_start:.2f} seconds")


    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    if data_assimilation == 1 and cal == 0:

        p = 0
        n = 0
        for p in range(0, n_obs):

            fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
            axes = axes.flatten()

            meteo_vars = [
                'Air Temperature (°C)',
                'Precipitation (mm)',
                'Relative Humidity (%)',
                'Shortwave Radiation (W/m²)',
                '1D Avg Temperature (°C)',
                '2D Avg Temperature (°C)'
            ]

            # Colors for ensemble members
            colors = plt.cm.viridis(np.linspace(0, 1, N))

            for i, var in enumerate(meteo_vars):
                ax = axes[i]

                # Plot ensemble members (perturbed)
                for n in range(N):
                    ax.plot(
                        Time,
                        meteo_matrix[:, n, i, p],
                        color=colors[n],
                        alpha=0.5,
                        linewidth=0.8
                    )

                # Plot original (unperturbed) meteo – assuming index 0 or separate array
                ax.plot(
                    Time,
                    meteo_input[:, i, p],
                    color='grey',
                    linewidth=1,
                    label='Original',
                    linestyle='--'
                )

                ax.set_title(var)
                ax.grid(True)

            # Optional legend only once
            axes[0].legend(loc='best')

            fig.savefig(
                os.path.join(
                    data_settings['data']['output_file']['folder_name'],
                    f'meteo_perturbations_{p}.png'
                )
            )
            plt.close(fig)

            # NOW PLOT THE open loop FOR THE POINT of output in position 10 ( swe) and 14 ( hs)
            fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
            # Plot SWE in the first subplot
            axes[0].plot(Time, output_vector_point[:, 10, p], color='black', linestyle='-', linewidth=1,
                         label='Deterministic Background')
            for n in range(N):
                axes[0].plot(Time, output_matrix_old[:, n, 0, p], color=colors[n], alpha=0.5, linewidth=0.8,
                             label='Perturbed Ensemble' if n == 0 else "")
            axes[0].plot(Time, output_a_mean[:, 0, p], color='red', linestyle='-', linewidth=2,
                         label='Analysis Ensemble Mean')
            axes[0].set_title('SWE_mm', fontsize=15)
            axes[0].set_xlabel('Time', fontsize=12)
            axes[0].set_ylabel('Value', fontsize=12)
            axes[0].grid()
            axes[0].legend()
            # Plot Snow Depth in the second subplot
            axes[1].plot(Time, output_vector_point[:, 14, p], color='black', linestyle='-', linewidth=1,
                         label='Deterministic Background')
            for n in range(N):
                axes[1].plot(Time, output_matrix_old[:, n, 1, p], color=colors[n], alpha=0.5, linewidth=0.8,
                             label='Perturbed Ensemble' if n == 0 else "")
            axes[1].plot(Time, output_a_mean[:, 1, p], color='red', linestyle='-', linewidth=1,
                         label='Analysis Ensemble Mean')
            axes[1].plot(Time, y[1, :, p], color='blue', marker='o', markersize=2, label='Observations')

            axes[1].set_title('H_S_m', fontsize=15)
            axes[1].set_xlabel('Time', fontsize=12)
            axes[1].set_ylabel('Value', fontsize=12)
            axes[1].grid()
            axes[1].legend()
            # Save the combined figure
            plt.savefig(os.path.join(data_settings['data']['output_file']['folder_name'],
                                     f'open_loop_point_{p}.png'))
            plt.close(fig)

            # now plot the state in a fig for the point p
            fieldnames_state = ["Swe_w", "Swe_d", "Rho_kg/m3", "albedo"]
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
            axes = axes.flatten()
            for i, var in enumerate(fieldnames_state):
                ax = axes[i]

                # Plot ensemble members (perturbed)
                for n in range(N):
                    ax.plot(
                        Time,
                        state_matrix_ensemble[:, n, i, p],
                        color=colors[n],
                        alpha=0.5,
                        linewidth=0.8
                    )

                # Plot original (unperturbed) meteo – assuming index 0 or separate array
                ax.plot(
                    Time,
                    state_vector_point[:, i, p],
                    color='grey',
                    linewidth=1,
                    label='Original',
                    linestyle='--'
                )

                # add also the analysis mean
                ax.plot(
                    Time,
                    Xa_mean[:, i, p],
                    color='red',
                    linewidth=1.5,
                    label='Analysis Mean'
                )
                ax.set_title(var)
                ax.grid(True)
            # Optional legend only once
            axes[0].legend(loc='best')
            fig.savefig(
                os.path.join(
                    data_settings['data']['output_file']['folder_name'],
                    f'state_perturbations_{p}.png'
                )
            )
            plt.close(fig)

        return  None

    # -----------------------------------------------------------------------------------------------------------------
    if cal == 0 and data_assimilation == 0:

        obs = pandas.read_pickle(obs)
        y_p = 0
        y = np.zeros((nt, 43))

        for (lat, lon), df in obs.items():
            df = df.set_index('time')
            df = df[~df.index.duplicated(keep='first')]
            df_res = df.resample('H').asfreq()
            # slice the dataframe to the start and end datetime
            df_1 = df_res.loc[start_datetime:end_datetime]
            # if the df is empty fill with nan
            if df_1.empty:
                y[:, y_p] = np.ones_like(y[:, y_p]) * np.nan

                # if df len is less than nt-1 resample missing dates over 1 hour frequency and fill with nan
            else:
                y[:, y_p] = df_1['Snow_depth_cm'] / 100

            y_p += 1

        fieldnames_output = [ 'SWE_mm', 'H_S_m' ]

        for i in range (0, nx):
            fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
            # Plot SWE in the first subplot
            axes[0].plot(Time, output_matrix[:, i, i, 10], color='black', linestyle='-', linewidth=0.5,
                         label='Deterministic Background')
            axes[0].set_title(fieldnames_output[0], fontsize=15)
            axes[0].set_xlabel('Time', fontsize=12)
            axes[0].set_ylabel('Value', fontsize=12)
            axes[0].grid()
            axes[0].legend()

            # Plot Snow Depth in the second subplot
            axes[1].plot(Time, output_matrix[:, i, i, 14], color='black', linestyle='-', linewidth=1,
                         label='Deterministic Background')
            axes[1].plot(Time, y[:, i], color='blue', marker='o', label='Observations')
            axes[1].set_title(fieldnames_output[1], fontsize=15)
            axes[1].set_xlabel('Time', fontsize=12)
            axes[1].set_ylabel('Value', fontsize=12)
            axes[1].grid()
            axes[1].legend()

            # Save the combined figure
            plt.savefig(os.path.join(data_settings['data']['output_file']['folder_name'],
                                     f'open_loop{i}.png'))
            plt.close()
            i += 1


        ds_output = xr.Dataset(
            {
                "Rainfall_mm": xr.DataArray(output_matrix[:, :, :, 0], dims=["time","lat", "lon" ],
                                            coords={"time": Time,"lat": lat_dem, "lon": lon_dem, }),
                "Snowfall_mm": xr.DataArray(output_matrix[:, :, :, 1], dims=["time","lat", "lon" ],
                                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "Melting_mm": xr.DataArray(output_matrix[:, :, :, 2], dims=["time","lat", "lon" ],
                                           coords={"time": Time,"lat": lat_dem, "lon": lon_dem, }),
                "Refreezing_mm": xr.DataArray(output_matrix[:, :, :, 3], dims=["time","lat", "lon"],
                                              coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "Outflow_mm": xr.DataArray(output_matrix[:, :, :, 4], dims=["time","lat", "lon"],
                                           coords={"time": Time,"lat": lat_dem, "lon": lon_dem, }),
                "Sf_daily_cum": xr.DataArray(output_matrix[:, :, :, 5], dims=["time","lat", "lon" ],
                                             coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "SWE_mm": xr.DataArray(output_matrix[:, :, :, 10], dims=["time","lat", "lon" ],
                                       coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "Snow_Age": xr.DataArray(output_matrix[:, :, :, 11], dims=["time","lat", "lon" ],
                                         coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "H_D_m": xr.DataArray(output_matrix[:, :, :, 12], dims=["time","lat", "lon" ],
                                      coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "Theta_w": xr.DataArray(output_matrix[:, :, :, 13], dims=["time","lat", "lon" ],
                                        coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "H_S_m": xr.DataArray(output_matrix[:, :, :, 14], dims=["time","lat", "lon" ],
                                      coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "Rho_S_kg_m3": xr.DataArray(output_matrix[:, :, :, 15], dims=["time","lat", "lon" ],
                                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
            }
        )
        output_file = os.path.join(data_settings['data']['output_file']['folder_name'], f"output_data_{start}_{end}.nc")
        ds_output.to_netcdf(output_file, engine='h5netcdf')
        print(f"Data saved to {output_file}")
        # Save the state
        ds_state = xr.Dataset(
            {
                "SWE_W_mm": xr.DataArray(state_matrix[:, :, :, 0], dims=["time","lat", "lon" ],
                                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "SWE_D_mm": xr.DataArray(state_matrix[:, :, :, 1], dims=["time","lat", "lon" ],
                                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "RHO_D_kg_m3": xr.DataArray(state_matrix[:, :, :, 2], dims=["time","lat", "lon" ],
                                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
                "albedo": xr.DataArray(state_matrix[:, :, :, 3],dims=["time","lat", "lon" ],
                                            coords={"time": Time, "lat": lat_dem, "lon": lon_dem, }),
            }
        )
        # Save the state to NetCDF
        state_file = os.path.join(data_settings['data']['output_file']['folder_name'], f"state_data_{start}_{end}.nc")
        ds_state.to_netcdf(state_file, engine='h5netcdf')
        print(f"State data saved to {state_file}")
        gc.collect()



        if comparison_ol==1:
            return 0,  state_matrix[-1, :, :,:], output_matrix[-1, :, :, :]

        rmse_hs, st_dev_hs = process_and_plot_snow_data(output_file, obs,
                                                        data_settings['data']['output_file']['folder_name_plots'], start, end)


        rmse_hs = log_and_return(cal, log_stream, start_time, rmse_hs, st_dev_hs)

        return rmse_hs,  state_matrix[-1, :, :,:], output_matrix[-1, :, :, :]
    # -----------------------------------------------------------------------------------------------------------------

    elif cal == 1 and data_assimilation == 0:
        # Convert snow depth series into a NumPy array
        snow_depth_array = np.array(output_matrix[:, :, :, 14])
        # Reshape snow_depth_array to 2D
        snow_depth_array_2d = np.squeeze(snow_depth_array)  # Removes dimensions of size 1
        # Convert snow depth series into a DataFrame
        snow_depth_df = pandas.DataFrame(snow_depth_array_2d, columns=['Snow Depth'])
        # Ensure Time is properly defined and matches the length of snow_depth_series
        # Set the index of the DataFrame to Time
        snow_depth_df.set_index(Time, inplace=True)
        # Ensure Time is a pandas Series or datetime range
        selected_data = values[(values['time'] >= Time.min()) & (values['time'] <= Time.max())]
        # Extract the values
        snow_depth_observed = selected_data['snd_interp'].values
        snow_depth_observed = snow_depth_observed.T

        # Create the comparison plot
        plt.figure(figsize=(10, 6))
        plt.plot(snow_depth_df.index, snow_depth_df['Snow Depth'], color='black',
                 label='Modeled Snow Depth')
        plt.plot(snow_depth_df.index, snow_depth_observed, color='red', label='Observed Snow Depth')
        # Add labels, title, and legend
        plt.xlabel('Time')
        plt.ylabel('Snow Depth (cm)')
        plt.title('Comparison of Modeled and Observed Snow Depth')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(data_settings['calibration']['folder_name'],
                                 f"time_series_LAT{lat}_LON{lon}.png"))
        plt.close()
        # Compute RMSE for snow depth
        rmse_hs = rmse(snow_depth_df['Snow Depth'], snow_depth_observed)
        st_dev_hs = snow_depth_df['Snow Depth'].std()

        return log_and_return(cal, log_stream, start_time, rmse_hs, st_dev_hs)
    # ------------------------------------------------------------------------------------------------------------------

    elif data_assimilation == 1 and downscaled == 1:
        print("Data assimilation with downscaled data not yet implemented")

    gc.collect()

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    default_n_threads = 30
    os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
    os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
    os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    alg_settings, alg_time_start, alg_time_end, alg_domain = get_args()
    data_settings = read_file_settings(alg_settings)
    calibrate = data_settings['calibration']['calibrate']
    data_assimilation = data_settings['data']['info_file']["perturbations"]["per_Obs"]
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    try:
        old_run_path = data_settings['data']['info_file']["old_run"]
        ds = xr.open_dataset(old_run_path)
        state_input = ds[['SWE_W', 'SWE_D', 'Rho_D','AlbedoS']]
        list_var = ['RainFall', 'SnowFall', 'MeltingS', 'RefreezingS', 'Outflow',
                    'SnowfallCum', 'SWE', 'AgeS', 'Theta_W', 'H_S', 'RhoS', 'MeltingG']
        output_input = ds[list_var]
    except:
        state_input = None
        output_input = None
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    try:
        start = alg_time_start
        end = alg_time_end
    except:
        print("analysis of dates not given")

    results = []
    if calibrate == 0 and data_assimilation == 0:
        # Define the time slices
        slice_list = [ ("2018-10-01", "2019-09-30")]
        for slice_start, slice_end in slice_list:

            mrad, mr, window_melting,alpha, lat, lon,values = [], [], [], [], [],[],[]
            rmse_hs = S3M_2D(mrad, mr, window_melting, alpha, lat, lon,values, slice_start, slice_end, state_input, output_input)
            gc.collect()

            print(rmse_hs)
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    elif data_assimilation == 1 and calibrate == 0:

        mrad, mr, window_melting,alpha, lat, lon,values = [], [], [], [], [],[],[]
        S3M_2D(mrad, mr, window_melting, alpha, lat, lon, values, start, end, state_input, output_input)
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    elif calibrate == 1 and data_assimilation == 0:

        logger_name = "S3M_2D_logger"
        log_stream = logging.getLogger(logger_name)
        log_folder_name = data_settings["calibration"]["folder_name"]

        file_name = data_settings["calibration"]["file_name_cal"]
        set_logging_file(logger_name=logger_name, logger_file=os.path.join(log_folder_name, file_name))

        project_name = 'PhD_project  S3M 2D -CALIBRATION'
        alg_type = 'PYTHON VERSION'
        log_stream.info(' ============================================================================')
        log_stream.info('[' + project_name + ' ' + alg_type + ' - ')
        log_stream.info(' ============================================================================')
        # open the pkl file with the data
        df_dict = pandas.read_pickle(data_settings["data"]["info_file"]["time_series_csnow"])
        # Define the file path for the text file
        output_csv = os.path.join(data_settings['calibration']["folder_name"], f"results_best.csv")

        df_results = pandas.DataFrame(columns=['Latitude', 'Longitude', 'SAMPLE', 'F_COST'])

        # Run the optimization in parallel
        results = Parallel(n_jobs=30)(delayed(optimize_for_key)(key,values, start, end) for key,values in df_dict.items())
        df_results = pandas.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        log_stream.info(f"Results saved to {output_csv}")
        log_stream.info("Calibration finished.")
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


