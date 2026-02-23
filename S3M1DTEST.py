# Library
import pdb
import joblib
from statistics import NormalDist

from S3M_1D_physics_test import S3M_1D_physics
from lib_utilis_data_proc import get_args
import matplotlib.pyplot as plt

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


def plot_ensemble(meteo_inputs, inputs, Time, meteo_matrix, states, state_matrix, state_vector, output_matrix,
                  outputs, tag, N, data_settings, output, fieldnames_state, snow_depth_obs):
    # Plot of meteo perturbations
    list_titles = ['Precipitation', 'Radiation', 'Temperature', 'Relative Humidity', 'T_10D', 'T_1D']
    inputs = np.array(inputs)  # Convert inputs to a numpy array
    plt.figure(figsize=(20, 12))

    for i in range(len(meteo_inputs)):
        plt.subplot(2, 3, i + 1)
        for j in range(N):
            plt.plot(Time, meteo_matrix[:, j, i], color='lightgrey', linewidth=0.5)
        plt.plot(Time, inputs[:, i], color='red', linewidth=1, linestyle='--', label='Deterministic')
        plt.title(list_titles[i], fontsize=15)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.savefig(
        fill_tags2string(os.path.join(output['folder_name'], 'meteo_ensemble.png'), data_settings['template'], tag))
    plt.close()
    # --------------------------------------------------------------------------------------------------------------
    # plot of states
    # plot each ensemble member for each variable inside state_matrix. using different colors for each ensemble
    # member and plot the state vector for the deterministic case in black
    # convert states in a matrix of same size of state_matrix

    states = np.array(states)
    colors = plt.cm.viridis(np.linspace(0, 1, state_matrix.shape[1]))

    plt.figure(figsize=(20, 12))
    for i in range(len(state_vector)):
        plt.subplot(3, 2, i + 1)
        for j in range(state_matrix.shape[1]):
            plt.plot(Time, state_matrix[:, j, i], color=colors[j], linewidth=1, linestyle='-')
        plt.plot(Time, states[:, i], color='black', linestyle='-', linewidth=2, label='Deterministic')
        plt.title(fieldnames_state[i], fontsize=15)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.savefig(
        fill_tags2string(os.path.join(output['folder_name'], 'state_ensemble.png'), data_settings['template'], tag))
    plt.close()
    # --------------------------------------------------------------------------------------------------------------

    # Compute matrices
    outputs = np.array(outputs)
    cum_precipitation = np.zeros((len(Time), N))
    cum_outflow = np.zeros((len(Time), N))
    swe_matrix = np.zeros((len(Time), N))

    cum_precipitation_det = np.cumsum(inputs[:, 0])
    cum_outflow_det = np.cumsum(outputs[:, 4])
    swe_det = outputs[:, 10]
    temperature = inputs[:, 2]
    radiation = inputs[:, 1]

    for i in range(N):
        cum_precipitation[:, i] = np.cumsum(meteo_matrix[:, i, 0])
        cum_outflow[:, i] = np.cumsum(output_matrix[:, i, 4])
        swe_matrix[:, i] = output_matrix[:, i, 10]

    # Plot and save cumulative precipitation
    plt.figure(figsize=(20, 12))
    plt.imshow(cum_precipitation, cmap='jet', interpolation='nearest', aspect='auto')
    plt.title('Cumulative Precipitation')
    plt.colorbar()
    plt.grid(False)
    plt.savefig(fill_tags2string(os.path.join(output['folder_name'], 'cumulative_precipitation_ensemble.png'),
                                 data_settings['template'], tag))
    plt.close()

    # Plot and save cumulative outflow
    plt.figure(figsize=(20, 12))
    plt.imshow(cum_outflow, cmap='jet', interpolation='nearest', aspect='auto')
    plt.title('Cumulative Outflow')
    plt.colorbar()
    plt.grid(False)
    plt.savefig(fill_tags2string(os.path.join(output['folder_name'], 'cumulative_outflow_ensemble.png'),
                                 data_settings['template'], tag))
    plt.close()

    # Plot and save SWE
    plt.figure(figsize=(20, 12))
    plt.imshow(swe_matrix, cmap='jet', interpolation='nearest', aspect='auto')
    plt.title('SWE')
    plt.colorbar()
    plt.grid(False)
    plt.savefig(
        fill_tags2string(os.path.join(output['folder_name'], 'SWE_ensemble.png'), data_settings['template'], tag))
    plt.close()

    # Plot and save temperature
    plt.figure(figsize=(20, 12))
    plt.imshow(meteo_matrix[:, :, 2], cmap='jet', interpolation='nearest', aspect='auto')
    plt.title('Temperature')
    plt.colorbar()
    plt.grid(False)
    plt.savefig(
        fill_tags2string(os.path.join(output['folder_name'], 'Temperature_ensemble.png'), data_settings['template'],
                         tag))
    plt.close()

    # Plot and save precipitation
    plt.figure(figsize=(20, 12))
    plt.imshow(meteo_matrix[:, :, 0], cmap='jet', interpolation='nearest', aspect='auto')
    plt.title('Precipitation')
    plt.colorbar()
    plt.grid(False)
    plt.savefig(
        fill_tags2string(os.path.join(output['folder_name'], 'Precipitation_ensemble.png'), data_settings['template'],
                         tag))
    plt.close()

    # Plot and save outflow
    plt.figure(figsize=(20, 12))
    plt.imshow(output_matrix[:, :, 4], cmap='jet', interpolation='nearest', aspect='auto')
    plt.title('Outflow')
    plt.colorbar()
    plt.grid(False)
    plt.savefig(
        fill_tags2string(os.path.join(output['folder_name'], 'Outflow_ensemble.png'), data_settings['template'], tag))
    plt.close()

    # Plot and save radiation
    plt.figure(figsize=(20, 12))
    plt.imshow(meteo_matrix[:, :, 1], cmap='jet', interpolation='nearest', aspect='auto')
    plt.title('Radiation')
    plt.colorbar()
    plt.grid(False)
    plt.savefig(
        fill_tags2string(os.path.join(output['folder_name'], 'Radiation_ensemble.png'), data_settings['template'], tag))
    plt.close()
    # --------------------------------------------------------------------------------------------------------------
    # Plot the cumulative precipitation, cumulative outflow, and SWE vectors for the deterministic case
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    data = [(cum_precipitation_det, 'Cumulative Precipitation'),
            (cum_outflow_det, 'Cumulative Outflow'),
            (swe_det, 'SWE')]

    for ax, (y_data, title) in zip(axes, data):
        ax.plot(Time, y_data, color='black', linewidth=1, linestyle='-')
        ax.set_title(title)
        ax.grid()

    plt.tight_layout()
    plt.savefig(
        fill_tags2string(os.path.join(output['folder_name'], 'deterministic.png'), data_settings['template'], tag))
    plt.close()

    # Plot the temperature and the radiation
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    data = [(temperature, 'Temperature'),
            (radiation, 'Radiation')]

    for ax, (y_data, title) in zip(axes, data):
        ax.plot(Time, y_data, color='black', linewidth=1, linestyle='-')
        ax.set_title(title)
        ax.grid()

    plt.tight_layout()
    plt.savefig(
        fill_tags2string(os.path.join(output['folder_name'], 'deterministic_case.png'), data_settings['template'], tag))
    plt.close()
    # --------------------------------------------------------------------------------------------------------------
    # Compute density_observe
    density_observe = np.zeros(len(Time))
    swe_obs = np.zeros(len(Time))

    # Data for plotting
    data = [
        (swe_det, swe_obs, swe_matrix, 'SWE'),
        (outputs[:, 14], snow_depth_obs, output_matrix[:, :, 14], 'Snow Depth'),
        (states[:, 2], density_observe, state_matrix[:, :, 2], 'Density')
    ]

    # Plot and save the figure
    plt.figure(figsize=(20, 12))
    for i, (det, obs, ens, title) in enumerate(data, 1):
        plt.subplot(3, 1, i)
        plt.plot(Time, det, color='black', linewidth=2, linestyle='-', label='Deterministic')
        plt.plot(Time, obs, color='red', marker='o', linestyle='None', label='Observed')
        plt.plot(Time, ens, color='lightgrey', linewidth=1, linestyle='-', label='Ensemble')
        plt.title(title)
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.savefig(
        fill_tags2string(os.path.join(output['folder_name'], 'observations_comparison.png'), data_settings['template'],
                         tag))
    plt.close()

    return None
def plot_assimilation(states, Xa, state_vector, Xb_old, Xa_mean, output, outputs, output_matrix_old, output_a,
                      output_a_mean, y, Time, fieldnames_state, data_settings, tag):

    # Plot the state pre-analysis xb and post-analysis xa for each member of ensemble and the correct trajectory
    deterministic = np.array(states)
    colors = ['lightgrey', 'lightgreen', 'green', 'black', 'red']
    labels = ['Prior Ensemble', 'Posterior Ensemble', 'Analysis', 'Deterministic run', 'Observations']

    fig, axes = plt.subplots(3, 2, figsize=(20, 12))
    for k, ax in enumerate(axes.flatten()):
        if k < len(state_vector):
            ax.set_title(fieldnames_state[k], fontsize=25, fontweight='bold')
            ax.plot(Time, Xb_old[:, :, k], color='lightgrey', linewidth=1.5)
            ax.plot(Time, Xa[:, :, k], color='lightgreen', linewidth=1.5)
            ax.plot(Time, Xa_mean[:, k], color='green', linewidth=2)
            ax.plot(Time, deterministic[:, k], color='black', linewidth=1)
            ax.set_xlabel('Time')
            ax.set_ylabel('State')
            ax.grid()

    fig.tight_layout()
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i],
                                  markerfacecolor=colors[i], markersize=15) for i in range(len(labels))]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=12, bbox_to_anchor=(0.5, 0.2))
    plt.savefig(fill_tags2string(os.path.join(output['folder_name'], 'state_pre_post_analysis.png'),
                                 data_settings['template'], tag))
    plt.close()

    # Plot the output pre-analysis and post-analysis for each member of ensemble and the correct trajectory
    deterministic_outputs = np.array(outputs)
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    data = [
        (output_matrix_old[:, :, 0], output_a[:, :, 0], output_a_mean[:, 0],
         deterministic_outputs[:, 10], 'SWE', 'SWE (mm)'),
        (output_matrix_old[:, :, 1], output_a[:, :, 1], output_a_mean[:, 1],
         deterministic_outputs[:, 14], 'SNOW DEPTH', 'Snow depth (m)')
    ]

    for ax, (prior, post, mean, det, title, ylabel) in zip(axes, data):
        ax.plot(Time, prior, color='lightgrey', linewidth=1.5, label='Prior ensemble trajectory')
        ax.plot(Time, post, color='lightgreen', linewidth=1.5, label='Posterior ensemble trajectory (Analysis)')
        ax.plot(Time, y[:, 0], color='red', linestyle='None', marker='o', markersize=0.8)
        ax.plot(Time, mean, color='green', linewidth=2)
        ax.plot(Time, det, color='black', linewidth=1)
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.grid()

    fig.tight_layout()
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i],
                                  markerfacecolor=colors[i], markersize=15) for i in range(len(labels))]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=12, bbox_to_anchor=(0.5, 0.001))
    plt.savefig(fill_tags2string(os.path.join(output['folder_name'], 'output_pre_post_analysis.png'),
                                 data_settings['template'], tag))
    plt.close()

    return None
# ---------------------------------------------------------------------------------------------------------------------

def S3M_2D(mrad, mr, window_melting,alpha ,lat, lon, values, start, end,  state_input, output_input):
    #  ------------------------------------------------------------------------------------------------------------------
    # trace time of the algorithm
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

    data_assimilation= data_settings['data']['info_file']["perturbations"]["per_Obs"]
    cal = data_settings['calibration']['calibrate']
    change_part = data_settings['data']['info_file']["change_part"]
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
    # -----------------------------------------TAGS---------------------------------------------------------------
    air_temp_tag = data_settings['data']["info_file"]['tags']['temperature_tag']
    precip_tag = data_settings['data']["info_file"]['tags']['prc_tag']
    rel_hum_tag = data_settings['data']["info_file"]['tags']['rh_tag']
    solar_rad_tag = data_settings['data']["info_file"]['tags']['rad_tag']
    specific_humidity_tag = data_settings['data']["info_file"]['tags']['specific_humidity_tag']
    pressure_tag = data_settings['data']["info_file"]['tags']['pressure_tag']
    var_list = [air_temp_tag, precip_tag,solar_rad_tag, specific_humidity_tag, "t1d" ,"t10d", pressure_tag]
    # ------------------------------------------------------------------------------------------------------------------
    dem_da = rxr.open_rasterio(dem, engine='rasterio').sel(band=1).drop_vars('band').chunk({"x": 100, "y": 100})
    lon_dem = dem_da.x.values
    lat_dem = dem_da.y.values
    ny, nx = dem_da.shape
    y_dem =dem_da["y"].values
    x_dem =dem_da["x"].values
    gc.collect()
    # ------------------------------------------------------------------------------------------------------------------
    # create a figure with two subplots
    if 1:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        values = {
            'val_x': [0],
            'val_swe_a': [0],
            'val_swe_b': [0],
            'val_swe_open': [0],
            'val_hs_a': [0],
            'val_hs_b': [0],
            'val_hs_open': [0],
            'val_hs_obs': [0],
        }


    trial = {'source_file_datetime_generic': start_time_str}
    log_folder_name = data_settings["log"]["folder_name"]
    logger_name = logger_name
    file_name = fill_tags2string(data_settings['log']['file_name'], data_settings['template'], trial)
    set_logging_file(logger_name=logger_name, logger_file=os.path.join(log_folder_name, file_name))
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
    state_limits = pandas.read_pickle(fill_tags2string(data_settings['data']['info_file']["perturbations"]['state_limits'], data_settings['template'], tag))
    inflation_deflation = [f for f in data_settings['data']['info_file']["perturbations"]['inflation_deflation'].values()]
    scale_mean_prec = data_settings['data']['info_file']["perturbations"]["scale_mean_prec"]
    pert_asymm_prec = data_settings['data']['info_file']["perturbations"]["pert_asymm_prec"]
    c_asymm_prec = data_settings['data']['info_file']["perturbations"]["c_asymm_prec"]
    pert_prec = data_settings['data']['info_file']["perturbations"]["pert_prec"]
    pert_rad = data_settings['data']['info_file']["perturbations"]["pert_rad"]
    pert_temp = data_settings['data']['info_file']["perturbations"]["pert_temp"]
    pert_rh = data_settings['data']['info_file']["perturbations"]["pert_rh"]
    sigma_err_swe = data_settings['data']['info_file']["perturbations"]["error_sigma_swe"]
    sigma_err_hs = data_settings['data']['info_file']["perturbations"]["error_sigma_snow_depth"]
    R_measures = np.array([[sigma_err_swe ** 2, 0], [0, sigma_err_hs ** 2]])
    L_tilde = pickle.load(open(data_settings['data']['info_file']["perturbations"]['L_tilde'], "rb"))
    L0 = pickle.load(open(data_settings['data']['info_file']["perturbations"]['L0'], "rb"))
    gc.collect()
    # ------------------------------------------------------------------------------------------------------------------
    ds = xr.open_dataset(comparison_ol_meteo)
    meteo_ds = ds[["AirTemperature", "IncRadiation", "Rain", "RelHumidity"]]
    # create a  time resolution from start to end with a frequency of 1 hour until the 23:00 of the end day
    Time = pandas.date_range(start=start_datetime, end= end_datetime , freq='h')
    meteo_ds = meteo_ds.sel(time=slice(start_datetime, end_datetime))
    nx_meteo, ny_meteo = len(meteo_ds.lon), len(meteo_ds.lat)
    # if the meteo grid is different from the dem grid , update nx and ny
    if nx_meteo != nx or ny_meteo != ny:
        Latitudes_1d = np.unique(meteo_ds.lat)
        Longitudes_1d = np.unique(meteo_ds.lon)
        AirTemperature_3D = np.repeat(
            meteo_ds['AirTemperature'].values[:, :, np.newaxis],  # shape (time, 43, 1)
            43, axis=2
        )  # now shape is (8760, 43, 43)

        Rain_3D = np.repeat(
            meteo_ds ['Rain'].values[:, :, np.newaxis],  # shape (time, 43, 1)
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
                "time": meteo_ds .time,
                "Latitude": Latitudes_1d,
                "Longitude": Longitudes_1d
            },
            data_vars={
                "AirTemperature": (("time", "Latitude", "Longitude"), AirTemperature_3D),
                "Rain": (("time", "Latitude", "Longitude"), Rain_3D),
                "IncRadiation": (("time", "Latitude", "Longitude"), IncRadiation_3D),
                "RelHumidity": (("time", "Latitude", "Longitude"), RelHumidity_3D)
            } )

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
    meteo = np.zeros((nt, ny, nx, 6), dtype=np.float32)

    meteo[:, :, :, 0] = meteo_ds["AirTemperature"]
    meteo[:, :, :, 1] = meteo_ds["Rain"]
    # set as limit 0 the negative precipitation values
    meteo[:, :, :, 1] = np.where(meteo[:, :, :, 1] < 0, 0, meteo[:, :, :, 1])
    meteo[:, :, :, 2] = meteo_ds["RelHumidity"]
    meteo[:, :, :, 2] = np.where(meteo[:, :, :, 2] < 0, 0, meteo[:, :, :, 2])
    meteo[:, :, :, 2] = np.where(meteo[:, :, :, 2] > 100, 100, meteo[:, :, :, 2])
    meteo[:, :, :, 3] = meteo_ds["IncRadiation"]
    meteo[:, :, :, 4] = meteo_ds["AirTemperature"].rolling(time=24, min_periods=1).mean()
    meteo[:, :, :, 5] = meteo_ds["AirTemperature"].rolling(time=48, min_periods=1).mean()
    gc.collect()
    #------------------------------------------------------------------------------------------------------------------
    # open pickle file with obs_mask
    obs_mask = pandas.read_pickle(obs_mask)
    ys, xs = zip(*obs_mask)
    lat = list(ys)
    lon = list(xs)
    lat_point = lat_dem[list(ys)]
    lon_point = lon_dem[list(xs)]
    n_obs = len(lat_point)
    obs = pandas.read_pickle(obs)
    y_p = 0
    y = np.zeros((nt, n_obs))
    for (lat, lon), df in obs.items():
        df = df.set_index('time')
        df = df[~df.index.duplicated(keep='first')]
        df_res = df.resample('h').asfreq()
        # slice the dataframe to the start and end datetime
        df_1 = df_res.loc[start_datetime:end_datetime]
        # if the df is empty fill with nan
        if df_1.empty:
            y[:, y_p] = np.ones_like(y[:, y_p]) * np.nan
            y_p += 1
            continue
            # if df len is less than nt-1 resample missing dates over 1 hour frequency and fill with nan
        else:
            y[:, y_p] = df_1['Snow_depth_cm'] / 100

        y_p += 1
    # ------------------------------------------------------------------------------------------------------------------
    # select only one point
    m = 6
    hs = y[:, 20]
    swe = np.ones_like(hs) * np.nan
    y = np.array([swe, hs]).T
    q = perturbations_data[(20, 20)]
    s= []
    for var in ["air_temp_degC", "prc_mm", "swin_wm", "rel_hum_perc", "T_albedo", "T_melting"]:
        df = q[var].sort_values("emp_rip")
        s.append(df.values)  # y, x for np.interp



    state_matrix = np.zeros((nt,N,4), dtype=np.float32)
    output_matrix = np.zeros(( nt, N, 17), dtype=np.float32)
    meteo_matrix = np.zeros(( nt, N, 6), dtype=np.float32)
    state_matrix[0, :, :] = state_vector
    output_matrix[0, :, :] = output_vector
    sampled_val = np.zeros((nt, N,m))
    prob_cum = np.zeros((nt, N,m))
    val_tilde = np.zeros((nt, N,m))
    epsilon = np.zeros((nt, N,m))
    epsilon_state = np.zeros((nt, N,len(state_vector)))
    temporary_val = np.zeros((nt, N,m))
    # Initialize output and state matrices
    output_matrix_old = np.zeros((nt, N, y.shape[1]))
    output_a = np.zeros((nt, N, y.shape[1]))
    output_a_mean = np.zeros((nt, y.shape[1]))
    Xb_old = np.zeros((nt, N, len(state_vector)))
    Xa = np.zeros((nt, N, len(state_vector)))
    Xa_old = np.zeros((nt, N, len(state_vector)))
    Xa_mean = np.zeros((nt, len(state_vector)))
    B = np.zeros((nt, len(state_vector), len(state_vector)))
    H = np.zeros((nt, N, y.shape[1], len(state_vector)))
    Pa = np.zeros((nt, N, len(state_vector), len(state_vector)))

    keys = list(statistics[(20, 20)]['statistics'].keys())
    states =[]
    outputs = []
    inputs = []
    N= int(N)
    R1= R_dict[(20,20)]
    R= R1["R"].values

    R1_state = R_state[(20, 20)]
    R_state = R1_state["R"].values

    # drop "key" if present
    if "key" in keys:
        keys.remove("key")

    inflat_deflat = np.zeros((nt, N, m))

    for p in range(m):
        inflat_deflat[:, :, p] = inflation_deflation[p]
    # ------------------------------------------------------------------------------------------------------------------
    for j in range(0, len(Time)):
        meteo_inputs = meteo[j, 20, 20, :]
        m = len(meteo_inputs)

        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:

            # Ensure symmetry
            R = (R + R.T) / 2

            # Regularization
            R_reg = R + np.eye(R.shape[0])
            L = np.linalg.cholesky(R_reg)

        temporary_val[j, :, :] = temporary_val[(j - 1), :, :] + (
            np.dot(L0, np.random.normal(0, 0.01, (m, N)))).T
        # compute inverse of L_tilde
        L_tilde_inv = np.linalg.inv(L_tilde)
        sampled_val[j, :, :] = np.dot(np.dot(L, L_tilde_inv), temporary_val[j, :, :].T).T
        # --------------------------------------------------------------------------------------------------

        prob_cum[j, :, :] = [[NormalDist(mu=0, sigma=1).cdf(sampled_val[j, i, p])
                              for p in range(m)] for i in range(N)]

        # --------------------------------------------------------------------------------------------------
        # INTERPOLATE
        # for each element inside s , sort it according to position 1

        for p in range(m):
            # convert sp into dataframe

            # Convert to DataFrame with columns 0, 1, 2
            s[p] = pandas.DataFrame(s[p], columns=[0, 1, 2])

            # Sort by column 1
            s[p].sort_values(by=1, inplace=True)

        val_tilde[j, :, :] = [[np.interp(prob_cum[j, i, p],
                                         s[p].iloc[:, 2],
                                         s[p].iloc[:, 0])
                               for p in range(m)]
                              for i in range(N)]

        # --------------------------------------------------------------------------------------------------
        # CREATE EPSILON
        for p in range(m):
            key = keys[p]
            epsilon[j, :,p] = (val_tilde[j, :,p] - statistics[(20, 20)]['statistics'][key]["mean"])
        # --------------------------------------------------------------------------------------------------
        # INFLATION AND DEFLATION
        epsilon[j, :, :] = epsilon[j, :, :] * inflat_deflat[j,:]
        for v in range(m):
            limits = statistics[(7, 7)]['statistics'][keys[v]]
            if v == 1 or v == 3:  # precipitation variable
                limits["min"] =0
            ep_min_val = np.min(epsilon[j, :, v])
            ep_max_val = np.max(epsilon[j, :, v])
            if ep_min_val + meteo_inputs[v] < limits["min"]:
                mask_neg = epsilon[j, :, v] < 0
                idx = np.where(mask_neg)[0]
                epsilon[j, idx, v] *= (meteo_inputs[v] - limits["min"]) / (-ep_min_val)

            if ep_max_val + meteo_inputs[v] > limits["max"]:
                mask_pos = epsilon[j, :, v] > 0
                idx_max = np.where(mask_pos)[0]
                epsilon[j, idx_max, v] *= (limits["max"] - meteo_inputs[v]) / ep_max_val

        # --------------------------------------------------------------------------------------------------
        # PERTURB METEO
        meteo_matrix[j,:,:] = meteo_inputs[:] + epsilon[j,:,:]
    # ------------------------------------------------------------------------------------------------------

        if meteo_inputs[0] == 0:
            meteo_matrix[j, :, 0] = 0

        epsilon_state[j, :, :] = np.random.multivariate_normal(np.zeros(len(state_vector)), R_state, N)
        for v in range(len(state_vector)):
            ep_min_val_state = np.min(epsilon_state[j, :, v])
            ep_max_val_state = np.max(epsilon_state[j, :, v])
            for i in range(N):
                if ep_min_val_state + state_matrix[j - 1, i, v] < state_limits[0][v]:
                    mask_neg = epsilon_state[j, :, v] < 0
                    idx_state = np.where(mask_neg)[0]
                    epsilon_state[j, idx_state, v] *= (state_matrix[j - 1, i, v] - state_limits[0][v]) / (
                        -ep_min_val)

                if ep_max_val_state + state_matrix[j - 1, i, v] > state_limits[1][v]:
                    mask_pos = epsilon_state[j, :, v] > 0
                    idx_max_state = np.where(mask_pos)[0]
                    epsilon_state[j, idx_max_state, v] *= (state_limits[1][v] - state_matrix[
                        j - 1, i, v]) / ep_max_val


        state_matrix[j - 1, :, :] = state_matrix[j - 1, :, :] + epsilon_state[j, :, :]
        i = 0
        for i in range(N):
            # set state limit for density and albedo
            state_matrix[j - 1, :, 2] = max(min(state_matrix[j - 1, i, 2], state_limits[1][2]),
                                            state_limits[0][2])
            state_matrix[j - 1, :, 3] = max(min(state_matrix[j - 1, i, 3], state_limits[1][3]),
                                            state_limits[0][3])

            output_matrix[j - 1, i, 0] = state_matrix[j - 1, i, 0] + state_matrix[j - 1, i, 1]

            if output_matrix[j - 1, i, 0] == 0:
                output_matrix[j - 1, i, 1] = 0
                state_matrix[j - 1, i, 2] = state_limits[0][2]

            else:
                output_matrix[j - 1, i, 1] = (state_matrix[j - 1, i, 0] / 997) + (
                        state_matrix[j - 1, i, 1] / state_matrix[j - 1, i, 2])

        if pert_prec == 0:
            meteo_matrix[j,:, 0] = meteo_inputs[0]
        if pert_rad == 0:
            meteo_matrix[j,:, 1] = meteo_inputs[1]
        if pert_temp == 0:
            meteo_matrix[j,:, 2] = meteo_inputs[2]
            meteo_matrix[j,:, 4] = meteo_inputs[4]
            meteo_matrix[j,:, 5] = meteo_inputs[5]
        if pert_rh == 0:
            meteo_matrix[j,:, 3] = meteo_inputs[3]
        # ------------------------------------------------------------------------------------------------------
        # RESCALE
        if scale_mean_prec == 1 and meteo_inputs[0] != 0:
            meteo_matrix[j,:, 0] =meteo_matrix[j,:, 0] * (meteo_inputs[0] / np.mean(meteo_matrix[j,:, 0]))

        # -----------------------------------------------------------------------------------------------------
        if pert_asymm_prec == 1 and meteo_inputs[0] != 0:
            mask =meteo_matrix[j,:, 0] > meteo_inputs[0]
            idx = np.where(mask)[0]
            meteo_matrix[j, idx, 0] *= c_asymm_prec

        # -----------------------------------------------------------------------------------------------------
        # FORWARD STEP
        result = joblib.Parallel(n_jobs=12)(joblib.delayed(S3M_1D_physics)(log_stream, meteo_matrix[j, i, :],
                                                                           parameters, state_matrix[j - 1, i, :],
                                                                           output_matrix[j - 1, i, :], Time[j],
                                                                           change_part,
                                                                           ensemble=i)
                                            for i in range(N))

        for i in range(N):
            meteo_matrix[j, i, :] = result[i][0]
            state_matrix[j, i, :] = result[i][1]
            output_matrix[j, i, :] = result[i][2]

        # ------------------------------------------------------------------------------------------------------
        # FORWARD STEP FOR THE DETERMINISTIC CASE
        meteo_inputs, state_vector, output_vector = S3M_1D_physics(log_stream, meteo_inputs[:],
                                                                   parameters, state_vector,
                                                                   output_vector,Time[j], change_part,
                                                                    ensemble=-1)
        states.append(state_vector)
        inputs.append(meteo_inputs)
        outputs.append(output_vector)
        # ------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------
        # ASSIMILATION STEP
        # ------------------------------------------------------------------------------------------------------
        # Background
        Xb_old[j, :, :] = state_matrix[j, :, :]
        output_matrix_old[j, :, 0] = output_matrix[j, :, 10]  # swe
        output_matrix_old[j, :, 1] = output_matrix[j, :, 14]  # hs
        # ------------------------------------------------------------------------------------------------------
        # Covariance matrix
        B[j, :, :] = np.cov(state_matrix[j, :, :], rowvar=False)
        flag = [0, 0, 0, 0]
        # CHECK FLAG
        if np.sum(output_matrix_old[j, :, 1] > 0) > 0.1 * N:
            std_dev = np.sqrt(np.diag((B[j, :, :])))

            if std_dev[2] == 0:
                std_dev[2] = 15
                flag[2] = 1

            if std_dev[0] == 0:
                std_dev[0] = 10
                flag[0] = 1

            if std_dev[1] == 0:
                std_dev[1] = 50
                flag[1] = 1

            if std_dev[3] == 0:
                std_dev[3] = 0.01
                flag[3] = 1

            if 1 in flag:
                std_matrix = np.outer(std_dev, std_dev.T)
                corr_matrix = np.divide(B[j, :, :], std_matrix)

                if flag[0] == 1:
                    corr_matrix[0, 0] = 1
                    corr_matrix[0, 1] = -0.2
                    corr_matrix[0, 2] = -0.3
                    corr_matrix[0, 3] = -0.1
                    corr_matrix[1, 0] = -0.2
                    corr_matrix[2, 0] = -0.3
                    corr_matrix[3, 0] = -0.1

                if flag[1] == 1:
                    corr_matrix[1, 1] = 1
                    corr_matrix[1, 0] = -0.2
                    corr_matrix[1, 2] = 0.1
                    corr_matrix[1, 3] = 0.1
                    corr_matrix[0, 1] = -0.2
                    corr_matrix[2, 1] = 0.3
                    corr_matrix[3, 1] = 0.1

                if flag[2] == 1:
                    corr_matrix[2, 2] = 1  # density
                    corr_matrix[2, 0] = -0.3  # swew
                    corr_matrix[2, 1] = 0.3  # swed
                    corr_matrix[2, 3] = -0.5  # albedo
                    corr_matrix[0, 2] = -0.3
                    corr_matrix[1, 2] = 0.3
                    corr_matrix[3, 2] = -0.5

                if flag[3] == 1:
                    corr_matrix[3, 3] = 1
                    corr_matrix[3, 0] = -0.1
                    corr_matrix[3, 1] = 0.1
                    corr_matrix[3, 2] = -0.5
                    corr_matrix[0, 3] = -0.1
                    corr_matrix[1, 3] = 0.1
                    corr_matrix[2, 3] = -0.5

                B[j, :, :] = np.multiply(corr_matrix, std_matrix)
        # ------------------------------------------------------------------------------------------------------
        inf_vect = np.array([1, 1, 1, 1])
        inf_matrix = np.outer(inf_vect, inf_vect.T)
        B[j, :, :] = np.multiply(inf_matrix, B[j, :, :])
        # ------------------------------------------------------------------------------------------------------
        # Definition of observed density and its inverse at time j
        try:
            rho_obs = ((y[j, 0] / y[j, 1]) - 0.1 * parameters['RhoW']) / 0.9
            rho_obs_inv = 1 / rho_obs
        except ZeroDivisionError:
            rho_obs = parameters['RhoSnowMin']
            rho_obs_inv = 1 / rho_obs
        if np.isnan(rho_obs):
            rho_obs = parameters['RhoSnowMin']
            rho_obs_inv = 1 / rho_obs
        # ------------------------------------------------------------------------------------------------------
        # Cycle over the ensemble members
        for i in range(N):
            # Definition of modelled density and its inverse at time j and ensemble member i
            if output_matrix_old[j, i, 0] > 0:

                if output_matrix_old[j, i, 1] == 0:
                    print('errore fuori physics')
                    # print(result[i])
                    pdb.set_trace()

                else:
                    rho_calc = Xb_old[j, i, 2]
                    rho_calc_inv = 1 / rho_calc

            else:
                rho_calc_inv = rho_obs_inv

            if rho_calc_inv == 0 or np.isinf(rho_calc_inv):
                rho_calc_inv = 1 / parameters['RhoSnowMin']

            d_alfa_swe = ((output_matrix[j, i, 8] * meteo_matrix[j, i, 3]) / (1000 * parameters[
                'RhoW'] * 0.334)) * parameters['dt']

            if output_matrix[j, i, 15] == 0:
                d_alfa_hs = d_alfa_swe / rho_obs
            else:
                d_alfa_hs = d_alfa_swe / output_matrix[j, i, 15]

            # --------------------------------------------------------------------------------------------------
            # Compute the observation operator matrix H for each ensemble member i

            if np.isnan(y[j, :]).all():
                # No assimilation
                Xa[j, i, :] = Xb_old[j, i, :]
                output_a[j, i, 0] = output_matrix_old[j, i, 0]
                output_a[j, i, 1] = output_matrix_old[j, i, 1]
                continue

            elif not np.isnan(y[j, 0]) and not np.isnan(y[j, 1]):
                H[j, i, :, :] = np.array([[1, 1, 0, d_alfa_swe], [1 / 997, rho_calc_inv,
                                                                  -(rho_calc_inv * rho_calc_inv), d_alfa_hs]])

            elif np.isnan(y[j, 1]) and not np.isnan(y[j, 0]):

                H[j, i, :, :] = np.array([[1, 1, 0, d_alfa_swe], [np.nan, np.nan, np.nan, np.nan]])

            elif np.isnan(y[j, 0]) and not np.isnan(y[j, 1]):
                H[j, i, :, :] = np.array([[np.nan, np.nan, np.nan, np.nan], [1 / 997, rho_calc_inv,
                                                                             -(rho_calc_inv * rho_calc_inv),
                                                                             d_alfa_hs]])

            # --------------------------------------------------------------------------------------------------
            Ht = H[j, i, :, :]
            idx = np.where(~np.isnan(y[j, :]))[0]
            CC = np.matmul(B[j, :, :], Ht[idx, :].T)
            HBH = np.matmul(np.matmul(Ht[idx, :], B[j, :, :]), Ht[idx, :].T)
            K = np.matmul(CC, np.linalg.inv(HBH + R_measures[idx][:, idx]))

            y_mod = np.matmul(Ht[idx, :], Xb_old[j, i, :])
            Xa[j, i, :] = Xb_old[j, i, :] + np.matmul(K, (y[j, idx] - y_mod))

            # --------------------------------------------------------------------------------------------------
            # Analysis check and rescale

            M_tot = 0
            idx_scale = 0
            rescale = 0

            for k in range(2):
                if Xa[j, i, k] < state_limits[0][k] or Xa[j, i, k] > state_limits[1][k]:
                    rescale = 1
                    M = max(state_limits[0][k] - Xa[j, i, k], Xa[j, i, k] - state_limits[1][k])
                    if M > M_tot:
                        M_tot = M
                        idx_scale = k

            if rescale == 1:
                # store the not correct analysis
                Xa_old[j, i, :2] = Xa[j, i, :2]
                Xa[j, i, :2] = Xb_old[j, i, :2] + ((Xa[j, i, :2] - Xb_old[j, i, :2]) * (
                        (np.abs(Xa[j, i, idx_scale] - Xb_old[j, i, idx_scale])) - M_tot) / np.abs(
                    Xa[j, i, idx_scale] - Xb_old[j, i, idx_scale]))

            if np.isnan(Xa[j, i, :]).any():
                Xa[j, i, :] = Xb_old[j, i, :]
            # --------------------------------------------------------------------------
            Xa[j, i, 0] = max(min(Xa[j, i, 0], state_limits[1][0]), state_limits[0][0])
            Xa[j, i, 1] = max(min(Xa[j, i, 1], state_limits[1][1]), state_limits[0][1])
            # --------------------------------------------------------------------------

            Xa[j, i, 2] = max(min(Xa[j, i, 2], state_limits[1][2]), state_limits[0][2])
            Xa[j, i, 3] = max(min(Xa[j, i, 3], state_limits[1][3]), state_limits[0][3])

            output_a[j, i, 0] = Xa[j, i, 0] + Xa[j, i, 1]

            if output_a[j, i, 0] == 0:
                output_a[j, i, 1] = 0
                Xa[j, i, 2] = state_limits[0][2]

            else:
                output_a[j, i, 1] = (Xa[j, i, 0] / 997) + (Xa[j, i, 1] / Xa[j, i, 2])

            # ------------------------------------------------------------------------------------------------------
        # Compute the mean of the analysis and the covariance matrix
        Xa_mean[j, :] = np.mean(Xa[j, :, :], axis=0)
        state_matrix[j, :, :] = Xa[j, :, :]
        Pa[j, :, :] = np.cov(Xa[j, :, :], rowvar=False)
        output_a_mean[j, :] = np.mean(output_a[j, :, :], axis=0)  # correct trajectory
        output_matrix[j, :, 10] = output_a[j, :, 0]
        output_matrix[j, :, 14] = output_a[j, :, 1]

        print(f'Time step {j} completed')

        # ------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------
        if 0:

            val_x = values['val_x']
            val_x.append(val_x[-1] + 1)

            new_values = {
                'val_swe_a': output_a_mean[j, 0],
                'val_swe_b': output_matrix_old[j, 0, 0],
                'val_swe_open': output_vector[10],
                'val_hs_a': output_a_mean[j, 1],
                'val_hs_b': output_matrix_old[j, 0, 1],
                'val_hs_open': output_vector[14],
                'val_hs_obs': y[j, 1]
            }

            for key, value in new_values.items():
                if key in values:
                    values[key].append(value)

            if j % 100 != 0:
                continue

            axs[0].clear()
            axs[1].clear()

            axs[0].plot(val_x, values['val_swe_a'], label='analysis', color='cyan')
            axs[0].plot(val_x, values['val_swe_b'], label='background', color='green')
            axs[0].plot(val_x, values['val_swe_open'], label='open loop', color='k')
            axs[0].legend()

            axs[1].plot(val_x, values['val_hs_a'], label='analysis', color='cyan')
            axs[1].plot(val_x, values['val_hs_b'], label='background', color='green')
            axs[1].plot(val_x, values['val_hs_open'], label='open loop', color='k')
            axs[1].plot(val_x, values['val_hs_obs'], label='obs', color='red', marker='o', linestyle='None', markersize=0.5)
            axs[1].legend()

            folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plot_update_point/"
            plt.savefig(os.path.join(folder, f'assimilation_point_{j}.png'))


    output = data_settings["data"]["output_file"]

    fieldnames_state = ["Swe_w", "Swe_d", "Rho_kg/m3", "albedo"]

    plot_ensemble(meteo_inputs, inputs, Time, meteo_matrix, states, state_matrix, state_vector,
                  output_matrix, outputs, tag, N, data_settings, output, fieldnames_state,
                  hs)

    # plot assimilation
    plot_assimilation(states, Xa, state_vector, Xb_old, Xa_mean, output, outputs, output_matrix_old, output_a,
                      output_a_mean, y, Time, fieldnames_state, data_settings, tag)




if __name__ == "__main__":
    alg_settings, alg_time_start, alg_time_end, alg_domain = get_args()
    data_settings = read_file_settings(alg_settings)
    state_input = None
    output_input = None
    try:
        start = alg_time_start
        end = alg_time_end
    except:
        print("analysis of dates not given")

    results = []
    mrad, mr, window_melting,alpha, lat, lon,values = [], [], [], [], [],[],[]
    S3M_2D(mrad, mr, window_melting, alpha, lat, lon, values, start, end, state_input, output_input)