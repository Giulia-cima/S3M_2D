# Re-writing  of S3M with python language. Project related to the doctoral thesis:Data assimilation
# and deep learning.I will try to re-write S3M in python and develop an EnKF data assimilation procedure to
# replace the current nudging procedure. Then I will use data assimilation data to train a neural network (LSTM)
# to exploit the computational capacity of DL to make this procedure comparable with operative times.
# In this version I will omit the glacial components. tHIS IS THE 2D VERSION
# ----------------------------------------------------------------------------------------------------------------------
"""
S3M 1D main
__date__ = '2024'
__version__ = '1.0.0'
__author__ =
        'Giulia Blandini'(giulia.blandini@cimafoundation.org',

__references__=
        'Francesco Avanzi' (francesco.avanzi@cimafoundation.org',
        'Fabio Delogu' (fabio.delogu@cimafoundation.org'

__library__ = 's3m libraries on fortran  and new : lib_utilis_flux , lib_utilis_pre_proc , diagnostic'

General command line:
python S3M_1P_run.py -settings_file "configuration_s3m_1D_package.json" -time_start "{ fill in }"
 -time_end {fill in -domain "{ fill in }"

"""
# Library
import logging
import numpy as np
import pandas
import time
import os
from datetime import datetime
from lib_utils_system import fill_tags2string
from lib_utils_logging import set_logging_file
from lib_data_io_json import read_file_settings
from S3M_2D_physics import S3M_2D_physics
from lib_utilis_data_proc import get_args, rmse, read_path, save_raster,ltln2idx_from_2dDataArray
import matplotlib.pyplot as plt


def S3M_2D(arg):
    # Get and set algorithm settings

    alg_settings, alg_time_start, alg_time_end, alg_domain = get_args()
    data_settings = read_file_settings(alg_settings)

    # Convert alg_time_start and alg_time_end to datetime objects
    start_datetime = datetime.strptime(alg_time_start, "%Y-%m-%d")
    end_datetime = datetime.strptime(alg_time_end, "%Y-%m-%d")

    # Get start year, end year, start month, start day, end month, and end day
    start_year = start_datetime.year
    end_year = end_datetime.year
    start_month = start_datetime.month
    start_day = start_datetime.day
    end_month = end_datetime.month
    end_day = end_datetime.day


    # ------------------------------------------------------------------------------------------------------------------
    # Set Logging file
    # Set the logging level for rasterio to WARNING to suppress DEBUG messages
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger_name = "S3M_2D_logger"
    log_stream = logging.getLogger(logger_name)
    # ------------------------------------------------------------------------------------------------------------------
    # Set algorithm settings
    current_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    start_code = 1
    # ------------------------------------------------------------------------------------------------------------------
    # Extract tags from the JSON file
    air_temp_tag = data_settings['data'][ "info_file"]['tags']['temperature_tag']
    precip_tag = data_settings['data'][ "info_file"]['tags']['prc_tag']
    rel_hum_tag = data_settings['data'][ "info_file"]['tags']['rh_tag']
    solar_rad_tag = data_settings['data'][ "info_file"]['tags']['rad_tag']
    snow_tag = data_settings['data'][ "info_file"]['tags']['snow_depth_tag']
    swe_tag = data_settings['data'][ "info_file"]['tags']['swe_tag']
    # ------------------------------------------------------------------------------------------------------------------
    input_path = data_settings['data']['info_file']['input_path']
    # Initialize  data and  Uptake common inputs from json
    time_series = []
    snow_depth_series = []
    swe_series = []
    n_days=1

    state_limits = pandas.read_pickle(data_settings['data']['info_file']["perturbations"]['state_limits'])
    out = data_settings['data']['info_file']['out']
    output = data_settings['data']['output_file']
    parameters = data_settings['data']['info_file']['parameters']
    change_part = data_settings['data']['info_file']["change_part"]
    state = data_settings['data']['info_file']['state']
    state_vector = [val for val in state.values()]
    output_vector = [val for val in out.values()]
    cal = data_settings['calibration']['calibrate']
    data_assimilation = data_settings['data']['info_file']["perturbations"]["per_Obs"]
    dem = parameters["coordinates_map"]
    lat = parameters["lat"]
    lon = parameters["lon"]
    # retrive index by coordinate
    row_index, col_index = ltln2idx_from_2dDataArray(dem, lat, lon)

    # create a ndarrya of size 1,parameters["window_melting"] of zeros

    # ------------------------------------------------------------------------------------------------------------------
    # choose between calibrate, open loop, data assimilation
    if cal == 1:
        project_name = 'PhD_project  S3M 2D -CALIBRATION'
        alg_type = 'PYTHON VERSION'
        log_stream.info(' ============================================================================')
        log_stream.info('[' + project_name + ' ' + alg_type + ' - ')
        parameters["mrad0"] = np.round(arg[0], 3)
        parameters["mr0"] = np.round(arg[1], 3)
        parameters["window_melting"] = int(arg[2])
        # WRITE in the log the info related to time
        log_stream.info('[' + 'mrad0' + ' ' + str(parameters["mrad0"]) + ' - ')
        log_stream.info('[' + 'mr0' + ' ' + str(parameters["mr0"]) + ' - ')
        log_stream.info('[' + 'window_melting' + ' ' + str(parameters["window_melting"]) + ' - ')

        return

    if data_assimilation == 1:
        project_name = 'PhD_project  S3M 2D -DATA ASSIMILATION'
        alg_type = 'PYTHON VERSION'
        log_stream.info(' ============================================================================')
        log_stream.info('[' + project_name + ' ' + alg_type + ' - ')

        return

    if data_assimilation == 0 and cal == 0:
        project_name = 'PhD_project  S3M 2D -OPEN LOOP'
        alg_type = 'PYTHON VERSION'
        log_stream.info(' ============================================================================')
        log_stream.info('[' + project_name + ' ' + alg_type + ' - ')


    # ------------------------------------------------------------------------------------------------------------------
    # List all year folders inside the input_path
    years = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
    # Iterate through each year folder
    for year in years:
        # Skip years before 2004 and after 2005
        if int(year) < start_year or int(year) > end_year:
            continue
        # create only one log for each year
        tag = {'source_file_datetime_generic_year': year}
        log_folder_name = fill_tags2string(data_settings["log"]["folder_name"], data_settings['template'], tag)
        logger_name = logger_name
        # Set logging file
        file_name = fill_tags2string(data_settings['log']['file_name'], data_settings['template'], tag)
        set_logging_file(logger_name=logger_name, logger_file=os.path.join(log_folder_name, file_name))
        year_path = os.path.join(input_path, year)
        months = [f for f in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, f))]
        for month in months:
            # Skip months before January 2004 and after September 2005
            if (year == str(start_year) and int(month) < start_month ) or (year == str(end_year) and int(month) > end_month):
                continue
            month_path = os.path.join(year_path, month)
            days = [f for f in os.listdir(month_path) if os.path.isdir(os.path.join(month_path, f))]

            # Iterate through each day folder
            for day in days:
                # Skip days before 01/01/2004 and after 30/09/2005
                if (year == start_year  and month ==start_month and int(day) < start_day) or (
                        year == end_year and month == end_month and int(day) > end_day):
                    continue
                # log_stream.info(f"{year}/{month}/{day}")
                day_path = os.path.join(month_path, day)
                air_temp = read_path(os.path.join(day_path, f"{air_temp_tag}_{year}{month}{day}.tif"))
                precip = read_path(os.path.join(day_path, f"{precip_tag}_{year}{month}{day}.tif"))
                rel_hum = read_path(os.path.join(day_path, f"{rel_hum_tag}_{year}{month}{day}.tif"))
                solar_rad = read_path(os.path.join(day_path, f"{solar_rad_tag}_{year}{month}{day}.tif"))
                # snow = read_path(os.path.join(day_path, f"{snow_tag}_{year}{month}{day}.tif"))
                # swe = read_path(os.path.join(day_path, f"{swe_tag}_{year}{month}{day}.tif"))
                if start_code ==1:
                    state_matrix = np.zeros((precip.shape[0], precip.shape[1], 4))
                    output_matrix = np.zeros((precip.shape[0], precip.shape[1], 16))
                    state_matrix_old = np.zeros((precip.shape[0], precip.shape[1], 4))
                    output_matrix_old = np.zeros((precip.shape[0], precip.shape[1], 16))
                    # initialize the first element of the state matrix and output matrix with  state and output vector
                    state_matrix_old[:, :, :] = state_vector
                    output_matrix_old[:, :, :] = output_vector
                    start_code = 0

                # -----------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------
                #  create a folder where to store the output from each day
                tag_day= f"{month}/{day}"
                output_folder_name = os.path.join(fill_tags2string(data_settings['data']['output_file']['folder_name'], data_settings['template'],tag), tag_day)

                # ------------------------------------------------------------------------------------------------------------------

                # create matrix of same dimension of the meteo map
                meteo = np.zeros((air_temp.shape[0], air_temp.shape[1], 6))
                # change the size of temp_average to be as air_temp , fill it all with -9999
                temp_average = np.zeros((air_temp.shape[0], air_temp.shape[1], parameters["window_melting"]))
                temp_average.fill(-9999)


                # in the first element of the meteo matrix I store the temperature that i have opened with rasterio
                # first convert air_temp to a 2D array of shape (nrows, ncols)

                temp_average[:, :, (n_days-1)] = air_temp
                # find which element of the third dimension of temp_average is -9999 AND REPLACE IT WITH THE VALUE OF air_temp
                # in the same position of firts two dimensions
                #for i in range(0, parameters["window_melting"]):
                # temp_average[:, :, i] = np.where(temp_average[:, :, i] == -9999, air_temp, temp_average[:, :, i])
                temp_average = np.where(temp_average == -9999, air_temp[:, :, np.newaxis], temp_average)


                meteo[:, :, 0] = air_temp
                meteo[:, :, 1] = precip
                meteo[:, :, 2] = rel_hum
                meteo[:, :, 3] = solar_rad
                meteo[:, :, 4] = air_temp
                # compute the average temperature , that is the average of temp_average along the third dimension
                meteo[:, :, 5] = np.mean(temp_average, axis=2)
                n_days += 1
                if n_days == parameters["window_melting"]:
                    n_days = 0  # reset the counter
                # ------------------------------------------------------------------------------------------------------
                # GET THE TIME FROM THE YEAR MONTH AND DAY AND TUORN INTO DATA TIME
                Time = pandas.to_datetime(f"{year}-{month}-{day}")

                # -------------------------------------------------------------------------------------------------------
                if data_assimilation ==0 and cal == 0:

                    meteo, state_matrix[:, :, :], output_matrix[:, :, :], mass_balance = S3M_2D_physics(log_stream,meteo,parameters,
                                                                                    state_matrix_old,output_matrix_old,Time,change_part)


                    # Extract snow_modelled from output_matrix
                    swe_modelled = output_matrix[:, :, 10]
                    snow_modelled = np.array(output_matrix[:, :, 14])
                    # Update state_matrix_old and output_matrix_old
                    state_matrix_old[:, :, :] = state_matrix[:, :, :]
                    output_matrix_old[:, :, :] = output_matrix[:, :, :]
                # ------------------------------------------------------------------------------------------------------
                if cal == 1:
                    # CALIBRATION
                    #N = swe.shape[0]
                    #for i in range(1, N):



                    # ------------------------------------------------------------------------------------------------------
                    # store variables
                    # ------------------------------------------------------------------------------------------------------
                    # plot variables
                    # ------------------------------------------------------------------------------------------------------
                    # compute RMSE
                   # rmse_swe = rmse(output[:, :,10], swe)
                    #rmse_hs = rmse(output[:, :,14], snow)

                    return
                # ------------------------------------------------------------------------------------------------------
                if data_assimilation == 1:
                    #for i in range(1, M):



                    # ------------------------------------------------------------------------------------------------------
                    # data assimilation
                    # ------------------------------------------------------------------------------------------------------
                    # store variables
                    # ------------------------------------------------------------------------------------------------------
                    # plot variables
                    # ------------------------------------------------------------------------------------------------------
                    # compute RMSE
                    # ------------------------------------------------------------------------------------------------------
                    return None
                # ------------------------------------------------------------------------------------------------------
                # save the map of snow and swe in the output folder as tif file
                snow_output_path = os.path.join(output_folder_name, 'snow_depth.tif')
                swe_output_path = os.path.join(output_folder_name, 'swe.tif')
                reference_file = os.path.join(day_path, f"{precip_tag}_{year}{month}{day}.tif")
                # Assuming `snow` and `swe` are the arrays you want to save
                save_raster(reference_file, snow_output_path, snow_modelled)
                save_raster(reference_file, swe_output_path, swe_modelled)
                # ------------------------------------------------------------------------------------------------------
                # COMPUTE RMSE FOR SWE AND SNOW DEPTH
                #rmse_swe = rmse(output_matrix[:, :, 10], swe)
                #rmse_hs = rmse(output_matrix[:, :, 14], snow)
                # WRITE THE INFOT IN A TXT FILE
                # with open(os.path.join(output_folder_name, 'RMSE.txt'), 'w') as f:
                    #f.write(f'RMSE SWE: {rmse_swe}\n')
                    #f.write(f'RMSE HS: {rmse_hs}\n')

                # ------------------------------------------------------------------------------------------------------
                # chose one point of the map and plot the time series of the snow depth and swe  in a self updating plot

                snow_depth = np.array(output_matrix[:, :, 14])[row_index, col_index]
                snow_depth_series.append(snow_depth)
                swe = np.array(output_matrix[:, :, 10])[row_index, col_index]
                swe_series.append(swe)
            # ------------------------------------------------------------------------------------------------------
        # first convert time_series, snow_depth_series, and swe_series to numpy arrays

    snow_depth_series = np.array(snow_depth_series)*100 # to convert inot cm
    swe_series = np.array(swe_series)


    # remove from the path tHE DAY AND MONTH
    path = fill_tags2string(data_settings['data']['output_file']['folder_name'], data_settings['template'],tag)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot snow depth time series
    ax1.plot(snow_depth_series)
    ax1.set_ylabel('Snow Depth')
    ax1.set_title('Snow Depth (cm)' )

    # Plot SWE time series
    ax2.plot(swe_series)
    ax2.set_ylabel('SWE (mm)')
    ax2.set_title('SWE Time Series')

    # Save the figure
    fig_path = os.path.join(path,'time_series_plots.png')
    plt.tight_layout()
    # insert title
    fig.suptitle('Valpelline-Chosoz\n')
    plt.savefig(fig_path)
    plt.close(fig)
        # ------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------
    return None


if __name__ == "__main__":
  arg = []
  S3M_2D(arg)
