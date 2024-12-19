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
import pdb
import logging
import numpy as np
import pandas
import time
import os
import csv
import pickle
import warnings
import joblib
from statistics import NormalDist
import netCDF4 as nc

from lib_utils_system import fill_tags2string
from lib_utils_logging import set_logging_file
from lib_data_io_json import read_file_settings
from lib_info_args import logger_name
from S3M_2D_physics import S3M_2D_physics
from lib_utilis_data_proc import pre_processing, get_args, rmse

import matplotlib.pyplot as plt
import tensorflow as tf


def S3M_2D(arg,Meteo_map_data, Nivometer_data):
    # Get and set algorithm settings
    alg_settings, alg_time_start, alg_time_end, alg_domain = get_args()
    data_settings = read_file_settings(alg_settings)
    alg_time_stamp = (alg_time_start[:10].replace('/', '').replace(' ', '') +
                      alg_time_end[:10].replace('/', '').replace(' ', ''))
    # ------------------------------------------------------------------------------------------------------------------
    # Set Logging file
    log_stream = logging.getLogger(logger_name)
    current_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    log_folder_name = data_settings['log']['folder_name'] + current_time
    # ------------------------------------------------------------------------------------------------------------------
    # Uptake common inputs from json
    tag = {'source_file_datetime_generic': alg_time_stamp}
    state_limits = pandas.read_pickle(data_settings['data']['info_file']["perturbations"]['state_limits'])
    out = data_settings['data']['info_file']['out']
    output = data_settings['data']['output_file']
    dynamic_inputs = data_settings['data']['info_file']['dynamic_inputs']
    parameters = data_settings['data']['info_file']['parameters']
    change_part = data_settings['data']['info_file']["change_part"]
    input_val = data_settings['data']['info_file']['input']
    state = data_settings['data']['info_file']['state']
    fieldnames_inputs = [key for key in input_val.keys()]
    fieldnames_output = [key for key in out.keys()]
    fieldnames_state = [key for key in state.keys()]
    state_vector = [val for val in state.values()]
    output_vector = [val for val in out.values()]
    cal = data_settings['calibration']['calibrate']
    data_assimilation = data_settings['data']['info_file']["perturbations"]["per_Obs"]
    # ------------------------------------------------------------------------------------------------------------------
    # choose between calibrate, open loop, data assimilation

    if cal == 1:
        project_name = 'PhD_project  S3M 2D -CALIBRATION'
        alg_type = 'PYTHON VERSION'
        log_stream.info(' ============================================================================')
        log_stream.info('[' + project_name + ' ' + alg_type + ' - ')
        log_stream.info(str(tag))
        parameters["mrad0"] = np.round(arg[0], 3)
        parameters["mr0"] = np.round(arg[1], 3)
        parameters["window_melting"] = int(arg[2])
        meteo = np.random.rand(3650, 6)
        state = np.random.rand(4, 3650)
        output = np.random.rand(16, 3650)

        return

    if data_assimilation == 1:
        project_name = 'PhD_project  S3M 2D -DATA ASSIMILATION'
        alg_type = 'PYTHON VERSION'
        log_stream.info(' ============================================================================')
        log_stream.info('[' + project_name + ' ' + alg_type + ' - ')
        log_stream.info(str(tag))
        return

    if data_assimilation == 0 and cal == 0:
        project_name = 'PhD_project  S3M 2D -OPEN LOOP'
        alg_type = 'PYTHON VERSION'
        log_stream.info(' ============================================================================')
        log_stream.info('[' + project_name + ' ' + alg_type + ' - ')
        log_stream.info(str(tag))
    # ------------------------------------------------------------------------------------------------------------------
    # Set logging file and output file
    file_name = fill_tags2string(data_settings['log']['file_name'], data_settings['template'], tag)
    log_folder_name = fill_tags2string(log_folder_name, data_settings['template'], tag)
    set_logging_file(logger_name=logger_name, logger_file=os.path.join(log_folder_name, file_name))
    output['folder_name'] = log_folder_name

    # ------------------------------------------------------------------------------------------------------------------
    # just to write the code
    # OPEN NET CDF FILE. it contains map of meteo data(prec, rad, temp, rh)  from 40 stations over 10 years
    # the file is 40x6x3650
    # CREATE A RANDOM MATRIX 40x6x3650
    meteo = np.random.rand(40, 6, 3650)
    # create a matrix of state variables 40x4x3650
    state_matrix = np.random.rand(40, 4, 3650)
    output_matrix = np.random.rand(40, 16, 3650)
    # OBSERVATION IS A MAP OF SWE AND SNOW DEPTH FROM 40 STATIONS OVER 10 YEARS
    # THE FILE IS 40x2x3650
    # CREATE A RANDOM MATRIX 40x2x3650
    observation = np.random.rand(40, 2, 3650)
    # Define the start and end times
    start_time = "2002-10-01 00:00:00"
    end_time = "2012-09-30 23:00:00"

    # Create a date range with 1-hour frequency
    time_vector = pandas.date_range(start=start_time, end=end_time, freq='H')

    # Convert to a list if needed
    Time = time_vector.tolist()

    # -------------------------------------------------------------------------------------------------------
    # initialize the first element of the state matrix and output matrix with  state and output vector
    state_matrix[:, :, 0] = state_vector
    output_matrix[:, :, 0] = output_vector
    M = meteo.shape[2]
    # -------------------------------------------------------------------------------------------------------
    if data_assimilation ==0 and cal == 0:
        # OPEN LOOP
        for i in range(1, M):
            meteo[:, :, i], state_matrix[:, :, i], output_matrix[:, :, i], mass_balance = S3M_2D_physics(log_stream,
                                                                                                         meteo[:, :, i],
                                                                                                         dynamic_inputs,
                                                                                                         parameters,
                                                                                                         state_matrix[:,
                                                                                                         :, i - 1],
                                                                                                         output_matrix[
                                                                                                         :, :, i - 1],
                                                                                                         Time[i],
                                                                                                         change_part)

    # ------------------------------------------------------------------------------------------------------
    if cal == 1:
        # CALIBRATION
        N = Nivometer_data.shape[0]
        for i in range(1, N):
                meteo[:, i], state[:, i], output[:, i], mass_balance = S3M_2D_physics(log_stream, meteo[:, i],
                                                                                      dynamic_inputs, parameters,
                                                                                      state[:, i - 1],
                                                                                      output[:, i - 1],
                                                                                      Time[i], change_part)

        # ------------------------------------------------------------------------------------------------------
        # store variables
        # ------------------------------------------------------------------------------------------------------
        # plot variables
        # ------------------------------------------------------------------------------------------------------
        # compute RMSE
        rmse_swe = rmse(output[:, 10], Nivometer_data[i, 0])
        rmse_hs = rmse(output[:, 14], Nivometer_data[i, 1])

        return rmse_swe, rmse_hs

    if data_assimilation == 1:
        for i in range(1, M):


            meteo[:, :, i],  state_matrix[:, :, i], output_matrix[:, :, i], mass_balance = S3M_2D_physics(log_stream, meteo[:, :, i],
                                                                                   dynamic_inputs, parameters,
                                                                                   state_matrix[:, :, i - 1],
                                                                                   output_matrix[:, :, i - 1],
                                                                                   Time[i], change_part)
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

    if data_assimilation == 0 and cal == 0:
        output_file = os.path.join(output['folder_name'], fill_tags2string(output['file_name'], data_settings['template'], tag))

        with nc.Dataset(output_file, 'w', format='NETCDF4') as dataset:
            # Define dimensions
            dataset.createDimension('station', 40)
            dataset.createDimension('variable', 6)
            dataset.createDimension('time', 3650)
            dataset.createDimension('state_var', 4)
            dataset.createDimension('output_var', 16)

            # Create variables
            meteo_var = dataset.createVariable('meteo', np.float32, ('station', 'variable', 'time'))
            state_var = dataset.createVariable('state_matrix', np.float32, ('station', 'state_var', 'time'))
            output_var = dataset.createVariable('output_matrix', np.float32, ('station', 'output_var', 'time'))

            # Assign data to variables
            meteo_var[:, :, :] = meteo
            state_var[:, :, :] = state_matrix
            output_var[:, :, :] = output_matrix

            # Add attributes (optional)
            dataset.description = 'Output data from S3M_2D model'
            meteo_var.units = 'unknown'
            state_var.units = 'unknown'
            output_var.units = 'unknown'
        # ------------------------------------------------------------------------------------------------------
        # COMPUTE RMSE FOR SWE AND SNOW DEPTH
        RMSE_SWE = rmse(output_matrix[:, 10, :], observation[:, 0, :])
        # RMSE FOR SNOW DEPTH
        RMSE_SNOW_DEPTH = rmse(output_matrix[:, 14, :], observation[:, 1, :])
        # plot the info on the log file
        log_stream.info('RMSE_SWE: ' + str(RMSE_SWE))
        log_stream.info('RMSE_SNOW_DEPTH: ' + str(RMSE_SNOW_DEPTH))
        file_txt = fill_tags2string((os.path.join(output['folder_name'], output['file_name_rmse'])),
                                         data_settings['template'], tag)
        # WRITE THE INFOT IN A TXT FILE
        with open(file_txt, 'w') as f:
            f.write('RMSE_SWE: ' + str(RMSE_SWE) + '\n')
            f.write('RMSE_SNOW_DEPTH: ' + str(RMSE_SNOW_DEPTH) + '\n')
        # ------------------------------------------------------------------------------------------------------
        # Plot with imshow the  SWE and SNOW DEPTH modelled and observed
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(output_matrix[:, 10, :], cmap='viridis', aspect='auto')
        axs[0].set_title('SWE modelled')
        axs[1].imshow(observation[:, 0, :], cmap='viridis', aspect='auto')
        axs[1].set_title('SWE observed')
        # store the plot
        file_plot = fill_tags2string((os.path.join(output['folder_name'], output["file_name_plot_swe"])),
                                              data_settings['template'], tag)
        plt.savefig(file_plot)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(output_matrix[:, 14, :], cmap='viridis', aspect='auto')
        axs[0].set_title('SNOW DEPTH modelled')
        axs[1].imshow(observation[:, 1, :], cmap='viridis', aspect='auto')
        axs[1].set_title('SNOW DEPTH observed')
        # store the plot
        file_plot = fill_tags2string((os.path.join(output['folder_name'], output["file_name_plot_hs"])),
                                              data_settings['template'], tag)
        plt.savefig(file_plot)
        # ------------------------------------------------------------------------------------------------------

    return None


if __name__ == "__main__":
  arg = []
  Meteo_map_data =  []
  Nivometer_data =  []
  S3M_2D(arg, Meteo_map_data, Nivometer_data)