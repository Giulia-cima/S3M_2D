# Description: This script contains the functions used to process the data for the S3M model.
import argparse
import logging
import pandas
import os
import rasterio
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Method to get script argument(s) revised
def get_args():
    parser_handle = argparse.ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-time_start', action="store", dest="alg_time_start")
    parser_handle.add_argument('-time_end', action="store", dest="alg_time_end")
    parser_handle.add_argument('-domain', action="store", dest="alg_domain")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:

        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.alg_time_start:

        alg_time_start = parser_values.alg_time_start
    else:
        alg_time_start = None

    if parser_values.alg_time_end:
        alg_time_end = parser_values.alg_time_end
    else:
        alg_time_end = None

    if parser_values.alg_domain:
        alg_domain = parser_values.alg_domain
    else:
        alg_domain = None

    return alg_settings, alg_time_start, alg_time_end, alg_domain


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def read_path(path):
    ext = os.path.splitext(path)[-1].lower()
    # Now we can simply use == to check for equality, no need for wildcards.
    if ext == ".csv":
        return pandas.read_csv(path)
    elif ext == ".xlsx" or ext == ".xls":
        return pandas.read_excel(path)
    elif ext == ".tiff" or ext == ".tif":
        with rasterio.open(path) as src:
            meta = src.meta
            # open tiff file
            array = src.read(1)

            return array
    else:
        logging.error("file path inconsistent")
        return None


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def rmse(simulation, observation):
    # return np.sqrt(np.mean((observation - simulation) ** 2))

    return np.sqrt(np.nanmean(np.square(observation - simulation)))


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def kge(simulations, evaluation):
    # modified from hydroeval ;removed cast as float64 and return only kge
    """Original Kling-Gupta Efficiency (KGE) and its three components
       (r, α, β) as per `Gupta et al., 2009
       <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.

       Note, all four values KGE, r, α, β are returned, in this order.

       Calculation Details:
            math::
              E_{\\text{KGE}} = 1 - \\sqrt{[r - 1]^2 + [\\alpha - 1]^2
              + [\\beta - 1]^2}
           math::
              r = \\frac{\\text{cov}(e, s)}{\\sigma({e}) \\cdot \\sigma(s)}
           math::
              \\alpha = \\frac{\\sigma(s)}{\\sigma(e)}
            math::
              \\beta = \\frac{\\mu(s)}{\\mu(e)}

           where *e* is the *evaluation* series, *s* is (one of) the
           *simulations* series, *cov* is the covariance, *σ* is the
           standard deviation, and *μ* is the arithmetic mean.

       """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = np.mean(simulations)
    obs_mean = np.mean(evaluation)

    r_num = np.sum((simulations - sim_mean) * (evaluation - obs_mean))
    r_den = np.sqrt(np.sum((simulations - sim_mean) ** 2)
                    * np.sum((evaluation - obs_mean) ** 2))
    r = r_num / r_den
    # calculate error in spread of flow alpha
    alpha = np.std(simulations) / np.std(evaluation)
    # calculate error in volume beta (bias of mean discharge)
    beta = (np.sum(simulations)
            / np.sum(evaluation))
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def save_raster(input_path, output_path, new_data, lat, lon, method):
    """
    Writes new data into a TIFF file while preserving metadata.

    Parameters:
        input_path (str): Path to the input TIFF file.
        output_path (str): Path to save the new TIFF file.
        new_data (numpy array): New data to write into the TIFF file.
        lat (list): List of latitudes.
        lon (list): List of longitudes.
        method (str): Method for selecting values.
        cal (int): Calibration flag.
    """

    # create a folder if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(input_path) as src:
        metadata = src.meta.copy()  # Read metadata
        transform = src.transform  # Read transform

    # Ensure new_data has the correct shape (bands, rows, columns)
    if new_data.ndim == 2:
        new_data = np.expand_dims(new_data, axis=0)  # Add a new axis for bands

    with rasterio.open(output_path, "w", **metadata) as dst:
        dst.write(new_data)

    title = output_path.split('/')[-1].replace('.tif', '')
    # Calculate extent (left, right, bottom, top) for the coordinate system
    left = transform[2]
    right = left + transform[0] * new_data.shape[2]
    bottom = transform[5] + transform[4] * new_data.shape[1]
    top = transform[5]
    extent = [left, right, bottom, top]

    # Create a figure with two subplots
    fig, axs = plt.subplots(figsize=(15, 7))

    # Plot the first subplot
    img1 = axs.imshow(new_data[0], cmap='Blues', extent=extent)
    axs.set_title(f"{title} ")
    fig.colorbar(img1, ax=axs, orientation='horizontal')
    if title == 'swe':
        axs.set_ylabel('mm')
    elif title == 'hs':
        axs.set_ylabel('cm')
    # Save the figure
    plt.savefig(output_path.replace('.tif', '_modelled.png'))
    plt.close()

    return

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def ltln2val_from_2dDataArray(input_map, lat, lon, method):

    input_map = rioxarray.open_rasterio(input_map)

    if input_map.dims[1] != 'y' or input_map.dims[2] != 'x':
        old_dim_names = input_map.dims
        new_dim_names = ['y', 'x']
        rename_dict = dict(zip(old_dim_names, new_dim_names))
        input_map = input_map.rename(rename_dict)

    lon_query = xr.DataArray(lon, dims="points")
    lat_query = xr.DataArray(lat, dims="points")

    values = input_map.sel(x=lon_query, y=lat_query, method=method)

    return values.data[0]

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# # # Calibration

from scipy.stats import truncnorm

# Function to get truncated normal distribution
def get_truncated_normal(mean, sd, lower, upper):
    return truncnorm(
        (lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------




# Calibration only during winter on only the 50% of the available data for each station

# i want to calibrate the model for each station using only 50% of the data points, and obtain a matrix of parameters with dimensions (m, 3)
# where m is the number of stations and the three columns contain the optimal values of mrad, mr and melting_window for each station


# df_dict is a dictionary containing  the dataframes of the nivometers , each key is a tuple (lat, lon)
# store the data in a 3D matrix Nivometer_data with dimensions (n, m, 2) where n is the number of data points, m is the number of stations and the third dimension contains the swe and hs values


# search correlation with altitude or with other meteo variables (e.g. temperature, precipitation, relative humidity, solar radiation)
# in case compute a regression . calcolo la retta di regressione e la sottraggo ai dati osservati per ottenere i residui dei paramtri .
# scompongo la varianza in più componenti , togliendo la dipendenza con le variabili.

# con la retta di regressione t = aq+b t = temperatura q = quota a = pendenza b = intercetta a tutte le quote, intero dem.
# ottengo una superficie di temperatura media . uguale per tutti gli istanti
# prendo i valori osservati all'istante e sottraggo i valori della superficie di temperatura media
# ottengo i residui della temperatura
# con un metodo a mia scelta interpolo i residui ottenendo una mappa di correzione da sommare alla temperatura media.
# ottengo una mappa di temperatura corretta.
# fare lo stesso procedimento per tutte le correlazioni se presenti. in caso di più correlazioni la retta diventa un piano.
# si sottraggono i piani e si interpolano i residui.
# interpoòlazione :  try kriging.


# INTERPOLATION ON THE MAP
# Now that we have the optimal parameters for each station, we  can interpolate them on the map

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

"""
def pre_processing(data_settings, name_tag, start, end):
    dataframe = read_path(os.path.join(data_settings['data']['info_file']['dynamic_inputs']['folder_name'],
                                       data_settings['data']['info_file']['dynamic_inputs']['file_name_input']))

    try:
        dataframe.set_index(pandas.to_datetime(dataframe['date']), inplace=True)
    except KeyError:
        dataframe.set_index(pandas.to_datetime(dataframe['Date']), inplace=True)

    dataframe.sort_index(inplace=True)

    df = dataframe.loc[start:end]

    # filter input file and uniform tags
    tag = list(data_settings['data']['info_file']['tags'].values())
    input_df = df[tag]
    uniform_names = list(data_settings['data']['info_file']["fieldnames"])
    input_df = input_df.set_axis(uniform_names, axis='columns')
    input_df = input_df.replace('NAN', np.nan)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Comparison with SM original
    if data_settings['data']['info_file']['resampling'] == 2:
        path_file_OL = fill_tags2string(
            (os.path.join(data_settings['data']['info_file']['dynamic_inputs']['folder_name'],
                          data_settings['data']['info_file']['dynamic_inputs']['file_cropped_name'])),
            data_settings['template'], name_tag)
        dataframe.to_csv(path_file_OL)

        tag = [f for f in data_settings['data']['info_file']['tags'].values()]
        input_df = dataframe.filter(tag, axis=1)
        uniform_names = [f for f in data_settings['data']['info_file']["fieldnames"]]
        input_df.set_axis(uniform_names)

        input_df = input_df.drop(['snow_depth_m', 'swe_mm'], axis=1)

        T_1D = input_df["tair_degC"].rolling(
            data_settings['data']['info_file']['parameters']['window_albedo'], min_periods=1).mean()
        T_10D = input_df["tair_degC"].rolling(
            data_settings['data']['info_file']['parameters']['window_melting'], min_periods=1).mean()

        input_df['T_1D'] = (pandas.Series(T_1D, index=input_df.index)).values
        input_df['T_10D'] = (pandas.Series(T_10D, index=input_df.index)).values

        path_file_input_new = fill_tags2string(
            (os.path.join(data_settings['data']['info_file']['dynamic_inputs']['folder_name'],
                          data_settings['data']['info_file']['dynamic_inputs']['file_post_processed_name'])),
            data_settings['template'], name_tag)

        input_df.to_csv(path_file_input_new)

        return path_file_input_new, input_df, 0, 0, path_file_OL
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    #  resampling with different time resolution value equal to 1, no resampling value equal to 0.
    if data_settings['data']['info_file']['resampling'] == 1:  # in case I should resample the file
        time_resolution = data_settings['data']['info_file']['time_resolution']
        price_mapping = data_settings['data']['info_file']['fieldnames']
        input_df = (input_df.resample(time_resolution).agg(price_mapping))
        input_df['prc_mm'] = input_df['prc_mm'].ffill()
        input_df["tair_degC"] = input_df["tair_degC"].ffill()
        input_df['swin_wm-2'] = input_df['swin_wm-2'].ffill()
        input_df['rh'] = input_df['rh'].ffill()

    elif data_settings['data']['info_file']['resampling'] == 0:
        input_df = input_df.replace('NAN', np.nan)
        input_df['prc_mm'] = input_df['prc_mm'].fillna(value=0)
        input_df["tair_degC"] = input_df["tair_degC"].ffill()
        input_df['swin_wm-2'] = input_df['swin_wm-2'].ffill()
        input_df['rh'] = input_df['rh'].ffill()
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # observation unit conversion for snow depth and swe.Conversion equal to 1 , no conversion equal to 0.

    if data_settings['data']['info_file']['unit_conversion'] == 1:
        snow_depth_obs = input_df['snow_depth_m'].astype('float64').apply(
            lambda x: x * data_settings['data']['info_file']['Snow_depth_con'])
        swe_obs = input_df['swe_mm'].astype('float64').apply(
            lambda x: x * data_settings['data']['info_file']['swe_con'])
    elif data_settings['data']['info_file']['unit_conversion'] == 0:
        swe_obs = input_df['swe_mm'].apply(lambda x: max(0, x) if not np.isnan(x) else np.nan)
        snow_depth_obs = input_df['snow_depth_m'].apply(lambda x: max(0, x) if not np.isnan(x) else np.nan)
    # --------------------------------------------------------------------------------------------------------------
    if data_settings['data']['info_file']['different_assilimation'] == 1:
        # put to nan the snow_depth_obs
        snow_depth_obs = np.nan * np.ones(len(input_df))
    if data_settings['data']['info_file']['different_assilimation'] == 2:
        swe_obs = np.nan * np.ones(len(input_df))
    if data_settings['data']['info_file']['different_assilimation'] ==3:
        # if the last 5 observations are equal to each other, set the swe_obs and snow_depth_obs to nan
        if len(swe_obs) > 5:
            if swe_obs[-5:].std() == 0:
                swe_obs = np.nan * np.ones(len(input_df))
            if snow_depth_obs[-5:].std() == 0:
                snow_depth_obs = np.nan * np.ones(len(input_df))

    # ------------------------------------------------------------------------------------------------------------------
    input_df = input_df.drop(['snow_depth_m', 'swe_mm'], axis=1)
    input_df['T_10D'] = input_df['tair_degC'].rolling(
        data_settings['data']['info_file']['parameters']['window_melting'],
        min_periods=1).mean()
    input_df['T_1D'] = input_df['tair_degC'].rolling(data_settings['data']['info_file']['parameters']['window_albedo'],
                                                     min_periods=1).mean()

    path_file_input_new = fill_tags2string(
        (os.path.join(data_settings['data']['info_file']['dynamic_inputs']['folder_name'],
                      data_settings['data']['info_file']['dynamic_inputs']['file_post_processed_name'])),
        data_settings['template'], name_tag)


    input_df.to_csv(path_file_input_new)

    # ------------------------------------------------------------------------------------------------------------------

    return path_file_input_new, input_df, swe_obs, snow_depth_obs, 0
    """