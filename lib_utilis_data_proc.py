
import pandas
import os
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
from lib_utils_system import fill_tags2string
from scipy.interpolate import interp1d

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


        """
        # take the path of the file with the assimilation data
        dataframe_assimilation = read_path(data_settings['data']['info_file']['file_assimilation'])
        try:
            dataframe_assimilation.set_index(pandas.to_datetime(dataframe_assimilation['date']), inplace=True)
        except KeyError:
            dataframe_assimilation.set_index(pandas.to_datetime(dataframe_assimilation['Date']), inplace=True)
        dataframe_assimilation.sort_index(inplace=True)

        # find the closest date in dataframe_assimilation to start and to end
        closest_start = dataframe_assimilation.index.get_loc(
            dataframe_assimilation.index.to_series().sub(pandas.Timestamp(start)).abs().idxmin())
        closest_end = dataframe_assimilation.index.get_loc(
            dataframe_assimilation.index.to_series().sub(pandas.Timestamp(end)).abs().idxmin())
        df_assimilation = dataframe_assimilation.iloc[closest_start:closest_end + 1]

        if data_settings['data']['info_file']['different_assilimation_time'] == 1:
            df_assimilation = df_assimilation.resample('1H').asfreq()
            df_assimilation = df_assimilation[:-1]
        # set the snow_depth_obs and swe_obs equal to the assimilation data
        try:
            swe_obs = df_assimilation['swe_mm'].apply(lambda x: max(0, x) if not np.isnan(x) else np.nan)
        except:
            # set the swe_obs as a column of nan
            swe_obs = np.nan * np.ones(len(input_df))
        try:
            snow_depth_obs = df_assimilation['snow_depth_m'].apply(lambda x: max(0, x) if not np.isnan(x) else np.nan)
        except:
            # set the snow_depth_obs as a column of nan
            snow_depth_obs = np.nan * np.ones(len(input_df))
            """

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


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def read_path(path):
    ext = os.path.splitext(path)[-1].lower()
    # Now we can simply use == to check for equality, no need for wildcards.
    if ext == ".csv":
        return pandas.read_csv(path)
    elif ext == ".xlsx" or ext == ".xls":
        return pandas.read_excel(path)
    else:
        logging.error("file path inconsistent")
        return None

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------



"""
def interpolation(s, x):
    Interpolates the data to the desired time step and returns the interpolated value.

    If the new x value is outside the range of the data, it returns the value indexed at the last value of the data.

    Args:
    - s: DataFrame or Series containing the data
    - x: Value for interpolation
    the interpolation method is linear by default
    the function numpy.interp takes as argument the x value, the x values of the data and the y values of the data
    numpy.interp(x, xp, fp, left=None, right=None, period=None)[source]


    Returns:
    - Interpolated value at the desired time step
    
    # already done by the np. interp function
    #if x < s.iloc[:, 1].min():
   #     return s.iloc[0, 0]
    #elif x > s.iloc[:, 1].max():
    #    return s.iloc[-1, 0]

    if not hasattr(interpolation, 'interpolation_func') or interpolation.s is not s:
        s.sort_values(s.columns[1], inplace=True)
        interpolation.s = s
        #interpolation.interpolation_func = interp1d(s.iloc[:, 1], s.iloc[:, 0], kind=kind)
        #return interpolation.interpolation_func(x)
        val_interp = np.interp(x, s.iloc[:, 1], s.iloc[:, 0])
    return val_interp
"""

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# # # Calibration data processing


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
