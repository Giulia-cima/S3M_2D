
import keras
from scipy import stats
import numpy as np
import pandas as pd
import xarray as xr
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import keras.optimizers
from keras.models import Sequential
from lib_data_io_json import read_file_settings
from lib_utilis_data_proc import get_args
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense
from keras.layers import Softmax, Multiply, Lambda
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit

# ==========================  Assign hydrological year ========
def hydrological_year(df):
    """
    Assign hydrological year to each timestamp.
    Hydrological year starts on Oct 1st (00:00) and ends on Sep 30th (23:00).
    """
    df = df.copy()
    df['hydro_year'] = np.where(
        df.index.month >= 10,
        df.index.year + 1,
        df.index.year
    )
    return df

def get_hydro_year(time, start_month=9):
    time = pd.to_datetime(time)

    # convert to numpy array so it is mutable
    year = time.year.values.copy()
    month = time.month.values

    # hydrological year shift
    year[month >= start_month] += 1

    return year

# ========================== Helper Functions =============
def compute_year_features(X, hydro_year):
    features = []
    year_list = []

    for y in np.unique(hydro_year):
        idx = hydro_year == y
        data = X[idx]

        # flatten spatial dims if needed
        data = data.reshape(data.shape[0], -1)

        feats = [
            np.nanmean(data),
            np.nanstd(data),
            np.nanpercentile(data, 90),
            np.nanpercentile(data, 10),
        ]

        features.append(feats)
        year_list.append(y)

    return np.array(features), np.array(year_list)

def smooth(y, box=5):
    if box <= 1:
        return y
    return np.convolve(y, np.ones(box) / box, mode='same')

def fill_tensors(station_ids, input_station, target_station,
                 input_tensor, target_tensor,
                 feature_names, target_names, nt):

    """
    Populate input and target tensors for given station IDs. Stacks data from multiple stations into a single tensor.
    :param station_ids: List of station IDs to include
    :param input_station: Dictionary of input data per station
    :param target_station: Dictionary of target data per station
    :param input_tensor: Preallocated input tensor to fill
    :param target_tensor: Preallocated target tensor to fill
    :param feature_names: List of feature names for input data
    :param target_names: List of target names for target data
    :param nt: Number of time steps per station
        :return: Filled input and target tensors
    """


    counter = 0

    for sid in station_ids:
        i0 = counter
        i1 = counter + nt

        for f, feat_name in enumerate(feature_names):
            input_tensor[i0:i1, f] = input_station[sid][feat_name]
        if target_tensor is None:
            continue
        else:
            for f, feat_name in enumerate(target_names):
                target_tensor[i0:i1, f] = target_station[sid][feat_name]

        counter += nt
    if target_tensor is None:
        return input_tensor, None
    else:
        return input_tensor, target_tensor

def zero_substitution_division(df, Tvalues, min_value, max_value):
    """
    This version of the function performs zero substitution followed by division normalization, taking the min and max values from the
    historical data statistics.
    :param df:  dataframe to process. it is important that the columns are in the same order as min_value and max_value lists
    :param Tvalues:  temperature values for zero substitution
    :param min_value: historical minimum values for division normalization
    :param max_value: historical maximum values for division normalization
    :return: processed dataframe
    """
    for i in range(0, df.shape[1]):
        df.iloc[:, i] = np.where(df.iloc[:, i] == 0, -Tvalues[i], df.iloc[:, i])

        if isinstance(min_value, float):
            min_val = min_value
            max_val = max_value
        else:
            min_val = min_value[i]
            max_val = max_value[i]

        df.iloc[:, i] = np.where(df.iloc[:, i] < 0, -(df.iloc[:, i] / min_val) * max_val, df.iloc[:, i])

    return df

def quantile_mapping(x, key=None):
    """
    Gaussianize a 1D variable using empirical CDF. First sorts the data, then computes the empirical CDF,
    and finally maps it to Gaussian quantiles using the inverse CDF (ppf) of the standard normal distribution.
    Before returning, the DataFrame is reordered to match the original input order.
    ️:param x: 1D array-like (pd.Series, pd.DataFrame column, or np.ndarray)
    :param key: column name if x is a DataFrame
    ️:return:  pd.Series of Gaussianized values
    """

    # -------------------- Input handling --------------------
    if isinstance(x, pd.DataFrame):
        if key is None:
            raise ValueError("key must be provided when x is a DataFrame")
        series = x[key]
    elif isinstance(x, pd.Series):
        series = x
    else:
        series = pd.Series(np.asarray(x))

    # -------------------- Remove NaNs --------------------
    is_valid = ~series.isna()
    values = series[is_valid].values
    valid_index = series[is_valid].index

    N = len(values)
    if N == 0:
        print(f"{key} Input contains only NaNs")
        # return an empty DataFrame
        return pd.DataFrame(columns=[key, "emp_cdf", "gauss_quant"])
    # -------------------- Create DataFrame --------------------
    mapping_df = pd.DataFrame({
        key: values}, index=valid_index)

    mapping_df_sorted = mapping_df.sort_values(key, ascending=True)

    # -------------------- Empirical CDF --------------------

    mapping_df_sorted["emp_cdf"] = np.arange(1, N + 1) / (N + 1)
    mapping_df_sorted["gauss_quant"] = stats.norm.ppf(mapping_df_sorted["emp_cdf"])

    # -------------------- Reorder mapping_df to original input order --------------------
    mapping_df_sorted_back = mapping_df_sorted.sort_index()

    return mapping_df_sorted_back["gauss_quant"]

def inverse_quantile_mapping(z, df_ref, key=None):

    # ---- 1. Attach index if missing
    if hasattr(z, "index"):
        idx = z.index
        z_vals = z.values
    else:
        idx = np.arange(len(z))
        z_vals = np.asarray(z)

    # ---- 2. Gaussian -> probability
    p = stats.norm.cdf(z_vals)

    # ---- 3. Empirical reference (must be sorted for interp)
    df_sorted =  df_ref.sort_values(by="emp_cdf", ascending=True)
    emp_sorted = df_sorted["emp_cdf"].values
    orig_sorted = df_sorted[key].values

    # ---- 4. Sort probabilities but keep track of order
    sort_p = np.argsort(p)
    p_sorted = p[sort_p]

    inv_sorted = np.interp(p_sorted, emp_sorted, orig_sorted)

    # ---- 5. Restore original order
    inv = np.empty_like(inv_sorted)
    inv[sort_p] = inv_sorted

    # sort by index
    inv_series = pd.Series(inv, index=idx)

    return  inv_series

# ==========================Cost Functions===============

def residual_entropy_tf(residuals, eps=1e-6):
    var = tf.math.reduce_variance(residuals)
    return tf.math.log(var + eps)

def physical_penalty(y_pred, lim_inf, lim_sup):
    """
    soft quadratic penalty outside bounds:
    zero inside bounds
    smooth gradients
    stable optimization
    interpretable physically

    """
    below = tf.nn.relu(lim_inf - y_pred)
    above = tf.nn.relu(y_pred - lim_sup)
    return tf.reduce_mean(below**2 + above**2)


def custom_loss(y_true, y_pred):

    """"
    RMSE → accuracy

    physics penalty → physical consistency

    entropy → predictability / noise control
    """
    global lim_inf, lim_sup

    # --------------------
    # RMSE
    # --------------------
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)

    # --------------------
    # Physical bounds penalty
    # --------------------
    lim_inf_tf = tf.constant([lim_inf[11], lim_inf[12], lim_inf[13], lim_inf[14]], dtype=y_pred.dtype)
    lim_sup_tf = tf.constant([lim_sup[11], lim_sup[12], lim_sup[13], lim_sup[14]], dtype=y_pred.dtype)

    phys_pen = physical_penalty(y_pred, lim_inf_tf, lim_sup_tf)


    # add another penalty to decrease error in snow density prediction (position 2)
    density_penalty = tf.reduce_mean(tf.square(y_pred[2] - y_true[2]))

    # --------------------
    # Entropy surrogate
    # --------------------
    residuals = y_true - y_pred
    entropy_term = residual_entropy_tf(residuals)

    # --------------------
    # Final loss
    # --------------------
    return rmse + 0.1 * phys_pen + 0.01 * entropy_term


# ======================== Hyperparameter Search ===========


def hyperparameter_search(
    train_X, train_y,
    val_X, val_y,
    custom_loss,
    search_space,
    n_trials=20,
    epochs=200,
    patience=20,
    verbose=0
):
    results = []
    best_val_loss = np.inf
    best_model = None
    best_params = None


    input_shape = (train_X.shape[1], train_X.shape[2])

    for trial in range(n_trials):
        print(f"\n🔍 Trial {trial+1}/{n_trials}")

        # Sample hyperparameters
        params = {
            "cell_1": int(np.random.choice(search_space["cell_1"])),
            "cell_2": int(np.random.choice(search_space["cell_2"])),
            "cell_3": int(np.random.choice(search_space["cell_3"])),
            "cell_4": int(np.random.choice(search_space["cell_4"])),
            "cell_dense": int(search_space["cell_dense"]),
            "initial_lr": float(np.random.choice(search_space["initial_lr"])),
            "decay_rate": float(np.random.choice(search_space["decay_rate"])),
            "batch_size": int(np.random.choice(search_space["batch_size"])),
        }

        model_params = {k: v for k, v in params.items() if k != "batch_size"}

        model = build_lstm_model(
            input_shape=input_shape,
            custom_loss=custom_loss,
            **model_params
        )

        callback = keras.callbacks.EarlyStopping(
            patience=patience,
            restore_best_weights=True
        )

        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=epochs,
            batch_size=params["batch_size"],
            shuffle=False,          # 🔴 critical for time series
            callbacks=[callback],
            verbose=verbose
        )

        min_val_loss = np.min(history.history["val_loss"])

        results.append({
            "params": params,
            "val_loss": min_val_loss
        })

        print(f"   ➜ val_loss = {min_val_loss:.4e}")

        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_model = model
            best_params = params
            best_history = history

    return best_model, best_params, best_history, results

#========================== Build LSTM Model =============

def build_lstm_model(
    input_shape,
    cell_1,
    cell_2,
    cell_3,
    cell_4,
    cell_dense,
    initial_lr,
    decay_rate,
    custom_loss
):
    model = Sequential()
    model.add(LSTM(cell_1, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(cell_2, return_sequences=True))
    model.add(LSTM(cell_3, return_sequences=True))
    model.add(LSTM(cell_4))
    model.add(Dense(cell_dense))

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=100000,
        decay_rate=decay_rate
    )

    optimizer = keras.optimizers.SGD(
        learning_rate=lr_schedule,
        clipnorm=1.0
    )

    model.compile(loss=custom_loss, optimizer=optimizer)
    return model

# ====================================================================

    # map month -> season
    def month_to_season(m):
        if m in [12, 1, 2]:
            return 0  # winter
        elif m in [3, 4, 5]:
            return 1  # spring
        elif m in [6, 7, 8]:
            return 2  # summer
        else:
            return 3  # autumn

# ====================================================
# ====================================================


# ========================== Main Model Functions ==========


def model_single_station():

    #Set a limits to cpu usage for TF
    tf.config.threading.set_intra_op_parallelism_threads(20)
    tf.config.threading.set_inter_op_parallelism_threads(20)

    # ========================== Load Settings and Data ==========================
    alg_settings, alg_time_start, alg_time_end, alg_domain = get_args()
    data_settings = read_file_settings(alg_settings)
    start_datetime = datetime.strptime(alg_time_start, "%Y-%m-%d %H:%M:%S")
    end_datetime = datetime.strptime(alg_time_end, "%Y-%m-%d %H:%M:%S")
    time_index = pd.date_range(start=start_datetime, end=end_datetime, freq='h')

    # ========================== Prepare Forcing Data ==========================
    meteo_ds = xr.open_dataset(data_settings['data']['info_file']['comparison_ol_meteo'])
    meteo_ds = meteo_ds.sel(time=slice(alg_time_start, alg_time_end))
    # ========================== Prepare  Model Data ==========================
    prior_state = (xr.open_dataset(data_settings['LSTM']['prior_state'])).sel(time=slice(alg_time_start, alg_time_end))
    posterior_state = (xr.open_dataset(data_settings['LSTM']['posterior_state'])).sel(time=slice(alg_time_start, alg_time_end))
    output_prior = (xr.open_dataset(data_settings['LSTM']['output_prior'])).sel(time=slice(alg_time_start, alg_time_end))
    output_post = (xr.open_dataset(data_settings['LSTM']['output_post'])).sel(time=slice(alg_time_start, alg_time_end))

    # ========================== Prepare Observed Data  ==========================
    obs = pd.read_pickle(data_settings['data']['info_file']["time_series_obs"])
    prior_col = list(prior_state.data_vars)
    nt = len(time_index)
    Snow_depth_obs = np.zeros((nt))
    y_p = 7
    # ========================== Load Limits and Stats ==========================
    global lim_inf, lim_sup
    lim_sup = joblib.load(data_settings['LSTM']['lim_sup'])[y_p]
    lim_inf =  joblib.load(data_settings['LSTM']['lim_inf'])[y_p]
    stat = joblib.load(data_settings['LSTM']['statistics'])
    gaussianization_data = joblib.load(data_settings['LSTM']['gaussianization_data'])

    # ========================== Prepare Data for LSTM ==========================
    m = len(meteo_ds.data_vars) + 2  # number of meteorological variables + 2 derived variables

    feature_names = [
        'Rain', 'AirTemperature', 'RelHumidity', 'IncRadiation',
        'T_albedo', 'T_melting',
        'SWE_W_prior', 'SWE_D_prior', 'RHO_D_prior','albedo_prior',
        'Snow_depth_obs']
    n_features = len(feature_names)

    target_names = [ 'SWE_W_post', 'SWE_D_post', 'albedo_post', 'RHO_D_post']
    target_names_test = [ 'SWE_W_mm_post', 'SWE_D_mm_post', 'RHO_D_kg_m3_post', 'albedo_post']
    n_target = len(target_names)

    # ========================== Find station data ==========================
    station_meteo = meteo_ds.sel(station=y_p)
    prior_state_station = prior_state.sel(point=y_p)
    posterior_state_station = posterior_state.sel(point=y_p)
    station_stats = stat[y_p]
    gauss_data = gaussianization_data[y_p]

    # ========================== Prepare Observed Snow Depth Data  ==========================
    keys_list = list(obs.keys())
    df = obs[keys_list[y_p]]
    df = df.set_index('time')
    df = df[~df.index.duplicated(keep='first')]
    df_res = df.resample('h').asfreq()
    df_1 = df_res.loc[start_datetime:end_datetime]
    years_to_remove = []
    df_1 = df_1.reindex(time_index)
    df_1 = hydrological_year(df_1)

    for hydro_year, group in df_1.groupby('hydro_year'):
        if group['Snow_depth_cm'].isna().all():
            years_to_remove.append(hydro_year)

    for year in years_to_remove:
        df_1 = df_1[df_1['hydro_year'] != year]
        station_meteo = station_meteo.sel(time=~station_meteo['time.year'].isin([year - 1, year]))
        prior_state_station = prior_state_station.sel(time=~prior_state_station['time.year'].isin([year - 1, year]))
        posterior_state_station = posterior_state_station.sel(
            time=~posterior_state_station['time.year'].isin([year - 1, year]))
        output_prior = output_prior.sel(time=~output_prior['time.year'].isin([year - 1, year]))
        output_post = output_post.sel(time=~output_post['time.year'].isin([year - 1, year]))

    Snow_depth_obs[:] = df_1['Snow_depth_cm'] / 100
    snow_depth_observed = Snow_depth_obs[:].copy()
    Snow_depth_obs[:] = pd.Series(Snow_depth_obs[:]).fillna(method='ffill', limit=3).fillna(method='bfill',
                                                                                            limit=3).fillna(0).values

    # ========================== Prepare Meteorological Data  ==========================
    meteo = np.zeros((nt, m), dtype=np.float32)
    meteo[:, 0] = station_meteo["AirTemperature"]
    meteo[:, 1] = station_meteo["Rain"]
    meteo[:, 1] = np.where(meteo[:, 1] < 0, 0, meteo[:, 1])
    meteo[:, 2] = station_meteo["RelHumidity"]
    meteo[:, 2] = np.clip(meteo[:, 2], 0, 100)
    meteo[:, 3] = station_meteo["IncRadiation"]
    meteo[:, 4] = station_meteo["AirTemperature"].rolling(time=24, min_periods=1).mean()
    meteo[:, 5] = station_meteo["AirTemperature"].rolling(time=48, min_periods=1).mean()

    # ========================== Zero substitution  ==========================
    prior_state_min = station_stats['prior_state_min']
    prior_state_max = station_stats['prior_state_max']
    posterior_state_min = station_stats['posterior_state_min']
    posterior_state_max = station_stats['posterior_state_max']
    obs_min = station_stats['obs_min']
    obs_max = station_stats['obs_max']
    prior_state_station_df = prior_state_station.to_dataframe()
    prior_state_station_df = prior_state_station_df.drop(columns=['point'])
    prior_state_station_df = zero_substitution_division(prior_state_station_df,
                                                        meteo[:, 0], prior_state_min,prior_state_max)
    posterior_state_station_df = posterior_state_station.to_dataframe()
    posterior_state_station_df = posterior_state_station_df.drop(columns=['point'])
    posterior_state_station_df = zero_substitution_division(posterior_state_station_df,
                                                            meteo[:, 0], posterior_state_min,posterior_state_max)
    Snow_depth_obs[:] = (zero_substitution_division(pd.DataFrame(Snow_depth_obs[:]),
                                                    meteo[:, 0], obs_min, obs_max)).values.flatten()

    # ==================Gaussianization ==================
    prc_mm = quantile_mapping(meteo[:, 1], 'Rain')
    T_air = quantile_mapping(meteo[:, 0], 'AirTemperature')
    RH = quantile_mapping(meteo[:, 2], 'RelHumidity')
    IR = quantile_mapping(meteo[:, 3], 'IncRadiation')
    T_albedo = quantile_mapping(meteo[:, 4], 'T_albedo')
    T_melting = quantile_mapping(meteo[:, 5], 'T_melting')
    prior_swe_w = quantile_mapping(prior_state_station_df['SWE_W_mm'], 'SWE_W_mm')
    prior_swe_d = quantile_mapping(prior_state_station_df['SWE_D_mm'], 'SWE_D_mm')
    prior_albedo = quantile_mapping(prior_state_station_df['albedo'], 'albedo')
    prior_rho = quantile_mapping(prior_state_station_df['RHO_D_kg_m3'], 'RHO_D_kg_m3')
    snow_depth = quantile_mapping(Snow_depth_obs[:], 'Snow_depth_cm')
    posterior_swe_w = quantile_mapping(posterior_state_station_df['SWE_W_mm'], 'SWE_W_mm')
    posterior_swe_d = quantile_mapping(posterior_state_station_df['SWE_D_mm'], 'SWE_D_mm')
    posterior_albedo = quantile_mapping(posterior_state_station_df['albedo'], 'albedo')
    posterior_rho = quantile_mapping(posterior_state_station_df["RHO_D_kg_m3"], 'RHO_D_kg_m3')

    qm_out = {
        'Rain': prc_mm,
        'AirTemperature': T_air,
        'RelHumidity': RH,
        'IncRadiation': IR,
        'T_albedo': T_albedo,
        'T_melting': T_melting,
        'SWE_W_prior': prior_swe_w,
        'SWE_D_prior':prior_swe_d,
        'RHO_D_prior': prior_rho,
        'albedo_prior':prior_albedo,
        'Snow_depth_obs':snow_depth
    }

    qm_out_target = {
        'SWE_W_post': posterior_swe_w,
        'SWE_D_post': posterior_swe_d,
        'RHO_D_post': posterior_rho,
        'albedo_post': posterior_albedo }

    # --- 3. Store station input ---
    input_station = {
        var_name: var_dict
        for var_name, var_dict in qm_out.items()
    }
    target_station = {
        var_name: var_dict
        for var_name, var_dict in qm_out_target.items()
    }
    print("Data preparation complete. Starting LSTM training...")
    if 0:
        val_out = { 'Rain': meteo[:, 1],
                'AirTemperature': meteo[:, 0],
                'RelHumidity': meteo[:, 2],
                'IncRadiation': meteo[:, 3],
                'T_albedo': meteo[:, 4],
                'T_melting': meteo[:, 5],
                'SWE_W_prior': prior_state_station_df['SWE_W_mm'].values,
                'SWE_D_prior':prior_state_station_df['SWE_D_mm'].values,
                'RHO_D_prior': prior_state_station_df['RHO_D_kg_m3'].values,
                'albedo_prior': prior_state_station_df['albedo'].values,
                'Snow_depth_obs': Snow_depth_obs[:]
            }

        val_out_target = {
                'SWE_W_post': posterior_state_station_df['SWE_W_mm'].values,
                'SWE_D_post': posterior_state_station_df['SWE_D_mm'].values,
                'RHO_D_post': posterior_state_station_df['RHO_D_kg_m3'].values,
                'albedo_post': posterior_state_station_df['albedo'].values
        }

        # --- 3. Store station input ---
        input_station = {
            var_name: var_dict
            for var_name, var_dict in val_out.items()}

        target_station = {
            var_name: var_dict
            for var_name, var_dict in val_out_target.items() }

    # ========================== Plotting prior and posterior comparison ==========================

    output_prior_station = output_prior.sel(point=y_p)
    output_post_station = output_post.sel(point=y_p)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(output_prior_station['SWE_mm'], label='Output Prior SWE', color='black')
    plt.plot(output_post_station['SWE_mm'], label='Output Post SWE', color='green')
    plt.title(f'SWE Comparison for Station ID: {y_p}')
    plt.xlabel('Time')
    plt.ylabel('SWE (mm)')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(output_prior_station['H_S_m'], label='Output Prior HS', color='black')
    plt.plot(output_post_station['H_S_m'], label='Output Post HS', color='green')
    plt.plot(snow_depth_observed, label='Observed Snow Depth', color='red', marker='o', linestyle='None', markersize=4)
    plt.title(f'HS Comparison for Station ID: {y_p}')
    plt.xlabel('Time')
    plt.ylabel('HS (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(data_settings['LSTM']['figure_initial_comparison'].replace('.png', f'_station_{y_p}.png'))
    plt.close()

    # ========================== Sample Split  before populating the tensor ==========================

    np.random.seed(42)

    X = np.column_stack([qm_out[name] for name in feature_names]).astype(np.float32)
    hydro_year = get_hydro_year(time_index)

    # 1️⃣ Compute year-level features
    year_feats, year_ids = compute_year_features(X, hydro_year)
    n_years = len(year_ids)

    # 2️⃣ Standardize year features for clustering
    scaler = StandardScaler()
    year_feats_scaled = scaler.fit_transform(year_feats)

    # 3️⃣ Decide on number of clusters
    n_clusters = min(4, n_years // 2)  # max 4 clusters or half the number of years

    use_random_split = False

    # 4️⃣ KMeans clustering if enough years
    if n_years >= 8:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(year_feats_scaled)

        # Check minimum cluster size
        if np.min(np.bincount(labels)) < 2:
            print("Some clusters <2 years → fallback to random year split")
            use_random_split = True
    else:
        print("Few hydrological years (<8) → fallback to random year split")
        use_random_split = True

    # 5️⃣ Random year split fallback
    if use_random_split:
        shuffled = np.random.permutation(n_years)
        n_train = int(n_years * 0.7)
        n_val = int(n_years * 0.15)
        train_years = year_ids[shuffled[:n_train]]
        val_years = year_ids[shuffled[n_train:n_train + n_val]]
        test_years = year_ids[shuffled[n_train + n_val:]]
    else:
        # 6️⃣ Stratified split using KMeans labels
        test_fraction = 0.25
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=42)
        train_idx, temp_idx = next(splitter.split(year_ids, labels))
        train_years = year_ids[train_idx]
        temp_years = year_ids[temp_idx]
        temp_labels = labels[temp_idx]

        # Check cluster sizes in temp set before stratifying val/test
        if np.min(np.bincount(temp_labels)) < 2:
            print("Clusters too small in temp set → random val/test split")
            shuffled = np.random.permutation(len(temp_years))
            n_val = int(len(temp_years) * 0.6)
            val_years = temp_years[shuffled[:n_val]]
            test_years = temp_years[shuffled[n_val:]]
        else:
            splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
            val_idx, test_idx = next(splitter2.split(temp_years, temp_labels))
            val_years = temp_years[val_idx]
            test_years = temp_years[test_idx]

    # 7️⃣ Map years back to actual indices for LSTM tensors
    train_ids = np.where(np.isin(hydro_year, train_years))[0]
    val_ids = np.where(np.isin(hydro_year, val_years))[0]
    test_ids = np.where(np.isin(hydro_year, test_years))[0]

    print(f"Train/Val/Test sizes → {len(train_ids)}, {len(val_ids)}, {len(test_ids)}")

    n_train = len(train_ids)
    n_val = len(val_ids)
    n_test = len(test_ids)

    input_tensor_train = np.zeros((n_train, n_features), dtype=np.float32)
    target_tensor_train = np.zeros((n_train, n_target), dtype=np.float32)
    input_tensor_val = np.zeros((n_val, n_features), dtype=np.float32)
    target_tensor_val = np.zeros((n_val, n_target), dtype=np.float32)
    input_tensor_test = np.zeros((n_test, n_features), dtype=np.float32)
    target_tensor_test = np.zeros((n_test, n_target), dtype=np.float32)
    inv_qm = np.zeros((n_test, n_target))
    # ========================== Populate Tensors considering the train,val and test ids ==========================
    for f, feat_name in enumerate(feature_names):
        input_tensor_train[:, f] = input_station[feat_name].iloc[train_ids].values
        input_tensor_val[:, f] = input_station[feat_name].iloc[val_ids].values
        input_tensor_test[:, f]  = input_station[feat_name].iloc[test_ids].values

    for f, feat_name in enumerate(target_names):
        target_tensor_train[:, f] = target_station[feat_name].iloc[train_ids].values
        target_tensor_val[:, f] = target_station[feat_name].iloc[val_ids].values
        target_tensor_test[:, f]  = target_station[feat_name].iloc[test_ids].values


    # ========================== Reshape Tensors for LSTM ==========================
    train_X = input_tensor_train.reshape((n_train, 1, n_features))
    train_y = target_tensor_train.reshape((n_train, 1,  n_target))
    val_X = input_tensor_val.reshape((n_val, 1, n_features))
    val_y = target_tensor_val.reshape((n_val, 1, n_target))

    # ========================= Verify mean and std of splits =========================
    y_ref = target_station[target_names[0]].values

    for name, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
        vals = y_ref[ids]
        print(f"{name}: mean={vals.mean():.3f}, std={vals.std():.3f}, n={len(vals)}")

    plt.figure()
    plt.hist(y_ref[train_ids], bins=50, alpha=0.5, label="train")
    plt.hist(y_ref[val_ids], bins=50, alpha=0.5, label="val")
    plt.hist(y_ref[test_ids], bins=50, alpha=0.5, label="test")
    plt.legend()
    plt.title("Distribution check of target variable")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig (data_settings['LSTM']['figure_distribution_check'].replace('.png', f'_station_{y_p}.png'))
    plt.close()

    # ========================== LSTM Model ==========================
    if 0:
        search_space = {
            "cell_1": [16,32,64],
            "cell_2": [16,64,128,256],
            "cell_3":  [16,32,64],
            "cell_4": [16,32,128],
            "cell_dense": 4,  # SWE components + density + albedo
            "initial_lr": [0.1, 1e-2, 1e-3],
            "decay_rate": [0.95, 0.99],
            "batch_size": [16, 32, 64,128,500]
        }

        best_model, best_params, best_history, results = hyperparameter_search(
            train_X, train_y,
            val_X, val_y,
            custom_loss=custom_loss,
            search_space=search_space,
            n_trials=30,
            epochs=500,
            patience=200,
            verbose=0)

        print("✅ Best hyperparameters:")
        for k, v in best_params.items():
            print(f"{k}: {v}")

    if 1:
        # ---- input settings ----
        input_shape = (train_X.shape[1], train_X.shape[2])

        cell_1 = data_settings['LSTM']['input']['cell_1']
        cell_2 = data_settings['LSTM']['input']['cell_2']
        cell_3 = data_settings['LSTM']['input']['cell_3']
        cell_dense = data_settings['LSTM']['input']['cell_dense']

        initial_lr = data_settings['LSTM']['input']['initial_learning_rate']
        decay_rate = data_settings['LSTM']['input']['decay_rate']
        batch_size = data_settings['LSTM']['input']['batch_size']
        patience = data_settings['LSTM']['input']['patience']
        epochs = data_settings['LSTM']['input']['epochs']
        drop = data_settings['LSTM']['input']['dropout']

        # infer output dimension automatically
        output_dim = train_y.shape[-1] if len(train_y.shape) > 1 else 1

        # ---- model ----
        inputs = Input(shape=input_shape)

        x = LSTM(cell_1, return_sequences=True)(inputs)
        x = LSTM(cell_2, return_sequences=True)(x)
        x = LSTM(cell_3, return_sequences=True)(x)
        x = Dropout(drop)(x)

        # ---- attention pooling ----
        # x.shape = (batch, cell_4)
        score = Dense(cell_3, activation="sigmoid")(x)  # match last LSTM units
        context = Multiply()([x, score])
        context = Lambda(lambda t: tf.reduce_sum(t, axis=1))(context)
        context = Dense(cell_dense, activation="relu")(context)

        # final prediction
        outputs = Dense(output_dim)(context)
        model = Model(inputs, outputs)

        # ---- optimizer ----
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=100000,
            decay_rate=decay_rate
        )

        optimizer = keras.optimizers.SGD(
            learning_rate=lr_schedule,
            clipnorm=1.0
        )

        # ---- callbacks ----
        callback = keras.callbacks.EarlyStopping(
            patience=patience,
            restore_best_weights=True
        )

        # ---- compile ----
        model.compile(loss=custom_loss, optimizer=optimizer, metrics=['mse'])

        # ---- train ----
        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,  # correct for time series
            callbacks=[callback],
            verbose=2
        )

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        # --- figure style ---
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11
        })

        fig, ax = plt.subplots(figsize=(5.5, 4))  # good for journal columns

        ax.plot(smooth(train_loss, 5), linewidth=2, label="Training")
        ax.plot(smooth(val_loss, 5), linewidth=2, label="Validation")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training History")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(data_settings['LSTM']['figure_Loss'] )

    # ========================== Predictions over single station on the list of test_ids ==========================

    test_X = input_tensor_test.reshape(( n_test,1, n_features))

    predictions = model.predict(test_X)

    output_prior_station = output_prior_station.isel(time=test_ids)
    output_post_station = output_post_station.isel(time=test_ids)
    posterior_state_station = posterior_state_station.isel(time=test_ids)
    prior_state_station = prior_state_station.isel(time=test_ids)

    obs = snow_depth_observed[test_ids]
    if 0:
        predictions = np.where(predictions < 0, 0, predictions)
        SWE_lstm = predictions[:, 0] + predictions[:, 1]
        HS_lstm = (predictions[:, 0] / 997) + (predictions[:, 1] / predictions[:, 2])
        HS_lstm = np.where(HS_lstm > 10,10, HS_lstm)


    if 1:
        for t in range(n_target):
            key = target_names_test[t]
            for var in gauss_data :
                 if key in gauss_data[var]:
                    inv_qm[:, t] = inverse_quantile_mapping(predictions[:, t], gauss_data[var], key)
                    print(f"Station ID: {y_p}, Target: {key}")

        inv_qm = np.where(inv_qm < 0, 0, inv_qm)

        SWE_lstm = inv_qm[:, 0] + inv_qm[:, 1]
        zero_mask = SWE_lstm == 0
        HS_lstm = (inv_qm[:, 0] / 997) + (inv_qm[:, 1] / inv_qm[:, 2])
        HS_lstm = np.where(zero_mask, 0, HS_lstm)
        HS_lstm = np.where(HS_lstm > 10,10, HS_lstm)

        # make a fig with 4 subplots for each target variable of inverse quantile mapping, use as reference the values from posterior_state_station and prior_state_station
        fig, axs = plt.subplots(4, 1, figsize=(12, 16))
        target_vars = ['SWE_W_mm', 'SWE_D_mm', 'RHO_D_kg_m3', 'albedo']
        for i, ax in enumerate(axs):
            target_var = target_vars[i]
            ax.plot(prior_state_station[target_var], label='Prior', color='black')
            ax.plot(posterior_state_station[target_var], label='Posterior', color='green')
            ax.plot(inv_qm[:, i], label='LSTM Prediction', color='orange')
            ax.set_title(f'{target_var} Comparison for Station ID: {y_p}')
            ax.set_xlabel('Time')
            ax.set_ylabel(target_var)
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(data_settings['LSTM']['figure_comparison_variables'].replace('.png', f'_station_{y_p}.png'))
        plt.close()


    # ========================== Compute RMSE ==========================
    # compute rmse for HS between HS_lstm and obs , ignoring nan values in obs
    valid_idx = ~np.isnan(obs) & ~np.isnan(HS_lstm)
    rmse_hs = np.sqrt(np.mean((HS_lstm[valid_idx] - obs[valid_idx]) ** 2))
    print(f"RMSE for HS on test set: {rmse_hs:.4f} m")
    #==========================Plotting comparison ==========================

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(output_prior_station['SWE_mm'], label='Output Prior SWE', color='black')
    plt.plot(output_post_station['SWE_mm'], label='Output Post SWE', color='green')
    plt.plot(SWE_lstm, label='LSTM Predicted SWE', color='orange')
    plt.title(f'SWE Comparison for Station ID: {y_p}')
    plt.xlabel('Time')
    plt.ylabel('SWE (mm)')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(output_prior_station['H_S_m'], label='Output Prior HS', color='black')
    plt.plot(output_post_station['H_S_m'], label='Output Post HS', color='green')
    plt.plot(HS_lstm, label='LSTM Predicted HS', color='orange')
    plt.plot(obs, label='Observed Snow Depth', color='red', marker='o', linestyle='None', markersize=4)
    plt.title(f'HS Comparison for Station ID: {y_p}')
    plt.xlabel('Time')
    plt.ylabel('HS (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(data_settings['LSTM']['figure_comparison'].replace('.png', f'_station_{y_p}.png'))

    # ========================== Finalize ==========================
    print("\Training complete.")
    print("Final validation loss:", history.history['val_loss'][-1])
    model.summary()
    model.save(data_settings['LSTM']['file_model'])
    return  None


def model_multiple_stations(station_ids):
    """
    Train a single LSTM for multiple stations and produce predictions, RMSE, and comparison plots per station.
    :param station_ids: list of station indices to include
    """
    # ------------------ TensorFlow config ------------------
    tf.config.threading.set_intra_op_parallelism_threads(20)
    tf.config.threading.set_inter_op_parallelism_threads(20)

    # ------------------ Load Settings and Data ------------------
    alg_settings, alg_time_start, alg_time_end, alg_domain = get_args()
    data_settings = read_file_settings(alg_settings)
    start_datetime = datetime.strptime(alg_time_start, "%Y-%m-%d %H:%M:%S")
    end_datetime = datetime.strptime(alg_time_end, "%Y-%m-%d %H:%M:%S")
    time_index = pd.date_range(start=start_datetime, end=end_datetime, freq='h')

    # Load meteorological and model datasets
    meteo_ds = xr.open_dataset(data_settings['data']['info_file']['comparison_ol_meteo']).sel(time=slice(alg_time_start, alg_time_end))
    prior_state = xr.open_dataset(data_settings['LSTM']['prior_state']).sel(time=slice(alg_time_start, alg_time_end))
    posterior_state = xr.open_dataset(data_settings['LSTM']['posterior_state']).sel(time=slice(alg_time_start, alg_time_end))
    output_prior = xr.open_dataset(data_settings['LSTM']['output_prior']).sel(time=slice(alg_time_start, alg_time_end))
    output_post = xr.open_dataset(data_settings['LSTM']['output_post']).sel(time=slice(alg_time_start, alg_time_end))

    # Load observations
    obs = pd.read_pickle(data_settings['data']['info_file']["time_series_obs"])

    # Load limits and statistics
    global lim_inf, lim_sup
    lim_sup = joblib.load(data_settings['LSTM']['lim_sup'])
    lim_inf = joblib.load(data_settings['LSTM']['lim_inf'])
    stat_all = joblib.load(data_settings['LSTM']['statistics'])
    gauss_all = joblib.load(data_settings['LSTM']['gaussianization_data'])

    # ------------------ Prepare features ------------------
    feature_names = [
        'Rain', 'AirTemperature', 'RelHumidity', 'IncRadiation',
        'T_albedo', 'T_melting',
        'SWE_W_prior', 'SWE_D_prior', 'RHO_D_prior','albedo_prior',
        'Snow_depth_obs'
    ]
    target_names = ['SWE_W_post', 'SWE_D_post', 'RHO_D_post', 'albedo_post']
    target_names_test = ['SWE_W_mm_post', 'SWE_D_mm_post', 'RHO_D_kg_m3_post', 'albedo_post']

    all_input = {}
    all_target = {}
    all_obs = {}
    all_output_prior = {}
    all_output_post = {}

    # ------------------ Loop to prepare data for all stations ------------------
    for sid in station_ids:
        # station-specific datasets
        station_meteo = meteo_ds.sel(station=sid)
        prior_state_station = prior_state.sel(point=sid)
        posterior_state_station = posterior_state.sel(point=sid)
        station_stats = stat_all[sid]
        gauss_data = gauss_all[sid]

        # Prepare observed snow depth
        df = obs[list(obs.keys())[sid]].set_index('time').resample('h').asfreq().reindex(time_index)
        df = hydrological_year(df)
        Snow_depth_obs = df['Snow_depth_cm'].fillna(method='ffill', limit=3).fillna(method='bfill', limit=3).fillna(0).values / 100
        all_obs[sid] = Snow_depth_obs

        # Prepare meteorological inputs
        m = len(meteo_ds.data_vars) + 2
        meteo = np.zeros((len(time_index), m), dtype=np.float32)
        meteo[:, 0] = station_meteo["AirTemperature"]
        meteo[:, 1] = np.clip(station_meteo["Rain"], 0, None)
        meteo[:, 2] = np.clip(station_meteo["RelHumidity"], 0, 100)
        meteo[:, 3] = station_meteo["IncRadiation"]
        meteo[:, 4] = station_meteo["AirTemperature"].rolling(time=24, min_periods=1).mean()
        meteo[:, 5] = station_meteo["AirTemperature"].rolling(time=48, min_periods=1).mean()

        # Zero substitution for prior, posterior, obs
        prior_df = prior_state_station.to_dataframe().drop(columns=['point'])
        posterior_df = posterior_state_station.to_dataframe().drop(columns=['point'])
        prior_df = zero_substitution_division(prior_df, meteo[:, 0], station_stats['prior_state_min'], station_stats['prior_state_max'])
        posterior_df = zero_substitution_division(posterior_df, meteo[:, 0], station_stats['posterior_state_min'], station_stats['posterior_state_max'])
        Snow_depth_obs[:] = zero_substitution_division(pd.DataFrame(Snow_depth_obs), meteo[:, 0], station_stats['obs_min'], station_stats['obs_max']).values.flatten()

        # Gaussianization
        qm_in = {
            'Rain': quantile_mapping(meteo[:, 1], 'Rain'),
            'AirTemperature': quantile_mapping(meteo[:, 0], 'AirTemperature'),
            'RelHumidity': quantile_mapping(meteo[:, 2], 'RelHumidity'),
            'IncRadiation': quantile_mapping(meteo[:, 3], 'IncRadiation'),
            'T_albedo': quantile_mapping(meteo[:, 4], 'T_albedo'),
            'T_melting': quantile_mapping(meteo[:, 5], 'T_melting'),
            'SWE_W_prior': quantile_mapping(prior_df['SWE_W_mm'], 'SWE_W_mm'),
            'SWE_D_prior': quantile_mapping(prior_df['SWE_D_mm'], 'SWE_D_mm'),
            'RHO_D_prior': quantile_mapping(prior_df['RHO_D_kg_m3'], 'RHO_D_kg_m3'),
            'albedo_prior': quantile_mapping(prior_df['albedo'], 'albedo'),
            'Snow_depth_obs': quantile_mapping(Snow_depth_obs, 'Snow_depth_cm')
        }
        qm_target = {
            'SWE_W_post': quantile_mapping(posterior_df['SWE_W_mm'], 'SWE_W_mm'),
            'SWE_D_post': quantile_mapping(posterior_df['SWE_D_mm'], 'SWE_D_mm'),
            'RHO_D_post': quantile_mapping(posterior_df['RHO_D_kg_m3'], 'RHO_D_kg_m3'),
            'albedo_post': quantile_mapping(posterior_df['albedo'], 'albedo')
        }

        all_input[sid] = pd.DataFrame(qm_in)
        all_target[sid] = pd.DataFrame(qm_target)
        all_output_prior[sid] = prior_state_station
        all_output_post[sid] = posterior_state_station

    # ------------------ Combine all stations into single tensors ------------------
    feature_dim = len(feature_names)
    target_dim = len(target_names)
    n_total = sum(len(all_input[sid]) for sid in station_ids)

    X_all = np.zeros((n_total, feature_dim), dtype=np.float32)
    y_all = np.zeros((n_total, target_dim), dtype=np.float32)
    station_map = []

    start = 0
    for sid in station_ids:
        n = len(all_input[sid])
        X_all[start:start+n, :] = all_input[sid][feature_names].values
        y_all[start:start+n, :] = all_target[sid][target_names].values
        station_map.extend([sid]*n)
        start += n

    X_all = X_all.reshape((n_total, 1, feature_dim))
    y_all = y_all.reshape((n_total, 1, target_dim))

    # ------------------ Split train/val/test (simple split across time) ------------------
    n_train = int(0.7*n_total)
    n_val = int(0.15*n_total)
    n_test = n_total - n_train - n_val

    train_X, val_X, test_X = X_all[:n_train], X_all[n_train:n_train+n_val], X_all[n_train+n_val:]
    train_y, val_y, test_y = y_all[:n_train], y_all[n_train:n_train+n_val], y_all[n_train+n_val:]
    station_test_map = np.array(station_map[n_train+n_val:])

    # ------------------ Build LSTM ------------------
    input_shape = (1, feature_dim)
    cell_1 = data_settings['LSTM']['input']['cell_1']
    cell_2 = data_settings['LSTM']['input']['cell_2']
    cell_3 = data_settings['LSTM']['input']['cell_3']
    cell_dense = data_settings['LSTM']['input']['cell_dense']
    drop = data_settings['LSTM']['input']['dropout']
    initial_lr = data_settings['LSTM']['input']['initial_learning_rate']
    decay_rate = data_settings['LSTM']['input']['decay_rate']
    batch_size = data_settings['LSTM']['input']['batch_size']
    patience = data_settings['LSTM']['input']['patience']
    epochs = data_settings['LSTM']['input']['epochs']

    inputs = Input(shape=input_shape)
    x = LSTM(cell_1, return_sequences=True)(inputs)
    x = LSTM(cell_2, return_sequences=True)(x)
    x = LSTM(cell_3, return_sequences=True)(x)
    x = Dropout(drop)(x)
    outputs = Dense(target_dim)(x[:, -1, :])  # last timestep
    model = Model(inputs, outputs)

    optimizer = keras.optimizers.SGD(learning_rate=initial_lr)
    model.compile(loss=custom_loss, optimizer=optimizer, metrics=['mse'])

    callback = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    history = model.fit(train_X, train_y, validation_data=(val_X, val_y),
                        epochs=epochs, batch_size=batch_size, callbacks=[callback], shuffle=False, verbose=2)

    # ------------------ Predictions and RMSE per station ------------------
    test_preds = model.predict(test_X)
    inv_qm_all = {}
    for sid in station_ids:
        idx = np.where(station_test_map == sid)[0]
        inv_qm = np.zeros((len(idx), target_dim))
        gauss_data = gauss_all[sid]
        for t, key in enumerate(target_names_test):
            for var in gauss_data:
                if key in gauss_data[var]:
                    inv_qm[:, t] = inverse_quantile_mapping(test_preds[idx, t], gauss_data[var], key)
        inv_qm_all[sid] = inv_qm

        # compute SWE/HS
        SWE_lstm = inv_qm[:, 0] + inv_qm[:, 1]
        zero_mask = SWE_lstm == 0
        HS_lstm = (inv_qm[:, 0]/997) + (inv_qm[:, 1]/inv_qm[:, 2])
        HS_lstm = np.where(zero_mask, 0, HS_lstm)

        obs_station = all_obs[sid][station_test_map==sid]
        valid_idx = ~np.isnan(obs_station) & ~np.isnan(HS_lstm)
        rmse_hs = np.sqrt(np.mean((HS_lstm[valid_idx] - obs_station[valid_idx])**2))
        print(f"Station {sid} → RMSE HS: {rmse_hs:.4f} m")

        # plot comparison
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        axs[0].plot(all_output_prior[sid]['SWE_mm'].values[station_test_map==sid], label='Prior SWE')
        axs[0].plot(all_output_post[sid]['SWE_mm'].values[station_test_map==sid], label='Posterior SWE')
        axs[0].plot(SWE_lstm, label='LSTM SWE')
        axs[0].set_title(f'SWE Comparison Station {sid}')
        axs[0].legend(); axs[0].grid(True)

        axs[1].plot(all_output_prior[sid]['H_S_m'].values[station_test_map==sid], label='Prior HS')
        axs[1].plot(all_output_post[sid]['H_S_m'].values[station_test_map==sid], label='Posterior HS')
        axs[1].plot(HS_lstm, label='LSTM HS')
        axs[1].plot(obs_station, label='Observed', marker='o', linestyle='None')
        axs[1].set_title(f'HS Comparison Station {sid}')
        axs[1].legend(); axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(data_settings['LSTM']['figure_comparison'].replace('.png', f'_station_{sid}.png'))
        plt.close()

    # ------------------ Save model ------------------
    model.save(data_settings['LSTM']['file_model'])
    print("✅ Multi-station model training complete.")

    return model, inv_qm_all

#=====================================================
# ====================================================




if __name__ == "__main__" :
    model_single_station()
