# Training LSTM
import os
import pickle
import joblib
import keras
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from numpy import concatenate
from keras.models import Sequential
from keras.layers import LSTM, Dense
from lib_data_io_json import read_file_settings
from lib_utilis_data_proc import get_args


# ========================== Custom Functions ==========================
def compute_relative_humidity(t, q, p):
    """
    Compute RH using simplified and numerically stable formula.
    """
    Tc = t - 273.15  # Convert K to °C
    e = q * p / (0.622 + 0.378 * q)
    es = 610.94 * np.exp((17.625 * Tc) / (243.04 + Tc))
    RH = (e / es * 100).clip(max=100)
    RH.name = 'RH'
    RH.attrs['long_name'] = 'Relative Humidity [%]'
    return RH

def zero_substitution_division(df, min_values, max_values):
    for i in range(6, df.shape[1]):
        df.iloc[:, i] = np.where(df.iloc[:, i] == 0, -df.iloc[:, 4], df.iloc[:, i])
        min_value = min_values[i]
        max_value = max_values[i]
        df.iloc[:, i] = np.where(df.iloc[:, i] < 0, -(df.iloc[:, i] / min_value) * max_value, df.iloc[:, i])
    return df


def custom_loss(y_true, y_pred):
    global lim_inf, lim_sup
    reg_loss = ((y_pred[:, 0] - lim_sup[12]) * (y_pred[:, 0] - lim_inf[12]) *
                (y_pred[:, 1] - lim_sup[13]) * (y_pred[:, 1] - lim_inf[13]) *
                (y_pred[:, 2] - lim_sup[14]) * (y_pred[:, 2] - lim_inf[14]) *
                (y_pred[:, 3] - lim_sup[15]) * (y_pred[:, 3] - lim_inf[15]))
    epsilon = 1e-7
    reg_term = 1 / (reg_loss + epsilon)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return tf.sqrt(mse) + reg_term



# ========================== Load Settings and Data ==========================
alg_settings, alg_time_start, alg_time_end, alg_domain = get_args()
data_settings = read_file_settings(alg_settings)

obs = pd.read_pickle(data_settings['data']['info_file']['time_series_csnow'])
df_dict = pd.read_pickle(data_settings['data']['info_file']['observation_vda'])
dem = data_settings['data']['info_file']['dem']
input_path = data_settings['data']['info_file']['input_path']
w = pd.read_pickle(data_settings['data']['info_file']['weights'])
dem_mask = pd.read_pickle(data_settings['data']['info_file']['dem_mask'])

with open(data_settings['LSTM']['input']['file_result_enkf'], 'rb') as file:
    data = pickle.load(file)

mean_values = joblib.load(data_settings['LSTM']['input']['mean_val'])
std_values = joblib.load(data_settings['LSTM']['input']['std_val'])
min_val = joblib.load(data_settings['LSTM']['input']['min_val'])
max_val = joblib.load(data_settings['LSTM']['input']['max_val'])
lim_sup = joblib.load(data_settings['LSTM']['input']['lim_sup'])
lim_inf = joblib.load(data_settings['LSTM']['input']['lim_inf'])

# ========================== Prepare Forcing Data ==========================
downscaled_ds = xr.open_dataset(input_path).sel(time=slice(alg_time_start, alg_time_end))
Time = pd.to_datetime(downscaled_ds.time.values)
nt = len(Time)
dem_da = xr.open_rasterio(dem, engine='rasterio').sel(band=1).drop_vars('band')
ny, nx = dem_da.shape

meteo = np.zeros((nt, ny, nx, 6), dtype=np.float32)

var_tags = data_settings['data']['info_file']['tags']
var_list = [var_tags['temperature_tag'], var_tags['prc_tag'], var_tags['rad_tag'], var_tags['specific_humidity_tag']]
multi_index_coarse = pd.MultiIndex.from_product([range(ny), range(nx)], names=["y", "x"])

mapped_results = {}

for var_name in var_list:
    var = downscaled_ds[var_name].values
    mapped = np.dot(w, var)
    mapped_results[var_name] = xr.DataArray(mapped,
                                            coords=[multi_index_coarse, Time],
                                            dims=['point', 'time'])

ds = xr.Dataset(mapped_results).unstack("point")

air_temp_celsius = ds[var_tags['temperature_tag']] - 273.15
meteo[:, :, :, 0] = air_temp_celsius
meteo[:, :, :, 1] = ds[var_tags['prc_tag']]
meteo[:, :, :, 2] = compute_relative_humidity(ds[var_tags['temperature_tag']], ds[var_tags['specific_humidity_tag']],
                                              ds[var_tags['prc_tag']])
meteo[:, :, :, 3] = ds[var_tags['rad_tag']]
meteo[:, :, :, 4] = uniform_filter1d(air_temp_celsius, size=24, axis=0)
meteo[:, :, :, 5] = uniform_filter1d(air_temp_celsius, size=240, axis=0)

# ========================== Prepare Data for LSTM ==========================
# Load prior and posterior state/output matrices
state_prior = xr.open_dataset(os.path.join(data_settings['data']['output_file']['folder_name'], "state_prior_data.nc"))
Target = xr.open_dataset(os.path.join(data_settings['data']['output_file']['folder_name'], "state_posterior_data.nc"))
output_prior = xr.open_dataset(os.path.join(data_settings['data']['output_file']['folder_name'], "output_prior_data.nc"))
output_posterior = xr.open_dataset(os.path.join(data_settings['data']['output_file']['folder_name'], "output_posterior_data.nc"))


# Extract variables
X_prior = np.array(data[8])
swe_prior = state_prior["SWE_W_mm"].values
snow_depth_prior = output_prior["H_S_m"].values
swe_enkf = output_posterior["SWE_W_mm"].values
snow_depth_enkf = output_posterior["H_S_m"].values
snow_depth_obs = obs['snd_interp'].values


# Convert Target to DataFrame
Target_df = Target.to_dataframe().reset_index()
# Prepare inputs
inputs = concatenate((meteo, X_prior, swe_prior[:, np.newaxis], snow_depth_prior[:, np.newaxis],
                      snow_depth_obs[:, np.newaxis]), axis=1)

input_columns = ['Precip_mm', 'Rad_wm2', 'Temp_C', 'RH_pct', 'T10D_C', 'T1D_C',
                 'SWE_W_mm', 'SWE_D_mm', 'RHO_D_kgm3', 'albedo',
                 'SWE_prior_mm', 'SnowDepth_prio_m', 'SnowDepth_obs_m']

values = pd.DataFrame(inputs, columns=input_columns)
values = pd.concat([values, Target_df], axis=1).dropna()

# Align other variables
idx = values.index
X_prior = pd.DataFrame(X_prior).iloc[idx]
swe_prior = pd.DataFrame(swe_prior.flatten()).iloc[idx]
snow_depth_prior = pd.DataFrame(snow_depth_prior.flatten()).iloc[idx]
swe_enkf = pd.DataFrame(swe_enkf.flatten()).iloc[idx]
snow_depth_enkf = pd.DataFrame(snow_depth_enkf.flatten()).iloc[idx]

# ========================== Normalization ==========================
values = zero_substitution_division(values.copy(), min_val, max_val)
scaled = (values - mean_values.values) / std_values.values
X = scaled.iloc[:, :-4].values.astype('float32')
y = scaled.iloc[:, -4:].values.astype('float32')
X = X.reshape((X.shape[0], 1, X.shape[1]))

# ========================== Train/Val Split ==========================
dates = pd.date_range(start=alg_time_start, periods=len(X))
train_mask = (dates >= data_settings['LSTM']['settings']['date_start_training']) & \
             (dates <= data_settings['LSTM']['settings']['date_end_training'])
val_mask = (dates >= data_settings['LSTM']['settings']['date_start_validation']) & \
           (dates <= data_settings['LSTM']['settings']['date_end_validation'])

train_X, train_y = X[train_mask], y[train_mask]
val_X, val_y = X[val_mask], y[val_mask]

# ========================== LSTM Model ==========================
model = Sequential()
model.add(LSTM(data_settings['LSTM']['input']['cell_1'], input_shape=(train_X.shape[1], train_X.shape[2]),
               return_sequences=True))
model.add(LSTM(data_settings['LSTM']['input']['cell_2']))
model.add(Dense(data_settings['LSTM']['input']['cell_dense']))

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=data_settings['LSTM']['input']['initial_learning_rate'],
    decay_steps=100000,
    decay_rate=data_settings['LSTM']['input']['decay_rate'])

optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, clipnorm=0.5)
model.compile(loss=custom_loss, optimizer=optimizer)

callback = keras.callbacks.EarlyStopping(patience=data_settings['LSTM']['input']['patience'],
                                         restore_best_weights=True)

history = model.fit(train_X, train_y, epochs=data_settings['LSTM']['input']['epochs'],
                    batch_size=data_settings['LSTM']['input']['batch_size'],
                    validation_data=(val_X, val_y), verbose=2, shuffle=False, callbacks=[callback])

# ========================== Plot Loss ==========================
plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(data_settings['LSTM']['output']['figure_Loss'])

# ========================== Predictions ==========================
yhat = model.predict(val_X)
val_X_flat = val_X.reshape((val_X.shape[0], val_X.shape[2]))
inv_yhat = (concatenate((val_X_flat, yhat), axis=1) * std_values.values) + mean_values.values
inv_yhat[inv_yhat < 0] = 0
inv_yhat = inv_yhat[:, -4:]

swe_lstm = inv_yhat[:, 0] + inv_yhat[:, 1]
snow_depth_lstm = (inv_yhat[:, 0] / 997) + (inv_yhat[:, 1] / inv_yhat[:, 2])
snow_depth_lstm[swe_lstm == 0] = 0
snow_depth_lstm = np.clip(snow_depth_lstm, 0, 10)

# ========================== Finalize ==========================
print("\Training complete.")
print("Final validation loss:", history.history['val_loss'][-1])
model.summary()
model.save(data_settings['LSTM']['output']['file_model'])