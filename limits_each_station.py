import pandas
import scipy.io
import os
import numpy as np
import pandas as pd
import pickle
import xarray as xr
import matplotlib.pyplot as plt

file_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/input_meteo/S3M_MeteoData_20071001_20240930.nc"
meteo_data = xr.open_dataset(file_path)
Latitudes_1d = np.unique(meteo_data.lat)
Longitudes_1d = np.unique(meteo_data.lon)

AirTemperature_3D = np.repeat(
    meteo_data['AirTemperature'].values[:, :, np.newaxis],  # shape (time, 43, 1)
    43, axis=2
)  # now shape is (8760, 43, 43)

Rain_3D = np.repeat(
    meteo_data['Rain'].values[:, :, np.newaxis],  # shape (time, 43, 1)
    43, axis=2
)  # now shape is (8760, 43, 43)

IncRadiation_3D = np.repeat(
    meteo_data['IncRadiation'].values[:, :, np.newaxis],  # shape (time, 43, 1)
    43, axis=2
)  # now shape is (8760, 43, 43)
RelHumidity_3D = np.repeat(
    meteo_data['RelHumidity'].values[:, :, np.newaxis],  # shape (time, 43, 1)
    43, axis=2
)  # now shape is (8760, 43, 43)

# Build the new dataset
meteo_ds = xr.Dataset(
    coords={
        "time": meteo_data.time,
        "Latitude": Latitudes_1d,
        "Longitude": Longitudes_1d
    },
    data_vars={
        "AirTemperature": (("time", "Latitude", "Longitude"), AirTemperature_3D),
        "Rain": (("time", "Latitude", "Longitude"), Rain_3D),
        "IncRadiation": (("time", "Latitude", "Longitude"), IncRadiation_3D),
        "RelHumidity": (("time", "Latitude", "Longitude"), RelHumidity_3D)
    })


file_csv ="/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/hs_stats_stations.csv"
hs_stats_stations = pd.read_csv(file_csv)
stations = hs_stats_stations['Station'].unique()

max_temp = {}
min_temp = {}
max_rain = {}
min_rain = {}
max_radiation = {}
min_radiation = {}
max_humidity = {}
min_humidity = {}


for station in stations:
    # CONVERT station from tuple to lat and lon
    lat, lon = eval(station)
    station_data = meteo_ds.sel(Latitude=lat, Longitude=lon, method='nearest')
    station_data = station_data.to_dataframe()
    max_temp[station] = station_data['AirTemperature'].max()
    min_temp[station] = station_data['AirTemperature'].min()
    max_rain[station] = station_data['Rain'].max()
    min_rain[station] = station_data['Rain'].min()
    max_radiation[station] = station_data['IncRadiation'].max()
    min_radiation[station] = station_data['IncRadiation'].min()
    max_humidity[station] = station_data['RelHumidity'].max()
    min_humidity[station] = station_data['RelHumidity'].min()

# merge all dictionaries into a single dataframe based on station
limits_df = pd.DataFrame({
    'Max_Temperature': pd.Series(max_temp),
    'Min_Temperature': pd.Series(min_temp),
    'Max_Rain': pd.Series(max_rain),
    'Min_Rain': pd.Series(min_rain),
    'Max_Radiation': pd.Series(max_radiation),
    'Min_Radiation': pd.Series(min_radiation),
    'Max_Humidity': pd.Series(max_humidity),
    'Min_Humidity': pd.Series(min_humidity),
})

# merge with hs_stats_stations
limits_df = limits_df.merge(hs_stats_stations, left_index=True, right_on='Station')
limits_df.to_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/limits_each_station.csv", index=False)


