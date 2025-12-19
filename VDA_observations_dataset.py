import pandas
import scipy.io
import os
import numpy as np
import pandas as pd
import pickle
import rioxarray as rxr
import matplotlib.pyplot as plt


def categorize_snow_depth(max_snow_depth,high, low ):
    if max_snow_depth >= high:
        return 'High Snow'
    elif max_snow_depth <= low:
        return 'Low Snow'
    else:
        return 'Medium Snow'
"""
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

os.chdir('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/DatiPuntualiDaDB_N/')
# list of files
lista_file = sorted(os.listdir('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/DatiPuntualiDaDB_N/'))
listing = len(lista_file)
measurements = np.empty((72, 52, 2260))
time = []
# Loop through each file and retrieve the measurements
j = 0

for fi_le in range(listing):
    f = scipy.io.loadmat(lista_file[fi_le])

    # Retrieve the specified information from the file
    a2dOss = f['a2dOss'].astype(float)  # Convert a2dOss to float
    code = f['a2iCalccode']
    time.append(f['a1sTempi_num'].flatten())
    lat =f['a1dLat']
    lon =f['a1dLon']
    Quota = f['a1dQuota']

    # where code is >4 set the measurement as nan
    a2dOss[code > 4] = -9998

    # Save each row of a2dOss as an array
    for i, row in enumerate(a2dOss):
        # Pad or truncate the row to match the shape (3000,)
        padded_row = np.full(2260, np.nan)
        padded_row[:min(len(row), 2260)] = row[:min(len(row), 2260)]
        measurements[j, i, :] = padded_row
    j+=1

# Initialize an empty dictionary to store the DataFrames
df_dict = {}

# Loop through each measurement
for i in range(measurements.shape[1]):
    n = measurements.shape[0] * measurements.shape[2]
    x = np.zeros((n,))  # Initialize x as a 1D array

    for j in range(measurements.shape[0]):
        start_idx = j * measurements.shape[2]
        end_idx = start_idx + measurements.shape[2]
        x[start_idx:end_idx] = measurements[j, i, :]
        # Assign the slice of measurements to x
    data = np.array(x)
    df = pd.DataFrame(data, columns=['Snow_depth_cm'])
    df.dropna(inplace=True)
    # Replace data < 0 with np.nan
    df['Snow_depth_cm'] = df['Snow_depth_cm'].mask(df['Snow_depth_cm'] < 0, np.nan)
    df['Snow_depth_cm'] = df['Snow_depth_cm'].mask(df['Snow_depth_cm'] >9999, np.nan)
    raw_time = np.concatenate(time) - 719529  # convert MATLAB → days since epoch
    raw_time = np.round(raw_time, 12)  # remove floating point noise
    df['time'] = pd.to_datetime(raw_time, unit='D')
    df['time'] = df['time'].dt.round('T')  # round to nearest minute
    df.dropna(inplace=True)
    df['a1dQuota'] = np.repeat(Quota[i], len(df))
    # Print or save the DataFrame as needed
    print(df.head())
# Use the coordinates as the key and store the DataFrame in the dictionary
    key = (round(float(lat[i]), 6), round(float(lon[i]), 6))
    df_dict[key] = df

# Read the list of valid station IDs
station_list = pd.read_csv('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/STATION_ID_VDA.csv')
station_list['lat'] = station_list['lat'].str.replace(',', '.').astype(float)
station_list['lon'] = station_list['lon'].str.replace(',', '.').astype(float)

# Filter df_dict to keep only the stations present in the station_list
valid_keys = set(station_list.apply(lambda row: (round(float(row['lat']), 6), round(float(row['lon']), 6)), axis=1))
# Create a new dictionary with only valid stations
filtered_df_dict = {key: df for key, df in df_dict.items() if key in valid_keys}

max_hs_stations ={}
min_hs_stations = {}

# considering all the stations, classify the hydrological year from  first October to  30 September  for key, df in filtered_df_dict.items():
for key, df in filtered_df_dict.items():
    df['HydroYear'] = np.where(df['time'].dt.month >= 10, df['time'].dt.year + 1, df['time'].dt.year)
    # add the max and min snow depth for each station in the dict using the key as the station identifier
    max_hs_stations[key] = df['Snow_depth_cm'].max()
    min_hs_stations[key] = df['Snow_depth_cm'].min()


# SAVE THE MAX AND MIN SNOW DEPTH FOR EACH STATION AS A CSV FILE
max_hs_df = pd.DataFrame(list(max_hs_stations.items()), columns=['Station', 'Max_Snow_depth_cm'])
min_hs_df = pd.DataFrame(list(min_hs_stations.items()), columns=['Station', 'Min_Snow_depth_cm'])
#MERGE THE TWO DATAFRAMES ON THE 'Station' COLUMN
hs_stats_df = pd.merge(max_hs_df, min_hs_df, on='Station')
hs_stats_df.to_csv(f'/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/hs_stats_stations.csv', index=False)





# now compute the mean snow depth for each hydrological year and each station for key, df in filtered_df_dict.items():
for key, df in filtered_df_dict.items():
    hydro_year_stats = df.groupby('HydroYear')['Snow_depth_cm'].agg(['mean']).reset_index()
    hydro_year_stats.rename(columns={'mean': 'Mean_Snow_depth_cm'}, inplace=True)

# COMPUTE THE 75 AND THE 25 PERCENTILE BASED ON THE MAX SNOW DEPTH FOR EACH HYDROLOGICAL
high_Percentile= hydro_year_stats['Mean_Snow_depth_cm'].quantile(0.75)
low_Percentile = hydro_year_stats['Mean_Snow_depth_cm'].quantile(0.25)

# SUBDIVIDE YEARS IN 3 CATEGORIES BASED ON THE 75 AND 25 PERCENTILE OF MAX SNOW DEPTH. ABOVE 75TH PERCENTILE IS "HIGH SNOW", BELOW 25TH PERCENTILE IS "LOW SNOW", IN BETWEEN IS "MEDIUM SNOW"
hydro_year_stats['Snow_Category'] = hydro_year_stats['Mean_Snow_depth_cm'].apply(lambda x: categorize_snow_depth(x, high_Percentile, low_Percentile))

# save hydro_year_stats as a csv file
hydro_year_stats.to_csv(f'/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/hydro_year_stats.csv', index=False)

# Save the filtered df_dict
with open('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/dict.pkl', 'wb') as f:
    pickle.dump(filtered_df_dict, f)

# Plot snow depth and SWE profile vs time for each station
for key, df in filtered_df_dict.items():
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], df['Snow_depth_cm'], label='Snow Depth (cm)')
    # Assuming SWE data is available in the DataFrame as 'SWE'
    if 'SWE' in df.columns:
        plt.plot(df['time'], df['SWE'], label='SWE')
    plt.xlabel('Time')
    plt.ylabel('Measurement')
    plt.title(f'Station {key}')
    plt.legend()
    # Save the figure with lat and lon as the filename
    plt.savefig(f'/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/plot_meteo/{key[0]}_{key[1]}.png')
    plt.close()

    
"""
"""
dem = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA.tif"
# open tif
dem = rxr.open_rasterio(dem)
lat_dem = dem.y.values
lon_dem = dem.x.values
# read station list
station_list = pd.read_csv('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/STATION_ID_VDA.csv')

# Replace commas with periods in the 'lat' and 'lon' columns
station_list['lat'] = station_list['lat'].str.replace(',', '.').astype(float)
station_list['lon'] = station_list['lon'].str.replace(',', '.').astype(float)

# crea una mask per tenere solo le stazioni presenti in station_list, tenendo conto che la matrice meteo ha shape (nt, ny, nx, 6)
obs_mask = []
for index, row in station_list.iterrows():
    lat_station = row['lat']
    lon_station = row['lon']
    # round to 6 decimal places
    lat_station = round(lat_station, 6)
    lon_station = round(lon_station, 6)
    # trova l'indice della latitudine più vicina
    ilat = np.abs(lat_dem - lat_station).argmin().item()
    ilon = np.abs(lon_dem - lon_station).argmin().item()
    obs_mask.append((ilat, ilon))


# print obs_mask
print(obs_mask)
# save obs_mask
with open('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/obs_mask.pkl', 'wb') as f:
    pickle.dump(obs_mask, f)


pandas.read_pickle('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/obs_mask.pkl')
"""

"""
# create an obs_mask.pkl file with the indices of the stations in  vda  STATION LIST
# read station list
station_list = pd.read_csv('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/STATION_ID_VDA.csv')

# Replace commas with periods in the 'lat' and 'lon' columns
station_list['lat'] = station_list['lat'].str.replace(',', '.').astype(float)
station_list['lon'] = station_list['lon'].str.replace(',', '.').astype(float)

lat_dem = station_list['lat'].values
lon_dem = station_list['lon'].values

# crea una mask per tenere solo le stazioni presenti in station_list, tenendo conto che la matrice meteo ha shape (nt, ny, nx, 6)
obs_mask = []
for index, row in station_list.iterrows():
    lat_station = row['lat']
    lon_station = row['lon']
    # round to 6 decimal places
    lat_station = round(lat_station, 6)
    lon_station = round(lon_station, 6)
    # trova l'indice della latitudine più vicina
    ilat = np.abs(lat_dem - lat_station).argmin().item()
    ilon = np.abs(lon_dem - lon_station).argmin().item()
    obs_mask.append((ilat, ilon))

# print obs_mask
print(obs_mask)
# save obs_mask
with open('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/obs_mask_vda.pkl', 'wb') as f:
    pickle.dump(obs_mask, f)

"""

pkl_file= "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/quantile_mapping/state_limits.pkl"
# open pkl file
with open(pkl_file, 'rb') as f:
    state_limits = pickle.load(f)
print(state_limits)

# modify state_limits
state_limits[1][1] = 800
state_limits[1][0] = 800
state_limits[0][2] =67.4
print(state_limits)

# save state_limits
with open(pkl_file, 'wb') as f:
    pickle.dump(state_limits, f)




