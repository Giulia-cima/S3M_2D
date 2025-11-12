
"""
These are the info i want to retrieve from the file.
a2dOss
a1sTempi_num
a1dQuota
a1dLat
a1dLon
"""
import pandas
import scipy.io
import os
import numpy as np
import pandas as pd
import pickle
import rioxarray as rxr
import matplotlib.pyplot as plt
"""
os.chdir('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/DatiPuntualiDaDB_N/')
# list of files
lista_file = sorted(os.listdir('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/DatiPuntualiDaDB_N/'))
listing = len(lista_file)

# Initialize an empty list to store the measurements
measurements = np.empty((72, 52, 2260))
time=[]
# Loop through each file and retrieve the measurements
j=0
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
    # Add the time column
    df['time'] = pd.to_datetime(np.concatenate(time) - 719529, unit='D')
    # drop nan values
    df.dropna(inplace=True)

    #df = df.resample('D', on='time').sum().reset_index()

    # Add the height
    df['a1dQuota'] = np.repeat(Quota[i], len(df))



    # Print or save the DataFrame as needed
    print(df.head())

# Use the coordinates as the key and store the DataFrame in the dictionary
    key = (round(float(lat[i]), 3), round(float(lon[i]), 3))
    df_dict[key] = df

# Read the list of valid station IDs
station_list = pd.read_csv('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/STATION_ID_VDA.csv')

# Replace commas with periods in the 'lat' and 'lon' columns
station_list['lat'] = station_list['lat'].str.replace(',', '.').astype(float)
station_list['lon'] = station_list['lon'].str.replace(',', '.').astype(float)

# crea una mask per tenere solo le stazioni presenti in station_list


# Filter df_dict to keep only the stations present in the station_list
#valid_keys = set(station_list.apply(lambda row: (round(float(row['lat']), 3), round(float(row['lon']), 3)), axis=1))
# not round
valid_keys = set(station_list.apply(lambda row: (float(row['lat']), float(row['lon'])), axis=1))

# Create a new dictionary with only valid stations
filtered_df_dict = {key: df for key, df in df_dict.items() if key in valid_keys}

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
    plt.savefig(f'/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/{key[0]}_{key[1]}.png')
    plt.close()
    
    
    
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
