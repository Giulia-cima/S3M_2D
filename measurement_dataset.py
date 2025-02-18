
"""
These are the info i want to retrieve from the file.
a2dOss
a1sTempi_num
a1dQuota
a1dLat
a1dLon
"""

import scipy.io
import os
import numpy as np
import pandas as pd
import pickle

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
    a2dOss = f['a2dOss']
    time.append(f['a1sTempi_num'].flatten())
    lat =f['a1dLat']
    lon =f['a1dLon']
    Quota = f['a1dQuota']

    # Save each row of a2dOss as an array
    for i, row in enumerate(a2dOss):
        # Pad or truncate the row to match the shape (3000,)
        padded_row =  np.full(2260, np.nan)
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
    # drop nan values
    df.dropna(inplace=True)
    # Replace data < 0 with np.nan
    df['Snow_depth_cm'] = df['Snow_depth_cm'].mask(df['Snow_depth_cm'] < 0, np.nan)

    # Add the time column
    df['time'] = pd.to_datetime(np.concatenate(time) - 719529, unit='D')
    # drop nan values
    df.dropna(inplace=True)

    df = df.resample('D', on='time').sum().reset_index()

    # Add the height
    df['a1dQuota'] = np.repeat(Quota[i], len(df))

    # Print or save the DataFrame as needed
    print(df.head())

# Use the coordinates as the key and store the DataFrame in the dictionary
    key = (float(lat[i]), float(lon[i]))
    df_dict[key] = df

# save df_dict
# save df_dict
with open('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/df_dict.pkl', 'wb') as f:
    pickle.dump(df_dict, f)

