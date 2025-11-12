
import os
import glob
import rioxarray

import pandas
import xarray as xr
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def merge_tiles(folders, output_folder):
    """
    Merge NetCDF files from multiple folders based on matching filenames.

    Parameters:
        folders (list): List of folder paths containing NetCDF files.
        output_folder (str): Path to the folder where merged files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_files = set()  # Track processed files to avoid duplicates

    # Iterate through all folders
    for folder in folders:
        files = glob.glob(os.path.join(folder, '*.nc'))
        # sort files by name of the file
        files.sort(key=lambda x: os.path.basename(x))
        for file in files:
            file_name = os.path.basename(file)

            # Skip if the file has already been processed
            if file_name in processed_files:
                continue

            # Generate matching file names across all folders
            matching_files = [os.path.join(f, file_name) for f in folders if
                              os.path.exists(os.path.join(f, file_name))]

            # Merge files if more than one exists
            if len(matching_files) > 1:
                try:
                    ds = xr.open_mfdataset(matching_files)
                    date_str = file_name.split('_')[1].split('.')[0]
                    ds['time'] = datetime.strptime(date_str, '%Y%m%d')
                    output_file = os.path.join(output_folder, file_name)
                    ds.to_netcdf(output_file)
                    ds.close()
                except Exception as e:
                    logging.error(f"Error merging files for {file_name}: {e}")
            else:
                logging.info(f"No files to merge for {file_name}")

            # Mark the file as processed
            processed_files.add(file_name)

def main():

    # Example usage
    folders = [
        '/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/y036x196/',
        '/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/y037x196/',
        '/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/y036x197/',
        '/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/y037x197/',
        '/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/y036x198/',
        '/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/y037x198/'
    ]
    output_folder = '/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/NETCDF/merged/'
    merge_tiles(folders, output_folder)


    merged_files = glob.glob(os.path.join(output_folder, '*.nc'))
    years = ['2015', '2016', '2017','2018', '2019', '2020', '2021', '2022']
    folder_yearly = '/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/NETCDF/merged_yearly/'

    # Create the output folder if it does not exist
    os.makedirs(folder_yearly, exist_ok=True)

    # Group files by year
    files_dict = {year: [] for year in years}
    for file in merged_files:
        # Extract the year from the filename assuming the format is snd_YYYYMMDD.nc
        year = os.path.basename(file).split('_')[1][:4]
        if year in years:
            files_dict[year].append(file)

    # Merge files for each year
    for year, files in files_dict.items():
        if not files:  # Skip if no files exist for the year
            logging.info(f"No files to merge for year {year}")
            continue
        try:
            # Sort files by name to ensure consistent time order
            files.sort(key=lambda x: os.path.basename(x))
            ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', coords='all')
            output_file = os.path.join(folder_yearly, f'snd_{year}_merged.nc')
            ds.to_netcdf(output_file)
            ds.close()
            logging.info(f"Merged files for year {year} into {output_file}")
        except Exception as e:
            logging.error(f"Error merging files for year {year}: {e}")


# -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------


def error_definition():
    """
  take the same station from the file combined_ds and the df_dict and calculate the error between c snow and the observations

"""
    input_folder = '/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/NETCDF/merged_yearly/'
    df_dict = pd.read_pickle("/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/dict.pkl")
    filtered_files = glob.glob(os.path.join(input_folder, '*_filtered.nc'))
    # Step 3: Extract time series for each valid point
    filtered_files.sort(key=lambda x: os.path.basename(x))  # Sort files by name

    # Step 3: Extract time series for each valid point
    combined_ds = xr.open_mfdataset(filtered_files, combine='nested', concat_dim='time', coords='all')
    # drop the duplicate coordinates
    combined_ds = combined_ds.drop_duplicates(dim=['lon', 'lat'])
    # sort the dataset by latitude
    # Ensure the dataset is sorted by 'lon' and 'lat'
    combined_ds = combined_ds.sortby(['lon', 'lat'])
    station_id =pandas.read_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/STATION_ID_VDA.csv")
    lat = station_id['lat'].values
    output_folder = '/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/observations/'
    error = []

    for i in range(len(lat)):
        # Get the station name
        station_name = station_id['name'].values[i]

        # Get the lon and lat of the station, ensuring they are floats
        lon = round(float(str(station_id['lon'].values[i]).replace(',', '.')), 3)
        lat = round(float(str(station_id['lat'].values[i]).replace(',', '.')), 3)

        # Get the data from the combined_ds
        ds = combined_ds.sel(lon=lon, lat=lat, method='nearest')

        # Get the data from the df_dict
        df = df_dict[(lat, lon)]

        # Get the time series from the ds
        ds = ds['snd'].to_dataframe().reset_index()

        # Get the time series from the df
        df = df.set_index('time')
        ds = ds.set_index('time')

        # Convert the time series of df in meters from cm
        df['Snow_depth_m'] = df['Snow_depth_cm'] / 100

        # Ensure the indices of df and ds are unique
        df = df[~df.index.duplicated(keep='first')]
        ds = ds[~ds.index.duplicated(keep='first')]

        # Align the time indices of both datasets
        common_time_index = df.index.union(ds.index)  # Combine time indices

        df = df.reindex(common_time_index)
        ds = ds.reindex(common_time_index)

        # Compute the difference only where ds['snd'] is not NaN
        valid_mask = ~ds['snd'].isna()
        difference = df['Snow_depth_m'][valid_mask] - ds['snd'][valid_mask]
        """ 
        # plot a comparison between the two time series
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Snow_depth_m'], 'o', label='Observations (cm)')
        plt.plot(ds.index, ds['snd'], 'o', label='Combined_ds (cm)')
        plt.xlabel('Time')
        plt.ylabel('Snow Depth (m)')
        plt.title(f'Comparison of Snow Depth at ({lon}, {lat})')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f'comparison_{station_name}.png'))
        plt.close()
        """
        error.append(difference)

        # Handle empty lists in the error list
    mean_error = [np.nanmean(e) if not e.empty else np.nan for e in error]
    std_error = [np.nanstd(e) if not e.empty else np.nan for e in error]

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Station': [station_id['name'].values[i] for i in range(len(mean_error))],
        'Mean Error': mean_error,
        'Standard Deviation': std_error
    })

    # Concatenate all differences into a single series
    all_differences = pd.concat(error, axis=0)

    # Compute the overall mean and standard deviation
    overall_mean = all_differences.mean()
    overall_std = all_differences.std()

    # Print the overall results in a txt file
    with open(os.path.join(output_folder, 'error_results.txt'), 'w') as f:
        f.write(f"Overall Mean Error: {overall_mean}\n")
        f.write(f"Overall Standard Deviation: {overall_std}\n")
        f.write("\nDetailed Results:\n")

# -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------


def check_duplicates_in_coords(ds):
    for coord in ['x', 'y', 'time']:
        if coord in ds.coords:
            values = ds.coords[coord].values
            if len(values) != len(np.unique(values)):
                print(f"Duplicates found in coordinate '{coord}'")
            else:
                print(f"No duplicates in coordinate '{coord}'")
def csnow_series():

    sampled_points_path = '/home/idrologia/share/PhD_GiuliaBlandini_dati//FILES/old/sample_points_kmeans.nc'
    input_folder = '/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/NETCDF/merged_yearly/'
    output_folder = '/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/observations/'

     # create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Read sampled points and extract valid points
    sampled_points = xr.open_dataset(sampled_points_path)
    valid_points = set(zip(sampled_points['x'].values, sampled_points['y'].values))
    valid_points = sorted(valid_points, key=lambda point: point[1])  # Sort by latitude
    print("Valid points extracted.")

    # Step 2: Filter NetCDF files by valid points
    files = glob.glob(os.path.join(input_folder, '*.nc'))
    files.sort(key=lambda x: os.path.basename(x))  # Sort files by name

    filtered_files = []
    for file in files:
        ds = xr.open_dataset(file)
        ds = ds.sortby('lat')  # Ensure latitude is sorted
        # Ensure lon and lat are 1D
        if ds['lon'].ndim == 1 and ds['lat'].ndim == 1:
            # Drop duplicates in coordinates (if any)
            _, unique_lon_idx = np.unique(ds['lon'], return_index=True)
            _, unique_lat_idx = np.unique(ds['lat'], return_index=True)

            # Select unique values using isel
            ds = ds.isel(lon=unique_lon_idx, lat=unique_lat_idx)

            # Now you can safely do selection
            ds_filtered = ds.sel(
                lon=[p[0] for p in valid_points],
                lat=[p[1] for p in valid_points],
                method='nearest'
            )

        filtered_file = os.path.join(input_folder, os.path.basename(file).replace('.nc', '_filtered.nc'))
        ds_filtered.to_netcdf(filtered_file)
        filtered_files.append(filtered_file)
        print(f"Filtered file saved: {filtered_file}")
        ds_filtered.close()


    # Step 3: Extract time series for each valid point
    filtered_files.sort(key=lambda x: os.path.basename(x))  # Sort files by name
    combined_ds = xr.open_mfdataset(filtered_files, combine='nested', concat_dim='time', coords='all')
    # drop the duplicate coordinates
    combined_ds = combined_ds.drop_duplicates(dim=['lon', 'lat'])
    # sort the dataset by latitude
    # Ensure the dataset is sorted by 'lon' and 'lat'
    combined_ds = combined_ds.sortby(['lon', 'lat'])
    df_dict = {}
    for lon, lat in valid_points:
        time_series = combined_ds['snd'].sel(lon=lon, lat=lat, method='nearest')
        df_dict[(lon, lat)] = time_series.to_dataframe().reset_index()

    print("Time series extracted for valid points.")

    for lon, lat in valid_points:
        df = df_dict[(lon, lat)]

        # Ensure time is in datetime format and sorted
        df['time'] =pd.to_datetime(df.time,unit='s')
        df = df.sort_values(by='time')  # Sort by time

        # Remove duplicate timestamps
        df = df[~df['time'].duplicated(keep='first')]

        # Set 'time' as the index for resampling
        df.set_index('time', inplace=True)

        # Resample to 1-hour intervals
        df = df.resample('1H').asfreq()
        df['time']=pd.to_datetime(df.index,unit='s')
        # Drop NaN values to fit interpolator
        valid = df['snd'].notna()

        if valid.sum() > 1:  # Need at least two points to interpolate
            # Convert time to UNIX timestamp for interpolation
            times_numeric = df['time'].astype(np.int64) // 10 ** 9

            # Get strictly increasing times and corresponding snd values
            times_valid = times_numeric[valid]
            snd_valid = df['snd'][valid].values

            # Sort by time again to be absolutely sure
            sorted_idx = np.argsort(times_valid)
            times_valid_sorted = times_valid.iloc[sorted_idx]
            snd_valid_sorted = snd_valid[sorted_idx]

            # Remove duplicate times (e.g., if multiple values map to same UNIX timestamp)
            _, unique_idx = np.unique(times_valid_sorted, return_index=True)
            times_unique = times_valid_sorted.iloc[unique_idx]
            snd_unique = snd_valid_sorted[unique_idx]

            # Now apply interpolation only if the result is still >= 2 points
            if len(times_unique) > 1:
                interpolator = PchipInterpolator(times_unique, snd_unique)
                df['snd_interp'] = interpolator(times_numeric)
            else:
                df['snd_interp'] = df['snd']

        # Update the dictionary with the processed DataFrame
        df_dict[(lon, lat)] = df


    # save the updated dictionary
    pd.to_pickle(df_dict, output_folder + 'time_series_csnow.pkl')
    return



def plot_meteo():

    path_pkl = '/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/observations/time_series.pkl'
    output_dir = '/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/observations/complete/'

    os.makedirs(output_dir, exist_ok=True)

    variables = ['SW', 'tp', 't', 'RH']

    # Load the pickled dictionary
    df_dict = pd.read_pickle(path_pkl)

    for key, df in df_dict.items():
        print(f"Key: {key}, Variables: {df.columns.tolist()}")

        # Define which variables to plot

        existing_vars = [var for var in variables if var in df.columns]

        if not existing_vars:
            print(f"No valid variables found for {key}. Skipping plot.")
            continue

        # Ensure 'time' column exists and is in datetime format
        df['time'] = pd.to_datetime(df.index)

        # convert tempeature from kelvin to celsius

        df['t']= df['t']-273.15
        # Plot
        fig, axs = plt.subplots(len(existing_vars), 1, figsize=(12, 3 * len(existing_vars)), sharex=True)
        fig.suptitle(f"Meteorological variables at location {key}", fontsize=16)

        if len(existing_vars) == 1:
            axs = [axs]  # Ensure axs is iterable

        for i, var in enumerate(existing_vars):
            axs[i].plot(df['time'], df[var], label=var)
            axs[i].set_ylabel(var)
            axs[i].legend(loc='upper right')
            axs[i].grid(True)

        axs[-1].set_xlabel('Time')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Sanitize key for filename
        filename = f"meteo_{key[0]:.4f}_{key[1]:.4f}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath)
        plt.close()
        print(f"Saved plot to {filepath}")

# -----------------------------------------------------------------------------
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    #main()
    csnow_series()








