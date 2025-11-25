import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray
import logging
from lib_utilis_data_proc import get_args
from lib_data_io_json import read_file_settings


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger()

def plot_map(data_2d, dem, title, colorbar_label, save_file, cmap):

    if isinstance(dem, str):
        dem = rioxarray.open_rasterio(dem, masked=True).squeeze()

    # SET values -9999 to NaN
    #data_2d = data_2d.where(data_2d != -9999.0)
    data_2d = np.where(data_2d != -9999.0, data_2d, np.nan)

    if "Snow Depth" in title:
        # clip to 10 m
        data_2d = np.where(data_2d <= 10.0, data_2d, np.nan)

    if "SWE" in title:
        data_2d = np.where(data_2d <= 10000.0, data_2d, np.nan)

    dem_lat = dem['y'].values
    dem_lon = dem['x'].values

    # Correct the extent definition
    extent = [dem_lon.min(), dem_lon.max(), dem_lat.min(), dem_lat.max()]  # Define map extent

    plt.figure(figsize=(12, 12))
    im = plt.imshow(data_2d, cmap=cmap, extent=extent, origin='lower')  # Use extent for geographic mapping
    plt.contour(dem_lon, dem_lat, dem.values, levels=[0], colors='black', linewidths=2,extent=extent)

    # Add title and labels
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Longitude (degrees)', fontsize=12)
    plt.ylabel('Latitude (degrees)', fontsize=12)

    # Add grid
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add colorbar
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label(colorbar_label, fontsize=12)

    # Save and close the plot
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()
    log.info(f"✅ Map saved: {save_file}")
    return None

def plot_time_series(obs_df, modeled_snow, modeled_swe, lat, lon,  save_file):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    try:
        obs_data = obs_df['snd_interp'].values
    except KeyError:
        try:
            obs_data = obs_df['Snow_depth_cm'].values / 100  # convert cm to m
        except:
            obs_data = np.zeros_like(modeled_snow)

    # Snow Depth
    ax1.plot(obs_df.index, obs_data, label='Observed Snow Depth', color='red', linewidth=0.8)
    ax1.plot(modeled_snow['time'], modeled_snow, label='Modeled Snow Depth', color='black', linestyle='--', linewidth=0.8)
    ax1.set_ylabel('Snow Depth (m)', fontsize=12)
    ax1.set_title('Snow Depth Time Series', fontsize=14, weight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # SWE
    ax2.plot(modeled_swe['time'], modeled_swe, label='Modeled SWE', color='blue', linewidth=0.8)
    ax2.set_ylabel('SWE (mm)', fontsize=12)
    ax2.set_title('SWE Time Series', fontsize=14, weight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Format x-axis
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

    # Add a super title
    fig.suptitle(f"Lat: {lat:.4f}, Lon: {lon:.4f} ", fontsize=16, weight='bold')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save and close the plot
    plt.savefig(save_file, dpi=300)
    plt.close()
    log.info(f"✅ Time series plot saved: {save_file}")
def process_and_plot_snow_data(file, data_path, save_path, start, end):
    save_path = Path(save_path) / f"{start}_{end}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Load DEM
    dem_file = '/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA.tif'
    dem = rioxarray.open_rasterio(dem_file, masked=True).squeeze()
    dem_lat, dem_lon = dem['y'].values, dem['x'].values

    # Load snow model
    ds = xr.open_dataset(file).rename({'lat': 'y', 'lon': 'x'}).assign_coords(x=dem_lon, y=dem_lat)
    snow, swe = ds['H_S_m'], ds['SWE_mm']
    time_index = pd.to_datetime(ds.time.values)

    # Plot snapshots
    for idx in [0, len(snow.time) // 2, len(snow.time) - 1]:
        tlabel = str(snow.time.isel(time=idx).values)
        # truncate the TIME TO 1H
        tlabel = tlabel[:13].replace('T', ' ')

        plot_map(snow.isel(time=idx), dem, f'Snow Depth ({tlabel})', 'Snow Depth (m)', save_path / f'snow_depth_map_{idx}.png')
        plot_map(swe.isel(time=idx), dem, f'SWE ({tlabel})', 'SWE (mm)', save_path / f'swe_map_{idx}.png')

    # Load point observations
    try:
        with open(data_path, 'rb') as f:
            obs_data = pickle.load(f)
    except Exception as e:
        log.error(f"❌ Failed to load observations from {data_path}: {e}")
        return

    rmse_list, std_list = [], []

    for (lon, lat), df in obs_data.items():
        log.info(f"▶ Site: lon = {lon}, lat = {lat}")
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[~df.index.duplicated()].reindex(time_index, method='nearest')

        try:
            snow_sel = snow.sel(x=lon, y=lat,  method='nearest')
            swe_sel = swe.sel(x=lon, y=lat, method='nearest')
        except Exception as e:
            log.warning(f"⚠️ Could not locate grid point for ({lon}, {lat}): {e}")
            continue

        lat_sel, lon_sel = snow_sel['y'].values, snow_sel['x'].values
        log.info(f"  Mapped to grid: Lat = {lat_sel:.3f}, Lon = {lon_sel:.3f}")

        if snow_sel.isnull().all():
            log.warning("  ⛔ No valid model data at this site.")
            continue

        fig_path = save_path / f"C-snow_time_series_LAT{lat:.4f}_LON{lon:.4f}_{start}_{end}.png"
        plot_time_series(df, snow_sel, swe_sel, lat, lon, start, end, fig_path)

        try:
            obs = df['snd_interp'].values
        except KeyError:
            try:
                obs = df['Snow_depth_cm'].values / 100
            except KeyError:
                obs = np.zeros_like(snow_sel)

        mod = snow_sel.values

        rmse = np.sqrt(np.mean((obs - mod) ** 2))
        std = np.std(mod)

        if rmse < 100 and std < 100:
            rmse_list.append(rmse)
            std_list.append(std)

    if rmse_list:
        log.info(f"\n📊 Mean RMSE Snow Depth CSNOW: {np.mean(rmse_list):.3f} m")
        log.info(f"📊 Mean STD  Snow Depth CSNOW: {np.mean(std_list):.3f} m")

    # Load VDA station observations
    rmse_list, std_list = [], []
    vda_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/dict.pkl"
    with open(vda_path, 'rb') as f:
        vda_st = pickle.load(f)

    for (lat,lon ), df in vda_st.items():
            log.info(f"📡 VDA Site: lon = {lon}, lat = {lat}")
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df[~df.index.duplicated()].reindex(time_index, method='nearest')

            try:
                snow_sel = snow.sel(x=lon, y=lat, method='nearest')
                swe_sel = swe.sel(x=lon, y=lat, method='nearest')
            except Exception as e:
                log.warning(f"⚠️ Could not locate grid point for VDA site ({lon}, {lat}): {e}")
                continue

            if snow_sel.isnull().all():
                continue

            fig_path = save_path / f"VDA_time_series_LAT{lat:.4f}_LON{lon:.4f}_{start}_{end}.png"
            plot_time_series(df, snow_sel, swe_sel, lat, lon, start, end, fig_path)
            obs = df['Snow_depth_cm'].values / 100
            mod = snow_sel.values

            rmse = np.sqrt(np.mean((obs - mod) ** 2))
            std = np.std(mod)

            if rmse < 100 and std < 100:
                rmse_list.append(rmse)
                std_list.append(std)

    if rmse_list:
        log.info(f"\n📊 Mean RMSE Snow Depth VDA: {np.mean(rmse_list):.3f} m")
        log.info(f"📊 Mean STD  Snow Depth VDA: {np.mean(std_list):.3f} m")

    return  rmse , std

def plot_meteo_maps(meteo_ds, dem, output_dir, idx,cmap='BuPu'):
    """
    Plots each variable in the meteo dataset as a map.

    Parameters:
        meteo_ds (xr.Dataset): The meteorological dataset containing variables to plot.
        dem (xr.DataArray): The DEM data for contour overlay.
        output_dir (str): Directory to save the plots.
        cmap (str): Colormap for the plots.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Extract DEM latitude and longitude
    dem_lat = dem['y'].values
    dem_lon = dem['x'].values
    # Correct the extent definition
    extent = [dem_lon.min(), dem_lon.max(), dem_lat.min(), dem_lat.max()] # Define map extent
    time_label = str(meteo_ds.time.isel(time=idx).values.astype('datetime64[h]'))
    log.info(f"▶ Plotting meteorological maps for time: {time_label}")
    # Create one figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(f"Downscaled Forcing - Time: {time_label}", fontsize=18, weight='bold')

    # Flatten axes for easier looping
    axes = axes.flatten()

    for ax, var_name in zip(axes, meteo_ds.data_vars):
        data_2d = meteo_ds[var_name].isel(time=idx)
        # SET values -9999 to NaN
        data_2d = data_2d.where(data_2d != -9999)
        title = var_name.replace('_', ' ').capitalize()
        colorbar_label = var_name.replace('_', ' ').capitalize()

        im = ax.imshow(data_2d.data, cmap=cmap, extent=extent, origin='lower')
        ax.contour(dem_lon, dem_lat, dem.values, levels=[0], colors='black', linewidths=2,extent=extent)

        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Add colorbar for each subplot
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
        cbar.set_label(colorbar_label)

    # Save the combined figure
    save_file = os.path.join(output_dir, "meteo_maps.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    log.info(f"✅ Combined map saved: {save_file}")

    """ 
    for var_name in meteo_ds.data_vars:
        data_2d = meteo_ds[var_name].isel(time=4000)
        title = f"{var_name.replace('_', ' ').capitalize()} Map"
        colorbar_label = var_name.replace('_', ' ').capitalize()
        save_file = os.path.join(output_dir, f"{var_name}_map.png")

        plt.figure(figsize=(12, 8))
        im = plt.imshow(data_2d, cmap=cmap, extent=extent, origin='upper')  # Use extent for geographic mapping
        plt.contour(dem, levels=[0], colors='black', linewidths=2, extent=extent)
        plt.title(title, fontsize=16, weight='bold')  # Make title bold and bigger
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        cbar = plt.colorbar(im, orientation='horizontal')  # Set colorbar to horizontal
        cbar.set_label(colorbar_label)
        plt.tight_layout()
        plt.savefig(save_file, dpi=300)
        plt.close()
        log.info(f"✅ Map saved: {save_file}")
    """

def plot_maps_s3m(ds, dem, output_dir, idx,cmap='BuPu'):
    """
    Plots each variable in the meteo dataset as a map.

    Parameters:
        meteo_ds (xr.Dataset): The meteorological dataset containing variables to plot.
        dem (xr.DataArray): The DEM data for contour overlay.
        output_dir (str): Directory to save the plots.
        cmap (str): Colormap for the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    dem_lat = dem['y'].values
    dem_lon = dem['x'].values
    # Correct the extent definition
    extent = [dem_lon.min(), dem_lon.max(), dem_lat.min(), dem_lat.max()] # Define map extent
    time_label = str(ds.time.isel(time=idx).values.astype('datetime64[h]'))
    log.info(f"▶ Plotting maps for time: {time_label}")


    fig, axes = plt.subplots( 3,3, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(f"Output - Time: {time_label}", fontsize=18, weight='bold')
    axes = axes.flatten()

    for ax, var_name in zip(axes, ds.data_vars):
        data_2d = ds[var_name].isel(time=idx)

        # SET values -9999 to NaN
        data_2d = data_2d.where(data_2d != -9999)
        data_2d = data_2d.where (data_2d >= 0)

        title = var_name.replace('_', ' ').capitalize()
        colorbar_label = var_name.replace('_', ' ').capitalize()
        im = ax.imshow(data_2d.values, cmap=cmap, origin='lower', extent=extent)
        ax.contour(dem_lon, dem_lat, dem.values, levels=[0], colors='black', linewidths=2,extent=extent)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
        cbar.set_label(colorbar_label)

    save_file = os.path.join(output_dir , f"maps_{time_label.replace(' ', '_').replace(':', '-')}.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    log.info(f"✅ Combined map saved: {save_file}")

def comparison_run():
    # ---------------------------------------------------------------------------
    dem_file = '/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA.tif'
    dem = rioxarray.open_rasterio(dem_file, masked=True).squeeze()
    dem_lat, dem_lon = dem['y'].values, dem['x'].values
    # ---------------------------------------------------------------------------
    outputs = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/output_data_2018-01-01_2018-01-31.nc"
    save_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plots/S3M/"
    save_path = Path(save_path)
    ds = xr.open_dataset(outputs).rename({'lat': 'y', 'lon': 'x'}).assign_coords(x=dem_lon, y=dem_lat)
    ds = ds[["Rainfall_mm", "Snowfall_mm", "Melting_mm", "Refreezing_mm",
             "Outflow_mm", "Sf_daily_cum", "Snow_Age", "H_S_m", 'SWE_mm']]

    for idx in [0, len(ds.time) // 2, len(ds.time) - 1]:
        tlabel = str(ds.time.isel(time=idx).values)
        # truncate the TIME TO 1H
        tlabel = tlabel[:13].replace('T', ' ')
        # Ensure save_path is a Path object
        plot_maps_s3m(ds, dem, save_path, idx, 'BuPu')
   #-----------------------------------------------------------------------------

    state = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/state_data_2018-01-01_2018-01-31.nc"
    save_path_state = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plots/S3M/state/"
    save_path_state = Path(save_path_state)
    ds_state = xr.open_dataset(state).rename({'lat': 'y', 'lon': 'x'}).assign_coords(x=dem_lon, y=dem_lat)
    ds_state = ds_state[['SWE_W_mm', 'SWE_D_mm', 'RHO_D_kg_m3', 'albedo']]

    for idx in [0, len(ds.time) // 2, len(ds.time) - 1]:
        tlabel = str(ds.time.isel(time=idx).values)
        # truncate the TIME TO 1H
        tlabel = tlabel[:13].replace('T', ' ')
        plot_maps_s3m(ds_state, dem, save_path_state, idx, 'BuPu')

    # ---------------------------------------------------------------------------
    file = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/output_201801.nc"
    save_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plots/S3M_FORTRAN/"
    save_path = Path(save_path)
    ds_fortran = xr.open_dataset(file).rename({'Latitude': 'y', 'Longitude': 'x'}).assign_coords(x=dem_lon, y=dem_lat)
    ds_fortran = ds_fortran[['RainFall', 'SnowFall', 'MeltingS', 'RefreezingS', "Outflow",
                             'SnowfallCum', 'AgeS', 'H_S', 'SWE']]

    for idx in [0, len(ds_fortran.time) // 2, len(ds_fortran.time) - 1]:
        tlabel = str(ds.time.isel(time=idx).values)
        # truncate the TIME TO 1H
        tlabel = tlabel[:13].replace('T', ' ')
        #plot_maps_s3m(ds_fortran, dem, save_path, idx, 'BuPu')
    # ---------------------------------------------------------------------------

    # compute the differencr between hs and hs_m AS A SINGLE VALUE
    error = ds_fortran['H_S'].values - ds['H_S_m'].values
    mae = np.nanmean(np.abs(error))
    rmse = np.sqrt(np.nanmean(error ** 2))
    print(f"MAE H_S: {mae:.3f} m")
    print(f"RMSE H_S: {rmse:.3f} m")
    # plot the error as a map for the 3 time steps

    for idx in [0, len(ds_fortran.time) // 2, len(ds_fortran.time) - 1]:
        tlabel = str(ds.time.isel(time=idx).values)
        # truncate the TIME TO 1H
        tlabel = tlabel[:13].replace('T', ' ')
        # Ensure save_path is a Path object
        save_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plots/"
        save_path = Path(save_path)
        plot_map(error[idx,:,:], dem, f'Error H_S (Fortran - Python) ({tlabel})', 'Error H_S (m)',
                 save_path / f'error_HS_map_{idx}.png', 'PuBu')
    # ---------------------------------------------------------------------------

    file = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/output_201801.nc"
    save_path_fortran_state = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plots/S3M_FORTRAN/state/"
    save_path_fortran_state = Path(save_path_fortran_state)
    ds_fortran = xr.open_dataset(file).rename({'Latitude': 'y', 'Longitude': 'x'}).assign_coords(x=dem_lon, y=dem_lat)
    ds_fortran_state = ds_fortran[['SWE_W','SWE_D','Rho_D','AlbedoS']]

    for idx in [0, len(ds_fortran.time) // 2, len(ds_fortran.time) - 1]:
        tlabel = str(ds.time.isel(time=idx).values)
        # truncate the TIME TO 1H
        tlabel = tlabel[:13].replace('T', ' ')
       # plot_maps_s3m(ds_fortran_state, dem, save_path_fortran_state , idx, 'BuPu')


    # ---------------------------------------------------------------------------
    lat, lon = 45.9, 7.2
    ds_fortran_new = xr.Dataset(
        {'Rainfall_mm': (('time','y', 'x'), ds_fortran['RainFall'].values),
         'Snowfall_mm': (('time','y', 'x'), ds_fortran['SnowFall'].values),
         'Melting_mm': (('time','y', 'x'), ds_fortran['MeltingS'].values),
         'Refreezing_mm': (('time','y', 'x'), ds_fortran['RefreezingS'].values),
         'Outflow_mm': (('time','y', 'x'), ds_fortran['Outflow'].values),
         'Sf_daily_cum': (('time','y', 'x'), ds_fortran['SnowfallCum'].values),
         'Snow_Age': (('time','y', 'x'), ds_fortran['AgeS'].values),
         'H_S_m': (('time','y', 'x'), ds_fortran['H_S'].values),
            'SWE_mm': (('time','y', 'x'), ds_fortran['SWE'].values)
         },
        coords={
            "time": ds.time.values,
            "y": dem_lat,
            "x": dem_lon

        }
    )
    ds_fortran_new_state = xr.Dataset(
        {'SWE_W_mm': (('time','y', 'x'), ds_fortran_state['SWE_W'].values),
            'SWE_D_mm': (('time','y', 'x'), ds_fortran_state['SWE_D'].values),
            'RHO_D_kg_m3': (('time','y', 'x'), ds_fortran_state['Rho_D'].values),
            'albedo': (('time','y', 'x'), ds_fortran_state['AlbedoS'].values)
            },
        coords={
            "time": ds.time.values,
            "y": dem_lat,
            "x": dem_lon
        }
    )
    ds_state_list = [ds_state[var].sel(x=lon, y=lat, method="nearest") for var in ds_state.data_vars]
    ds_fortran_state_list = [ds_fortran_new_state[var].sel(x=lon, y=lat, method="nearest") for var in ds_fortran_new_state.data_vars]

    # Extract nearest grid point for each variable
    ds_list = [ds[var].sel(x=lon, y=lat, method="nearest") for var in ds.data_vars]
    ds_fortran_list = [ds_fortran_new[var].sel(x=lon, y=lat, method="nearest") for var in ds_fortran_new.data_vars]

    # Convert time
    time_index = pd.to_datetime(ds.time.values)
    time_index_fortran = pd.to_datetime(ds_fortran_new.time.values)

    # Variable names (must match both ds and ds_fortran_new)
    var_names = list(ds.data_vars.keys())
    y_labels = [
        "Rainfall (mm)", "Snowfall (mm)", "Melting (mm)", "Refreezing (mm)",
        "Outflow (mm)", "Snowfall Cumulative (mm)", "Snow Age (days)", "Snow Depth H_S (m)", "SWE (mm)"
    ]
    colors = ["blue", "green", "red", "orange", "purple", "brown", "gray", "teal", "cyan"]

    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)
    fig.suptitle(f"Python vs Fortran Comparison at Lat: {lat:.4f}, Lon: {lon:.4f}", fontsize=16, weight="bold")
    axes = axes.flatten()

    # Loop over variables and plot
    for ax, var_name, ds_var, ds_fortran_var, y_label, color in zip(
            axes, var_names, ds_list, ds_fortran_list, y_labels, colors):
        ax.plot(time_index, ds_var, label=f"Python {var_name}", color=color,linewidth=1.2)
        ax.plot(time_index_fortran, ds_fortran_var, label=f"Fortran {var_name}", color='k', linestyle ="--", linewidth=1.2, alpha=0.7)

        ax.set_title(f"{var_name} Time Series", fontsize=13, weight="bold")
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(alpha=0.5, linestyle="--")
        ax.legend(fontsize=9)

        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))

    # Save plot
    save_path = Path("/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plots/time_series_comparison_fluxes.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    log.info(f"✅ Saved comparison time series plot to {save_path}")

    state_var_names = list(ds_state.data_vars.keys())
    state_y_labels = [
        "SWE Wet (mm)", "SWE Dry (mm)", "Density Dry (kg/m³)", "Albedo"]
    state_colors = ["blue", "green", "red", "orange"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"State Variables Comparison at Lat: {lat:.4f}, Lon: {lon:.4f}", fontsize=16, weight="bold")
    axes = axes.flatten()

    for ax, var_name, ds_var, ds_fortran_var, y_label, color in zip(
            axes, state_var_names, ds_state_list, ds_fortran_state_list, state_y_labels, state_colors):
        ax.plot(time_index, ds_var, label=f"Python {var_name}", color=color,linewidth=1.2)
        ax.plot(time_index_fortran, ds_fortran_var, label=f"Fortran {var_name}", color='k', linestyle ="--", linewidth=1.2, alpha=0.7)

        ax.set_title(f"{var_name} Time Series", fontsize=13, weight="bold")
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(alpha=0.5, linestyle="--")
        ax.legend(fontsize=9)

        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    # Save plot
    save_path = Path("/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plots/time_series_comparison_state.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    log.info(f"✅ Saved comparison time series plot to {save_path}")

    # make a plot of snowfall and albedo for the two models
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"Snowfall and Albedo Comparison at Lat: {lat:.4f}, Lon: {lon:.4f}", fontsize=16, weight="bold")
    axes = axes.flatten()
    axes[0].plot(time_index, ds_list[1], label=f"Python Snowfall", color='blue',linewidth=1.2)
    axs = axes[0].twinx()
    axs.plot(time_index, ds_state_list[3], label=f"Python Albedo", color='orange', linestyle ="--", linewidth=1.2)
    axes[0].set_title(f"Snowfall and Albedo Time Series", fontsize=13, weight="bold")
    axes[0].set_ylabel("Snowfall (mm)", fontsize=11)
    axs.set_ylabel("Albedo", fontsize=11)
    axes[0].grid(alpha=0.5, linestyle="--")
    axes[0].legend(fontsize=9, loc='upper left')
    axs.legend(fontsize=9, loc='upper right')
    axes[0].tick_params(axis="x", rotation=45, labelsize=9)
    axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))

    axes[1].plot(time_index_fortran, ds_fortran_list[1], label=f"Fortran Snowfall", color='gray', linestyle ="--", linewidth=1.2, alpha=0.7)
    axs1 = axes[1].twinx()
    axs1.plot(time_index_fortran, ds_fortran_state_list[3], label=f"Fortran Albedo", color='green', linestyle ="--", linewidth=1.2, alpha=0.7)
    axes[1].set_title(f"Snowfall and Albedo Time Series", fontsize=13, weight="bold")
    axes[1].set_ylabel("Snowfall (mm)", fontsize=11)
    axs1.set_ylabel("Albedo", fontsize=11)
    axes[1].grid(alpha=0.5, linestyle="--")
    axes[1].legend(fontsize=9, loc='upper left')
    axs1.legend(fontsize=9, loc='upper right')
    axes[1].tick_params(axis="x", rotation=45, labelsize=9)
    axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    # Save plot
    save_path = Path("/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plots/time_series_comparison_snowfall_albedo.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"Snow Age and Snowfall Comparison at Lat: {lat:.4f}, Lon: {lon:.4f}", fontsize=16, weight="bold")

    # --- Python ---
    axes[0].plot(time_index, ds_list[5], label="Python Snowfall", color='blue', linewidth=1.2)
    ax_age = axes[0].twinx()
    ax_age.plot(time_index, ds_list[6], label="Python Snow Age", color='orange', linestyle='--', linewidth=1.2)
    axes[0].set_title("Python Model", fontsize=13, weight="bold")
    axes[0].set_ylabel("Snowfall (mm)", fontsize=11)
    ax_age.set_ylabel("Snow Age (days)", fontsize=11)
    axes[0].grid(alpha=0.5, linestyle="--")
    axes[0].legend(fontsize=9, loc='upper left')
    ax_age.legend(fontsize=9, loc='upper right')
    axes[0].tick_params(axis="x", rotation=45, labelsize=9)
    axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))

    # --- Fortran ---
    axes[1].plot(time_index_fortran, ds_fortran_list[5], label="Fortran Snowfall", color='gray', linestyle="--",
                 linewidth=1.2, alpha=0.7)
    ax_age_f = axes[1].twinx()
    ax_age_f.plot(time_index_fortran, ds_fortran_list[6], label="Fortran Snow Age", color='green', linestyle="--",
                  linewidth=1.2, alpha=0.7)
    axes[1].set_title("Fortran Model", fontsize=13, weight="bold")
    axes[1].set_ylabel("Snowfall (mm)", fontsize=11)
    ax_age_f.set_ylabel("Snow Age (days)", fontsize=11)
    axes[1].grid(alpha=0.5, linestyle="--")
    axes[1].legend(fontsize=9, loc='upper left')
    ax_age_f.legend(fontsize=9, loc='upper right')
    axes[1].tick_params(axis="x", rotation=45, labelsize=9)
    axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))

    # Save figure
    save_path = Path(
        "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plots/time_series_comparison_snowage_snowfall.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    log.info(f"✅ Saved Snow Age and Snowfall comparison plot to {save_path}")
    return None


def plot_time_series_open_loop_vda():
    file_output = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/output_data_2018-10-01_2019-09-30.nc"
    vda_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/dict.pkl"
    save_path = Path("/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/plots/open_loop/")
    # open nc
    ds = xr.open_dataset(file_output)
    ds = ds[["Rainfall_mm", "Snowfall_mm", "Melting_mm", "Refreezing_mm",
             "Outflow_mm", "Sf_daily_cum", "Snow_Age", "H_S_m", 'SWE_mm']]
    snow = ds['H_S_m']
    swe = ds['SWE_mm']
    time_index = pd.to_datetime(ds.time.values)
    start = "2018-10-01"
    end = "2019-09-30"

    # Load VDA station observations
    rmse_list, std_list = [], []

    with open(vda_path, 'rb') as f:
        vda_st = pickle.load(f)

    print(vda_st)

    for (lat, lon), df in vda_st.items():
        log.info(f"📡 VDA Site: lon = {lon}, lat = {lat}")
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[~df.index.duplicated()].reindex(time_index, method='nearest')

        try:
            snow_sel = snow.sel(lat=lat, lon=lon,method='nearest')
            swe_sel = swe.sel( lat=lat, lon=lon, method='nearest')
        except Exception as e:
            log.warning(f"⚠️ Could not locate grid point for VDA site ({lon}, {lat}): {e}")
            continue

        if snow_sel.isnull().all():
            continue

        fig_path = save_path / f"VDA_time_series_LAT{lat:.4f}_LON{lon:.4f}_{start}_{end}.png"
        plot_time_series(df, snow_sel, swe_sel, lat, lon,fig_path)
        obs = df['Snow_depth_cm'].values / 100
        mod = snow_sel.values

        rmse = np.sqrt(np.mean((obs - mod) ** 2))
        std = np.std(mod)

        if rmse < 100 and std < 100:
            rmse_list.append(rmse)
            std_list.append(std)

    if rmse_list:
        log.info(f"\n📊 Mean RMSE Snow Depth VDA: {np.mean(rmse_list):.3f} m")
        log.info(f"📊 Mean STD  Snow Depth VDA: {np.mean(std_list):.3f} m")

def plot_meteo_ensemble(meteo_ensemble, Time, output_folder,meteo_original):
    """
    Plot the time series of the meteorological ensemble for the first observed point.
    """
    num_vars = meteo_ensemble.shape[2]
    num_ensemble = meteo_ensemble.shape[3]
    list_titles = [ 'Temperature','Precipitation', 'Relative Humidity',  'Radiation','T_10D', 'T_1D']
    plt.figure(figsize=(20, 12))
    colorors = plt.cm.viridis(np.linspace(0, 1, num_ensemble))

    for i in range(num_vars):
        plt.subplot(2, 3, i + 1)
        for n in range(num_ensemble):
            plt.plot(Time, meteo_ensemble[:, 0, i, n], color= colorors[n],  alpha=0.5)

        plt.plot(Time, meteo_original[:,0, 0, i], color='red', label='Original', linewidth=1)

        plt.title(list_titles[i], fontsize=15)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid()
        plt.legend()
        # --- Styling ---
    plt.suptitle('Meteorological Ensemble Time Series', fontsize=20)
    plt.tight_layout()
    # Save the figure
    output_path = os.path.join(output_folder, 'meteo_ensemble_time_series.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved to {output_path}")

def plot_ensemble(state_matrix_ensemble,state_matrix_prior, state_matrix, data_settings, start, end, Time,
                          output_matrix_ensemble, output_matrix, output_matrix_prior,obs):

    for i in range(state_matrix.shape[1]):

        # Plot state ensemble and output ensemble
        colors = plt.cm.viridis(np.linspace(0, 1, state_matrix_ensemble.shape[3]))
        fieldnames_state = ["SWE_w", "SWE_d", "rho_d", "albedo"]

        plt.figure(figsize=(20, 12))
        for p in range(state_matrix_ensemble.shape[2]):
            plt.subplot(3, 2, p + 1)
            for q in range(state_matrix_ensemble.shape[3]):  # Ensure q is within bounds
                plt.plot(Time, state_matrix_ensemble[1:, i, p, q], color='lightgrey', linewidth=1, linestyle="-", alpha=0.5,
                         label=f'Ensemble {q + 1}' if p == 0 else "")
            plt.plot(Time, state_matrix_prior[1:, i, i, p], color='black', linestyle='-', linewidth=2, label='Deterministic')
            plt.plot (Time, state_matrix[1:, i, i, p], color='red', linestyle='-', linewidth=2, label='Posterior mean')
            plt.title(fieldnames_state[p], fontsize=15)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.grid()
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_settings['data']['output_file']['folder_name'],
                                 f'state_ensemble_timeseries_{start}_{end}.png'))
        plt.close()

        fieldnames_output = ["SWE_mm", "H_S_m"]
        # now do the same for swe and snow depth . you find it in the position 10 and 14  of output matrix and output vector and output matrix ensemble
        # Create a single figure with two subplots for SWE and Snow Depth
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), constrained_layout=True)

        # Plot SWE in the first subplot
        for q in range(output_matrix_ensemble.shape[3]):  # Ensure q is within bounds
            axes[0].plot(Time, output_matrix_ensemble[1:, i, 10, q], color='lightgreen', linewidth=1)

        axes[0].plot(Time, output_matrix[1:, i, i, 10], color='red', linestyle='--', linewidth=0.5,
                     label='Posterior mean')
        axes[0].plot(Time, output_matrix_prior[1:, i, i, 0], color='black', linestyle='--', linewidth=1,
                     label='Deterministic')
        axes[0].set_title(fieldnames_output[0], fontsize=15)
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].grid()
        axes[0].legend()

        # Plot Snow Depth in the second subplot
        for q in range(output_matrix_ensemble.shape[3]):  # Ensure q is within bounds
            axes[1].plot(Time, output_matrix_ensemble[1:, i, 14, q], color='lightgreen', linewidth=1)
        axes[1].plot(Time, output_matrix[1:, i, i, 14], color='red', linestyle='-', linewidth=2, label='Posterior mean')
        axes[1].plot(Time, output_matrix_prior[1:, i, i, 1], color='black', linestyle='--', linewidth=1,
                     label='Deterministic')
        axes[1].plot(Time, obs[:,i], color='blue', linestyle='-', linewidth=2, label='Observations')
        axes[1].set_title(fieldnames_output[1], fontsize=15)
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('Value', fontsize=12)
        axes[1].grid()
        axes[1].legend()

        # Save the combined figure
        plt.savefig(os.path.join(data_settings['data']['output_file']['folder_name'],
                                 f'output_ensemble_combined_timeseries_{i}_{start}_{end}.png'))
        plt.close()
    return


if __name__ == "__main__":
    None

