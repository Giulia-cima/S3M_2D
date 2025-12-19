import os
import warnings
import numpy as np
import pandas
import pickle
import time
import rioxarray
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d
import xarray as xr
import rioxarray as rxr
from geopy.distance import geodesic


warnings.filterwarnings("ignore")
def compute_relative_humidity(temp , qair, pressure):
    """
    Compute relative humidity  using specific humidity, temperature, and pressure.
    qair: specific humidity (kg/kg)
    temp: temperature (K)
    pressure: pressure (Pa)
    Returns relative humidity in percentage (0-100%).
    """
    mr = qair / (1 - qair)
    e = mr * pressure / (0.62197 + mr)
    es = 611.2 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))
    rh = e / es
    rh = np.clip(rh, 0, 1)
    rh_percentage = rh * 100
    return rh_percentage


def stochastic_process():
    # write a script to emulate a stochastic process with a given covariance matrix R
    # and plot the result
    # -------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------

    n = 6  # dimensione del vettore di stato
    x0 = np.zeros((n, 1))  # condizione iniziale
    N = 24 * 365 * 20  # numero di passi

    # matrice di covarianza di primo tentativo
    r = np.random.randn(n, n)
    R0 = np.dot(r, r.T)

    # Catena di Markov
    X = np.zeros((n, N))
    x = x0
    L0 = np.linalg.cholesky(R0)  # use the transpose to match MATLAB's 'lower' option
    for t in range(1, N):
        x = x + np.dot(L0, np.random.normal(0, 0.1, (n, 1)))
        X[:, t] = x.flatten()

    plt.plot(X.T)
    # save figure in the current directory
    plt.savefig('/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/quantile_mapping/"stochastic_process.png')

    R_tilde = np.cov(X)
    # obtain the cholesky decomposition of R_tilde
    L_tilde = np.linalg.cholesky(R_tilde)
    # save the matrix L_tilde in a pkl file
    pickle.dump(L_tilde, open( "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/quantile_mapping/L_tilde.pkl", "wb"))
    # save the matrix L0 in a pkl file
    pickle.dump(L0, open( "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/quantile_mapping/L0.pkl", "wb"))

    # Initialize or load the `data` variable
    data = [[0] * 4 for _ in range(2)]  # Example initialization with a 2x4 matrix

    # Modify the data
    data[0][0]=0
    data[0][1] =0
    data[0][2] =67.4
    data[0][3] = 0.5
    data[1][0] =500
    data[1][0] =500
    data[1][2] = 600
    data[1][3] = 0.95

    # Save the modified data
    with open('/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/quantile_mapping/state_limits.pkl', 'wb') as file:
        pickle.dump(data, file)
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
def quantile_mapping(dataframe, key):
    """
    Apply quantile mapping to a DataFrame column.
    """
    df_sorted = dataframe.sort_values(key, ascending=True)
    N = len(df_sorted)
    df_sorted['emp_rip'] = [(i + 1) / (N + 1) for i in range(N)]
    df_sorted['gauss_quant'] = [stats.norm.ppf(df_sorted['emp_rip'].iloc[i]) for i in range(N)]
    return df_sorted

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def closest_station(y, x, processed_keys):
    """
    y, x: lat/lon of the current station
    processed_keys: list of tuples of (lat, lon) already processed
    """
    min_dist = float('inf')
    closest = None
    for py, px in processed_keys:
        dist = geodesic((y, x), (py, px)).km
        if dist < min_dist:
            min_dist = dist
            closest = (py, px)
    return closest



def dataset_quantile_mapping():
    t0 = time.time()
    quantile_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/quantile_mapping/"
    os.makedirs(quantile_path, exist_ok=True)
    meteo_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/input_meteo/S3M_MeteoData_20181001_20190930.nc"
    # Open the dataset
    ds = xr.open_dataset(meteo_path)
    # Define the time slices
    slice_list = [("2018-10-01 00:00:00", "2019-09-30 23:00:00")]
    quantile_data_list = []
    R_dict_list = []
    statistics_dict_list = []


    Latitudes_1d = np.unique(ds.lat)
    Longitudes_1d = np.unique(ds.lon)

    AirTemperature_3D = np.repeat(
        ds['AirTemperature'].values [:, :, np.newaxis],  # shape (time, 43, 1)
        43, axis=2
    )  # now shape is (8760, 43, 43)

    Rain_3D = np.repeat(
        ds['Rain'].values [:, :, np.newaxis],  # shape (time, 43, 1)
        43, axis=2
    )  # now shape is (8760, 43, 43)

    IncRadiation_3D = np.repeat(
        ds['IncRadiation'].values [:, :, np.newaxis],  # shape (time, 43, 1)
        43, axis=2
    )  # now shape is (8760, 43, 43)
    RelHumidity_3D = np.repeat(
        ds['RelHumidity'].values [:, :, np.newaxis],  # shape (time, 43, 1)
        43, axis=2
    )  # now shape is (8760, 43, 43)


    # Build the new dataset
    ds_new = xr.Dataset(
        coords={
            "time": ds.time,
            "Latitude": Latitudes_1d,
            "Longitude": Longitudes_1d
        },
        data_vars={
            "AirTemperature": (("time", "Latitude", "Longitude"), AirTemperature_3D),
            "Rain": (("time", "Latitude", "Longitude"), Rain_3D),
            "IncRadiation": (("time", "Latitude", "Longitude"), IncRadiation_3D),
            "RelHumidity": (("time", "Latitude", "Longitude"), RelHumidity_3D)
        }

    )
    print(ds_new.coords)

    station_list = pandas.read_csv('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/STATION_ID_VDA.csv')
    # Replace commas with periods in the 'lat' and 'lon' columns
    station_list['lat'] = station_list['lat'].str.replace(',', '.').astype(float)
    station_list['lon'] = station_list['lon'].str.replace(',', '.').astype(float)

    i = 0

    # crea una mask per tenere solo le stazioni presenti in station_list, tenendo conto che la matrice meteo ha shape (nt, ny, nx, 6)
    obs_mask = []

    for index, row in station_list.iterrows():
        lat_station = row['lat']
        lon_station = row['lon']
        # round at 6 number after comma
        lat_station = round(lat_station, 6)
        lon_station = round(lon_station, 6)
        obs_mask.append((lat_station, lon_station))

    t0 = time.time()
    for start, end in slice_list:
        for y,x  in obs_mask:

            # Select the point in the DataArray
            meteo_ds = ds_new.sel(Latitude=y, Longitude = x, method="nearest")
            print(f"Values selected for station at lon: {x}, lat: {y}")

            # Remove values of precipitation and radiation that are less than or equal to 0
            meteo_ds['Rain'] = meteo_ds['Rain'].where(meteo_ds['Rain'] > 0, np.nan)

            meteo_ds['IncRadiation'] = meteo_ds['IncRadiation'].where(meteo_ds['IncRadiation'] > 0, np.nan)

            temperature = meteo_ds["AirTemperature"]
            # set values below -35 to -35

            temperature = temperature.where(temperature > -30, -15)

            # remove values below -9999c for temperature
            temperature = temperature.where(temperature > -9999, np.nan)

            radiation = meteo_ds["IncRadiation"]
            precipitation = meteo_ds["Rain"]
            relative_humidity = meteo_ds["RelHumidity"]
            # clip relative humidity between 0 and 100
            relative_humidity = relative_humidity.clip(0, 100)

            # Convert the smoothed arrays back to xarray.DataArray
            T_albedo = meteo_ds["AirTemperature"].rolling(time=24, min_periods=1).mean()
            T_melting = meteo_ds["AirTemperature"].rolling(time=48, min_periods=1).mean()

            # Assign a name to the DataArray
            T_albedo.name = 'T_albedo'
            T_melting.name = 'T_melting'

            print(f"Variables selected for station at lon: {x}, lat: {y} in {time.time() - t0:.2f} seconds")

            # convert each variable into a pandas DataFrame
            prc_mm = pandas.DataFrame(meteo_ds["Rain"].values, columns=['prc_mm'])
            temperature = pandas.DataFrame(temperature.values, columns=['air_temp_degC'])
            radiation = pandas.DataFrame(radiation.values, columns=['swin_wm-2'])
            relative_humidity = pandas.DataFrame(relative_humidity.values, columns=['rel_hum_perc'])
            relative_humidity['rel_hum_perc'] = relative_humidity['rel_hum_perc'].apply(lambda v: min(max(v, 0), 100))
            T_albedo = pandas.DataFrame(T_albedo.values, columns=['T_albedo'])
            T_melting = pandas.DataFrame(T_melting.values, columns=['T_melting'])


            # remove nan from all the variables
            prc_mm = prc_mm.dropna()
            temperature = temperature.dropna()
            radiation = radiation.dropna()
            relative_humidity = relative_humidity.dropna()
            T_albedo = T_albedo.dropna()
            T_melting = T_melting.dropna()


            # Apply quantile mapping
            prc_mm = quantile_mapping(prc_mm, 'prc_mm')
            temperature = quantile_mapping(temperature, 'air_temp_degC')
            radiation = quantile_mapping(radiation, 'swin_wm-2')
            relative_humidity = quantile_mapping(relative_humidity, 'rel_hum_perc')
            T_albedo = quantile_mapping(T_albedo, 'T_albedo')
            T_melting = quantile_mapping(T_melting, 'T_melting')

            print(f"quantile mapping computed for station at lon: {x}, lat: {y} in {time.time() - t0:.2f} seconds")

            df = pandas.DataFrame(columns=["prc_mm_rif", "prc_mm", "swin_wm", "air_temp_degC", "rel_hum_perc","T_albedo", "T_melting"
                                           ])

            df["prc_mm_rif"] = prc_mm["prc_mm"].astype(np.float64)
            df["prc_mm"] = prc_mm["gauss_quant"].astype(np.float64)
            df["air_temp_degC"] = temperature['gauss_quant'].astype(np.float64)
            df["rel_hum_perc"] = relative_humidity["gauss_quant"].astype(np.float64)
            df["swin_wm"] = radiation["gauss_quant"].astype(np.float64)
            df["T_albedo"] = T_albedo["gauss_quant"].astype(np.float64)
            df["T_melting"] = T_melting["gauss_quant"].astype(np.float64)

            # remove row with precipitation equal to 0
            df = df.drop(df[df["prc_mm_rif"] ==0].index)
            df = df.drop("prc_mm_rif", axis=1)

            # if the df is empty  or there are NaN values in precipitation column or temperature column
            if df.empty or df["prc_mm"].isnull().any() or df["air_temp_degC"].isnull().any():
                nearest_key = closest_station(y, x, [entry["key"] for entry in quantile_data_list])
                print(f"DataFrame is empty after removing zero precipitation for station at lon: {x}, lat: {y}. Using previous R matrix.")
                # find the R matrix and statistics from the nearest station
                R = next(entry for entry in R_dict_list if entry["key"] == nearest_key)["R"]
                # Use its quantile data and statistics
                statistics = next(entry for entry in statistics_dict_list if entry["key"] == nearest_key)
                variables = next(entry for entry in quantile_data_list if entry["key"] == nearest_key)
            else:
                # continue from here
                R = df.corr()
                print(R)

                print(f"Correlation matrix computed for station at  lon: {x}, lat: {y} in {time.time() - t0:.2f} seconds")

                # Compute statistics dynamically
                variables = {
                     "air_temp_degC": temperature,
                    "prc_mm": prc_mm,
                    "rel_hum_perc": relative_humidity,
                    "swin_wm": radiation,
                    "T_albedo": T_albedo,
                    "T_melting": T_melting
                }
                vars = {

                    "air_temp_degC": temperature.iloc[:,0].values,
                    "prc_mm": prc_mm.iloc[:,0].values,
                    "rel_hum_perc": relative_humidity.iloc[:, 0].values,
                    "swin_wm": radiation.iloc[:,0].values,
                    "T_albedo": T_albedo.iloc[:,0].values,
                    "T_melting": T_melting.iloc[:,0].values

                }

                statistics = {
                    var: {
                        "mean": float(np.nanmean(data)),
                        "std": float(np.nanstd(data)),
                        "min": float(np.nanmin(data)),
                        "max": float(np.nanmax(data)),
                        "median" : float(np.nanmedian(data))
                    }
                    for var, data in vars.items()
                }

            # Do the same for R and statistics
            R_dict = { **{"R": R}, "key": (i,i)}
            R_dict_list.append(R_dict)

            statistics_dict = {**statistics,   "key": (i,i)}
            statistics_dict_list.append(statistics_dict)

            # Assemble quantile data
            quantile_data = {**variables,  "key": (i,i)}
            quantile_data_list.append(quantile_data)

            # Create quantile dictionary using multindex as key
            quantile_dict = {entry["key"]: {
                var: entry[var] for var in variables
            }
                for entry in quantile_data_list
            }
            # Create R dictionary using multindex as key
            R_dict = {entry["key"]: {
                "R": entry["R"]
            }
                for entry in R_dict_list
            }
            # Create statistics dictionary using multindex as key
            statistics_dict = {entry["key"]: {
                "statistics": entry
            }
                for entry in statistics_dict_list
            }
            i += 1

        # Save output files
        output_files = {
            f'R_matrices_{start}_{end}.pkl': R_dict,
            f'statistics_{start}_{end}.pkl': statistics_dict,
            f'quantile_data_{start}_{end}.pkl': quantile_dict
        }

        for filename, data in output_files.items():
            with open(os.path.join(quantile_path, filename), 'wb') as f:
                pickle.dump(data, f)

        print(f"Quantile mapping for time slice {start} to {end} completed in {time.time() - t0:.2f} seconds")


    return

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def R_state_matrix():
    R_dict_list = []
    # Specify the file path
    file_path = '/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/state_data_2018-10-01_2019-09-30.nc'
    # Load the NetCDF file
    data = xr.open_dataset(file_path)
    station_list = pandas.read_csv('/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/STATION_ID_VDA.csv')

    # Replace commas with periods in the 'lat' and 'lon' columns
    station_list['lat'] = station_list['lat'].str.replace(',', '.').astype(float)
    station_list['lon'] = station_list['lon'].str.replace(',', '.').astype(float)

    lat_dem = data.lat.values
    lon_dem = data.lon.values
    # crea una mask per tenere solo le stazioni presenti in station_list, tenendo conto che la matrice meteo ha shape (nt, ny, nx, 6)
    obs_mask = []
    for index, row in station_list.iterrows():
        lat_station = row['lat']
        lon_station = row['lon']
        # round at 6 number after comma
        lat_station = round(lat_station, 6)
        lon_station = round(lon_station, 6)
        # trova l'indice della latitudine più vicina
        ilat = np.abs(data.lat - lat_station).argmin().item()
        ilon = np.abs(data.lon - lon_station).argmin().item()
        obs_mask.append((ilat, ilon))

    i = 0
    for  y, x  in obs_mask:
        # Select the point in the DataArray
        point = data.isel(lat=y,lon=x)
        # Compute covariance across space -> shape (variable, variable)
        # select the variables of interest
        var = ["SWE_W_mm", "SWE_D_mm", "RHO_D_kg_m3", "albedo"]
        # Extract data for the selected variables
        data_df = point[var].to_dataframe().reset_index()
        # Drop unnecessary columns
        data_df = data_df.drop(columns=["lat", "lon", "time"])
        data_df = data_df.dropna()  # Drop rows with NaN values
        data_df = data_df[(data_df["SWE_W_mm"] > -9999) & (data_df["SWE_D_mm"] > -9999) &
                            (data_df["RHO_D_kg_m3"] > -9999) & (data_df["albedo"] > -9999)]

        # if  data_df is empty after removing invalid values, use the previous R matrix. do also in the case that data_df["SWE_W_mm"] is all zeros
        if data_df.empty or data_df["SWE_W_mm"].eq(0).all():
            nearest_key = closest_station(y, x, [entry["key"] for entry in R_dict_list])
            print(
                f"DataFrame is empty after removing zero precipitation for station at lon: {x}, lat: {y}. Using previous R matrix.")
            # find the R matrix and statistics from the nearest station
            R_state_reconstructed = next(entry for entry in R_dict_list if entry["key"] == nearest_key)["R"]

        else:
            # Compute the covariance matrix
            R_state = data_df.cov()
            R_state_corr = data_df.corr()
            # Compute sigma values (square root of the diagonal of the covariance matrix)
            sigma = np.sqrt(np.diag(R_state))

            # Build the reconstructed covariance matrix
            R_state_reconstructed = R_state_corr * (sigma[:, None] * sigma) ** 0.1
            print( R_state_reconstructed)

        #print(R_state_reconstructed)
        R_dict = {**{"R": R_state_reconstructed}, "key": (i, i)}
        R_dict_list.append(R_dict)
        i += 1
    # Create R dictionary using multindex as key
    R_dict = {entry["key"]: {
        "R": entry["R"]
    }
        for entry in R_dict_list
    }
    # Save output files
    with open('/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/quantile_mapping/R_state_2018-10-01_2019-09-30.pkl', 'wb') as f:
        pickle.dump(R_dict, f)
    return
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def downscaled_quantile_mapping():
    t0 = time.time()
    quantile_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/quantile_mapping/"
    os.makedirs(quantile_path, exist_ok=True)
    meteo_path = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/outputs/output.nc"
    dem = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA.tif"
    obs = pandas.read_pickle( "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/observations/time_series_csnow.pkl")
    # Load the weights matrix
    w =pandas.read_pickle( "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/weights.pkl")
    # ----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    # Load the DEM
    dem = rioxarray.open_rasterio(dem, masked=True).squeeze()
    # Ensure the DEM has the correct dimensions
    if 'band' in dem.dims:
        dem = dem.rename({'band': 'z'})
    # Get the dimensions of the DEM
    ny, nx = dem.shape
    print(f"DEM shape: {ny} rows, {nx} columns")
    # Get geographical coordinates from DEM (assume 1D arrays)
    y_coords = dem["y"].values  # e.g., northing
    x_coords = dem["x"].values  # e.g., easting

    # Create 2D meshgrid of coordinate values
    yy_coarse, xx_coarse = np.meshgrid(y_coords, x_coords, indexing='ij')  # 'ij' keeps shape as (ny, nx)

    # Create MultiIndex using raveled coordinate grids
    multi_index_coarse = pandas.MultiIndex.from_arrays(
        [yy_coarse.ravel(), xx_coarse.ravel()],
        names=("y", "x")  # here y and x are the spatial coordinates
    )

    # Create a lookup table from (y, x) coordinate pair → index
    lookup_table = {
        (y_val, x_val): idx
        for idx, (y_val, x_val) in enumerate(multi_index_coarse)
    }
    print(f"lookup table created")
    # ----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    # Open the dataset
    data = xr.open_dataset(meteo_path)

    # Define the time slices
    slice_list = [("2017-10-01", "2018-09-30")]
    quantile_data_list = []
    R_dict_list = []
    statistics_dict_list = []


    for start, end in slice_list:
        t0 = time.time()

        # Select data within the specified time range
        data_slice = data.sel(time=slice(start, end))
        Time = data_slice['time']
        # Define the variable list
        var_list = ['t', 'SW', 'tp', 'q']

        # Initialize a dictionary to store mapped results
        mapped_results = {}
        n_clusters =1500

        for var in var_list:
            data = data_slice[var].values  # shape: (n_clusters, n_time)
            n_time = data.shape[1]
            mapped = np.full((ny, nx, n_time), np.nan)
            for i in range(ny):  # x-direction
                for j in range(nx):  # y-direction
                    if (i, j) in w:
                        wij = w[i, j]  # dict {cluster_id: freq}
                        keys = list(wij.keys())

                        # Filter only valid cluster IDs
                        valid_keys = [k for k in keys if 0 <= k < n_clusters]

                        if len(valid_keys) > 0:
                            weights = np.array([wij[k] for k in valid_keys])
                            values = np.array([data[k, :] for k in valid_keys])  # shape: (n_valid, n_time)

                            # Weighted average across clusters → shape: (n_time,)
                            mapped[i, j, :] = np.average(values, axis=0, weights=weights)


                mapped_results[var] = (("y", "x", "time"), mapped)


         # Combine mapped variables into a single Dataset
        ds = xr.Dataset(mapped_results,coords={"y":y_coords.astype(np.float64),"x":x_coords.astype(np.float64), "time": Time})
        print(f"ERA5 variables mapped in {time.time() - t0:.2f} seconds")
        # ----------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------
        t0 = time.time()

        for lon, lat in obs.keys():
            # Select the point in the DataArray
            point_data = ds.sel(y=lat,x=lon, method="nearest", tolerance=200)
            matched_lat = float(point_data.y.values)
            matched_lon = float(point_data.x.values)
            idx = lookup_table.get((matched_lat, matched_lon), None)


            print(f"Values selected for station at lon: {lon}, lat: {lat}")
             # Convert Kelvin to Celsius
            temperature = point_data['t'] - 273.15
            radiation = point_data['SW'].where( point_data['SW'] <= 5000, 0)
            precipitation = point_data['tp']
            point_data['RH'] = compute_relative_humidity(point_data['t'],  point_data["q"], precipitation)
            relative_humidity = point_data['RH']
            # Convert the smoothed arrays back to xarray.DataArray
            T_albedo = xr.DataArray(
                uniform_filter1d(temperature.values, size=24, axis=0, mode='nearest'),
                coords=temperature.coords,
                dims=temperature.dims
            )
            T_melting = xr.DataArray(
                uniform_filter1d(temperature.values, size=48, axis=0, mode='nearest'),
                coords=temperature.coords,
                dims=temperature.dims
            )

            # Assign a name to the DataArray
            T_albedo.name = 'T_albedo'
            T_melting.name = 'T_melting'

            print( f"Variables selected for station at lon: {lon}, lat: {lat} in {time.time() - t0:.2f} seconds")

            # convert each variable into a pandas DataFrame
            prc_mm = pandas.DataFrame(point_data['tp'].values, columns=['prc_mm'])
            temperature = pandas.DataFrame(temperature.values, columns=['air_temp_degC'])
            radiation = pandas.DataFrame(radiation.values, columns=['swin_wm-2'])
            relative_humidity = pandas.DataFrame(relative_humidity.values, columns=['rel_hum_perc'])
            T_albedo = pandas.DataFrame(T_albedo.values, columns=['T_albedo'])
            T_melting= pandas.DataFrame(T_melting.values, columns=['T_melting'])

            # Apply quantile mapping
            prc_mm = quantile_mapping(prc_mm, 'prc_mm')
            temperature = quantile_mapping(temperature, 'air_temp_degC')
            radiation = quantile_mapping(radiation, 'swin_wm-2')
            relative_humidity = quantile_mapping(relative_humidity, 'rel_hum_perc')
            T_albedo = quantile_mapping(T_albedo, 'T_albedo')
            T_melting = quantile_mapping(T_melting, 'T_melting')

            print(f"quantile mapping computed for station at lon: {lon}, lat: {lat} in {time.time() - t0:.2f} seconds")

            df = pandas.DataFrame(columns=["prc_mm_rif", "prc_mm", "swin_wm", "air_temp_degC", "rel_hum_perc",
                                           "T_albedo", "T_melting"])

            df["prc_mm_rif"] = prc_mm["prc_mm"].astype(np.float64)
            df["prc_mm"] = prc_mm["gauss_quant"].astype(np.float64)
            df["air_temp_degC"] = temperature['gauss_quant'].astype(np.float64)
            df["rel_hum_perc"] = relative_humidity["gauss_quant"].astype(np.float64)
            df["swin_wm"] = radiation["gauss_quant"].astype(np.float64)
            df["T_albedo"] = T_albedo["gauss_quant"].astype(np.float64)
            df["T_melting"] = T_melting["gauss_quant"].astype(np.float64)

            # remove row with precipitation equal to 0
            #df = df.drop(df[df["prc_mm_rif"] == 0].index)
            df = df.drop("prc_mm_rif", axis=1)

            # continue from here
            R = df.corr()
            print(R)
            print(f"Correlation matrix computed for station at  lon: {lon}, lat: {lat} in {time.time() - t0:.2f} seconds")


            # Compute statistics dynamically
            variables = {
                "prc_mm": prc_mm,
                "swin_wm": radiation,
                "air_temp_degC": temperature,
                "rel_hum_perc": relative_humidity,
                "T_albedo": T_albedo,
                "T_melting": T_melting
            }
            vars = {
                "prc_mm": prc_mm.iloc[:, 0].values,
                "swin_wm": radiation.iloc[:, 0].values,
                "air_temp_degC": temperature.iloc[:, 0].values,
                "rel_hum_perc": relative_humidity.iloc[:, 0].values,
                "T_albedo": T_albedo.iloc[:, 0].values,
                "T_melting": T_melting.iloc[:, 0].values

            }

            statistics = {
                var: {
                    "mean": float(np.nanmean(data)),
                    "std": float(np.nanstd(data)),
                    "min": float(np.nanmin(data)),
                    "max": float(np.nanmax(data)),
                }
                for var, data in vars.items()
            }

            # Do the same for R and statistics
            R_dict =  { **{"R": R.values}, "multindex": idx}
            R_dict_list.append(R_dict)

            statistics_dict = {**statistics , "multindex": idx}
            statistics_dict_list.append(statistics_dict)

            # Assemble quantile data
            quantile_data = {**variables, "multindex": idx}
            quantile_data_list.append(quantile_data)

            # Create quantile dictionary using multindex as key
            quantile_dict = {entry["multindex"]: {
                var: entry[var] for var in variables
            }
                for entry in quantile_data_list
            }
            # Create R dictionary using multindex as key
            R_dict = {entry["multindex"]: {
                "R": entry["R"]
            }
                for entry in R_dict_list
            }
            # Create statistics dictionary using multindex as key
            statistics_dict = {entry["multindex"]: {
                "statistics": entry
            }
                for entry in statistics_dict_list
            }


        # Save output files
        output_files = {
            f'R_matrices_{start}_{end}.pkl': R_dict,
            f'statistics_{start}_{end}.pkl': statistics_dict,
            f'quantile_data_{start}_{end}.pkl': quantile_dict
        }

        for filename, data in output_files.items():
            with open(os.path.join(quantile_path, filename), 'wb') as f:
                pickle.dump(data, f)

        print(f"Quantile mapping for time slice {start} to {end} completed in {time.time() - t0:.2f} seconds")

    return
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
def R_state_matrix_downscaled():
    R_state_list = []
    # Specify the file path
    file_path = '/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/state_data_2018-10-01_2019-09-30.nc'
    # Load the NetCDF file
    data = xr.open_dataset(file_path)

    obs = pandas.read_pickle(
        "/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/dict.pkl")

    for lon, lat in obs.keys():
        # Select the point in the DataArray
        point = data.sel(lat=lat,lon=lon, method='nearest')
        print(f"Values selected for station at lon: {lon}, lat: {lat}")
        # Compute covariance across space -> shape (variable, variable)
        # select the variables of interest
        var = ["SWE_W_mm", "SWE_D_mm", "RHO_D_kg_m3", "albedo"]
        # Extract data for the selected variables
        data_df = point[var].to_dataframe().reset_index()
        # Drop unnecessary columns
        data_df = data_df.drop(columns=["lat", "lon", "time"])

        # Handle NaN values by dropping rows with NaN or filling them
        if data_df.isnull().values.any():
            print(f"NaN values detected for station at lon: {lon}, lat: {lat}. Handling missing data...")
            data_df = data_df.dropna()  # Drop rows with NaN values
            # Alternatively, you can fill NaN values with the mean of each column:
            # data_df = data_df.fillna(data_df.mean())

        # Compute the covariance matrix
       # R_state = data_df.cov()
        R_state_corr = data_df.corr()

        print(R_state_corr)

        # Compute sigma values (square root of the diagonal of the covariance matrix)
        #sigma = np.sqrt(np.diag(R_state))

        # Build the reconstructed covariance matrix
       # R_state_reconstructed = R_state_corr * (sigma[:, None] * sigma) ** 0.1

        #print(R_state_reconstructed)

        R_dict = {**{"R": R_state_corr.values}, "key": (lon, lat)}
        R_state_list.append(R_dict)
    # Create R dictionary using multindex as key
    R_dict = {entry["key"]: {
        "R": entry["R"]
    }
        for entry in R_state_list
    }

    return



if __name__ == "__main__":
    #downscaled_quantile_mapping()
    #stochastic_process()
    dataset_quantile_mapping()
    #R_state_matrix()

