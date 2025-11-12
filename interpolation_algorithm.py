import os
import geopandas as gpd
from shapely.geometry import shape
import warnings
import rasterio
import logging
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
import rioxarray as rxr
import xarray as xr
import numpy as np
import pickle
import rasterio.features
from pyproj import Transformer
from collections import Counter, defaultdict


def process_variable(var_name, ds_downscaled, cluster_ids_flat, positions_dict, lat, lon, time_len):
    data_var = ds_downscaled[var_name].values  # (n_points, time)
    remapped_array = np.full((cluster_ids_flat.size, time_len), np.nan)

    for cid, pos in positions_dict.items():
        remapped_array[pos, :] = data_var[cid, :]

    remapped_array = remapped_array.T.reshape((time_len, lat, lon)).transpose(1, 2, 0)
    return var_name, remapped_array

def convert_xy_to_latlon(x, y, crs_proj):
    """
        Convert 1D projected x, y coordinates to 1D geographic lat/lon.
        Assumes a regular grid where x varies along columns and y along rows.
        """
    transformer = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)

    # Convert x (columns) → lon (keep y fixed)
    lon_1d, _ = transformer.transform(x, np.full_like(x, y[0]))

    # Convert y (rows) → lat (keep x fixed)
    _, lat_1d = transformer.transform(np.full_like(y, x[0]), y)

    return lat_1d, lon_1d

def weights(cluster):

    # cluster è una matrice 1283x1773 con I CLUSTER POINT DI TOPOPYSCALE CHE VOGLIO MAPPARE. hO UNA DIFFERENZA DI RISOLUZIONE.
    # CLUSTER  ha la  stessa copertura del DEM fine , che copre la vda e altre zone intorno.
    # voglio tagliarla solo sulla vda cosi da riscalare poi sul dem coarse 206*446 che considera solo la vda.
     # per farlo ho pensato di settare a nan i valori che non sono contenuti come coordinate inel dem fine cut ( vda +zona nera buffer)
     # e poi settare a nan i valori nella zone nera di buffer ovvere dove binaty mask ==1.


      # APERTURA FILE

    dem_coarse_path = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA_1km.tif"
    dem_fine_path = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/VDA_PADDED_utm.tif"

    dem_coarse = rxr.open_rasterio(dem_coarse_path, masked=True).squeeze()
    dem_fine = rxr.open_rasterio(dem_fine_path, masked=True).squeeze()

    cluster_ids = cluster['point_ind'].values
    # === Caricamento cluster_ids ===
    cluster_ids = cluster['point_ind']

    # === Assegna CRS e proietta a EPSG:4326 ===
    cluster_ids.rio.write_crs(dem_fine.rio.crs, inplace=True)
    cluster_ids = cluster_ids.rio.reproject("EPSG:4326")

    # === Clip: crea bounding box in EPSG:4326 ===
    bbox_coords = [
        [7.9446622, 45.461001],
        [7.9446622, 45.9929342],
        [6.793001, 45.9929342],
        [6.793001, 45.461001],
        [7.9446622, 45.461001]
    ]
    polygon = shape({"type": "Polygon", "coordinates": [bbox_coords]})
    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")

    # === Clip del dataset cluster_ids proiettato ===
    cluster_ids_cropped = cluster_ids.rio.clip(gdf.geometry, gdf.crs, drop=True)

    ny_coarse = dem_coarse['y'].shape[0]
    nx_coarse = dem_coarse['x'].shape[0]
    # Convert cluster_da_valid to numpy
    cluster_array = cluster_ids_cropped.values

    w_dict = {}

    # Calculate coarsening factor along lat/lon by comparing lengths
    lat_factor = int(np.round(len(cluster_ids_cropped['y']) / len(dem_coarse['y'])))
    lon_factor = int(np.round(len(cluster_ids_cropped['x']) / len(dem_coarse['x'])))

    for i in range(ny_coarse):
        for j in range(nx_coarse):
            # Define bounds of the current window in fine-resolution indices
            lat_start = i * lat_factor
            lat_end = (i + 1) * lat_factor
            lon_start = j * lon_factor
            lon_end = (j + 1) * lon_factor

            # Handle edges (avoid out-of-bounds)
            if lat_end > cluster_array.shape[0] or lon_end > cluster_array.shape[1]:
                continue

            # Extract subwindow
            window = cluster_array[ lat_start:lat_end, lon_start:lon_end]

            # Flatten and remove NaNs
            window_flat = window.flatten()
            window_flat = window_flat[~np.isnan(window_flat)]

            if window_flat.size == 0:
                continue

            counts = Counter(window_flat)
            total = window_flat.size
            freqs = {int(k): v / total for k, v in counts.items()}

            w_dict[(i, j)] = freqs

    output_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/weights.pkl"

    with open(output_path, 'wb') as f:
        pickle.dump(w_dict, f)
    return
#*------------------------------------------------------------------------------------------------------------------------------------------------
#*------------------------------------------------------------------------------------------------------------------------------------------------
#*------------------------------------------------------------------------------------------------------------------------------------------------
#*------------------------------------------------------------------------------------------------------------------------------------------------

def load_rasters(base_path, file_dict):
    rasters = {}
    transform, shape = None, None
    for key, fname in file_dict.items():
        raster, transform, shape = read_and_normalize_raster(Path(base_path) / fname)
        rasters[key] = raster
    return rasters, transform, shape

def reshape_param(param_name, data, shape_latlon):
    values = data[param_name].values
    if len(values) != np.prod(shape_latlon):
        raise ValueError(f"Cannot reshape '{param_name}' to shape {shape_latlon}")
    return values.reshape(shape_latlon)
def predict_and_fill_calibrated(base_path, raster_files, gp_models, point_coords, true_values_dict, nodata_val=-9999):
    """
    Predict over full domain and insert observed values at known points.

    Parameters:
    - base_path: folder containing rasters
    - raster_files: dict of terrain rasters
    - gp_models: dict with keys 'mrad', 'mr', 'window_melting'
    - point_coords: list of (lon, lat) used in calibration
    - true_values_dict: dict of arrays: {param_name: [true values at point_coords]}
    - nodata_val: fill value in rasters
    """
    all_features = []
    with rasterio.open(Path(base_path) / list(raster_files.values())[0]) as ref:
        rows, cols = ref.height, ref.width
        transform = ref.transform

    # Load and normalize all rasters
    for key, filename in raster_files.items():
        with rasterio.open(Path(base_path) / filename) as src:
            data = src.read(1).astype(float)
            data[data == nodata_val] = np.nan
            min_val, max_val = np.nanmin(data), np.nanmax(data)
            normalized = (data - min_val) / (max_val - min_val)
            all_features.append(normalized)

    # Stack and flatten
    feature_stack = np.stack(all_features, axis=0)  # (n_features, rows, cols)
    n_features = feature_stack.shape[0]
    features_flat = feature_stack.reshape(n_features, -1).T  # (n_pixels, n_features)

    total_pixels = rows * cols
    all_indices = np.arange(total_pixels)

    # Get calibration indices
    with rasterio.open(Path(base_path) / list(raster_files.values())[0]) as src:
        calib_indices = [src.index(lon, lat) for lon, lat in point_coords]
        calib_flat_indices = np.array([r * cols + c for r, c in calib_indices])

    # Find prediction indices
    predict_flat_indices = np.setdiff1d(all_indices, calib_flat_indices)
    features_predict = features_flat[predict_flat_indices]
    valid_mask = ~np.isnan(features_predict).any(axis=1)

    predict_flat_valid = predict_flat_indices[valid_mask]
    X_predict_valid = features_predict[valid_mask]

    # Predict and fill
    predictions_full = {}

    for param_name, gp_model in gp_models.items():
        full_array = np.full(total_pixels, np.nan)

        # GP predictions
        y_pred = gp_model.predict(X_predict_valid)
        full_array[predict_flat_valid] = y_pred

        # Fill in calibration points
        true_vals = true_values_dict[param_name]
        full_array[calib_flat_indices] = true_vals

        predictions_full[param_name] = full_array.reshape(rows, cols)

    return predictions_full

def read_and_normalize_raster(file_path, points, nodata_val=-9999):
    """
    Reads raster values at given (lon, lat) points and normalizes them to 0–1.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(float)
        data[data == nodata_val] = np.nan

        selected_values = []
        for lon, lat in points:
            try:
                row, col = src.index(lon, lat)
                value = data[row, col]
                selected_values.append(value)
            except IndexError:
                selected_values.append(np.nan)

        selected_values = np.array(selected_values)
        norm = (selected_values - np.nanmin(selected_values)) / (np.nanmax(selected_values) - np.nanmin(selected_values))
        return norm


def evaluate_predictions(y_true, y_pred, name='parameter'):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)

    print(f"\nPerformance for {name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Bias: {bias:.4f}")
    print(f"  R² Score: {r2:.4f}")


#*------------------------------------------------------------------------------------------------------------------------------------------------
#*------------------------------------------------------------------------------------------------------------------------------------------------

def parameter_cal():
    # === Setup Logging ===
    logging.basicConfig(level=logging.INFO)

    # === File paths ===
    base_path = Path('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES')
    param_csv = Path('/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/calibration/results_best.csv')
    output_nc = Path('/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/interpolated_parameters.nc')
    points_file = Path('/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/observations/time_series_csnow.pkl')

    # === File dictionary ===
    raster_files = {
        'elevation': 'DEM_VDA.tif',
        'slope': 'SLOPE_VDA.tif',
        'aspect': 'ASPECT_VDA.tif',
        'tpi': 'TPI_VDA.tif',
        'csmod': 'CSMOD_VDA_reprojected.tif'
    }

    points_data = pd.read_pickle(points_file)
    selected_data = []

    # Extract points (lon, lat) from keys of points_data
    point_coords = [(key[0], key[1]) for key in points_data.keys()]

    # Read and normalize raster values at point locations
    for key, raster_file in raster_files.items():
        file_path = base_path / raster_file
        normalized_values = read_and_normalize_raster(file_path, point_coords)
        selected_data.append(normalized_values)

    # Stack features into NumPy array (features, samples)
    features_raw = np.array(selected_data)

    # === Read calibration CSV and parse SAMPLE column ===
    data = pd.read_csv(param_csv)
    if 'SAMPLE' not in data.columns:
        raise ValueError("'SAMPLE' column not found in CSV.")

    sample_dicts = data['SAMPLE']
    data['mrad'] = sample_dicts.apply(lambda d: d.get('mrad', np.nan))
    data['mr'] = sample_dicts.apply(lambda d: d.get('mr', np.nan))
    data['window_melting'] = sample_dicts.apply(lambda d: d.get('window_melting', np.nan))
    data = data.drop(columns=['SAMPLE', 'F_COST'], errors='ignore')

    # === Train/test split ===

    # Convert calibration parameters to arrays
    mrad_vals = data['mrad'].values
    mr_vals = data['mr'].values
    melting_vals = data['window_melting'].values

    # Remove samples (columns) with any NaNs in features or targets
    valid_mask = ~np.isnan(features_raw).any(axis=0) & ~np.isnan(mrad_vals) & ~np.isnan(mr_vals) & ~np.isnan(
        melting_vals)

    features_clean = features_raw[:, valid_mask]
    mrad_clean = mrad_vals[valid_mask]
    mr_clean = mr_vals[valid_mask]
    melting_clean = melting_vals[valid_mask]

    # Split into training and prediction sets
    num_samples = features_clean.shape[1]
    train_indices = np.random.choice(num_samples, size=int(0.8 * num_samples), replace=False)
    predict_indices = np.setdiff1d(np.arange(num_samples), train_indices)

    features_train = features_clean[:, train_indices].T  # Transpose for sklearn (n_samples, n_features)
    features_predict = features_clean[:, predict_indices].T

    mrad_train = mrad_clean[train_indices]
    mr_train = mr_clean[train_indices]
    melting_train = melting_clean[train_indices]

    # Ground truth for testing
    mrad_true = mrad_clean[predict_indices]
    mr_true = mr_clean[predict_indices]
    melting_true = melting_clean[predict_indices]

    # === Train and Predict GP model separately for each parameter ===
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0,
                                                                                  noise_level_bounds=(1e-5, 1e1))

    # MRAD
    logging.info("Training GP model for mrad")
    gp_model_mrad = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
    gp_model_mrad.fit(features_train, mrad_train)
    predicted_mrad = gp_model_mrad.predict(features_predict)
    evaluate_predictions(mrad_true, predicted_mrad, name='mrad')

    # MR
    logging.info("Training GP model for mr")
    gp_model_mr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
    gp_model_mr.fit(features_train, mr_train)
    predicted_mr = gp_model_mr.predict(features_predict)
    evaluate_predictions(mr_true, predicted_mr, name='mr')

    # WINDOW MELTING
    logging.info("Training GP model for window_melting")
    gp_model_window = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
    gp_model_window.fit(features_train, melting_train)
    predicted_window = gp_model_window.predict(features_predict)
    evaluate_predictions(melting_true, predicted_window, name='window_melting')

    gp_models = {
        'mrad': gp_model_mrad,
        'mr': gp_model_mr,
        'window_melting': gp_model_window
    }

    true_values_dict = {
        'mrad': mrad_vals,
        'mr': mr_vals,
        'window_melting': melting_vals
    }

    predicted_fields = predict_and_fill_calibrated(base_path, raster_files, gp_models, point_coords, true_values_dict)

    ds = xr.Dataset({
        'mrad': (['y', 'x'], predicted_fields['mrad']),
        'mr': (['y', 'x'], predicted_fields['mr']),
        'window_melting': (['y', 'x'], predicted_fields['window_melting']),
    })

    ds.to_netcdf(output_nc)
    logging.info(f"Saved full parameter fields to {output_nc}")

#*------------------------------------------------------------------------------------------------------------------------------------------------
#*------------------------------------------------------------------------------------------------------------------------------------------------
def interpolate_correction(corrections):
    # === Setup Logging ===
    logging.basicConfig(level=logging.INFO)

    # === File paths ===
    base_path = Path('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES')
    param_csv = Path('/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/calibration/results_best.csv')
    output_nc = Path('/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/interpolated_parameters.nc')
    points_file = Path('/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/observations/time_series_csnow.pkl')

    # === File dictionary ===
    raster_files = {
        'elevation': 'DEM_VDA.tif',
        'slope': 'SLOPE_VDA.tif',
        'aspect': 'ASPECT_VDA.tif',
        'tpi': 'TPI_VDA.tif',
        'csmod': 'CSMOD_VDA_reprojected.tif'
    }

    points_data = pd.read_pickle(points_file)
    selected_data = []

    # Extract points (lon, lat) from keys of points_data
    point_coords = [(key[0], key[1]) for key in points_data.keys()]

    # Read and normalize raster values at point locations
    for key, raster_file in raster_files.items():
        file_path = base_path / raster_file
        normalized_values = read_and_normalize_raster(file_path, point_coords)
        selected_data.append(normalized_values)

    # Stack features into NumPy array (features, samples)
    features_raw = np.array(selected_data)

    # === GP regression for corrections ===
    # write a gp regression for the corrections
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0,
                                                                                  noise_level_bounds=(1e-5, 1e1))
    gp_model_correction = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
    # Assuming you have a target variable for corrections
    # For example, let's say you have a target variable 'correction_target'
    correction_target = corrections.flatten()  # Flatten if needed
    # Fit the GP model
    gp_model_correction.fit(features_raw.T, correction_target)
    return gp_model_correction, features_raw

#*------------------------------------------------------------------------------------------------------------------------------------------------
#*------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    default_n_threads =20  # Reduce the number of threads
    os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
    os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
    os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
    cluster =xr.open_dataset("/home/idrologia/PhD_GiuliaBlandini/S3M_2D/outputs/ds_param.nc")
    weights (cluster)
   # sky_view_factor()
    # --------------------------------------------------------------

