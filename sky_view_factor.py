import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import shape, mapping
import json
from pyproj import Transformer

import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import box, shape
import json
import matplotlib.pyplot as plt
import numpy as np

# === Percorsi file ===
svf_nc = '/home/idrologia/PhD_GiuliaBlandini/S3M_2D/outputs/ds_param.nc'
solar_nc = '/home/idrologia/PhD_GiuliaBlandini/S3M_2D/outputs/ds_solar.nc'
dem_coarse_path = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA_1km.tif"
dem_fine_path = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/VDA_PADDED_utm.tif"
dem_fine_cut_path = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/dem_cropped.tif"
out_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/masked_svf.nc"

# === Caricamento raster ===
dem_fine = rxr.open_rasterio(dem_fine_path, masked=True).squeeze()
dem_fine_cut = rxr.open_rasterio(dem_fine_cut_path, masked=True).squeeze()
dem_coarse = rxr.open_rasterio(dem_coarse_path, masked=True).squeeze()

# === Caricamento SVF da NetCDF ===
ds = xr.open_dataset(svf_nc)
svf = ds['svf']

solar = xr.open_dataset(solar_nc)


# === Assegna CRS e proietta a EPSG:4326 ===
svf.rio.write_crs(dem_fine.rio.crs, inplace=True)
svf = svf.rio.reproject("EPSG:4326")

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

# === Clip del dataset SVF proiettato ===
svf_cropped = svf.rio.clip(gdf.geometry, gdf.crs, drop=True)

# === Plot (opzionale) ===
plt.figure(figsize=(8, 4))
svf_cropped.plot(cmap="viridis")
plt.title("SVF Cropped")
plt.tight_layout()
plt.savefig("svf_cropped.png", dpi=150)

# === Calcolo fattore coarsificazione (assumendo pixel quadrati) ===

lat_factor = int(np.round(len(svf_cropped ['y']) / len(dem_coarse['y'])))
lon_factor = int(np.round(len(svf_cropped ['x'])/ len(dem_coarse['x'])))

# === Coarsifica lo SVF ritagliato ===
coarse_svf = svf_cropped.coarsen(x=lon_factor, y=lat_factor, boundary='trim').mean()
# === Controllo finale ===
print("coarse_svf_aligned shape:", coarse_svf.shape)
print("dem_coarse shape:", dem_coarse.shape)

coarse_svf= coarse_svf.sel(x=dem_coarse['x'], y=dem_coarse['y'], method='nearest')


# save coarse_svf to NetCDF
coarse_svf.to_netcdf(out_path, mode='w', format='NETCDF4')

# === Visualizzazione
# === Visualizzazione
plt.figure(figsize=(10, 8))
im = plt.pcolormesh(coarse_svf['x'], coarse_svf['y'], coarse_svf.values, cmap='viridis', shading='auto')
# Contour del DEM
plt.contour(dem_coarse['x'], dem_coarse['y'], dem_coarse.values, colors='black', linewidths=0.5)
# Titolo e assi
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("SVF (color) and DEM contours (black)")
plt.grid(True)
plt.tight_layout()
plt.savefig("svf_dem_contour.png", dpi=150)

# Process all variables in the SVF dataset
for var_name in ds.data_vars:
    print(f"Processing variable: {var_name}")
    variable= ds[var_name]

    # Assign CRS and reproject
    variable.rio.write_crs(dem_fine.rio.crs, inplace=True)
    variable = variable.rio.reproject("EPSG:4326")

    # Clip the variable
    var_cropped = variable.rio.clip(gdf.geometry, gdf.crs, drop=True)

    # Coarsen the variable
    lat_factor = int(np.round(len(var_cropped['y']) / len(dem_coarse['y'])))
    lon_factor = int(np.round(len(var_cropped['x']) / len(dem_coarse['x'])))
    coarse_var = var_cropped.coarsen(x=lon_factor, y=lat_factor, boundary='trim').mean()

    # Align with DEM grid
    coarse_var = coarse_var.sel(x=dem_coarse['x'], y=dem_coarse['y'], method='nearest')

    # Save to NetCDF
    output_path = out_path.replace("masked_svf.nc", f"masked_{var_name}.nc")
    coarse_var.to_netcdf(output_path, mode='w', format='NETCDF4')

    # Visualization
    plt.figure(figsize=(10, 8))
    im = plt.imshow(coarse_var.values, cmap='Blues')
    plt.contour(dem_coarse['x'], dem_coarse['y'], dem_coarse.values, colors='black', linewidths=0.5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f" {var_name} (color) and DEM contours (black)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/{var_name}_dem_contour.png", dpi=150)
    plt.close()

# Process all variables in the solar dataset
for var_name in solar.data_vars:
    print(f"Processing variable: {var_name}")
    var = solar[var_name]

    # Assign CRS and reproject
    var.rio.write_crs(dem_fine.rio.crs, inplace=True)
    var = var.rio.reproject("EPSG:4326")

    # Clip the variable
    var_cropped = var.rio.clip(gdf.geometry, gdf.crs, drop=True)

    # Coarsen the variable
    lat_factor = int(np.round(len(var_cropped['y']) / len(dem_coarse['y'])))
    lon_factor = int(np.round(len(var_cropped['x']) / len(dem_coarse['x'])))
    coarse_var = var_cropped.coarsen(x=lon_factor, y=lat_factor, boundary='trim').mean()

    # Align with DEM grid
    coarse_var = coarse_var.sel(x=dem_coarse['x'], y=dem_coarse['y'], method='nearest')

    # Save to NetCDF
    output_path = out_path.replace("masked_svf.nc", f"masked_{var_name}.nc")
    coarse_var.to_netcdf(output_path, mode='w', format='NETCDF4')

    # Visualization
    plt.figure(figsize=(10, 8))
    im = plt.imshow(coarse_var['x'], coarse_var['y'], coarse_var.values, cmap='Blues', shading='auto')
    plt.contour(dem_coarse['x'], dem_coarse['y'], dem_coarse.values, colors='black', linewidths=0.5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{var_name} (color) and DEM contours (black)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f" /home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/{var_name}_dem_contour.png", dpi=150)
    plt.close()
