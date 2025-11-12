import os
import geopandas as gpd
import rioxarray as rxr
from rasterio import features
import xarray as xr
import pickle


# Load shapefile and DEM
scratchM = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/dem/confini/"
shapefile = gpd.read_file(os.path.join(scratchM, 'vda.shp'))
shapefile = shapefile.set_crs("EPSG:4326") if shapefile.crs is None else shapefile.to_crs(epsg=4326)

dem_low = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA.tif"
dem = rxr.open_rasterio(dem_low, engine='rasterio').sel(band=1).drop_vars('band')

# Reproject shapefile to DEM CRS
shapefile = shapefile.to_crs(dem.rio.crs)

# Create a mask: 1 inside shape, 0 outside
mask_array = features.rasterize(
    [(geom, 1) for geom in shapefile.geometry],
    out_shape=dem.shape,
    transform=dem.rio.transform(),
    fill=0,
    dtype='uint8'
)

# Convert to DataArray and align to DEM
mask = xr.DataArray(mask_array, coords=dem.coords, dims=dem.dims)

# Apply the mask: set NaN outside the shape
dem_masked = dem.where(mask == 1)

# Convert mask to a NumPy array of booleans (True inside the shape, False outside)
binary_mask = (mask == 1).values

mask_data = {
    "mask": binary_mask,
    "x": mask.x.values,
    "y": mask.y.values
}

with open("/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/mask_vda_with_coords.pkl", "wb") as f:
    pickle.dump(mask_data, f)

# Optional: Save masked DEM
# dem_masked.rio.to_raster("masked_dem.tif")
""" 
import matplotlib.pyplot as plt

# Plot the masked DEM
fig, ax = plt.subplots(figsize=(10, 8))
dem_masked.plot(ax=ax, cmap='terrain')
shapefile.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
ax.set_title("Masked DEM within VDA Boundary")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig("/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/mask.png")
"""



