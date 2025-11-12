from rasterio.warp import reproject, Resampling
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import xarray as xr

# Paths to the .tif files
elevation_tif = '/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/DEM_VDA.tif'
slope_tif = '/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/SLOPE_VDA.tif'
aspect_tif = '/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/ASPECT_VDA.tif'
tpi_tif = '/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/TPI_VDA.tif'
# Define the input and output file paths
input_tif = '/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/DEM_VDA.tif'
CSMOD = '/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/CSMOD_VDA.tif'

# open all the tif files and store the data in a matrix at the correct coordinates
with rasterio.open(input_tif) as src:
    elevation = src.read(1)
    transform = src.transform
    profile = src.profile
    # Get the coordinates of the raster
    height, width = elevation.shape
    x = np.arange(transform[2], transform[2] + width * transform[0], transform[0])
    y = np.arange(transform[5], transform[5] + height * transform[4], transform[4])
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    elevation = elevation.flatten()
    elevation_plot = elevation
    # Set the profile for the output GeoTIFF


# do the same for the other rasters
with rasterio.open(slope_tif) as src:
    slope = src.read(1)
    slope = slope.flatten()
    slope[slope <0] = np.nan  # Set -9999 to NaN

with rasterio.open(aspect_tif) as src:
    aspect = src.read(1)
    aspect = aspect.flatten()
    aspect[aspect<0] = np.nan  # Set -9999 to

with rasterio.open(tpi_tif) as src:
    tpi = src.read(1)
    tpi = tpi.flatten()
    tpi[tpi <0] = np.nan  # Set -9999 to NaN

with rasterio.open(CSMOD) as src:
    csmod_data = src.read(1)
    csmod_transform = src.transform
    csmod_crs = src.crs

# Reproject the CSMOD data to match the DEM
with rasterio.open(elevation_tif) as dem_src:
    dem_transform = dem_src.transform
    dem_crs = dem_src.crs
    dem_shape = dem_src.shape

    reprojected_csmod = np.empty(dem_shape, dtype='float32')
    reproject(
        source=csmod_data,
        destination=reprojected_csmod,
        src_transform=csmod_transform,
        src_crs=csmod_crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=Resampling.nearest
    )
    # save as a new geotiff
    profile = dem_src.profile
    profile.update({
        'dtype': 'float32',
        'count': 1,
        'crs': dem_crs,
        'transform': dem_transform
    })
    with rasterio.open('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/CSMOD_VDA_reprojected.tif', 'w', **profile) as dst:
        dst.write(reprojected_csmod, 1)

# Plot the csmod data
plt.figure(figsize=(10, 6))
plt.imshow(reprojected_csmod, cmap='viridis')
plt.colorbar(label='CSMOD Value')
plt.title('CSMOD Data')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.savefig('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/csmod_plot.png')

# Flatten the reprojected data and handle nodata values
reprojected_csmod = reprojected_csmod.flatten()


# Set -9999 to NaN for elevation
elevation[elevation == -9999] = np.nan  # Set -9999 to NaN
# Set -9999 to NaN for slope
slope[slope == -9999] = np.nan  # Set -9999 to NaN
# Set -9999 to NaN for aspect
aspect[aspect == -9999] = np.nan  # Set -9999 to NaN
# Set -9999 to NaN for tpi
tpi[tpi == -9999] = np.nan  # Set -9999 to NaN
# Set -9999 to NaN for csmod
reprojected_csmod[reprojected_csmod == -9999] = np.nan  # Set -9999 to NaN

# using min and max values to normalize the data
elevation_min = np.nanmin(elevation)
elevation_max = np.nanmax(elevation)
slope_min = np.nanmin(slope)
slope_max = np.nanmax(slope)
aspect_min = np.nanmin(aspect)
aspect_max = np.nanmax(aspect)
tpi_min = np.nanmin(tpi)
tpi_max = np.nanmax(tpi)
csmod_min = np.nanmin(reprojected_csmod)
csmod_max = np.nanmax(reprojected_csmod)
# Normalize the data
elevation = (elevation - elevation_min) / (elevation_max - elevation_min)
slope = (slope - slope_min) / (slope_max - slope_min)
aspect = (aspect - aspect_min) / (aspect_max - aspect_min)
tpi = (tpi - tpi_min) / (tpi_max - tpi_min)
reprojected_csmod = (reprojected_csmod - csmod_min) / (csmod_max - csmod_min)

slope_sin =np.sin(slope)

aspect_cos= np.cos(aspect)


# Combine data into a feature matrix
coordinates = np.vstack((x, y)).T
features = np.vstack((elevation, tpi,reprojected_csmod)).T

#features = np.vstack((slope_sin, aspect_cos)).T

# set -9999 to NaN
features[features == -9999] = np.nan
# Remove rows with NaN values (if any)
valid_mask = ~np.isnan(features).any(axis=1)
coordinates = coordinates[valid_mask]
features = features[valid_mask]

# Perform KMeans clustering on the feature matrix
optimal_n_clusters =50
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# Add cluster labels to the coordinates
data_with_clusters = pd.DataFrame({
    'x': coordinates[:, 0],
    'y': coordinates[:, 1],
    'cluster': clusters
})

# Calculate the centroid for each cluster
centroids = data_with_clusters.groupby('cluster')[['x', 'y']].mean().reset_index()

# Save the centroids to a CSV file
centroids.to_csv('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/centroids_kmeans.csv', index=False)
# plot centroids

import matplotlib.pyplot as plt

# Calculate centroids
centroids = data_with_clusters.groupby('cluster')[['x', 'y']].mean().reset_index()

# Save centroids to a CSV file
centroids.to_csv('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/centroids_kmeans.csv', index=False)



# Plot centroids
plt.figure(figsize=(10, 6))
plt.scatter(centroids['x'], centroids['y'], c='blue', marker='o', s=50, label='Centroids')
plt.title('Centroids of Clusters')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.savefig('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/centroids_plot.png')


# Randomly sample one point that belongs to each cluster
sampled_points = []
selected_points =30
for cluster_label in range(selected_points):
    # Get the indices of points in the current cluster
    cluster_indices = np.where(clusters == cluster_label)[0]
    # Randomly sample one index from the cluster
    sampled_index = np.random.choice(cluster_indices)

    sampled_point = {
        'x': coordinates[sampled_index, 0],
        'y': coordinates[sampled_index, 1],
        'row': ((coordinates[sampled_index, 1] - transform[5]) / transform[4]).astype(int),
        'col': ((coordinates[sampled_index, 0] - transform[2]) / transform[0]).astype(int),
        'slope': features[sampled_index, 0],
        'aspect': features[sampled_index, 1],
        'cluster': cluster_label
    }
    sampled_points.append(sampled_point)

# Convert to DataFrame and save sampled points
sampled_points_df = pd.DataFrame(sampled_points)

# Convert coordinates to row and column indices
sampled_points_df['row'] = ((sampled_points_df['y'] - transform[5]) / transform[4]).astype(int)
sampled_points_df['col'] = ((sampled_points_df['x'] - transform[2]) / transform[0]).astype(int)


# Improved plot of the DEM and sampled points with coordinates in degrees
plt.figure(figsize=(12, 8))
plt.imshow(elevation_plot.reshape(height, width), cmap='cividis', extent=[
    transform[2], transform[2] + width * transform[0],
    transform[5] + height * transform[4], transform[5]
])
plt.colorbar(label='Elevation (m)', orientation='horizontal', pad=0.05)
plt.scatter(sampled_points_df['x'], sampled_points_df['y'], c='red', marker='o', s=20, label='Sampled Points')
plt.title('Sampled Points on Digital Elevation Model (KMeans)', fontsize=14)

# Set axis labels and ticks
plt.xlabel('Longitude (degrees)', fontsize=12)
plt.ylabel('Latitude (degrees)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add gridlines
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Add legend and save the plot
plt.legend(fontsize=10, loc='upper right')
plt.savefig('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/sample_points_kmeans_with_coords.png', bbox_inches='tight', dpi=300)

# Create a DataArray for the sampled points
sampled_points_da = xr.DataArray(
    sampled_points_df[['slope', 'aspect', 'cluster']].values,
    dims=['sample', 'feature'],  # Specify two dimensions
    coords={
        'x': ('sample', sampled_points_df['x'].values),
        'y': ('sample', sampled_points_df['y'].values),
        'feature': ['slope', 'aspect', 'cluster']  # Add feature names as a coordinate
    }
)

# Create a Dataset and assign the DataArray to it
sampled_points_ds = xr.Dataset(
    {
        'sampled_points': sampled_points_da
    },
    coords={
        'x': ('sample', sampled_points_df['x'].values),
        'y': ('sample', sampled_points_df['y'].values)
    }
)

# Save the Dataset to a NetCDF file
sampled_points_ds.to_netcdf('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/sample_points_kmeans.nc')



import rasterio
import matplotlib.pyplot as plt
import xarray as xr

# File paths
points = '/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/sample_points_kmeans.nc'
dem = ('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/DEM_VDA.tif')

# Open the NetCDF file and extract the points
ds = xr.open_dataset(points)
x_coords = ds['x'].values
y_coords = ds['y'].values

# Open the DEM file
with rasterio.open(dem) as src:
    dem_data = src.read(1)  # Read the first band
    dem_data[dem_data < 0] = np.nan  # Set values below 0 to NaN
    transform = src.transform

    # Plot the DEM
    plt.figure(figsize=(10, 10))
    plt.imshow(dem_data, cmap='terrain', extent=[
        transform[2], transform[2] + src.width * transform[0],
        transform[5] + src.height * transform[4], transform[5]
    ])
    plt.colorbar(label='Elevation (m)', orientation='horizontal', pad=0.08)

    # Overlay the points
    plt.scatter(x_coords, y_coords, c='red', marker='o', s=40, label='Sampled Points')

    # Add title, labels, and legend
    plt.title('Sampled Points on DEM', fontsize=14)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude ', fontsize=12)
    plt.legend(fontsize=10, loc='lower right')

    # Save the plot
    plt.savefig('/home/idrologia/share/PhD_GiuliaBlandini_dati/FILES/old/dem_with_points.png', bbox_inches='tight', dpi=300)
