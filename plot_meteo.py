
# open net cdf
import pickle
import pandas
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr

era5 = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/climate/SURF_201806.nc"
era5_downscale = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/outputs/output.nc"
observation_vda= "/home/idrologia/share/PhD_GiuliaBlandini_dati/DATI/dict.pkl"
dem ="/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA_1km.tif"
w = pandas.read_pickle("/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/weights.pkl")



ds = xr.open_dataset(era5)
downscaled_ds = xr.open_dataset(era5_downscale)
# select time from 2018-01-01 to 2018-01-31 of downscaled data
downscaled_ds = downscaled_ds.sel(time=slice("2018-06-01", "2018-06-30"))

dem_da = rxr.open_rasterio(dem, masked=True).squeeze()

ds_vda = pickle.load(open(observation_vda, "rb"))
y_plot, x_plot = list(ds_vda.keys())[0]

# find the closest point in the downscaled dataset and era5 using sel
dem_da = rxr.open_rasterio(dem, engine='rasterio').sel(band=1).drop_vars('band').chunk({"x": 100, "y": 100})
# Extract longitude and latitude coordinates
lon_dem = dem_da.x.values
lat_dem = dem_da.y.values
ny, nx = dem_da.shape
n_clusters = len(w)

var_list = ["SW"]
for var in var_list:
    data = downscaled_ds[var].values  # shape: (n_clusters, n_time)
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


# plot the 3 map of radiation to make a comparison, use same fig different subplot
fig, axs = plt.subplots(1, 2, figsize=(30,15))

# Plot the original ERA5 data
im1 = axs[0].imshow(ds['ssrd'].values[12, :, :], cmap='Blues')
axs[0].set_title('Original ERA5 Data')
cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
cbar1.set_label('Solar Radiation (W/m²)')

# Plot the downscaled data
# invert ass 0 of mapped

im2 = axs[1].imshow(mapped[:, :, 12], cmap='Blues')
axs[1].set_title('Downscaled Data')
cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
cbar2.set_label('Solar Radiation (W/m²)')
plt.tight_layout()
# save the figure
plt.savefig("/home/idrologia/PhD_GiuliaBlandini/S3M_2D/outputs/mapped_data_comparison.png")



plt.figure(figsize=(10, 8))


im = plt.imshow(mapped[:, :, 12], cmap='Blues')
plt.contour(dem_da['x'], dem_da['y'], dem_da.values, colors='black', linewidths=2)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f" Radiation  (color) and DEM contours (black)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"/home/idrologia/share/PhD_GiuliaBlandini_dati/OUTPUT_2D/radiation_dem_contour.png", dpi=150)


""" 
# create a dataxarray with mapped and add the dem coordinate

# Create an xarray.DataArray for the mapped data
mapped_da = xr.DataArray(
    mapped,
    dims=["y", "x", "time"],
    coords={
        "x": lon_dem,
        "y": lat_dem,
        "time": downscaled_ds["time"].values
    },
    name="mapped_data"
)

# nord point 45.804, 7.494
# sud point 45.582, 7.193
# find the two point in the ds_vda  keys

north_point = (45.804, 7.494)
south_point = (45.582, 7.193)

# use these two point to select the mapped da and confront the radiation
# Select the radiation data for the north and south points
north_radiation = mapped_da.sel(x=north_point[1], y=north_point[0], method="nearest")
south_radiation = mapped_da.sel(x=south_point[1], y=south_point[0], method="nearest")

# Create a single figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot for the North Point
axs[0].plot(mapped_da["time"], north_radiation, label="North Point Radiation", color="blue")
axs[0].axhline(y=north_radiation.mean(), color="blue", linestyle="--", label="North Point Mean")
axs[0].set_ylabel("Radiation (W/m²)")
axs[0].set_title("Radiation Time Series - North Point (45.804, 7.494)")
axs[0].legend()
axs[0].grid()

# Plot for the South Point
axs[1].plot(mapped_da["time"], south_radiation, label="South Point Radiation", color="red")
axs[1].axhline(y=south_radiation.mean(), color="red", linestyle="--", label="South Point Mean")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Radiation (W/m²)")
axs[1].set_title("Radiation Time Series - South Point (45.582, 7.193)")
axs[1].legend()
axs[1].grid()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("/home/idrologia/PhD_GiuliaBlandini/S3M_2D/outputs/radiation_comparison_subplot.png")
"""
