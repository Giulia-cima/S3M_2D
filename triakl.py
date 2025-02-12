import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np



# Load the air temperature raster to extract metadata
with rasterio.open("/home/giulia/Desktop/prova/daily_summaries/2004/01/01/AirTemperature_MEDIA_20040101.tif") as src:
    air_temp_meta = src.meta.copy()  # Copy metadata
    air_temp_data = src.read(1)  # Read temperature data (assumes single band)

# Compute snow depth (replace with your actual computation)
snow_depth_data = np.maximum(0, air_temp_data * -0.1 + 10)  # Example formula

# Update metadata for the new raster
air_temp_meta.update(dtype=rasterio.float32, nodata=-9999)

# Save the snow depth data as a new GeoTIFF
with rasterio.open("/home/giulia/Desktop/prova/snow_depth.tif", "w", **air_temp_meta) as dst:
    dst.write(snow_depth_data.astype(np.float32), 1)

print("Snow depth map saved as 'snow_depth.tif'")

# Open the newly created snow depth raster
with rasterio.open("/home/giulia/Desktop/prova/snow_depth.tif") as src:
    snow_depth_data = src.read(1)  # Read the first band
    snow_depth_meta = src.meta  # Get metadata

# Plot the snow depth map
plt.figure(figsize=(10, 6))
plt.imshow(snow_depth_data, cmap="Blues", origin="upper")
plt.colorbar(label="Snow Depth (mm)")
plt.title("Snow Depth Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()



with rasterio.open("/home/giulia/Desktop/prova/daily_summaries/2004/01/01/Rain_SOMMA_20040101.tif") as src:
    rain_data = src.read(1)
    # ADD 50 MM TO RAINFALL
    rain_data += 50
    # Update metadata for the new raster
    rain_meta = src.meta.copy()
    rain_meta.update(dtype=rasterio.float32, nodata=-9999)

    # Save the updated rainfall data as a new GeoTIFF
    with rasterio.open("/home/giulia/Desktop/prova/rainfall_updated_20040101.tif", "w", **rain_meta) as dst:
        dst.write(rain_data.astype(np.float32), 1)
