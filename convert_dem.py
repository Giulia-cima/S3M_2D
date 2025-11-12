import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS
import os
import math

import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

"""
def is_utm(crs):
    
    try:
        crs_obj = CRS.from_user_input(crs)
        return crs_obj.to_dict().get('proj') == 'utm'
    except:
        return False

def convert_to_utm(input_path, output_path=None):
    with rasterio.open(input_path) as src:
        src_crs = src.crs
        print(f"Original CRS: {src_crs}")

        if is_utm(src_crs):
            print("The file is already in UTM.")
            return input_path  # No conversion needed

        # Automatically choose a UTM zone based on raster bounds
        centroid = src.transform * (src.width // 2, src.height // 2)
        utm_zone = math.floor((centroid[0] + 180) / 6) + 1
        utm_crs = CRS.from_epsg(32600 + utm_zone)  # EPSG code for UTM zones in the northern hemisphere

        transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': utm_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + "_utm.tif"

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.nearest
                )
        print(f"File converted to UTM and saved at: {output_path}")
        return output_path

def print_tif_coordinates(tif_path):
    with rasterio.open(tif_path) as src:
        print(f"\n--- TIFF Metadata ---")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Transform: {src.transform}")
        print(f"Width, Height: {src.width}, {src.height}")

        # Get a few sample coordinates (e.g. corners)
        top_left = (src.bounds.left, src.bounds.top)
        top_right = (src.bounds.right, src.bounds.top)
        bottom_left = (src.bounds.left, src.bounds.bottom)
        bottom_right = (src.bounds.right, src.bounds.bottom)

        print("\n--- Corner Coordinates ---")
        print(f"Top Left: {top_left}")
        print(f"Top Right: {top_right}")
        print(f"Bottom Left: {bottom_left}")
        print(f"Bottom Right: {bottom_right}")

        # Optional: print the coordinates of the pixel center at (row=0, col=0)
        row, col = 0, 0
        x, y = src.transform * (col + 0.5, row + 0.5)
        print(f"\nCenter of top-left pixel (0, 0): ({x}, {y})")





input_tif = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/VDA_PADDED.tif"
convert_to_utm(input_tif)
# Usage
print_tif_coordinates( "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/VDA_PADDED_utm.tif")
"""

# Function to resample raster to a new resolution
def resample_raster(input_path, output_path, scale_factor):
    with rasterio.open(input_path) as src:
        original_resolution_x = src.transform.a
        original_resolution_y = -src.transform.e
        print(f"Original resolution of {input_path}: {original_resolution_x} x {original_resolution_y} degrees")

        new_height = int(src.height // scale_factor)
        new_width = int(src.width // scale_factor)

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.average
        )

        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        profile = src.profile
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })

        print(f"New resolution of {output_path}: {new_transform.a} x {-new_transform.e} degrees")

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)


scale_factor = 0.01 / 0.0025822  # circa 0.01° ~ 1 km at average latitude

dem = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA.tif"
output_dem = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA_1km.tif"
resample_raster(dem, output_dem, scale_factor)

# plot the dem as a map
with rasterio.open(output_dem) as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax, cmap='terrain')
    ax.set_title('Digital Elevation Model (DEM)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig("/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA_1km.png")

slope = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/SLOPE_VDA.tif"
aspect = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/ASPECT_VDA.tif"
csmod = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/CSMOD_VDA.tif"
tpi = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/TPI_VDA.tif"
output_slope = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/SLOPE_VDA_1km.tif"
output_aspect = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/ASPECT_VDA_1km.tif"
output_csmod = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/CSMOD_VDA_1km.tif"
output_tpi = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/TPI_VDA_1km.tif"

# Resample slope, aspect, csmod, and tpi rasters
resample_raster(slope, output_slope, scale_factor)
resample_raster(aspect, output_aspect, scale_factor)
resample_raster(tpi, output_tpi, scale_factor)


# first open csmod and check the resolution and the reference system
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

src_path = csmod
dst_path  = "/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/CSMOD_VDA_wgs84.tif"

dst_crs = "EPSG:4326"  # target CRS

with rasterio.open(src_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )

    profile = src.profile.copy()
    profile.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    with rasterio.open(dst_path, "w", **profile) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear  # or bilinear / cubic
            )

# check the resolution of the new file
with rasterio.open(dst_path) as src:
    print(f"New resolution of {dst_path}: {src.transform.a} x {-src.transform.e} degrees")
    scale_factor = 0.01 / src.transform.a  # circa 0.01° ~ 1 km at average latitude

resample_raster(dst_path, output_csmod, scale_factor)
