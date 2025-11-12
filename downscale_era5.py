"""
to run in server ZEUS
"""

# Get the current event loop
import asyncio
import os
from TopoPyScale import topoclass as tc
from TopoPyScale.prepare_climate_data_TPS_zarr import get_zarr
fetch = 0
loop = asyncio.get_event_loop()
loop.close()

default_n_threads =30
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
# Set the CDS API configuration
os.environ['CDSAPIRC'] = '/home/idrologia/PhD_GiuliaBlandini/.cdsapirc'

config_file = 'config.yml'
mp = tc.Topoclass(config_file)
if fetch ==1:
    # download the ERA5 data
    result = get_zarr(
        start_date="2017-10-01",
        end_date="2022-09-30",
        refrence_area_path="/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/dem/DEM_VDA_UTM.tif",
        plevels=[500, 600, 700, 850, 1000],
        output_dir="/home/idrologia/PhD_GiuliaBlandini/S3M_2D/inputs/climate/",
        dataset_surf_path="https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        PAT="edh_pat_61df66e3f10aa2e2de793de541ed4c55259bc1f7c37b54303073a1dd38952db257a6f74b388cf9d44578d01ace9c3928"
    )
    mp.process_SURF_file('./inputs/climate')
    mp.remap_netcdf('./inputs/climate')

# check if the dem file is in meters, if it is not, convert it from degrees to meters
mp.compute_dem_param()
mp.extract_topo_param()
mp.compute_horizon()
mp.compute_solar_geometry()
mp.downscale_climate()
mp.to_netcdf()

