import netCDF4 as nc
import numpy as np




# UPLOAD INFO FROM JSON
# SET THE LOGGING
# CHOOSE BETWEEN OPEN LOOP, CALIBRATION OR DATA ASSIMILATION
# RUN THE MODEL FOR EACH TIME STEP (upload meteo map and obtain Xb map). METEO MAPS ARE NETCDF FILES. USE VECTORIZED OPERATIONS

""" 
# Open the NetCDF file
file_path = 'path_to_your_file.nc'
dataset = nc.Dataset(file_path, 'r')

# Read the variable you are interested in
# Replace 'variable_name' with the actual variable name in your NetCDF file
variable_data = dataset.variables['variable_name'][:]

# Convert the variable data to a NumPy matrix
matrix = np.array(variable_data)

# Close the dataset
dataset.close()
"""



# APPLY POINTY ENKF
# OBTAIN THE CORRECTION DELTA_X
# INTERPOLATE THE CORRECTION DELTA_X TO OBTAIN THE CORRECTION MAP
# OBTAIN THE ANALYSIS MAP Xa
# COMPUTE A SCORE BETWEEN OBSERVATIONS AND ANALYSIS AND OPEN LOOP
# PLOT THE MAPS
# SAVE THE MAPS IN NETCDF FILES