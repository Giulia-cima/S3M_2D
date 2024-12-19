import numpy as np
from scipy.optimize import minimize
from S3M_2D_forcing import S3M_2D

# check the start and end of  winter season using a snow depth threshold of 0.3 m , P > 0 and T <= 0.5
def winter_season(SWE, P, T):
    mask = (SWE > 0.3) & (P > 0) & (T <= 0.5)
    return mask

# Define the cost function
def f_cost(params, Meteo_map_data, Nivometer_data, stdev_swe, stdev_hs):
    mrad, mr, melting_window = params
    arg = [mrad, mr, melting_window]
    rmse_swe, rmse_hs = S3M_2D(arg, Meteo_map_data, Nivometer_data)

    return rmse_swe / stdev_swe + rmse_hs / stdev_hs

# Calibration only during winter on only the 50% of the available data for each station
# Nivometer data  are stored in a matrix Nivometer_data with dimensions (n,m, 2) where n is the number of data points
# and m is the number of stations and the third dimension
# contains the swe and the snow height data

Nivometer_data = np.random.rand(100, 10, 2)

# there is also the meteo map data stored in a matrix Meteo_map_data with dimensions (n,m, 6) where n is the number of data points
# and m is the number of stations and the third dimension contains the precipitation, temperature,, relative humidity, solar radiation
# and two averaged temperature values

Meteo_map_data = np.random.rand(100, 10, 6)

# i want to calibrate the model for each station using only 50% of the data points, and obtain a matrix of parameters with dimensions (m, 3)
# where m is the number of stations and the three columns contain the optimal values of mrad, mr and melting_window for each station

n = Nivometer_data.shape[0]
m = Nivometer_data.shape[1]
n_half = int(n/2)
params = np.zeros((m, 3))
mrad_bounds = (0.5, 3)
mr_bounds = (0.8, 2.5)
melting_window_bounds = (24, 240)
# using vectorized operations  , optimize the cost function for each station using a random 50% of the data points

for i in range(m):
    # compute the standard deviation of the swe and hs data
    stdev_swe_obs = np.std(Nivometer_data[:, i, 0])
    stdev_hs_obs = np.std(Nivometer_data[:, i, 1])

    # select a random 50% of the data points from the matrices
    idx = np.random.choice(n, n_half, replace=False)
    swe = Nivometer_data[idx, i, 0]
    hs = Nivometer_data[idx, i, 1]
    Nivometer_data_cal = Nivometer_data[idx]
    Meteo_map_data_cal = Meteo_map_data[idx]

    # compute winter season mask
    mask = winter_season(swe, Meteo_map_data_cal[:, 0], Meteo_map_data_cal[:, 1])
    swe = swe[mask]
    hs = hs[mask]
    Nivometer_data_cal = Nivometer_data_cal[mask]
    Meteo_map_data_cal = Meteo_map_data_cal[mask]

    # use a constraint optimization algorithm to find the optimal parameters using the mwthod L-BFGS-B
    res = minimize(f_cost, x0=np.array([1, 1, 120]), args=(Meteo_map_data_cal, Nivometer_data, stdev_swe_obs, stdev_hs_obs ), bounds=[mrad_bounds, mr_bounds, melting_window_bounds], method='L-BFGS-B')
    params[i] = res.x

