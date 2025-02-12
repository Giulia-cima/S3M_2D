# Re-writing of a punctual version of S3M with python language. Project related to the doctoral thesis:Data assimilation
# and deep learning·I will try to re-write S3M in python and develop an EnKF or PF data assimilation procedure to
# replace the current nudging procedure. Then I will use data assimilation data to train a neural network (CNN/LSTM/RNN)
# to exploit the computational capacity of DL to make this procedure comparable with operative times.
# In this version I will omit the glacial components.
# #----------------------------------------------------------------------------------------------------------------------
# ## Input data of S3M are:
# 1) Land data, (mandatory but not for punctual model)
# 2) Meteorological observations, (mandatory)
# 3) SCA satellite images, (optional)
# 4) SWE independent estimates for assimilation, (optional)
# 5) Ice thickness and a number of other ancillary glacier data, (optional).[ This will not be needed]

# ## Output results are (among others):
# 1) Snowpack runoff,
# 2) SWE (dry and wet), snow density (dry and wet), snow depth, bulk liquid water content
# 3) Snowfall, Rainfall, and Precipitation rates, as well as fresh-snow density
# 4) Snow age, albedo, melt, and snowpack runoff
# 5) Ice thickness. [ This will not be needed]

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# Library

from lib_utilis_flux import PhasePart, density, melting, refreezing, Hydraulics
from solar_radiation import solar_radiation, solarhours
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def S3M_2D_physics(log_stream, meteo, parameters, state_vector, output_vector, Time, change_part):
    # ----------------------------------------------------------------------------------------
    # Meteorological input upload
    T_air = meteo[:, :,0]
    P = meteo[:,:,1]
    RH = meteo[:,:,2]
    lat = parameters["latitude"]
    lon = parameters["longitude"]
    # make griD of P.shape[0]x P.shape[1] with the same value of LAT
    lat = np.tile(lat, (P.shape[0], P.shape[1]))
    # make griD of P.shape[0]x P.shape[1] with the same value of LON
    lon = np.tile(lon, (P.shape[0], P.shape[1]))
    # make griD of P.shape[0]x P.shape[1] with  0
    Outflow_ExcessRain = np.zeros((P.shape[0], P.shape[1]))
    Outflow_ExcessMelt = np.zeros((P.shape[0], P.shape[1]))
    # Solar radiation
    Radiation = meteo[:, :, 3]
    # IN CASE I HAVE DAILY TIME STEP HOW TO CHECK RADIAITON FOR THE DAY as i have not the hour?

    """
    # Day of the year and hour of the day from the timestamp,but Time is a vector of 40 elements, so also h and doy
    h = Time.hour
    doy = Time.timetuple().tm_yday
    # apply the solar radiation function to the matrix of latitudes and longitudes
    Rtoa = solar_radiation(lat, lon, doy, h) # Rtoa is a vector of 40 elements
    hrise, hset = solarhours(lat, lon, doy)

    # Calculate the solar radiation for the day , if the hour is between sunrise and sunset the min between the actual
    # solar radiation and the maximum solar radiation is taken, otherwise 0. consider thar Rtoa is a vector of 40 elements
    #and so is the output vector and hrise and hset
    Radiation = np.minimum(Rtoa, meteo[:,3])
    mask = (h < hrise) | (h > hset)
    Radiation[mask] = 0
    """
    # ----------------------------------------------------------------------------------------
    """ 
    # first sanity check
    # Vectorized operation to set output_vector[10] to 0 if it is less than 0.01
    mask = output_vector[:, :,10] < 0.01
    output_vector[mask, 10] = 0

    # Vectorized operation to set state_vector elements to 0
    state_vector[mask, :-1] = 0

    # Vectorized operation to set output_vector elements to 0 from index 11 onwards
    output_vector[mask, 11:] = 0
"""
    # ----------------------------------------------------------------------------------------
    # first sanity check
    mask = output_vector[:, :, 10] < 0.01
    output_vector[mask, 10] = 0
    state_vector[mask, :-1] = 0
    output_vector[mask, 11:] = 0
    SWE_W, SWE_D = state_vector[:, :, 0], state_vector[:, :, 1]
    Sf_daily_cum = output_vector[:, :, 5]
    # -----------------------------------------------------------------------------------
    # Precipitation phase partitioning
    alpha, beta, gamma = parameters["alpha"], parameters["beta"], parameters["gamma"]
    Snowfall, Rainfall = PhasePart(P, alpha, beta, gamma, T_air, RH, change_part)
    Sf_daily_cum += Snowfall
    # -----------------------------------------------------------------------------------
    # Update SWE with rainfall and snowfall
    SWE_D += Snowfall
    SWE_W += Rainfall
    # check the shape
    print("sf shape after ", Sf_daily_cum.shape)
    print("snowfall shape after ", Snowfall.shape)
    print("rainfall shape after ", Rainfall.shape)

    # -----------------------------------------------------------------------------------------
    # Vectorized operation to handle Rainfall and SWE_D
    mask_rainfall = Rainfall > 0
    mask_swe_d = SWE_D >= 10.0

    # Update SWE_W where Rainfall > 0 and SWE_D >= 10.0
    SWE_W[mask_rainfall & mask_swe_d] += Rainfall[mask_rainfall & mask_swe_d]

    # Update Outflow_ExcessRain and SWE_W where Rainfall > 0 and SWE_D < 10.0
    mask_swe_d_less = ~mask_swe_d
    Outflow_ExcessRain[mask_rainfall & mask_swe_d_less] = Rainfall[mask_rainfall & mask_swe_d_less] + SWE_W[
        mask_rainfall & mask_swe_d_less]

    SWE_W[mask_rainfall & mask_swe_d_less] = 0.0

    # Ensure SWE_D and SWE_W are non-negative
    SWE_D = np.maximum(SWE_D, 0)
    SWE_W = np.maximum(SWE_W, 0)

    # Calculate SWE
    SWE = SWE_D + SWE_W
    # ------------------------------------------------------------------------------------------------------------------

    # Compute snow density
    Rho_D_min, Rho_D_max, Rho_S_max, RhoW, dt = parameters["RhoSnowMin"], parameters["RhoSnowMax"], parameters[
        "RhoFreshSnowMax"], parameters["RhoW"], parameters["dt"]

    Rho_D, RhoS0, SnowTemp, H_D = density(Rho_D_min, Rho_D_max, Rho_S_max, RhoW, dt, state_vector, output_vector, SWE_D,
                                          Snowfall, T_air)
    # ------------------------------------------------------------------------------------------------------------------
    # Compute melting and refreezing
    cm = dt / 3600
    T_10D, T_1D, Ttau, mrad0, mr0 = meteo[:,:,4], meteo[:,:,5], parameters["Ttau"], parameters["mrad0"], parameters["mr0"]
    As, albedo, multiplicative_term = output_vector[:,:,11], state_vector[:,:,3], parameters["multiplicative_albedo"]

    Melting, albedo, As, Sf_daily_cum, mrad, mr = melting(Time, mrad0, mr0, T_air, T_10D, T_1D, Ttau, Radiation, RhoW,
                                                          dt, cm, SWE_D, albedo, As, SWE, Sf_daily_cum,
                                                          multiplicative_term)
    Refreezing = refreezing(T_air, T_10D, SWE_W, mr0, cm, Ttau)

    # -------------------------------------------------------------------------------------

    # Ensure Melting and Refreezing are non-negative
    Melting = np.maximum(Melting, 0)
    Refreezing = np.maximum(Refreezing, 0)

    # Vectorized operation for Melting
    mask_melting = (0 < SWE_D) & (SWE_D <= Melting)
    Melting[mask_melting] = SWE_D[mask_melting]
    Outflow_ExcessMelt[mask_melting] = SWE_D[mask_melting] + SWE_W[mask_melting]
    SWE_D[mask_melting] = 0.0
    SWE_W[mask_melting] = 0.0
    SWE[mask_melting] = 0.0

    mask_no_melting = ~mask_melting
    SWE_D[mask_no_melting] -= Melting[mask_no_melting]
    SWE_W[mask_no_melting] += Melting[mask_no_melting]
    SWE[mask_no_melting] = SWE_D[mask_no_melting] + SWE_W[mask_no_melting]

    # Vectorized operation for Refreezing
    mask_refreezing = (0 < SWE_W) & (SWE_W <= Refreezing)
    Rho_D[mask_refreezing] = (SWE_D[mask_refreezing] + Refreezing[mask_refreezing]) / \
                             ((Refreezing[mask_refreezing] / 917) + (SWE_D[mask_refreezing] / Rho_D[mask_refreezing]))
    SWE_D[mask_refreezing] += Refreezing[mask_refreezing]
    SWE_W[mask_refreezing] = 0.0
    SWE[mask_refreezing] = SWE_D[mask_refreezing]

    mask_partial_refreezing = (0 < Refreezing) & (Refreezing < SWE_W)
    Rho_D[mask_partial_refreezing] = (SWE_D[mask_partial_refreezing] + Refreezing[mask_partial_refreezing]) / \
                                     ((Refreezing[mask_partial_refreezing] / 917) + (
                                                 SWE_D[mask_partial_refreezing] / Rho_D[mask_partial_refreezing]))
    SWE_D[mask_partial_refreezing] += Refreezing[mask_partial_refreezing]
    SWE_W[mask_partial_refreezing] -= Refreezing[mask_partial_refreezing]
    SWE[mask_partial_refreezing] = SWE_D[mask_partial_refreezing] + SWE_W[mask_partial_refreezing]

    # Ensure SWE_D and SWE_W are non-negative
    SWE_D = np.maximum(SWE_D, 0)
    SWE_W = np.maximum(SWE_W, 0)
    SWE = SWE_D + SWE_W
    # -------------------------------------------------------------------------------------
    # Update height
    H_D = ((SWE_D / 1000) * RhoW) / Rho_D
    # -------------------------------------------------------------------------------------
    # Outflow
    outflow, H_S = Hydraulics(Rho_D, RhoW, SWE_D, SWE_W, H_D, dt)
    # -------------------------------------------------------------------------------------
    # Ensure SWE_W is non-negative and update SWE
    mask_swe_w_positive = SWE_W > 0
    SWE_W[mask_swe_w_positive] -= outflow[mask_swe_w_positive]
    SWE[mask_swe_w_positive] = SWE_D[mask_swe_w_positive] + SWE_W[mask_swe_w_positive]
    # -------------------------------------------------------------------------------------
    # Update outflow
    outflow += Outflow_ExcessMelt + Outflow_ExcessRain
    # -------------------------------------------------------------------------------------
    # Handle cases where SWE_W is less than or equal to 0
    mask_swe_w_non_positive = SWE_W <= 0
    outflow[mask_swe_w_non_positive] += SWE_W[mask_swe_w_non_positive]
    SWE_W[mask_swe_w_non_positive] = 0.0
    SWE[mask_swe_w_non_positive] = SWE_D[mask_swe_w_non_positive]
    # -------------------------------------------------------------------------------------
    # Log error if SWE_W is negative
    mask_swe_w_negative = SWE_W < 0
    if np.any(mask_swe_w_negative):
        log_stream.error('negative swe_w ' + str(SWE_W[mask_swe_w_negative]) + ' at time ' + str(Time))
    # -------------------------------------------------------------------------------------
    # Handle cases where SWE is less than 0
    mask_swe_negative = SWE < 0
    if np.any(mask_swe_negative):
        log_stream.error('negative swe ' + str(SWE[mask_swe_negative]) + ' at time ' + str(Time))
    SWE[mask_swe_negative] = 0
    # -------------------------------------------------------------------------------------
    # Update height
    H_D = ((SWE_D / 1000) * RhoW) / Rho_D
    H_S = np.where((SWE_W / 1000) >= ((1 - Rho_D / 917) * H_D),
                   H_D + (SWE_W / 1000) - (1 - Rho_D / 917) * H_D,
                   H_D)
    # -------------------------------------------------------------------------------------
    # Calculate theta_w and Rho_s
    mask_h_s_positive = H_S > 0
    theta_w = np.where(mask_h_s_positive, SWE_W / 1000 / H_S, 0)
    Rho_s = np.where(mask_h_s_positive, (Rho_D * H_D + RhoW * (SWE_W / 1000)) / H_S, 0)

    # Mass Balance
    mass_balance = np.where(
        (np.round((SWE - output_vector[:,:, 10]), 2) != np.round((Snowfall + Rainfall - outflow), 2)) &
        (np.round((SWE - output_vector[:, :,10]), 1) != np.round((Snowfall + Rainfall - outflow), 1)),
        1, 0
    )
    # -------------------------------------------------------------------------------------

    # Create new vectors
    variables = np.array([SWE_W, SWE_D, Rho_D, albedo])
    state_vector_new = np.stack(variables, axis=-1)
    # convert time into a float
    Time = Time.hour + Time.minute / 60
    output_variables = [Rainfall, Snowfall, Melting, Refreezing, outflow, Sf_daily_cum, Time, mass_balance,
                        mrad, mr, SWE, As, H_D, theta_w, H_S, Rho_s]


#  make output_vector_new a 3D array with the same shape as output_vector but with an additional dimension
    output_vector_new = np.zeros((output_vector.shape[0], output_vector.shape[1], len(output_variables)))
    for i, output_variable in enumerate(output_variables):
        output_vector_new[:, :, i] = output_variable

    return meteo, state_vector_new, output_vector_new, mass_balance
