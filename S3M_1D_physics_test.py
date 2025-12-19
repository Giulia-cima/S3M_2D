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

from lib_utilis_flux_1D_test import PhasePart, density, melting, refreezing, Hydraulics
from solar_radiation_test import solar_radiation, solarhours
import numpy as np


def S3M_1D_physics(log_stream, meteo, parameters, state_vector, output_vector, Time, change_part,ensemble):

    mass_balance = 0
    Outflow_ExcessRain = 0
    Outflow_ExcessMelt = 0

    T_air = meteo[0]
    P = meteo[1]
    RH = meteo[2]
    Radiation = meteo[3]

    # Day of the year and hour of the day from the timestamp
    h = Time.hour
    doy = Time.timetuple().tm_yday
    Rtoa = solar_radiation(h, doy, parameters["latitude"], parameters["longitude"])
    hrise, hset = solarhours(parameters["latitude"], parameters["longitude"], doy)

    if hrise <= Time.hour <= hset:

        Radiation = np.minimum(Rtoa, Radiation)

    else:
        Radiation =0

    # ----------------------------------------------------------------------------------------

    # first sanity check
    if output_vector[10] < 0.01:
        output_vector[10] = 0
        state_vector[:len(state_vector)-1] = [0] * (len(state_vector)-1)
        output_vector[11:] = [0] * (len(output_vector)-11)

    SWE_W, SWE_D = state_vector[0], state_vector[1]
    Sf_daily_cum = output_vector[5]

    # ------------------------------------------------------------------------------------

    # Precipitation phase partitioning
    alpha, beta, gamma = parameters["alpha_p"], parameters["beta"], parameters["gamma"]
    Snowfall, Rainfall = PhasePart(P, alpha, beta, gamma, T_air, RH, change_part)
    Sf_daily_cum += Snowfall

    # ---------------------------------------------------------------------------------------------
    # Update SWE with rainfall and snowfall
    SWE_D += Snowfall
    if Rainfall > 0:
        if SWE_D >= 10.0:
            SWE_W += Rainfall
        else:
            Outflow_ExcessRain = Rainfall + SWE_W
            SWE_W = 0.0
    # ---------------------------------------------------------------------------------------------
    SWE_D = max(SWE_D, 0)
    SWE_W = max(SWE_W, 0)
    SWE = SWE_D + SWE_W

    # Compute snow density
    Rho_D_min, Rho_D_max, Rho_S_max, RhoW, dt = parameters["RhoSnowMin"], parameters["RhoSnowMax"], parameters[
        "RhoFreshSnowMax"], parameters["RhoW"], parameters["dt"]
    Rho_D, RhoS0, SnowTemp, H_D = density(Rho_D_min, Rho_D_max, Rho_S_max, RhoW, dt, state_vector, output_vector, SWE_D,
                                          Snowfall, T_air)

    # Compute melting and refreezing
    cm = dt / 86400
    T_10D, T_1D, Ttau, mrad0, mr0 = meteo[4], meteo[5], parameters["Ttau"], parameters["mrad0"], parameters["mr0"]
    As, albedo, multiplicative_term = output_vector[11], state_vector[3], parameters["multiplicative_albedo"]

    Melting, albedo, As, Sf_daily_cum, mrad, mr = melting(Time, mrad0, mr0, T_air, T_10D, T_1D, Ttau, Radiation, RhoW,
                                                          dt, cm, SWE_D, albedo, As, SWE, Sf_daily_cum,
                                                          multiplicative_term)
    Refreezing = refreezing(T_air, T_10D, SWE_W, mr0, cm, Ttau)

    # -------------------------------------------------------------------------------------

    Melting = max(Melting, 0)
    if 0 < SWE_D <= Melting:
        Melting = SWE_D
        Outflow_ExcessMelt = SWE_D + SWE_W
        SWE_D = SWE_W = SWE = 0.0
    else:
        SWE_D -= Melting
        SWE_W += Melting
        SWE = SWE_D + SWE_W

    Refreezing = max(Refreezing, 0)
    if 0 < SWE_W <= Refreezing:
        Rho_D = (SWE_D + Refreezing) / ((Refreezing / 917) + (SWE_D / Rho_D))
        SWE_D += Refreezing
        SWE_W = 0.0
        SWE = SWE_D
    elif 0 < Refreezing < SWE_W:
        Rho_D = (SWE_D + Refreezing) / ((Refreezing / 917) + (SWE_D / Rho_D))
        SWE_D += Refreezing
        SWE_W -= Refreezing
        SWE = SWE_D + SWE_W

    SWE_D = max(SWE_D, 0)
    SWE_W = max(SWE_W, 0)
    SWE = SWE_D + SWE_W
    # -------------------------------------------------------------------------------------

    # Update height
    H_D = ((SWE_D / 1000) * RhoW) / Rho_D

    # Outflow
    outflow, H_S = Hydraulics(Rho_D, RhoW, SWE_D, SWE_W, H_D, dt)

    if SWE_W > 0:
        SWE_W -= outflow
        SWE = SWE_D + SWE_W

    outflow += Outflow_ExcessMelt + Outflow_ExcessRain

    if SWE_W <= 0:
        if SWE_W < 0:
            log_stream.error('negative swe_w ' + str(SWE_W) + ' at time ' + str(Time))
        outflow += SWE_W
        SWE_W = 0.0
        SWE = SWE_D

    if SWE < 0:
        log_stream.error('negative swe ' + str(SWE) + ' at time ' + str(Time))
        SWE = 0

    H_D = ((SWE_D / 1000) * RhoW) / Rho_D
    H_S = H_D + (SWE_W / 1000) - (1 - Rho_D / 917) * H_D if (SWE_W / 1000) >= ((1 - Rho_D / 917) * H_D) else H_D

    if H_S > 0:
        theta_w = SWE_W / 1000 / H_S
        Rho_s = (Rho_D * H_D + RhoW * (SWE_W / 1000)) / H_S
    else:
        theta_w = Rho_s = 0

    # Mass Balance
    if (round((SWE - output_vector[10]), 2) != round((Snowfall + Rainfall - outflow), 2)
            and round((SWE - output_vector[10]), 1) != round((Snowfall + Rainfall - outflow), 1)):
        mass_balance = 1
        # log_stream.error('mass balance error at time ' + str(Time) + ' ensemble member ' + str(ensemble))

    input_vector_new = [P, Radiation, T_air, RH, T_10D, T_1D]
    state_vector_new = [SWE_W, SWE_D, Rho_D, albedo]
    Time = 0
    output_vector_new = [Rainfall, Snowfall, Melting, Refreezing, outflow, Sf_daily_cum, Time, mass_balance,
                         mrad, mr, SWE, As, H_D, theta_w, H_S, Rho_s]

    return input_vector_new, state_vector_new, output_vector_new
