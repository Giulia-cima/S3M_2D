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

from lib_utilis_flux import PhasePart, density, melting, refreezing, Hydraulics, Sterrain
from solar_radiation import solar_radiation, solarhours
import numpy as np
from lib_utilis_data_proc import read_path


def S3M_2D_physics(meteo, parameters, state_vector, output_vector, Time, change_part, Ice_flag, lat, lon, slope, svf):
    # ------------------------------------------------------------------------------------------------------------------
    # Meteorological input upload

    T_air = meteo[:, :,  0]
    P = meteo[:, :, 1]
    RH = meteo[:, :,  2]
    Radiation = meteo[:, :,  3]


    # ------------------------------------------------------------------------------------------------------------------
    mask = P < 0  # True where P < 0

    Outflow_ExcessRain = np.zeros_like(P, dtype=float)
    Outflow_ExcessRain[mask] = np.nan

    Outflow_ExcessMelt = np.zeros_like(P, dtype=float)
    Outflow_ExcessMelt[mask] = np.nan

    IceThickness_WE = np.zeros_like(P, dtype=float)
    IceThickness_WE[mask] = np.nan

    MeltingGCumWY = np.zeros_like(P, dtype=float)
    MeltingGCumWY[mask] = np.nan

    MeltingDayCum = np.zeros_like(P, dtype=float)
    MeltingDayCum[mask] = np.nan
    IceMassBalance = parameters["IceMassBalance"]
    try:
        Ice_thickness = state_vector[:, :, 4]
    except:
        Ice_thickness = np.zeros_like(P, dtype=float)
        Ice_thickness[mask] = np.nan
    # ------------------------------------------------------------------------------------------------------------------

    if parameters["dt"] == 3600:
        h = Time.hour
        doy = Time.dayofyear
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij', sparse=True)
        Rtoa = solar_radiation(h, doy, lat_grid, lon_grid)
        hrise, hset = solarhours(lat_grid, lon_grid, doy)
        mask_hour = (hrise <= h) & (h <= hset)
        Radiation[mask_hour] = np.minimum(Rtoa[mask_hour] , Radiation[mask_hour] )
        Radiation[~mask_hour] = 0.0
    # ------------------------------------------------------------------------------------------------------------------
    # Sanity check
    mask_sanity = output_vector[ :, :,  10] < 0.01
    state_vector[mask_sanity, :-1] = 0
    state_vector[mask_sanity, 3] = 0.5
    output_vector[mask_sanity, 10:] = 0
    # ------------------------------------------------------------------------------------------------------------------
    if IceMassBalance == 1 or IceMassBalance == 2:
         IceThickness_WE = Ice_thickness*917
    # ------------------------------------------------------------------------------------------------------------------
    SWE_W, SWE_D = state_vector[:, :, 0], state_vector[:, :, 1]
    Sf_daily_cum = output_vector[:, :, 5]
    # ------------------------------------------------------------------------------------------------------------------
    # Precipitation phase partitioning
    alpha, beta, gamma = parameters["alpha_p"], parameters["beta"], parameters["gamma"]
    Snowfall, Rainfall = PhasePart(P, alpha, beta, gamma, T_air, RH, change_part)
    Sf_daily_cum += Snowfall
    # ------------------------------------------------------------------------------------------------------------------
    SWE_D += Snowfall  # This is applied element-wise
    # Where SWE_D >= 10, rainfall goes to SWE_W
    mask_snow_storage = (Rainfall > 0)  & (SWE_D >= 10.0)
    SWE_W[mask_snow_storage] += Rainfall[mask_snow_storage]
    # Where SWE_D < 10, rainfall contributes to outflow
    mask_excess = (Rainfall > 0) & (SWE_D < 10.0)
    Outflow_ExcessRain[mask_excess] = Rainfall[mask_excess] + SWE_W[mask_excess]
    SWE_W[mask_excess] = 0.0
    # ---------------------------------------------------------------------------------------------
    SWE_D = np.maximum(SWE_D, 0)
    SWE_W = np.maximum(SWE_W, 0)
    SWE = SWE_D + SWE_W
    # ------------------------------------------------------------------------------------------------------------------
    # Compute snow density
    Rho_D_min, Rho_D_max, Rho_S_max, RhoW, dt = parameters["RhoSnowMin"], parameters["RhoSnowMax"], parameters[
        "RhoFreshSnowMax"], parameters["RhoW"], parameters["dt"]
    IceMeltingCoeff = parameters["IceMeltingCoeff"]
    Rho_D, RhoS0, SnowTemp, H_D = density(Rho_D_min, Rho_D_max, Rho_S_max, RhoW, dt, state_vector, output_vector, SWE_D,
                                          Snowfall, T_air)
    # ------------------------------------------------------------------------------------------------------------------
    # Compute melting and refreezing
    cm = dt / 86400
    mrad0, mr0 = parameters["mrad0"], parameters["mr0"]

    As, albedo, multiplicative_term = (output_vector[:, :, 11], state_vector[:, :,  3],
                                       parameters["multiplicative_albedo"])

    T_albedo,T_melting, Ttau = (meteo[:, :, 4], meteo[:, :,  5], parameters["Ttau"])
    if parameters["OL"] == 0:
        Radiation = Sterrain(As, albedo, T_albedo, Time, multiplicative_term, Ice_thickness, Ice_flag, SWE_D, Radiation, slope,
                         svf)
# ------------------------------------------------------------------------------------------------------------------
    Melting, albedo, As, mrad, mr, Melting_g, Sf_daily_cum = melting(Time,mrad0, mr0, T_air, T_melting, T_albedo, Ttau,
                                                                     Radiation, RhoW, dt, cm, SWE_D, albedo, As, SWE,
                                                                     Sf_daily_cum, multiplicative_term, Ice_flag, Ice_thickness, IceMeltingCoeff)
    Refreezing = refreezing(T_air, T_melting, SWE_W, mr0, cm, Ttau)
    # ------------------------------------------------------------------------------------------------------------------
    Melting = np.maximum(Melting, 0)
    Refreezing = np.maximum(Refreezing, 0)
    # ------------------------------------------------------------------------------------------------------------------
    Outflow_ExcessMelt = np.zeros_like(SWE_D)

    # Case 1: SWE_D <= Melting
    mask_melt_all = (SWE_D > 0) & (SWE_D <= Melting)
    Melting[mask_melt_all] = SWE_D[mask_melt_all]
    Outflow_ExcessMelt[mask_melt_all] = SWE_D[mask_melt_all] + SWE_W[mask_melt_all]
    SWE_D[mask_melt_all] = 0
    SWE_W[mask_melt_all] = 0
    SWE[mask_melt_all] = 0

    # Case 2: SWE_D > Melting
    mask_melt_part = ~mask_melt_all
    SWE_D[mask_melt_part] -= Melting[mask_melt_part]
    SWE_W[mask_melt_part] += Melting[mask_melt_part]
    SWE[mask_melt_part]  = SWE_D[mask_melt_part]  + SWE_W[mask_melt_part]

    # Refreezing
    Refreezing = np.maximum(Refreezing, 0)
    mask_refreeze_all = (SWE_W > 0) & (SWE_W <= Refreezing)
    Rho_D[mask_refreeze_all] = (SWE_D[mask_refreeze_all] + Refreezing[mask_refreeze_all]) / (
            (Refreezing[mask_refreeze_all] / 917) + (SWE_D[mask_refreeze_all] / Rho_D[mask_refreeze_all])
    )
    SWE_D[mask_refreeze_all] += Refreezing[mask_refreeze_all]
    SWE_W[mask_refreeze_all] = 0
    SWE[mask_refreeze_all] = SWE_D[mask_refreeze_all]

    #Case 2: 0 < Refreezing < SWE_W
    mask_refreeze_part = (Refreezing > 0) & (Refreezing < SWE_W)
    Rho_D[mask_refreeze_part] = (SWE_D[mask_refreeze_part] + Refreezing[mask_refreeze_part]) / (
            (Refreezing[mask_refreeze_part] / 917) + (SWE_D[mask_refreeze_part] / Rho_D[mask_refreeze_part])
    )
    SWE_D[mask_refreeze_part] += Refreezing[mask_refreeze_part]
    SWE_W[mask_refreeze_part] -= Refreezing[mask_refreeze_part]
    SWE[mask_refreeze_part] = SWE_D[mask_refreeze_part] + SWE_W[mask_refreeze_part]

    # Ensure non-negative SWE
    SWE_D = np.maximum(SWE_D, 0)
    SWE_W = np.maximum(SWE_W, 0)
    SWE = SWE_D + SWE_W

    # ------------------------------------------------------------------------------------------------------------------
    """ 
    # Correct the logical operations with NumPy arrays
    if Ice_flag == 1:
        # PIXELS W/O GLACIERS AND WITH SNOW
        mask = (SWE_D > 0.0) & (SWE_D <= Melting) & (IceThickness_WE <= 0.0)
        Melting[mask] = SWE_D[mask]
        Outflow_ExcessMelt[mask] = SWE_D[mask] + SWE_W[mask]
        SWE_D[mask] = 0.0
        SWE_W[mask] = 0.0
        SWE[mask] = 0.0

        mask = (SWE_D > 0.0) & (IceThickness_WE <= 0.0)
        SWE_D[mask] -= Melting[mask]
        SWE_W[mask] += Melting[mask]
        SWE[mask] = SWE_D[mask] + SWE_W[mask]
        # --------------------------------------------------------------------------------------------------------------
        # PIXELS W GLACIERS AND WITH SNOW
        mask = (SWE_D > 0) & (SWE_D <= Melting) & (IceThickness_WE > 0)
        Outflow_ExcessMelt[mask] = SWE_D[mask] + SWE_W[mask]
        IceThickness_WE[mask] -= (Melting[mask] - SWE_D[mask])
        Melting_g[mask] += (Melting[mask] - SWE_D[mask])
        SWE_D[mask] = 0
        SWE_W[mask] = 0
        SWE[mask] = 0
        Ice_flag[mask] = 0

        mask = (SWE_D > 0) & (IceThickness_WE > 0)
        SWE_D[mask] -= Melting[mask]
        SWE_W[mask] += Melting[mask]
        SWE[mask] = SWE_D[mask] + SWE_W[mask]
        # --------------------------------------------------------------------------------------------------------------
        # PIXELS W GLACIERS BUT NO SNOW
        mask = (SWE_D == 0) & (IceThickness_WE > 0) & (Melting_g > 0) & (Ice_flag > 0)
        IceThickness_WE[mask] -= Melting_g[mask]
        # --------------------------------------------------------------------------------------------------------------
    elif Ice_flag == 2:
    
        #This second case regards a simulation with mass balance AND movement according to the deltaH parametrization,
        #so here WE DO NOT subtract glacier melt from IceThickness_WE as simulation time passes. We store melt into
        #MeltingGCumWY
        
        # PIXELS W/O GLACIERS AND WITH SNOW
        mask = (SWE_D > 0) & (SWE_D <= Melting) & (IceThickness_WE <= 0)
        Melting[mask] = SWE_D[mask]
        Outflow_ExcessMelt[mask] = SWE_D[mask] + SWE_W[mask]
        SWE_D[mask] = 0.0
        SWE_W[mask] = 0.0
        SWE[mask] = 0.0

        mask = (SWE_D > 0) & (IceThickness_WE <= 0)
        SWE_D[mask] -= Melting[mask]
        SWE_W[mask] += Melting[mask]
        SWE[mask] = SWE_D[mask] + SWE_W[mask]
        # --------------------------------------------------------------------------------------------------------------
        # PIXELS W GLACIERS AND WITH SNOW
        mask = (SWE_D > 0.0) & (SWE_D <= Melting) & (IceThickness_WE > 0.0)
        Outflow_ExcessMelt[mask] = SWE_D[mask] + SWE_W[mask]
        Melting_g[mask] += (Melting[mask] - SWE_D[mask])
        MeltingGCumWY[mask] += Melting_g[mask]
        SWE_D[mask] = 0.0
        SWE_W[mask] = 0.0
        SWE[mask] = 0.0
        Ice_flag[mask] = 0.0

        mask = (SWE_D > 0.0) & (IceThickness_WE > 0.0)
        SWE_D[mask] -= Melting[mask]
        SWE_W[mask] += Melting[mask]
        SWE[mask] = SWE_D[mask] + SWE_W[mask]
        # --------------------------------------------------------------------------------------------------------------
        # PIXELS W GLACIERS BUT NO SNOW
        mask = (SWE_D == 0.0) & (IceThickness_WE > 0.0) & (Melting_g > 0.0) & (Ice_flag > 0.0)
        MeltingGCumWY[mask] += Melting_g[mask]
    else:
        # PIXELS where Melt > SWE_D
        mask_melt = (SWE_D > 0) & (SWE_D <= Melting)
        Melting[mask_melt] = SWE_D[mask_melt]
        Outflow_ExcessMelt[mask_melt] = SWE_D[mask_melt] + SWE_W[mask_melt]
        SWE_D[mask_melt] = 0
        SWE_W[mask_melt] = 0
        SWE[mask_melt] = 0

        mask_melt_1 = SWE_D > 0 & (SWE_D > Melting)
        SWE_D[mask_melt_1] -= Melting[mask_melt_1]
        SWE_W[mask_melt_1] += Melting[mask_melt_1]
        SWE[mask_melt_1] = SWE_D[mask_melt_1] + SWE_W[mask_melt_1]
        """
    # ------------------------------------------------------------------------------------------------------------------
    # Compute daily cumulated melting
    mask = Melting > 0.0
    MeltingDayCum[mask] += Melting[mask]
    # ------------------------------------------------------------------------------------------------------------------
    # Update height of dry snow layer (H_D)
    H_D = ((SWE_D / 1000) * RhoW) / Rho_D

    # Compute outflow and updated height (H_S)
    outflow, H_S = Hydraulics(Rho_D, RhoW, SWE_D, SWE_W, H_D, dt)

    # Reduce SWE_W by outflow where SWE_W > 0
    mask_water = SWE_W > 0
    SWE_W[mask_water] -= outflow[mask_water]
    SWE[mask_water] = SWE_D[mask_water] + SWE_W[mask_water]

    # Add excess melt and rain to outflow
    outflow += Outflow_ExcessMelt + Outflow_ExcessRain

    # Handle negative or zero SWE_W
    mask_zero = SWE_W <= 0
    outflow[mask_zero] += SWE_W[mask_zero]  # negative contribution
    SWE_W[mask_zero] = 0
    SWE[mask_zero] = SWE_D[mask_zero]

    # Ensure non-negative SWE
    SWE = np.maximum(SWE, 0)

    # Dry snow height
    H_D = (SWE_D / 1000) * RhoW / Rho_D

    # Total snow height including liquid water
    water_contrib = SWE_W / 1000
    comp_factor = (1 - Rho_D / 917) * H_D
    H_D = ((SWE_D / 1000) * RhoW) / Rho_D
    mask_height = water_contrib >= comp_factor
    H_S[mask_height] = H_D[mask_height] + water_contrib[mask_height] - comp_factor[mask_height]
    H_S[~mask_height] = H_D[~mask_height]

    # Volumetric water content and bulk snow density
    mask_snow = H_S > 0
    theta_w = np.zeros_like(H_S)
    Rho_s = np.zeros_like(H_S)

    theta_w[mask_snow] = (SWE_W[mask_snow] / 1000) / H_S[mask_snow]
    Rho_s[mask_snow] = (Rho_D[mask_snow] * H_D[mask_snow] + RhoW * (SWE_W[mask_snow] / 1000)) / H_S[mask_snow]

    # ------------------------------------------------------------------------------------------------------------------

    delta_swe = np.round(SWE - output_vector[:, :,10], 2)
    delta_flux = np.round(Snowfall + Rainfall - outflow, 2)
    cond_balance = np.abs(np.round(delta_swe[:,:] - delta_flux[:,:] ,1))

    if np.any(cond_balance > 0):
        mass_balance = np.ones_like(SWE)  # or any variable with the shape you want
        #print("Mass balance check failed at some pixels in open loop.")
    else:
        mass_balance = np.zeros_like(SWE)
    """ 
    if IceMassBalance== 2 :

        mask = SWE >0

        IceThickness_WE[mask] = IceThickness_WE[mask]+ SWE[mask]
        SWE = 0.0
        SWE_D = 0.0
        SWE_W = 0.0

        # IceThickness_WE = GlacierDeltaH(dt, Rows, Cols, iRows_Pivot, IceThickness_WE, MeltingGCumWY, Mask, PivotTable, DEM, Glaciers_ID, AreaCell)

        Melting_g_CumWY = 0

    if IceMassBalance == 1 or IceMassBalance == 2:
        # Conversion from mm of water equivalent to m of ice
            ChangeThickness = IceThickness_WE/917 -Ice_thickness
            Ice_thickness = IceThickness_WE/917
    """
    # ------------------------------------------------------------------------------------------------------------------

    Time = Time.hour + Time.minute / 60
    Time = np.tile(Time, (P.shape))
    # ------------------------------------------------------------------------------------------------------------------
    # create a new state vector and output vector
    state_vector_new = np.zeros(state_vector.shape)
    output_vector_new = np.zeros(output_vector.shape)

    state_vector_new[:, :, 0] = SWE_W
    state_vector_new[:, :, 1] = SWE_D
    state_vector_new[:, :, 2] = Rho_D
    state_vector_new[:, :, 3] = albedo
    #state_vector_new[:, :, 4] = Ice_thickness
    # ------------------------------------------------------------------------------------------------------------------
    output_vector_new[:, :, 0] = Rainfall
    output_vector_new[:, :, 1] = Snowfall
    output_vector_new[:, :, 2] = Melting
    output_vector_new[:, :, 3] = Refreezing
    output_vector_new[:, :, 4] = outflow
    output_vector_new[:, :, 5] = Sf_daily_cum
    output_vector_new[:, :, 6] = Time
    output_vector_new[:, :, 7] = mass_balance
    output_vector_new[:, :, 8] = mrad
    output_vector_new[:, :, 9] = mr
    output_vector_new[:, :, 10] = SWE
    output_vector_new[:, :, 11] = As
    output_vector_new[:, :, 12] = H_D
    output_vector_new[:, :, 13] = theta_w
    output_vector_new[:, :, 14] = H_S
    output_vector_new[:, :, 15] = Rho_s
    output_vector_new[:, :, 16] = Melting_g

    # sanity check

    mask = output_vector_new[:, :, 10] < 0.01
    state_vector_new[mask, 0] = 0
    state_vector_new[mask, 1] = 0
    state_vector_new[mask, 2] = 0
    state_vector_new[mask, 3] = 0.5
    output_vector_new[mask, 10:] = 0



    # ------------------------------------------------------------------------------------------------------------------

    return meteo, state_vector_new, output_vector_new, mass_balance
