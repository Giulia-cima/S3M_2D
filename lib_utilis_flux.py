import numpy as np
import pandas
import rasterio
import  scipy.special as sp
# -----------------------------------------------------
# -----------------------------------------------------
# Froidurot et. al 2014 PRECIPITATION-PHASE partitioning
def PhasePart(P, alpha, beta, gamma, T_air, RH, change_part):
        """
        Vectorized computation of snowfall and rainfall.
        Parameters:
            P       : precipitation array
            T_air   : air temperature array
            RH      : relative humidity array
            alpha, beta, gamma : coefficients for SepCoeff
            change_part : 1 for simple threshold, else sigmoid separation

        Returns:
            Snowfall, Rainfall : arrays of the same shape as P
        """
        mask_p = P > 0
        Snowfall = np.zeros_like(P, dtype=float)
        Rainfall = np.zeros_like(P, dtype=float)

        if change_part == 1:
            # simple threshold
            Snowfall= np.where(T_air <= 0.5, P, 0)
            Rainfall= np.where(T_air > 0.5, P, 0)

        else:
            # sigmoid separation
            SepCoeff = 1 / (1 + np.exp(alpha + (beta * T_air[mask_p]) + (gamma * RH[mask_p])))
            Snowfall[mask_p] = (1 - SepCoeff) * P[mask_p]
            Rainfall[mask_p] = SepCoeff*P[mask_p]

        # apply minimum threshold
        Snowfall[Snowfall < 0.01] = 0
        Rainfall[Rainfall < 0.01] = 0

        return Snowfall, Rainfall

# -----------------------------------------------------
# -----------------------------------------------------

def density(Rho_D_min, Rho_D_max, Rho_S_max, RhoW, dt, state_vector, output_vector, SWE_D, Snowfall, T_air):
        """
        Vectorized computation of dry-snow density and height.

        Parameters:
            SWE_D      : array of SWE (mm)
            Snowfall   : array of snowfall (mm)
            T_air      : array of air temperatures (°C)
            state_vector : array with at least 3 elements, state_vector[2] = Rho_D
            output_vector: array with at least 16 elements, output_vector[15] = RhoS0
            dt         : timestep in seconds
            Rho_D_min, Rho_D_max, Rho_S_max, RhoW : scalars

        Returns:
            Rho_D, RhoS0, SnowTemp, H_D : arrays of same shape as SWE_D
        """
        Rho_D =state_vector[:, :, 2]
        RhoS0 = output_vector[:, :, 15]
        H_D = output_vector[:, :, 12]
        SnowTemp = np.zeros_like(SWE_D)


        # Fresh snow density
        mask_new_snow = Snowfall > 0
        RhoS0[mask_new_snow] = 67.9 + 51.3 * np.exp(T_air[mask_new_snow] / 2.6)
        RhoS0 = np.clip(RhoS0, Rho_D_min, Rho_S_max)

        RhoS0[Snowfall == 0] = 0

        # Dry-snow density update
        mask_cond1 = (SWE_D - Snowfall > 1) & (Snowfall > 1) & (Rho_D > Rho_D_min)
        Rho_D[mask_cond1] = SWE_D[mask_cond1] / (
                (Snowfall[mask_cond1] / RhoS0[mask_cond1]) + (
                    (SWE_D[mask_cond1] - Snowfall[mask_cond1]) / Rho_D[mask_cond1])
        )

        mask_cond2 = (SWE_D - Snowfall <= 1) & (Snowfall > 0)

        Rho_D[mask_cond2] = RhoS0[mask_cond2]

        # enforce limits
        Rho_D = np.clip(Rho_D, Rho_D_min, Rho_D_max)

        # compute snow height
        mask_cond3 = Rho_D > 0
        H_D[mask_cond3] = ((SWE_D[mask_cond3] / 1000) * RhoW) / Rho_D[mask_cond3]
        mask_not_cond3 = Rho_D <= 0
        H_D[mask_not_cond3] = 0

        # snow temperature
        mask_cond4 = T_air >= 0
        SnowTemp[mask_cond4] = 0
        mask_not_cond4 = T_air < 0
        SnowTemp[mask_not_cond4] = 0.5 * T_air[mask_not_cond4]

        # dry-snow compaction
        mask_snow = SWE_D > 0
        Rho_D[mask_snow] = Rho_D[mask_snow]+ 0.66 * (dt / 3600) * 0.001 * H_D[mask_snow] * (Rho_D[mask_snow] ** 2) * \
                            np.exp(0.08 * SnowTemp[mask_snow] - 0.021 * Rho_D[mask_snow])

        # enforce limits again
        Rho_D = np.clip(Rho_D, Rho_D_min, Rho_D_max)

        # recompute snow height after compaction
        mask_cond5 = Rho_D > 0
        mask_not_cond5 = Rho_D <= 0
        H_D[mask_cond5] = ((SWE_D [mask_cond5] / 1000) * RhoW )/ Rho_D[mask_cond5]
        H_D[mask_not_cond5] = 0

        return Rho_D, RhoS0, SnowTemp, H_D

# -----------------------------------------------------
# -----------------------------------------------------

# compute snowpack outflow based on Darcy's
def Hydraulics(Rho_D, RhoW, SWE_D, SWE_W, H_D, dt):
    """
    Vectorized computation of snow porosity, saturation, permeability, and outflow.

    Parameters:
        SWE_D : dry snow SWE (array)
        SWE_W : liquid SWE (array)
        H_D   : dry snow height (array)
        Rho_D : dry snow density (array)
        RhoW  : water density (scalar)
        dt    : timestep (s)

    Returns:
        H_S      : total snow height (array)
        Outflow_K: outflow due to percolation (array)
    """
    # Porosity

    Porosity = np.where(H_D > 0, 1 - Rho_D / 917, 0)


    # Initialize output arrays
    H_S = np.zeros_like(H_D, dtype=float)
    Sr = np.zeros_like(H_D, dtype=float)

    cond1 = (H_D > 0) & (((SWE_W / 1000) - (Porosity * H_D)) >= 0)
    cond2 = (H_D > 0) & ~cond1  # H_D > 0 but condition 1 not satisfied
    cond3 = ~((H_D > 0))  # H_D <= 0

    # Apply conditions
    # saturated
    H_S[cond1] = H_D[cond1] + ((SWE_W[cond1] / 1000) - (Porosity[cond1] * H_D[cond1]))
    Sr[cond1] = 1.0

    # partially saturated
    H_S[cond2] = H_D[cond2]
    Sr[cond2] = (SWE_W[cond2] / 1000) / (Porosity[cond2] * H_D[cond2])

    H_S[cond3] = 0
    Sr[cond3] = 0

    # Irreducible saturation and effective saturation
    mask_porosity = Porosity > 0
    Sr_irr = np.zeros_like(Rho_D, dtype=float)
    Sr_irr [mask_porosity] = 0.02 * ((Rho_D[mask_porosity] / RhoW) / Porosity[mask_porosity])
    Sr_star = np.where(Porosity > 0, (Sr - Sr_irr) / (1 - Sr_irr), 0)

    # --- SSA, r_e, permeability, conductivity ---
    SSA = np.zeros_like(Rho_D, dtype=float)
    e = np.zeros_like(Rho_D, dtype=float)
    Perm = np.zeros_like(Rho_D, dtype=float)
    Cond = np.zeros_like(Rho_D, dtype=float)

    # Condition where Rho_D > 0
    mask_rho = Rho_D > 0

    SSA[mask_rho] = -308.2 * np.log(Rho_D[mask_rho] / 1000) - 206
    SSA[mask_rho] /= 10  # SSA in m2/kg
    e[mask_rho] = 3 / (SSA[mask_rho] * 917)
    Perm[mask_rho] = 3 * (e[mask_rho] ** 2) * np.exp(-0.013 * Rho_D[mask_rho])

    # Cond only defined where Rho_D > 0
    Cond[mask_rho & (Sr >= Sr_irr)] = Perm[mask_rho & (Sr >= Sr_irr)] * (Sr_star[mask_rho & (Sr >= Sr_irr)] ** 3)
    Cond[mask_rho & (Sr < Sr_irr)] = 0

    # --- Outflow computation ---
    Outflow_K = np.zeros_like(SWE_W, dtype=float)

    # Case 1: Sr >= 0.5 or SWE_D < 10
    mask1 = (Sr >= 0.5) | (SWE_D < 10)
    Outflow_K[mask1] = SWE_W[mask1]

    # Case 2: SWE_W < 5.47e5 * Cond * dt * 1000
    mask2 = (~mask1) & (SWE_W < (5.47e5 * Cond * dt * 1000))
    Outflow_K[mask2] = SWE_W[mask2]

    # Case 3: otherwise
    mask3 = (~mask1) & (~mask2)
    Outflow_K[mask3] = 5.47e5 * Cond[mask3] * dt * 1000

    # Case 4: Outflow <= 0.01 → set to zero
    Outflow_K[Outflow_K <= 0.01] = 0

    return Outflow_K, H_S

# -----------------------------------------------------
# -----------------------------------------------------

def refreezing(T_air, T_melting, SWE_W, mr0, cm, Ttau):
    """
    Compute refreezing amount R (mm/h) over a 2D domain.

    Parameters:
    - T_air     : 2D array of air temperature (°C)
    - T_melting : 2D array of melt temperature (°C)
    - SWE_W     : 2D array of liquid water in the snowpack (mm)
    - mr0       : scalar or 2D array, melt rate calibration constant
    - cm        : scalar, degree-day melt factor (mm/°C/h)
    - Ttau      : scalar, temperature threshold for melting/refreezing (°C)

    Returns:
    - R         : 2D array of refreezing amount (mm/h)
    """
    # Step 1: Compute mr
    mr = 0.598862 * np.arctan(0.27439 * T_melting - 0.5988) - 0.598862 * (np.pi/ 2) + mr0

    # mr cannot be negative
    mr = np.where(mr < 0, 0, mr)

    # Step 2: Compute R (refreezing energy)
    R = np.where((T_air < Ttau) & (SWE_W > 0), -cm * mr * (T_air - Ttau), 0)

    return R
# -----------------------------------------------------
# -----------------------------------------------------

def melting(ref_time,mrad0, mr0, T_air, T_melting, T_albedo, Ttau, Radiation, RhoW, dt, cm,
            SWE_D, albedo, As, SWE, Sf_daily_cum,
            multiplicative_term, Ice_flag, Ice_thickness, IceMeltingCoeff):
    """
    Compute snow and glacier melt using a temperature-index approach.

    Returns:
        M      -- Snow melt (mm)
        albedo -- Updated surface albedo
        As     -- Updated snow age
        mrad   -- Radiation melt coefficient
        mr     -- Temperature melt coefficient
        M_g    -- Glacier melt (mm)
    """
    # Initialize melt arrays as all zeros

    lambdaf = 0.334  # latent heat of fusion [MJ/kg]

    # === Update snow age and albedo ===
    As, Sf_daily_cum = snow_age(As, ref_time, SWE, Sf_daily_cum)

    albedo = alb(As, albedo, T_albedo, ref_time, multiplicative_term,Ice_thickness,Ice_flag)

    # === Melt coefficients (temperature & radiation based) ===
    mrad =0.49338 * np.arctan(0.27439 * T_melting - 0.5988) - 0.49338 * (np.pi / 2) + mrad0
    mr = 0.598862 * np.arctan(0.27439 * T_melting - 0.5988) - 0.598862 * (np.pi/ 2) + mr0

    # Enforce non-negativite
    mrad = np.maximum(mrad, 0)
    mr = np.maximum(mr, 0)

    # --- Radiative and temperature melt ---
    # Compute both, then mask later
    M_rad = mrad * ((Radiation * (1.0 - albedo)) / (1000.0 * RhoW * lambdaf)) * dt
    M_rad = np.where(M_rad < 0.01, 0, M_rad)

    M_temp = cm * mr * (T_air - Ttau)
    M_temp = np.where(M_temp < 0.01, 0, M_temp)

    # Total melt only where melting conditions are met
    cond = (T_air >= Ttau) & (T_melting >= Ttau) & (SWE_D > 0)
    M = np.where(cond, (M_rad + M_temp), 0)

    # === Glacier melt ===
    # Glacier melt coefficients
    mr_g = IceMeltingCoeff * mr
    mrad_g = mrad
    # Enforce non-negativity
    mr_g = np.maximum(mr_g, 0)
    mrad_g = np.maximum(mrad_g, 0)

    # Radiative and temperature glacier melt
    M_rad_g = mrad_g * ((Radiation * (1.0 - albedo)) / (1000.0 * RhoW * lambdaf)) * dt
    M_rad_g = np.where(M_rad_g < 0.01, 0, M_rad_g)
    M_temp_g = cm * mr_g * (T_air - Ttau)
    M_temp_g = np.where(M_temp_g < 0.01, 0, M_temp_g)

    # Total glacier melt only where melting conditions are met
    cond_g = (T_air >= Ttau) & (T_melting >= Ttau) & (SWE_D == 0)
    M_g = np.where(cond_g, M_rad_g + M_temp_g, 0)

    return M, albedo, As, mrad, mr, M_g, Sf_daily_cum


# -----------------------------------------------------
# ----------------------------------------------------


# To compute snow age consider 1 d with at least 3 mm of total snow fall
def snow_age(As, ref_time, SWE, Sf_daily_cum):
    """
    Update snow age (As) based on cumulative daily snowfall and SWE.

    Parameters:
    - As           : snow age array (2D)
    - SWE          : snow water equivalent array (2D)
    - Snowfall_cum : cumulative daily snowfall array (2D)
    - ref_time     : datetime object (used to check if it's the last hour of the day)

    Returns:
    - As           : updated snow age array (2D)
    - Snowfall_cum : reset cumulative snowfall array (2D if ref_time.hour == 23)
    """
    # If no snow (SWE == 0), reset age to 0
    mask_zero = SWE == 0
    As[mask_zero] = 0

    if isinstance(ref_time, np.ndarray):
        ref_time = pandas.to_datetime(ref_time)
    # Ensure ref_time is a pandas.Timestamp

    if ref_time.hour == 23:

        mask_up = Sf_daily_cum <= 3
        As[mask_up] = As[mask_up] + 1

        # If strong snowfall (> 3 mm/day), snow age resets to 0
        mask_reset = Sf_daily_cum > 3
        As[mask_reset] = 0

        # reset Sf every day
        Sf_daily_cum = np.zeros_like(SWE)



    return As,  Sf_daily_cum

# -----------------------------------------------------
# ----------------------------------------------------

# compute albedo Laramie and Schaake 1972
# compute a value of albedo based on daily mean temperature

def alb(As, albedo, T_albedo, ref_time, multiplicative_term,Ice_thickness,Ice_flag):
    """
    Update albedo based on snow age and daily mean temperature.

    Parameters:
    - As                 : 2D array of snow age (days)
    - albedo             : 2D array of current albedo
    - T_albedo           : 2D array of daily mean temperature (°C)
    - ref_time           : datetime.datetime object
    - multiplicative_term: scalar or 2D array (adjustment coefficient)

    Returns:
    - albedo_new         : 2D array of updated albedo
    """
    # Ensure ref_time is a pandas.Timestamp
    if isinstance(ref_time, np.ndarray):
        ref_time = pandas.to_datetime(ref_time)

    if ref_time.hour == 23:
        # --- Pivot albedo curves (Laramie & Schaake, 1972) ---
        days = np.arange(1, 501)
        albedo_pivot_wet = 0.5 + 0.45 * np.exp(-days * 0.12)
        albedo_pivot_dry = 0.5 + 0.45 * np.exp(-days * 0.05)

        albedo_old = albedo.copy()

            # --- WET condition (Ta > 0°C) ---
        wet_mask = T_albedo > 0

        if np.any(wet_mask):
            diff = np.abs(albedo_pivot_wet[:, None, None] - albedo_old)
            idx = np.argmin(diff, axis=0)
            next_idx = np.minimum(idx + 1, len(albedo_pivot_wet) - 1)
            albedo[wet_mask] = albedo_pivot_wet[next_idx[wet_mask]]

        # --- DRY condition (Ta ≤ 0°C) ---
        dry_mask = T_albedo <= 0
        if np.any(dry_mask):
            diff = np.abs(albedo_pivot_dry[:, None, None] - albedo_old)
            idx = np.argmin(diff, axis=0)
            next_idx = np.minimum(idx + 1, len(albedo_pivot_dry) - 1)
            albedo[dry_mask] = albedo_pivot_dry[next_idx[dry_mask]]

        # --- Boundary conditions ---
        albedo= np.clip(albedo, 0.5, 0.95)

        # --- New snow condition (Age = 0) ---
        new_snow_mask = As== 0
        albedo[new_snow_mask] = 0.95

    return albedo

"""
def alb(As, albedo, T_albedo, ref_time, multiplicative_term,Ice_thickness,Ice_flag):
   
        Update albedo based on snow age and daily mean temperature.

        Parameters:
        - As                 : 2D array of snow age (days)
        - albedo             : 2D array of current albedo
        - T_albedo           : 2D array of daily mean temperature (°C)
        - ref_time           : datetime.datetime object
        - multiplicative_term: scalar or 2D array (adjustment coefficient)

        Returns:
        - albedo_new         : 2D array of updated albedo
        
    # Ensure ref_time is a pandas.Timestamp
    if isinstance(ref_time, np.ndarray):
        ref_time = pandas.to_datetime(ref_time)

    if ref_time.hour == 23:

        albedo[As==0] = 0.95
        mask_1 = (As >0) & (T_albedo > 0)
        mask_2 =  (As >0) & (T_albedo < 0)

        albedo[mask_1] = albedo[mask_1]  - (((0.12 + As[mask_1] ) / 24) * 0.45 * np.exp(-0.12 * As[mask_1]))
        albedo[mask_2] =albedo[mask_2]  - (((0.05 + As[mask_2] ) / 24) * 0.45 * np.exp(-0.05 * As[mask_2]))

        # clip albedo between o.5 and 0.95
        albedo = np.clip(albedo, 0.5, 0.95)

    return  albedo
"""
# -----------------------------------------------------
# -----------------------------------------------------

def Sterrain(As, albedo, T_albedo, ref_time, multiplicative_term,Ice_thickness,Ice_flag,SWE_D,SWout,slope,sky_view_factor) :
    """
    Compute terrain-reflected shortwave radiation and update SWout.

    Parameters:
    - slope: Slope angle in degrees.
    - sky_view_factor: Sky view factor (dimensionless).
    - SWout: Incoming shortwave radiation (W/m²).

    Returns:
    - Updated SWout including terrain-reflected shortwave radiation.
    """
    # Convert slope to radians
    slp = np.radians(slope)
    # Flatten the array
    slp_flattened = slp.values.flatten()  # or use slp.ravel()
    svf = sky_view_factor['svf'].values  # Assuming sky_view_factor is a DataFrame
    svf= svf.flatten()

    albedo_new =alb(As, albedo, T_albedo, ref_time, multiplicative_term,Ice_thickness,Ice_flag)

    # Eq. 9b from Dozier and Frew (1990)
    Ct = 0.5 * (1 + np.cos(slp_flattened)) - svf.flatten()

    # Terrain-reflected shortwave radiation
    Sterrain = Ct * (albedo_new * (1 - albedo_new)) * SWout

    # Update SWout with terrain-reflected component
    SWout += Sterrain

    return SWout

""" 
def GlacierDeltaH(dt, Rows, Cols, iRows_Pivot, IceThickness_WE, MeltingGCumWY, Mask, PivotTable, DEM, Glaciers_ID, AreaCell):
    # Rearrange the pivot table
    DiscStepsDeltaH = PivotTable[1:iRows_Pivot, 0]
    GlacierIDinPivotTable = PivotTable[0, 1:iCols_Pivot]
    PivotTableValues = PivotTable[1:iRows_Pivot, 1:iCols_Pivot]

    for iI in range(iCols_Pivot - 1):  # Loop over the number of glaciers in the pivot table
        TempMatrix_DEM = np.copy(DEM)

        mask_glacier = (Glaciers_ID == GlacierIDinPivotTable[iI]) & (DEM >= 0.0)
        dMaxInitialThicknessGlacier = np.max(IceThickness_WE[mask_glacier])

        if dMaxInitialThicknessGlacier > 0.0:
            mask_glacier_thick = mask_glacier & (IceThickness_WE > 0.0)
            dMinElevGlacier = np.min(TempMatrix_DEM[mask_glacier_thick])
            dMaxElevGlacier = np.max(TempMatrix_DEM[mask_glacier_thick])
            dMeanElevGlacier = np.sum(TempMatrix_DEM[mask_glacier_thick]) / np.count_nonzero(mask_glacier_thick)

            if dMaxElevGlacier > dMinElevGlacier:
                TempMatrix_DEMnorm = np.where(mask_glacier_thick,
                                              (dMaxElevGlacier - TempMatrix_DEM) / (dMaxElevGlacier - dMinElevGlacier),
                                              -9999.0)

                TempMatrix_deltaH = np.full((Rows, Cols), -9999.0)
                for iI_Rows in range(Rows):
                    for iI_Cols in range(Cols):
                        if TempMatrix_DEMnorm[iI_Rows, iI_Cols] >= 0.0:
                            Index_for_deltaH = np.argmin(np.abs(DiscStepsDeltaH - TempMatrix_DEMnorm[iI_Rows, iI_Cols]))
                            TempMatrix_deltaH[iI_Rows, iI_Cols] = PivotTableValues[Index_for_deltaH, iI]

                dTemp_MaxdeltaH = np.max(TempMatrix_deltaH[mask_glacier_thick])

                if dTemp_MaxdeltaH < 0.0:
                    IceThickness_WE[mask_glacier_thick] -= MeltingGCumWY[mask_glacier_thick]
                else:
                    dTemp_GlacierArea = np.sum(AreaCell[mask_glacier_thick])
                    dTempMatric_MeltingGCumWY_this_glacier = np.sum(MeltingGCumWY[mask_glacier_thick] * (AreaCell[mask_glacier_thick] / dTemp_GlacierArea))
                    dTempMatrix_deltaH_by_Area = np.sum((AreaCell[mask_glacier_thick] / dTemp_GlacierArea) * TempMatrix_deltaH[mask_glacier_thick])
                    dTemp_f_s = dTempMatric_MeltingGCumWY_this_glacier / dTempMatrix_deltaH_by_Area

                    mask_tongue = mask_glacier_thick & (DEM < dMeanElevGlacier) & (dTemp_f_s * TempMatrix_deltaH > MeltingGCumWY)
                    IceThickness_WE[mask_tongue] -= MeltingGCumWY[mask_tongue]

                    mask_elsewhere = mask_glacier_thick & ~mask_tongue
                    IceThickness_WE[mask_elsewhere] -= dTemp_f_s * TempMatrix_deltaH[mask_elsewhere]
            else:
                IceThickness_WE[mask_glacier_thick] -= MeltingGCumWY[mask_glacier_thick]

    mask_non_dynamic = (Glaciers_ID <= 0.0) & (IceThickness_WE > 0.0) & (DEM >= 0.0)
    IceThickness_WE[mask_non_dynamic] -= MeltingGCumWY[mask_non_dynamic]

    IceThickness_WE[IceThickness_WE < 0.0] = 0.0

    return IceThickness_WE
"""

# -----------------------------------------------------
# -----------------------------------------------------

