import math
import numpy as np
import pdb
# -----------------------------------------------------
# -----------------------------------------------------
# Froidurot et. al 2014 PRECIPITATION-PHASE partitioning

def PhasePart(P, alpha, beta, gamma, T_air, RH, change_part):
    Snowfall = np.zeros_like(P)
    Rainfall = np.zeros_like(P)

    mask_positive_P = P > 0

    if change_part == 1:
        Snowfall[mask_positive_P] = np.where(T_air[mask_positive_P] <= 0.5, P[mask_positive_P], 0)
        Rainfall[mask_positive_P] = np.where(T_air[mask_positive_P] > 0.5, P[mask_positive_P], 0)
    else:
        SepCoeff = 1 / (1 + np.exp(alpha + beta * T_air + gamma * RH))
        Snowfall[mask_positive_P] = P[mask_positive_P] * (1 - SepCoeff[mask_positive_P])
        Rainfall[mask_positive_P] = P[mask_positive_P] * SepCoeff[mask_positive_P]

    Snowfall = np.where(Snowfall >= 0.01, Snowfall, 0)
    Rainfall = np.where(Rainfall >= 0.01, Rainfall, 0)

    return Snowfall, Rainfall
# -----------------------------------------------------
# -----------------------------------------------------


# Update density
import numpy as np

def density(Rho_D_min, Rho_D_max, Rho_S_max, RhoW, dt, state_vector, output_vector, SWE_D, Snowfall, T_air):
    Rho_D = state_vector[:, 2]
    RhoS0 = output_vector[:, 15]

    # New snow events
    mask_snowfall_positive = Snowfall > 0
    RhoS0[mask_snowfall_positive] = 67.9 + 51.3 * np.exp(T_air[mask_snowfall_positive] / 2.6)
    RhoS0 = np.clip(RhoS0, Rho_D_min, Rho_S_max)

    mask_snowfall_zero = Snowfall == 0
    RhoS0[mask_snowfall_zero] = 0

    mask_swe_d_greater = (SWE_D - Snowfall > 1) & (Snowfall > 1) & (Rho_D > Rho_D_min)
    Rho_D[mask_swe_d_greater] = SWE_D[mask_swe_d_greater] / (
        (Snowfall[mask_swe_d_greater] / RhoS0[mask_swe_d_greater]) +
        ((SWE_D[mask_swe_d_greater] - Snowfall[mask_swe_d_greater]) / Rho_D[mask_swe_d_greater])
    )

    mask_swe_d_less_equal = (SWE_D - Snowfall <= 1) & (Snowfall > 0)
    Rho_D[mask_swe_d_less_equal] = RhoS0[mask_swe_d_less_equal]

    # Check snow limits
    Rho_D = np.clip(Rho_D, Rho_D_min, Rho_D_max)

    # Compute updated dry-snow height
    H_D = np.where(Rho_D > 0, ((SWE_D / 1000) * RhoW) / Rho_D, 0)

    # Linear approximation for snow temperature in ˚C
    SnowTemp = np.where(T_air >= 0, 0, 0.5 * T_air)

    # Dry-snow density compaction
    mask_swe_d_positive = SWE_D > 0
    Rho_D[mask_swe_d_positive] += 0.66 * (dt / 3600) * 0.001 * H_D[mask_swe_d_positive] * (Rho_D[mask_swe_d_positive] ** 2) * np.exp(
        0.08 * SnowTemp[mask_swe_d_positive] - 0.021 * Rho_D[mask_swe_d_positive]
    )

    # Check snow limits again
    Rho_D = np.clip(Rho_D, Rho_D_min, Rho_D_max)

    # Updated snow height
    H_D = np.where(Rho_D > 0, ((SWE_D / 1000) * RhoW) / Rho_D, 0)

    return Rho_D, RhoS0, SnowTemp, H_D


# -----------------------------------------------------
# -----------------------------------------------------

# compute snowpack outflow based on Darcy's

def Hydraulics(Rho_D, RhoW, SWE_D, SWE_W, H_D, dt):
    Cond = np.zeros_like(Rho_D)

    Porosity = np.where(H_D > 0, 1 - Rho_D / 917, 0)

    # Compute control volume and saturation degree
    H_S = np.where((H_D > 0) & ((SWE_W / 1000) - (Porosity * H_D) >= 0),
                   H_D + ((SWE_W / 1000) - (Porosity * H_D)),
                   np.where(H_D > 0, H_D, 0))
    Sr = np.where((H_D > 0) & ((SWE_W / 1000) - (Porosity * H_D) >= 0), 1.0,
                  np.where(H_D > 0, (SWE_W / 1000) / (Porosity * H_D), 0))

    Sr_irr = np.where(Porosity > 0, 0.02 * ((Rho_D / RhoW) / Porosity), 0)
    Sr_star = np.where(Porosity > 0, (Sr - Sr_irr) / (1 - Sr_irr), 0)

    # Compute SSA, r_e, permeability, and conductivity
    SSA = np.where(Rho_D > 0, -308.2 * np.log((Rho_D / 1000)) - 206, 0)
    SSA = SSA / 10  # now SSA is in m2/kg
    e = np.where(Rho_D > 0, 3 / (SSA * 917), 0)
    Perm = np.where(Rho_D > 0, 3 * (e ** 2) * np.exp(-0.013 * Rho_D), 0)

    Cond = np.where(Sr < Sr_irr, 0, Perm * (Sr_star ** 3))

    # Compute outflow
    Outflow_K = np.where((Sr >= 0.5) | (SWE_D < 10), SWE_W,
                         np.where(SWE_W < 5.47 * (10 ** 5) * Cond * dt * 1000, SWE_W,
                                  5.47 * (10 ** 5) * Cond * dt * 1000))

    Outflow_K = np.where(Outflow_K <= 0.01, 0, Outflow_K)

    return Outflow_K, H_S

# -----------------------------------------------------
# -----------------------------------------------------

def refreezing(T_air, T_melting, SWE_W, mr0, cm, Ttau):
    mr = 0.598862 * np.arctan(0.27439 * T_melting - 0.5988) - 0.598862 * (np.pi / 2) + mr0
    mr = np.maximum(mr, 0)

    R = np.where((T_air < Ttau) & (SWE_W > 0), -cm * mr * (T_air - Ttau), 0)

    return R
# addition

# -----------------------------------------------------
# -----------------------------------------------------


def melting(ref_time, mrad0, mr0, T_air, T_melting, T_albedo, Ttau, Radiation, RhoW, dt, cm, SWE_D, albedo, As, SWE,
            Sf_daily_cum, multiplicative_term):
    lambdaf = 0.334
    As, Sf_daily_cum = snow_age(As, ref_time, SWE, Sf_daily_cum)
    albedo = alb(As, albedo, T_albedo, ref_time, multiplicative_term)

    mrad = np.where(mrad0 == 0, 0, 0.49338 * np.arctan(0.27439 * T_melting - 0.5988) - 0.49338 * (np.pi / 2) + mrad0)
    mr = 0.598862 * np.arctan(0.27439 * T_melting - 0.5988) - 0.598862 * (np.pi / 2) + mr0

    mr = np.maximum(mr, 0)
    mrad = np.maximum(mrad, 0)

    mask_melting_conditions = (T_air >= Ttau) & (T_melting >= Ttau) & (SWE_D > 0)

    M_rad = np.where(mask_melting_conditions, mrad * ((Radiation * (1.0 - albedo)) / (1000.0 * RhoW * lambdaf)) * dt, 0)
    M_rad = np.where(M_rad < 0.01, 0, M_rad)

    M_temp = np.where(mask_melting_conditions, cm * mr * (T_air - Ttau), 0)
    M_temp = np.where(M_temp < 0.01, 0, M_temp)

    M = M_temp + M_rad

    return M, albedo, As, Sf_daily_cum, mrad, mr


# -----------------------------------------------------
# ----------------------------------------------------

# To compute snow age consider 1 d with at least 3 mm of total snow fall
def snow_age(As, ref_time, SWE, Sf_daily_cum):
    """Update snow age and daily cumulative snowfall."""
    # Assuming ref_time is a vector of shape (40)
    mask_hour_23 = (ref_time == 23)
    mask_snowfall_low = (Sf_daily_cum < 0.1)

    # Ensure As and Sf_daily_cum are vectors of shape (40)
    As = np.where(mask_hour_23 & mask_snowfall_low, As + 1, As)
    Sf_daily_cum = np.where(mask_hour_23, 0, Sf_daily_cum)

    return As, Sf_daily_cum


# -----------------------------------------------------
# ----------------------------------------------------

# compute albedo Laramie and Schaake 1972
# compute a value of albedo based on daily mean temperature

def alb(As, albedo, T_albedo, ref_time, multiplicative_term):
    # Create masks for the conditions
    mask_hour_23 = ref_time.hour == 23
    mask_as_zero = As == 0
    mask_t_albedo_positive = T_albedo > 0

    # Initialize albedo_new with the current albedo values
    albedo_new = np.copy(albedo)

    # Update albedo_new based on the conditions
    albedo_new[mask_hour_23 & mask_as_zero] = 0.95

    tau = np.where(mask_t_albedo_positive, 0.12, 0.05)
    albedo_update = albedo - multiplicative_term * (((tau + As) / 24) * 0.45 * np.exp(-tau * As))
    albedo_new[mask_hour_23 & ~mask_as_zero] = albedo_update[mask_hour_23 & ~mask_as_zero]

    # Ensure albedo_new is within the specified bounds
    albedo_new = np.clip(albedo_new, 0.5, 0.95)

    return albedo_new


# -----------------------------------------------------
# -----------------------------------------------------

