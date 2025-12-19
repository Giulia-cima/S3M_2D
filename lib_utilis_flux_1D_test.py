import math
import numpy as np
# import pdb
# -----------------------------------------------------
# -----------------------------------------------------
# Froidurot et. al 2014 PRECIPITATION-PHASE partitioning
""""
def PhasePart(P, alpha, beta, gamma, T_air, RH, change_part):
    if P > 0:

        if change_part == 1:

            if T_air > 0.5:
                Snowfall = 0
                Rainfall = P

            else:
                Snowfall = P
                Rainfall = 0
        else:
            SepCoeff = 1 / (1 + math.exp(alpha + beta * T_air + gamma * RH))

            Snowfall = P * (1 - SepCoeff)
            Rainfall = SepCoeff * P

        if Snowfall < 0.01:
            Snowfall = 0

        if Rainfall < 0.01:
            Rainfall = 0
    else:
        Snowfall = 0
        Rainfall = 0

    return Snowfall, Rainfall
"""


def PhasePart(P, alpha, beta, gamma, T_air, RH, change_part):
    if P > 0:
        if change_part == 1:
            Snowfall = P if T_air <= 0.5 else 0
            Rainfall = P if T_air > 0.5 else 0
        else:
            SepCoeff = 1 / (1 + math.exp(alpha + beta * T_air + gamma * RH))
            Snowfall = P * (1 - SepCoeff)
            Rainfall = SepCoeff * P

        Snowfall = Snowfall if Snowfall >= 0.01 else 0
        Rainfall = Rainfall if Rainfall >= 0.01 else 0
    else:
        Snowfall = 0
        Rainfall = 0

    return Snowfall, Rainfall
# -----------------------------------------------------
# -----------------------------------------------------


# Update density
def density(Rho_D_min, Rho_D_max, Rho_S_max, RhoW, dt, state_vector, output_vector, SWE_D, Snowfall, T_air):
    Rho_D = state_vector[2]
    RhoS0 = output_vector[15]

    # new snow events

    if Snowfall > 0:
        RhoS0 = 67.9 + 51.3 * math.exp(T_air / 2.6)  # fresh snow density

        if RhoS0 > Rho_S_max:
            RhoS0 = Rho_S_max

        if RhoS0 < Rho_D_min:
            RhoS0 = Rho_D_min

    elif Snowfall == 0:
        RhoS0 = 0

    if (SWE_D - Snowfall > 1) and Snowfall > 1 and Rho_D > Rho_D_min:
        Rho_D = SWE_D / ((Snowfall / RhoS0) + ((SWE_D - Snowfall) / Rho_D))  # kg/m^3

    elif (SWE_D - Snowfall <= 1) and Snowfall > 0:
        Rho_D = RhoS0  # kg/m^3

    # check snow limits
    if Rho_D > Rho_D_max:
        Rho_D = Rho_D_max

    if Rho_D < Rho_D_min:
        Rho_D = Rho_D_min

    # Compute  updated dry - snow height
    if Rho_D > 0:
        H_D = ((SWE_D / 1000) * RhoW) / Rho_D  # m
    else:
        H_D = 0

        # linear approximation for snow temperature in ˚C
    if T_air >= 0:
        SnowTemp = 0
    else:
        SnowTemp = 0.5 * T_air

    # dry-snow density compaction
    if SWE_D > 0:
        Rho_D = Rho_D + 0.66 * (dt / 3600) * 0.001 * H_D * (Rho_D ** 2) * math.exp(
            0.08 * SnowTemp - 0.021 * Rho_D)

    # check snow limits
    if Rho_D > Rho_D_max:
        Rho_D = Rho_D_max

    if Rho_D < Rho_D_min:
        Rho_D = Rho_D_min

    # updated snow height
    if Rho_D > 0:
        H_D = ((SWE_D / 1000) * RhoW) / Rho_D
    else:
        H_D = 0

    return Rho_D, RhoS0, SnowTemp, H_D


# -----------------------------------------------------
# -----------------------------------------------------

# compute snowpack outflow based on Darcy's
def Hydraulics(Rho_D, RhoW, SWE_D, SWE_W, H_D, dt):
    Cond = 0

    if H_D > 0:
        Porosity = 1 - Rho_D / 917  # ice density = 917 kg/m3 # no unit for porosity
    else:
        Porosity = 0

    # Compute control volume and saturation degree
    # divide by 1000 to convert  mm to m and so  compare SWE_W with H_D
    if H_D > 0 and ((SWE_W / 1000) - (Porosity * H_D)) >= 0:

        H_S = H_D + ((SWE_W / 1000) - (Porosity * H_D))
        Sr = 1.0

    elif H_D > 0:

        H_S = H_D
        Sr = (SWE_W / 1000) / (Porosity * H_D)

    else:
        H_S = 0
        Sr = 0

    if Porosity > 0:

        Sr_irr = 0.02 * ((Rho_D / RhoW) / Porosity)
        Sr_star = (Sr - Sr_irr) / (1 - Sr_irr)

    else:
        Sr_irr = 0
        Sr_star = 0

    # Compute SSA, r_e, permeability, and conductivity
    if Rho_D > 0:
        SSA = -308.2 * np.log((Rho_D / 1000)) - 206  # SSA sm2/g so we change rho into g/cm3
        SSA = SSA / 10  # now SSA is in m2/kg
        e = 3 / (SSA * 917)
        Perm = 3 * (e ** 2) * (math.exp(-0.013 * Rho_D))

        if Sr < Sr_irr:
            Cond = 0
        else:
            Cond = Perm * (Sr_star ** 3)

    # compute outflow
    if Sr >= 0.5 or SWE_D < 10:
        Outflow_K = SWE_W

    elif SWE_W < 5.47 * (10 ** 5) * Cond * dt * 1000:  # potential drainage ,if not respected I can get swe<0
        Outflow_K = SWE_W

    else:
        Outflow_K = 5.47 * (10 ** 5) * Cond * dt * 1000

    if Outflow_K <= 0.01:
        Outflow_K = 0

    return Outflow_K, H_S


# -----------------------------------------------------
# -----------------------------------------------------

def refreezing(T_air, T_melting, SWE_W, mr0, cm, Ttau):
    mr = 0.598862 * np.arctan(0.27439 * T_melting - 0.5988) - 0.598862 * (3.14 / 2) + mr0
    if mr < 0:
        mr = 0

    if T_air < Ttau and SWE_W > 0:
        R = -cm * mr * (T_air - Ttau)

    else:
        R = 0

    return R
# addition

# -----------------------------------------------------
# -----------------------------------------------------


def melting(ref_time, mrad0, mr0, T_air, T_melting, T_albedo, Ttau, Radiation, RhoW, dt, cm, SWE_D, albedo, As, SWE,
            Sf_daily_cum, multiplicative_term):
    lambdaf = 0.334
    As, Sf_daily_cum = snow_age(As, ref_time, SWE, Sf_daily_cum)
    albedo = alb(As, albedo, T_albedo, ref_time, multiplicative_term)

    if mrad0 == 0:  # IN CASE OF CALIBRATING A DEGREE DAY MODEL
        mrad = 0
    else:
        mrad = 0.49338 * np.arctan(0.27439 * T_melting - 0.5988) - 0.49338 * (math.pi / 2) + mrad0  # previously 2.07

    mr = 0.598862 * np.arctan(0.27439 * T_melting - 0.5988) - 0.598862 * (math.pi / 2) + mr0

    if mr < 0:
        mr = 0

    if mrad < 0:
        mrad = 0

    if T_air >= Ttau and T_melting >= Ttau and SWE_D > 0:

        M_rad = mrad * ((Radiation * (1.0 - albedo)) / (1000.0 * RhoW * lambdaf)) * dt  # what if not dt but time_step

        if M_rad < 0.01:
            M_rad = 0

        M_temp = cm * mr * (T_air - Ttau)          # *(1+0.01*Rainfall)  # cm is correct otherwise I will not get mm
        if M_temp < 0.01:
            M_temp = 0

        M = (M_temp + M_rad)

    else:
        M = 0

    return M, albedo, As, Sf_daily_cum, mrad, mr


# -----------------------------------------------------
# ----------------------------------------------------

# To compute snow age consider 1 d with at least 3 mm of total snow fall
def snow_age(As, ref_time, SWE, Sf_daily_cum):
    if ref_time.hour == 23:
        if Sf_daily_cum <= 3:
            As = As + 1

        elif Sf_daily_cum > 3:
            As = 0

        Sf_daily_cum = 0

    if SWE == 0:
        As = 0

    return As, Sf_daily_cum


# -----------------------------------------------------
# ----------------------------------------------------

# compute albedo Laramie and Schaake 1972
# compute a value of albedo based on daily mean temperature
def alb(As, albedo, T_albedo, ref_time, multiplicative_term):
    if ref_time.hour == 23:  # newly add

        if As == 0:
            albedo_new = 0.95
        else:

            if T_albedo > 0:
                tau = 0.12    # 1/days while As is in days
            else:
                tau = 0.05  # 1/days while As is in days

            albedo_new = albedo - multiplicative_term * (((tau+As)/24)*0.45 * math.exp(-tau * As)) # changed ,more coherent


        if albedo_new <= 0.5:
            albedo_new = 0.5

        elif albedo_new > 0.95:
            albedo_new = 0.95

    else:
        albedo_new = albedo

    return albedo_new


# -----------------------------------------------------
# -----------------------------------------------------

""""
def sanity_check(state_vector, output_vector):
    if output_vector[10] < 0.01:
        output_vector[10] = 0
        for i in range(len(state_vector)):
            state_vector[i] = 0
        for i in range(11, len(output_vector)):
            output_vector[i] = 0
    return state_vector
"""
# -----------------------------------------------------
# -----------------------------------------------------
