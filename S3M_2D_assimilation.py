from statistics import NormalDist
import numpy as np
import pandas
import joblib
from scipy.stats import norm
import  pdb

def perturb_point(
    meteo_inputs,                 # shape (m,)
    state_matrix_ensemble,        # shape (time, N, n_state)
    temporary_val,
    meteo_matrix,
    output_matrix_ensemble,
    s,
    statistics,
    keys,
    inflat_deflat,                # shape (time, N)
    R,
    R_state,
    state_limits,
    state_vector,
    pert_prec,
    pert_rad,
    pert_temp,
    pert_rh,
    scale_mean_prec,
    pert_asymm_prec,
    c_asymm_prec,
    L0,
    L_tilde,
    statistics_state
):

    """
    Operates on ONE spatial point.
    """

    N = state_matrix_ensemble.shape[0]
    m = meteo_inputs.shape[0]
    epsilon = np.zeros((N, m))

    # ----------------------------------------------------------------------------------
    # Cholesky of R
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        R = 0.5 * (R + R.T)
        R_reg = R + np.eye(R.shape[0])
        L = np.linalg.cholesky(R_reg)

    # ----------------------------------------------------------------------------------
    temporary_val = (
            temporary_val
            + np.matmul(
        L0,
        np.random.normal(0, 0.01, (m, N))
    ).T
    )

    L_tilde_inv = np.linalg.inv(L_tilde)

    sampled_val = np.linalg.multi_dot([L, L_tilde_inv, temporary_val.T]).T

    mean = sampled_val.mean(axis=0, keepdims=True)
    std = sampled_val.std(axis=0, keepdims=True)
    sampled_val = (sampled_val- mean) / std

    # ----------------------------------------------------------------------------------
    prob_cum= [
        [NormalDist(0, 1).cdf(sampled_val[ i, p]) for p in range(m)]
        for i in range(N)
    ]

    # ----------------------------------------------------------------------------------
    for p in range(m):
        s[p] = pandas.DataFrame(s[p], columns=[0, 1, 2]).sort_values(by=1)

    val_tilde = [
        [
            np.interp(prob_cum[i][p], s[p].iloc[:, 2], s[p].iloc[:, 0])
            for p in range(m)
        ]
        for i in range(N)
    ]
    # transform to numpy array
    val_tilde = np.array(val_tilde)

    # ---------------------------------------------------------------------------
    for p in range(m):
        key = keys[p]
        epsilon [:, p] = (val_tilde[:, p] - statistics['statistics'][key]["mean"])
    epsilon*= inflat_deflat
    # ----------------------------------------------------------------------------
    for v in range(m):
        limits = statistics['statistics'][keys[v]]
        if v in (1, 3):  # precipitation
            limits["min"] = 0

        ep_min = epsilon[ :, v].min()
        ep_max = epsilon[:, v].max()

        if ep_min + meteo_inputs[v] < limits["min"]:
            mask = epsilon[:, v] < 0
            epsilon[ mask, v] *= ((meteo_inputs[v] - limits["min"]) / (-ep_min) )

        if ep_max + meteo_inputs[v] > limits["max"]:
            mask = epsilon[ :, v] > 0
            epsilon[ mask, v] *= ((limits["max"] - meteo_inputs[v]) / ep_max )
    # ----------------------------------------------------------------------------
    for val in range(m):
         meteo_matrix[:, val] = meteo_inputs[val] + epsilon[:, val]

    # if precipitation is equal to zero, set all ensemble members to zero
    if meteo_inputs[1] == 0:
        meteo_matrix[ :, 1] = 0

    # ----------------------------------------------------------------------------------

    epsilon_state = np.random.multivariate_normal(np.zeros(len(state_vector)), R_state, N)
    keys = list(statistics_state['statistics'].keys())
    # ----------------------------------------------------------------------------------
    for v in range(len(state_vector)):
        ep_min = epsilon_state[:, v].min()
        ep_max = epsilon_state[:, v].max()
        limits = statistics_state['statistics'][keys[v]]

        base_min = state_matrix_ensemble[:, v].min()
        base_max = state_matrix_ensemble[:, v].max()

        # lower bound
        if ep_min + base_min < limits["min"]:
            mask = epsilon_state[:, v] < 0
            epsilon_state[mask, v] *= (
                    (base_min - limits["min"]) / (-ep_min)
            )

        # upper bound
        if ep_max + base_max > limits["max"]:
            mask = epsilon_state[:, v] > 0
            epsilon_state[mask, v] *= (
                    (limits["max"] - base_max) / ep_max
            )

    state_matrix_ensemble += epsilon_state

    # ----------------------------------------------------------------------------------
    output_matrix_ensemble[:, 10] = (
    state_matrix_ensemble[:, 0] + state_matrix_ensemble[:, 1]
    )

    mask_output = output_matrix_ensemble[:, 10] <= 0

    output_matrix_ensemble[mask_output, 0] = 0
    output_matrix_ensemble[mask_output, 14] = 0
    state_matrix_ensemble[mask_output, 2] = 0
    state_matrix_ensemble[mask_output, 3] = 0.5

    # compute ONLY where valid
    valid = ~mask_output
    zero_density = valid & (state_matrix_ensemble[:, 2] == 0)
    state_matrix_ensemble[zero_density, 2] = state_limits[0][2]
    output_matrix_ensemble[valid, 14] = (
    state_matrix_ensemble[valid, 0] / 997
    + state_matrix_ensemble[valid, 1] / state_matrix_ensemble[valid, 2])
        # ----------------------------------------------------------------------------------
    if  pert_prec == 0:
        meteo_matrix[:, 1] = meteo_inputs[1]
    if  pert_rad==0:
        meteo_matrix[ :, 3] = meteo_inputs[3]
    if  pert_temp==0:
        meteo_matrix[ :, 0] = meteo_inputs[0]
        meteo_matrix[ :, 4] = meteo_inputs[4]
        meteo_matrix[ :, 5] = meteo_inputs[5]
    if  pert_rh==0:
        meteo_matrix[ :, 2] = meteo_inputs[2]

    if scale_mean_prec and meteo_inputs[1] != 0:
        meteo_matrix[ :, 1] *= meteo_inputs[1] / meteo_matrix[:, 1].mean()

    if pert_asymm_prec and meteo_inputs[1] != 0:
        mask = meteo_matrix[ :, 1] > meteo_inputs[1]
        meteo_matrix[ mask, 1] *= c_asymm_prec

    # ----------------------------------------------------------------------------------
    return meteo_matrix, state_matrix_ensemble, output_matrix_ensemble, temporary_val


def assimilation_point(meteo_matrix,state_matrix,output_matrix, y,parameters,state_limits, R_measures, Xb_old, Xa_mean,Pa,output_a_mean,N,statistics_state):
    """
    :param meteo_matrix: size (N, m)
    :param state_matrix:   size (N, n_state)
    :param output_matrix:  size (N, n_output)
    :param output_matrix_old:   size (N, 2)
    :param y:          size (2,)
    :param parameters:
    :param state_limits:
    :param B:
    :param H:
    :param R_measures:
    :param Xb_old:   size (N, n_state)
    :param Xa:    size (N, n_state)
    :param output_a:  size (N, 2)
    :param Xa_mean:         size (n_state,)
    :param Pa:     size (n_state, n_state)
    :param output_a_mean:   size (2,)
    :param N:
    :return:
    1. Xa_mean: analysis mean state vector (size (n_state,))
    2. Pa: analysis covariance matrix (size (n_state, n_state))
    3. output_a_mean: analysis mean output vector (size (2,))
    4. Xb_old: background state matrix (size (N, n_state))

    """
    Xb_old[:, :] = state_matrix[ :, :]
    Xa = np.zeros((N, 4))
    output_a = np.zeros((N, 2))
    # ------------------------------------------------------------------------------------------------------
    # Covariance matrix
    B = np.cov(state_matrix[:, :], rowvar=False)
    flag = [0, 0, 0, 0]
    # CHECK FLAG
    if np.sum(output_matrix[:, 10] > 0) > 0.1 * N:
        std_dev = np.sqrt(np.diag((B[:, :])))

        if std_dev[2] == 0:
            std_dev[2] = 15
            flag[2] = 1

        if std_dev[0] == 0:
            std_dev[0] = 10
            flag[0] = 1

        if std_dev[1] == 0:
            std_dev[1] = 50
            flag[1] = 1

        if std_dev[3] == 0:
            std_dev[3] = 0.01
            flag[3] = 1

        if 1 in flag:
            std_matrix = np.outer(std_dev, std_dev.T)
            corr_matrix = np.divide(B[:, :], std_matrix)

            if flag[0] == 1:
                corr_matrix[0, 0] = 1
                corr_matrix[0, 1] = -0.2
                corr_matrix[0, 2] = -0.3
                corr_matrix[0, 3] = -0.1
                corr_matrix[1, 0] = -0.2
                corr_matrix[2, 0] = -0.3
                corr_matrix[3, 0] = -0.1

            if flag[1] == 1:
                corr_matrix[1, 1] = 1
                corr_matrix[1, 0] = -0.2
                corr_matrix[1, 2] = 0.1
                corr_matrix[1, 3] = 0.1
                corr_matrix[0, 1] = -0.2
                corr_matrix[2, 1] = 0.3
                corr_matrix[3, 1] = 0.1

            if flag[2] == 1:
                corr_matrix[2, 2] = 1  # density
                corr_matrix[2, 0] = -0.3  # swew
                corr_matrix[2, 1] = 0.3  # swed
                corr_matrix[2, 3] = -0.5  # albedo
                corr_matrix[0, 2] = -0.3
                corr_matrix[1, 2] = 0.3
                corr_matrix[3, 2] = -0.5

            if flag[3] == 1:
                corr_matrix[3, 3] = 1
                corr_matrix[3, 0] = -0.1
                corr_matrix[3, 1] = 0.1
                corr_matrix[3, 2] = -0.5
                corr_matrix[0, 3] = -0.1
                corr_matrix[1, 3] = 0.1
                corr_matrix[2, 3] = -0.5

            B = np.multiply(corr_matrix, std_matrix)
    # ------------------------------------------------------------------------------------------------------
    inf_vect = np.array([1, 1, 1, 1])
    inf_matrix = np.outer(inf_vect, inf_vect.T)
    B = np.multiply(inf_matrix, B)
    # ------------------------------------------------------------------------------------------------------
    # Definition of observed density and its inverse at time j
    try:
        rho_obs = ((y[0] / y[1]) - 0.1 * parameters['RhoW']) / 0.9
        rho_obs_inv = 1 / rho_obs
    except ZeroDivisionError:
        rho_obs = parameters['RhoSnowMin']
        rho_obs_inv = 1 / rho_obs
    if np.isnan(rho_obs):
        rho_obs = parameters['RhoSnowMin']
        rho_obs_inv = 1 / rho_obs
    # ------------------------------------------------------------------------------------------------------
    # Cycle over the ensemble members
    n = 0
    for n  in range(N):

        # Definition of modelled density and its inverse at time j and ensemble member i
        if output_matrix[n,10] > 0:

            if output_matrix[n, 14] == 0:
                print('errore fuori physics')
                # print(result[i])
                pdb.set_trace()

            else:
                rho_calc = Xb_old[ n, 2]
                rho_calc_inv = 1 / rho_calc

        else:
            rho_calc_inv = rho_obs_inv

        if rho_calc_inv == 0 or np.isinf(rho_calc_inv):
            rho_calc_inv = 1 / parameters['RhoSnowMin']

        d_alfa_swe = ((output_matrix[ n, 8] * meteo_matrix[ n, 3]) / (  1000*parameters[
            'RhoW'] * 0.334)) * parameters['dt']

        #d_alfa_swe = ((output_matrix[ n, 8] * meteo_matrix[ n, 3] *(1-state_matrix[n,3])) / (1000 * parameters['RhoW'] * 0.334))

        if output_matrix[ n, 15] == 0:
            d_alfa_hs = d_alfa_swe / rho_obs
        else:
            d_alfa_hs = d_alfa_swe / output_matrix[n, 15]

        # --------------------------------------------------------------------------------------------------
        # Compute the observation operator matrix H for each ensemble member i

        if np.isnan(y[ :]).all():
            # No assimilation
            Xa[ n, :] = Xb_old[n, :]
            output_a[n, 0] = output_matrix[ n, 10]
            output_a[ n, 1] = output_matrix[ n, 14]

            continue

        elif not np.isnan(y[0]) and not np.isnan(y[1]):
            H = np.array([[1, 1, 0, d_alfa_swe], [1 / 997, rho_calc_inv,
                                                              -(rho_calc_inv * rho_calc_inv), d_alfa_hs]])

        elif np.isnan(y[1]) and not np.isnan(y[0]):

            H = np.array([[1, 1, 0, d_alfa_swe], [np.nan, np.nan, np.nan, np.nan]])

        elif np.isnan(y[ 0]) and not np.isnan(y[1]):
            H= np.array([[np.nan, np.nan, np.nan, np.nan], [1 / 997, rho_calc_inv,
                                                                         -(rho_calc_inv * rho_calc_inv),
                                                                         d_alfa_hs]])

        # --------------------------------------------------------------------------------------------------
        Ht = H[ :, :]
        idx = np.where(~np.isnan(y[:]))[0]
        CC = np.matmul(B[:, :], Ht[idx, :].T)
        HBH = np.matmul(np.matmul(Ht[idx, :], B[ :, :]), Ht[idx, :].T)
        K = np.matmul(CC, np.linalg.inv(HBH + R_measures[idx][:, idx]))

        y_mod = np.matmul(Ht[idx, :], Xb_old[ n, :])
        Xa[ n, :] = Xb_old[n, :] + np.matmul(K, (y[idx] - y_mod))

        # --------------------------------------------------------------------------------------------------
        # Analysis check and rescale

        M_tot = 0
        idx_scale = 0
        rescale = 0
        keys = list(statistics_state['statistics'].keys())

        for k in range(2):
            limits = statistics_state['statistics'][keys[k]]
            if Xa[n, k] < limits["min"] or Xa[ n, k] > limits["max"]:
                rescale = 1
                M = max(limits["min"] - Xa[ n, k], Xa[n, k] -limits["max"])
                if M > M_tot:
                    M_tot = M
                    idx_scale = k

        if rescale == 1:

            Xa[ n, :2] = Xb_old[n, :2] + ((Xa[n, :2] - Xb_old[n, :2]) * (
                    (np.abs(Xa[n, idx_scale] - Xb_old[ n, idx_scale])) - M_tot) / np.abs(
                Xa[ n, idx_scale] - Xb_old[ n, idx_scale]))

        if np.isnan(Xa[ n, :]).any():
            Xa[ n, :] = Xb_old[n, :]
        # --------------------------------------------------------------------------
        output_a[n, 0] = Xa[n, 0] + Xa[n, 1]

        Xa[ n, 2] = max(min(Xa[ n, 2], statistics_state['statistics'][keys[2]]["max"]), statistics_state['statistics'][keys[2]]["min"])
        if output_a[ n, 0] > 0:
             Xa[ n, 2] = max(min(Xa[ n, 2],state_limits[1][2]), state_limits[0][2])

        Xa[ n, 3] = max(min(Xa[ n, 3], statistics_state['statistics'][keys[3]]["max"]), statistics_state['statistics'][keys[3]]["min"])

        if output_a[ n, 0] <= 0:
            output_a[ n, 0] = 0
            output_a[ n, 1] = 0
            Xa[ n, 2] = 0
            Xa[ n, 3] = 0.5

        else:

            output_a[n, 1] = (Xa[ n, 0] / 997) + (Xa[ n, 1] / Xa[ n, 2])

        # ------------------------------------------------------------------------------------------------------
    # Compute the mean of the analysis and the covariance matrix
    Xa_mean[:] = np.mean(Xa[ :, :], axis=0)
    Pa[ :, :] = np.cov(Xa[:, :], rowvar=False)
    output_a_mean [:] = np.mean(output_a[ :, :], axis=0)  # correct trajectory


    if output_a_mean[0] <= 0:
        output_a_mean[0] = 0
        output_a_mean[1] = 0
        Xa_mean[2] = 0
        Xa_mean[3] = 0.5
    else:
        if Xa_mean[2] < state_limits[0][2]:
            Xa_mean[2] = state_limits[0][2]
        elif Xa_mean[2] > state_limits[1][2]:
             Xa_mean[2] = state_limits[1][2]
        if Xa_mean[3] < state_limits[0][3]:
            Xa_mean[3] = state_limits[0][3]
        elif Xa_mean[3] > state_limits[1][3]:
            Xa_mean[3] = state_limits[1][3]

    # ------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    return Xa_mean, Pa, output_a_mean, Xb_old




