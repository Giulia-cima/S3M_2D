from statistics import NormalDist
import numpy as np
import pandas
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

    N = state_matrix_ensemble.shape[0]
    m = meteo_inputs.shape[0]
    epsilon = np.zeros((N, m))
    s_size =state_matrix_ensemble.shape[1]
    prob_cum = np.zeros((N, m))
    sampled_val = np.zeros((N, m))

    # ----------------------------------------------------------------------------------
    # Cholesky of R
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # Ensure symmetry and regularization
        R = (R + R.T) / 2
        R_reg = R + np.eye(R.shape[0])
        L = np.linalg.cholesky(R_reg)

    # Temporary value update
    temporary_val[:, :] = temporary_val[ :, :] + (
        np.dot(L0, np.random.normal(0, 0.01, (m, N)))).T

    # Compute inverse of L_tilde
    L_tilde_inv = np.linalg.inv(L_tilde)

    sampled_val[:,:]= np.dot(np.dot(L, L_tilde_inv), temporary_val[ :, :].T).T

    mean = sampled_val.mean(axis=0, keepdims=True)
    std = sampled_val.std(axis=0, keepdims=True)
    sampled_val = (sampled_val - mean) / std

    # ----------------------------------------------------------------------------------
    # Compute cumulative probabilities
    prob_cum[:,:] = [[NormalDist(mu=0, sigma=1).cdf(sampled_val[i, p])
                          for p in range(m)] for i in range(N)]

    # ----------------------------------------------------------------------------------
    for p in range(m):
        s[p] = pandas.DataFrame(s[p], columns=[0, 1, 2])
        s[p].sort_values(by=1, inplace=True)

    val_tilde = [
        [
            np.interp(prob_cum[i][p], s[p].iloc[:, 2].values, s[p].iloc[:, 0].values)
            for p in range(m)
        ]
        for i in range(N)
    ]
    # transform to numpy array 100 x m
    val_tilde =  np.array(val_tilde)

    # ---------------------------------------------------------------------------
    for p in range(m):
        key = keys[p]
        epsilon [:, p] = (val_tilde[:, p] - statistics['statistics'][key]["mean"])

    epsilon= epsilon* inflat_deflat
    # ----------------------------------------------------------------------------
    for v in range(m):
        limits = statistics['statistics'][keys[v]]
        if v in (1, 3):  # precipitation
            limits["min"] = 0

        ep_min = epsilon[ :, v].min()
        ep_max = epsilon[:, v].max()

        if ep_min + meteo_inputs[v] < limits["min"]:
            mask = epsilon[:, v] < 0
            epsilon[ mask, v] = epsilon[ mask, v]* ((meteo_inputs[v] - limits["min"]) / (-ep_min) )

        if ep_max + meteo_inputs[v] > limits["max"]:
            mask = epsilon[ :, v] > 0
            epsilon[ mask, v] *= ((limits["max"] - meteo_inputs[v]) / ep_max )
    # ----------------------------------------------------------------------------
    for val in range(m):
         meteo_matrix[:, val] = meteo_inputs[val] + epsilon[:, val]

    # if precipitation is equal to zero, set all ensemble members to zero
    if meteo_inputs[1] == 0:
        meteo_matrix[ :, 1] = 0

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

    if scale_mean_prec == 1 and meteo_inputs[1] != 0:
        meteo_matrix[ :, 1] *= meteo_inputs[1] / meteo_matrix[:, 1].mean()

    if pert_asymm_prec ==1 and meteo_inputs[1] != 0:
        mask = meteo_matrix[ :, 1] > meteo_inputs[1]
        meteo_matrix[ mask, 1] *= c_asymm_prec

    # ----------------------------------------------------------------------------------
    if 1:

        epsilon_state = np.random.multivariate_normal(np.zeros(s_size), R_state, N)
        for v in range(s_size):
            ep_min_val_state = np.min(epsilon_state[ :, v])
            ep_max_val_state = np.max(epsilon_state[:, v])
            for i in range(N):
                if ep_min_val_state + state_matrix_ensemble[i, v] < state_limits[0][v]:
                    mask_neg = epsilon_state[ :, v] < 0
                    idx_state = np.where(mask_neg)[0]
                    epsilon_state[ idx_state, v] *= (state_matrix_ensemble[ i, v] - state_limits[0][v]) / (
                        -ep_min_val_state)

                if ep_max_val_state + state_matrix_ensemble[i, v] > state_limits[1][v]:
                    mask_pos = epsilon_state[ :, v] > 0
                    idx_max_state = np.where(mask_pos)[0]
                    epsilon_state[idx_max_state, v] *= (state_limits[1][v] - state_matrix_ensemble[
                        i, v]) / ep_max_val_state

        state_matrix_ensemble[ :, :] = state_matrix_ensemble[ :, :] + epsilon_state[ :, :]
        i = 0
        for i in range(N):
            # set state limit for density and albedo
            state_matrix_ensemble[ :, 2] = max(min(state_matrix_ensemble[ i, 2], state_limits[1][2]),
                                            state_limits[0][2])
            state_matrix_ensemble[ :, 3] = max(min(state_matrix_ensemble[ i, 3], state_limits[1][3]),
                                            state_limits[0][3])

            output_matrix_ensemble[ i, 0] = state_matrix_ensemble[ i, 0] + state_matrix_ensemble[ i, 1]

            if output_matrix_ensemble[ i, 0] == 0:
                output_matrix_ensemble[i, 1] = 0
                state_matrix_ensemble[ i, 2] = state_limits[0][2]

            else:
                output_matrix_ensemble[ i, 1] = (state_matrix_ensemble[ i, 0] / 997) + (
                        state_matrix_ensemble[ i, 1] / state_matrix_ensemble[ i, 2])

        # check if there is any xb_old below zero
        mask_state = state_matrix_ensemble[:, :] < 0
        if np.any(mask_state):
            print("warning: negative xb_old values after perturbation")
    # ----------------------------------------------------------------------------------
    return meteo_matrix, state_matrix_ensemble, output_matrix_ensemble, temporary_val


def assimilation_point(meteo_matrix,state_matrix,output_matrix, y,parameters ,state_limits, R_measures, N,statistics_state):
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

    Xa = np.zeros((N, state_matrix.shape[1]))  # analysis state matrix
    output_a = np.zeros((N, 2))  # analysis output matrix
    H = np.zeros((N, 2, state_matrix.shape[1]))  # observation operator matrix for each ensemble member
    # covariance Matrix
    # Background

    # ------------------------------------------------------------------------------------------------------
    # Covariance matrix
    B = np.cov(state_matrix[:, :], rowvar=False)
    flag = [0, 0, 0, 0]
    # CHECK FLAG
    if np.sum( output_matrix[:, 14] > 0) > 0.1 * N:
        std_dev = np.sqrt(np.diag((B)))

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
            corr_matrix = np.divide(B, std_matrix)

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

            B= np.multiply(corr_matrix, std_matrix)
    # ------------------------------------------------------------------------------------------------------
    inf_vect = np.array([1, 1, 1, 1])
    inf_matrix = np.outer(inf_vect, inf_vect.T)
    B= np.multiply(inf_matrix, B)
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
    for i in range(N):
        # Definition of modelled density and its inverse at time j and ensemble member i
        if output_matrix[i, 10] > 0:

            if output_matrix[ i, 14] == 0:
                print('errore fuori physics')
                # print(result[i])
                pdb.set_trace()

            else:
                rho_calc = state_matrix[i, 2]
                rho_calc_inv = 1 / rho_calc

        else:
            rho_calc_inv = rho_obs_inv

        if rho_calc_inv == 0 or np.isinf(rho_calc_inv):
            rho_calc_inv = 1 / parameters['RhoSnowMin']

        d_alfa_swe = ((output_matrix[i, 8] * meteo_matrix[i, 3]) / (1000 * parameters[
            'RhoW'] * 0.334)) * parameters['dt']

        if output_matrix[i, 15] == 0:
            d_alfa_hs = d_alfa_swe / rho_obs
        else:
            d_alfa_hs = d_alfa_swe / output_matrix[i, 15]

        # --------------------------------------------------------------------------------------------------
        # Compute the observation operator matrix H for each ensemble member i

        if np.isnan(y[:]).all():
            # No assimilation
            Xa[i, :] = state_matrix[ i, :]
            output_a[i, 0] = output_matrix[i, 10]
            output_a[i, 1] = output_matrix[ i, 14]
            continue

        elif not np.isnan(y[0]) and not np.isnan(y[1]):
            H[ i, :, :] = np.array([[1, 1, 0, d_alfa_swe], [1 / 997, rho_calc_inv,
                                                              -(rho_calc_inv * rho_calc_inv), d_alfa_hs]])

        elif np.isnan(y[1]) and not np.isnan(y[0]):

            H[i, :, :] = np.array([[1, 1, 0, d_alfa_swe], [np.nan, np.nan, np.nan, np.nan]])

        elif np.isnan(y[0]) and not np.isnan(y[1]):
            H[i, :, :] = np.array([[np.nan, np.nan, np.nan, np.nan], [1 / 997, rho_calc_inv,
                                                                         -(rho_calc_inv * rho_calc_inv),
                                                                         d_alfa_hs]])

        # --------------------------------------------------------------------------------------------------
        Ht = H[i, :, :]
        idx = np.where(~np.isnan(y[ :]))[0]
        CC = np.matmul(B[:, :], Ht[idx, :].T)
        HBH = np.matmul(np.matmul(Ht[idx, :], B[ :, :]), Ht[idx, :].T)
        K = np.matmul(CC, np.linalg.inv(HBH + R_measures[idx][:, idx]))

        y_mod = np.matmul(Ht[idx, :], state_matrix[ i, :])
        Xa[i, :] = state_matrix[ i, :] + np.matmul(K, (y[ idx] - y_mod))

        # --------------------------------------------------------------------------------------------------
        # Analysis check and rescale

        M_tot = 0
        idx_scale = 0
        rescale = 0

        for k in range(2):
            if Xa[i, k] < state_limits[0][k] or Xa[ i, k] > state_limits[1][k]:
                rescale = 1
                M = max(state_limits[0][k] - Xa[i, k], Xa[ i, k] - state_limits[1][k])
                if M > M_tot:
                    M_tot = M
                    idx_scale = k

        if rescale == 1:
            # store the not correct analysis

            Xa[ i, :2] = state_matrix[ i, :2] + ((Xa[ i, :2] - state_matrix[ i, :2]) * (
                    (np.abs(Xa[ i, idx_scale] - state_matrix[i, idx_scale])) - M_tot) / np.abs(
                Xa[i, idx_scale] - state_matrix[ i, idx_scale]))

        if np.isnan(Xa[ i, :]).any():
            Xa[ i, :] = state_matrix[i, :]
        # --------------------------------------------------------------------------
        Xa[ i, 0] = max(min(Xa[ i, 0], state_limits[1][0]), state_limits[0][0])
        Xa[i, 1] = max(min(Xa[ i, 1], state_limits[1][1]), state_limits[0][1])
        # --------------------------------------------------------------------------

        Xa[ i, 2] = max(min(Xa[ i, 2], state_limits[1][2]), state_limits[0][2])
        Xa[i, 3] = max(min(Xa[i, 3], state_limits[1][3]), state_limits[0][3])

        output_a[ i, 0] = Xa[ i, 0] + Xa[ i, 1]

        if output_a[ i, 0] == 0:
            output_a[ i, 1] = 0
            Xa[ i, 2] = state_limits[0][2]

        else:
            output_a[i, 1] = (Xa[i, 0] / 997) + (Xa[ i, 1] / Xa[i, 2])

    # ------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    return  Xa, output_a


