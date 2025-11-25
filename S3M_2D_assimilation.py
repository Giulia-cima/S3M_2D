
import numpy as np
from scipy.stats import norm
import  pdb
def pre_perturbation_val(R_dict, quantile_data_dict, obs_mask):

        L_dict = {}
        s_list = {}

        for (lat,lon) in obs_mask:

            R1 = R_dict[(lat,lon)]
            R= R1["R"].values

            try:
                L_dict[(lat, lon)] = np.linalg.cholesky(R)
            except np.linalg.LinAlgError:
                print("Matrix not positive definite, applying fix.")
                R = (R + R.T) / 2
                R = R+ np.eye(R.shape[0])
                L_dict[(lat, lon)] = np.linalg.cholesky(R)

            q = quantile_data_dict[(lat, lon)]
            s_list[(lat, lon)] = []  # initialize

            for var in ["air_temp_degC", "prc_mm", "swin_wm", "rel_hum_perc", "T_albedo", "T_melting"]:
                df = q[var].sort_values("emp_rip")
                s_list[(lat, lon)].append((df.iloc[:, 1].values, df.iloc[:, 0].values))  # y, x for np.interp

        return L_dict, s_list
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def perturb_meteo_state(
        meteo,  m, L_dict, N, L0, L_tilde, s_dict, statistics_matrix,
        inflat_deflat, R_state, state_matrix,
        state_limits, pert_prec, pert_rad,
        pert_temp, pert_rh, scale_mean_prec, c_asymm_prec,pert_asymm_prec, temp_old,obs_mask):
    """
    Perturb the meteorological forcing fields and state matrix using quantile mapping and multivariate normal sampling.
    """

    state_matrix_pert = np.zeros((state_matrix.shape[0], state_matrix.shape[1], N))
    meteo_ensemble = np.zeros((meteo.shape[0], meteo.shape[1], N))
    temp_new = np.zeros_like(temp_old)
    epsilon = np.zeros((m, N))
    val_tilde = np.zeros((m, N))
    num_i = meteo.shape[0]

    inflat_deflat_array = np.array(inflat_deflat).reshape(m,1) * np.ones((1,N))
    rand_norm = np.random.normal(0, 0.01, (num_i, m, N))

    keys = list(statistics_matrix[(0, 0)]['statistics'].keys())
    i=0
    for (lat,lon) in obs_mask:

        L = L_dict[(lat,lon)]

        temp_new[i, :, :] = temp_old[i, :, :] + L0 @ rand_norm[i]
        L_tilde_inv = np.linalg.inv(L_tilde)

        sampled_val = np.dot(np.dot(L, L_tilde_inv),  temp_new[i, :, :])
        mean = sampled_val.mean(axis=1, keepdims=True)
        std = sampled_val.std(axis=1, keepdims=True)
        sampled_val = (sampled_val - mean) / std

        prob_cum = norm.cdf(sampled_val)

        for p in range(m):
            y_vals, x_vals = s_dict[(lat, lon)][p]
            val_tilde[p, :] = np.interp(prob_cum[p, :], y_vals, x_vals)
            key = keys[p]
            epsilon[p, :] = (val_tilde[p, :] - statistics_matrix[(lat, lon)]['statistics'][key]["mean"])

        # rescale using inflation/deflation factor
        epsilon = epsilon * inflat_deflat_array

        for v in range(m):
            limits = statistics_matrix[(lat, lon)]['statistics'][keys[v]]
            ep_min = np.min(epsilon[v, :])
            ep_max = np.max(epsilon[v, :])
            if ep_min + meteo[i, v] < limits["min"]:
                mask_v_min = epsilon[v, :] < 0
                epsilon[v, mask_v_min] *= (meteo[i, v] - limits["min"]) / (-ep_min)
            if ep_max + meteo[i, v] > limits["max"]:
                mask_v_max = epsilon[v, :] > 0
                epsilon[v, mask_v_max] *= (limits["max"] - meteo[i, v]) / ep_max

        # Step 6: Apply perturbation
        # sum to each variable the perturbation to get the perturbed meteo ensemble
        met_pert = meteo[i, :, np.newaxis] + epsilon

        # Step 7: Apply flags
        if not pert_prec:
            met_pert[:, 0] = meteo[i, 0]
        if not pert_rad:
            met_pert[:, 1] = meteo[i, 1]
        if not pert_temp:
            met_pert[:, [2, 4, 5]] = meteo[i, [2, 4, 5]]
        if not pert_rh:
            met_pert[:, 3] = meteo[i, 3]

        # Step 8: Rescale if needed
        if scale_mean_prec and np.mean(meteo[i, 0]) > 0:
            met_pert[:, 0] *= np.mean(meteo[i, 0]) / np.mean(met_pert[:, 0])
        if pert_asymm_prec and np.mean(meteo[i, 0]) > 0:
            mask_prec = met_pert[:, 0] > meteo[i, 0]
            met_pert[:, 0][mask_prec] *= c_asymm_prec

        meteo_ensemble[i,:] = met_pert
        temp_old[i, :, :] = temp_new[i, :, :]

        # -------- STATE PERTURBATION --------
        R_state_p = R_state[(lat,lon)]["R"].values
        eps_state = np.random.multivariate_normal(
                np.zeros(state_matrix[i, :].shape[0]), R_state_p, N)
        state_pert = state_matrix[i, :] + eps_state
        np.clip(state_pert, state_limits[0], state_limits[1], out=state_pert)
        state_matrix_pert[i,:] = state_pert.T

        i +=1

    return  meteo_ensemble, state_matrix_pert, temp_new
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def enkf_assimilation_step(i,state_ens, output_ens, meteo_ens, y_obs,
                           parameters, R_measures, state_limits,Time):
    """
    Perform the EnKF assimilation step for a single time step using ensemble slices.
    Parameters:
        state_ens: np.ndarray - State ensemble array of shape (num_points, state_dim, N).
        output_ens: np.ndarray - Output ensemble array of shape (num_points, output_dim, N).
        meteo_ens: np.ndarray - Meteorological ensemble array of shape (num_points, meteo_dim, N).
        y_obs: np.ndarray - Observations array of shape (num_points, obs_dim).
        parameters: dict - Dictionary of model parameters.
        R_measures: np.ndarray - Observation error covariance matrix of shape (obs_dim, obs_dim).
        state_limits: tuple - Tuple containing min and max limits for state variables.
    Returns:
        Xa_mean: np.ndarray - Analysis state mean of shape (num_points, state_dim).
        output_a_mean: np.ndarray - Analysis output mean of shape (num_points, output_dim).
        Pa: np.ndarray - Analysis state covariance of shape (num_points, state_dim, state_dim).
    """


    state_dim = 4
    N = meteo_ens.shape[2]
    num_points = meteo_ens.shape[0]
    Xa_all = np.zeros((num_points,  state_dim,N))
    output_a_all = np.zeros((num_points,  2, N))
    Xa_mean = np.zeros((num_points, state_dim))
    output_a_mean = np.zeros((num_points, 2))
    Pa = np.zeros((num_points, state_dim, state_dim))
    corrections = np.zeros((state_dim, N))
    k= 0

    for (lat, lon), df in y_obs.items():
        # set df['time'] as time index
        t= Time[i]
        # convert t to pandas datetime if necessary
        t = np.datetime64(t)
        df = df.set_index('time')
        # drop duplicates
        df = df[~df.index.duplicated(keep='first')]
        if t in df.index:
            y_row = df.loc[t]
            y = y_row['Snow_depth_cm']/100
            try:

                y = float(y)
            except:
                print(y)
        else:
            y = np.nan


        state_ens_point = state_ens[k]
        output_ens_point = output_ens[k]
        meteo_ens_point = meteo_ens[k]
        Xb_old = state_ens[k].copy()
        output_matrix_old = output_ens[k].copy()

        B = np.cov(state_ens_point)
        std_dev = np.sqrt(np.diag(B))
        flag = np.zeros(state_dim)

        for q in range(state_dim):
            if std_dev[q] == 0:
                flag[q] = 1
                std_dev[q] = [10, 50, 15, 0.01][q]

        if flag.any():
            std_matrix = np.outer(std_dev, std_dev)
            corr_matrix = np.divide(B, std_matrix, out=np.zeros_like(B), where=std_matrix != 0)

            # Adjust correlation matrix based on flags
            if flag[0]:
                corr_matrix[0, :] = [-1 if idx != 0 else 1 for idx in range(state_dim)]
                corr_matrix[:, 0] = corr_matrix[0, :]
            if flag[1]:
                corr_matrix[1, :] = [-0.2, 1, 0.1, 0.1]
                corr_matrix[:, 1] = corr_matrix[1, :]
            if flag[2]:
                corr_matrix[2, :] = [-0.3, 0.3, 1, -0.5]
                corr_matrix[:, 2] = corr_matrix[2, :]
            if flag[3]:
                corr_matrix[3, :] = [-0.1, 0.1, -0.5, 1]
                corr_matrix[:, 3] = corr_matrix[3, :]

            B = np.multiply(corr_matrix, std_matrix)

        rho_inv = 1 / parameters['RhoSnowavg']


        Xa = np.zeros((state_dim, N))
        Xa_old = np.zeros((state_dim, N))
        output_a = np.zeros((2, N))
        H = np.full((state_dim, 2, N), np.nan)

        for n in range(N):

            rho_calc_inv = rho_inv  # Default value
            if output_ens_point[10, n] > 0:
                rho_calc = state_ens_point[ 2, n]
                rho_calc_inv = 1 / rho_calc if rho_calc != 0 else rho_inv

            d_alfa_swe = ((output_ens_point[8, n] * meteo_ens_point[1, n]) /
                          (1000 * parameters['RhoW'] * 0.334)) * parameters['dt']
            d_alfa_hs = d_alfa_swe / (
                output_ens_point[15, n] if output_ens_point[ 15, n] != 0 else parameters['RhoSnowavg'])

            if  np.isnan(y) or (y < 0):
                # No assimilatio
                Xa[:, n] = Xb_old[:, n]
                output_a[0, n] = output_matrix_old[0, n]
                output_a[1, n] = output_matrix_old[1, n]
                continue

            else:
                idx = 1
                H[:, :, n] = np.array([[np.nan, np.nan, np.nan, np.nan], [1 / 997, rho_calc_inv,
                                                                          -(rho_calc_inv * rho_calc_inv),
                                                                          d_alfa_hs]]).T

            """ 
            if np.isnan(y_obs).all():
                # No assimilation
                Xa[:, n] = Xb_old[:, n]
                output_a[0, n]= output_matrix_old[0, n]
                output_a[1, n] = output_matrix_old[1, n]
                continue

            elif not np.isnan(y[0]) and not np.isnan(y[1]):
                H[ :, :,n ] = np.array([[1, 1, 0, d_alfa_swe], [1 / 997, rho_calc_inv,
                                                                  -(rho_calc_inv * rho_calc_inv), d_alfa_hs]]).T

            elif np.isnan(y[1]) and not np.isnan(y[0]):

                H[:, :, n] = np.array([[1, 1, 0, d_alfa_swe], [np.nan, np.nan, np.nan, np.nan]]).T

            elif np.isnan(y[0]) and not np.isnan(y[1]):
                H[:, :, n] = np.array([[np.nan, np.nan, np.nan, np.nan], [1 / 997, rho_calc_inv,
                                                                             -(rho_calc_inv * rho_calc_inv),
                                                                             d_alfa_hs]]).T
                    
           """

            Ht = H[:,:, n]
            CC = np.matmul(B, Ht[:,idx])
            HBH = np.matmul(np.matmul(Ht[:,idx].T, B), Ht[:,idx])
            try :
                K = np.matmul(CC, np.linalg.inv(HBH + R_measures[:, idx]))
            except :
                K = CC * 1.0 / (HBH + R_measures[ idx, idx])

            y_mod = np.matmul(Ht[:,idx].T, state_ens_point[:, n])
            try:
                Xa[:,n] = state_ens_point[:, n] + np.matmul(K, (y - y_mod))
                corrections = np.matmul(K, (y - y_mod))
            except:
                Xa[:, n] = state_ens_point[:, n] + K*(y - y_mod)
                corrections = K*(y - y_mod)


            Xa_all[k,:,:] = Xa[:,:]

            output_a_all[k,:,:]= output_a[:,:]
            # --------------------------------------------------------------------------

            # Analysis check and rescale
            rescale = 0
            M_tot = 0
            idx_scale = -1

            for p in range(2):
                if Xa[p,n] < state_limits[0][p] or Xa[p, n] > state_limits[1][p]:
                    rescale = 1
                    M = max(state_limits[0][p] - Xa[p, n], Xa[p, n] - state_limits[1][p])
                    if M > M_tot:
                        M_tot = M
                        idx_scale = p

            if rescale == 1:
                # store the not correct analysis
                Xa_old[:2, n] = Xa[:2,n]
                Xa[:2, n] = Xb_old[:2, n] + ((Xa[:2,n] - Xb_old[:2,n]) * (
                        (np.abs(Xa[idx_scale,n] - Xb_old[idx_scale, n])) - M_tot) / np.abs(
                    Xa[idx_scale, n] - Xb_old[idx_scale, n]))

            if np.isnan(Xa[:, n]).any():
                Xa[:, n] = Xb_old[:, n]
           # --------------------------------------------------------------------------

            for p in range(state_dim):
                Xa[p, n] = np.clip(Xa[p, n], state_limits[0][p], state_limits[1][p])

            output_a[0, n] = Xa[0, n] + Xa[0, n]

            if output_a[0, n]  == 0:
                output_a[1, n] = 0
                Xa[2, n] = state_limits[0][2]

            else:
                output_a[1, n] = 0 if output_a[0, n] == 0 else (Xa[0,n] / 997 + Xa[1,n] / Xa[2, n])
        # --------------------------------------------------------------------------

        Xa_mean[k,:] = np.mean(Xa_all[k,:,:], axis=1)

        output_a_mean[k,:] = np.mean(output_a_all[k,:,:], axis=1)
        Pa[k, :, :] = np.cov(Xa_all[k,:,:], rowvar=True)

        k += 1

    return Xa_mean, output_a_mean, Pa, corrections


#------- End of S3M_2D_assimilation.py -------

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


# Globals container for large shared state (read-only inside workers)
_GLOBALS = {}

def _init_worker(global_data):
    global _GLOBALS
    _GLOBALS = global_data

def _perturb_site_worker(args):
    (
        idx, latlon,meteo,state_matrix, temp_old, m, N, L0, L_tilde_inv, inflat_deflat_array,
        pert_prec, pert_rad, pert_temp, pert_rh,
        scale_mean_prec, c_asymm_prec, pert_asymm_prec,
        state_limits, keys, rng_seed_base
    ) = args

    # Load large global objects
    L_dict           = _GLOBALS["L_dict"]
    s_dict           = _GLOBALS["s_dict"]
    statistics_matrix = _GLOBALS["statistics_matrix"]
    R_state          = _GLOBALS["R_state"]

    lat, lon = latlon
    rng = np.random.RandomState(rng_seed_base + idx)

    # Allocate
    epsilon = np.zeros((m, N))
    val_tilde = np.zeros((m, N))

    # --- Meteo perturbation ---
    L = L_dict[(lat, lon)]
    rand_norm_i = rng.normal(0, 0.01, (m, N))
    temp_new_i = temp_old[idx] + L0 @ rand_norm_i

    sampled_val = L @ (L_tilde_inv @ temp_new_i)
    mean = sampled_val.mean(axis=1, keepdims=True)
    std = sampled_val.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    sampled_val = (sampled_val - mean) / std
    prob_cum = norm.cdf(sampled_val)

    for p in range(m):
        y_vals, x_vals = s_dict[(lat, lon)][p]
        val_tilde[p] = np.interp(prob_cum[p], y_vals, x_vals)
        key = keys[p]
        epsilon[p] = val_tilde[p] - statistics_matrix[(lat, lon)]['statistics'][key]["mean"]

    epsilon *= inflat_deflat_array

    for v in range(m):
        limits = statistics_matrix[(lat, lon)]['statistics'][keys[v]]
        # ensure scalars
        met_val = float(np.asarray(meteo[idx, v]).mean())
        lim_min = float(np.asarray(limits["min"]).min())
        lim_max = float(np.asarray(limits["max"]).max())
        ep_min = np.min(epsilon[v, :])
        ep_max = np.max(epsilon[v, :])

        # enforce limits
        if ep_min + met_val < lim_min:
            mask_v_min = epsilon[v] < 0
            if ep_min != 0:
                epsilon[v, mask_v_min] *= (met_val - lim_min) / (-ep_min)

        if ep_max + met_val > lim_max:
            mask_v_max = epsilon[v] > 0
            if ep_max != 0:
                epsilon[v, mask_v_max] *= (lim_max - met_val) / ep_max

    met_pert = meteo[idx, :, None] + epsilon

    if not pert_prec:
        met_pert[:, 0] = meteo[idx, 0]
    if not pert_rad:
        met_pert[:, 1] = meteo[idx, 1]
    if not pert_temp:
        met_pert[:, [2, 4, 5]] = meteo[idx, [2, 4, 5]]
    if not pert_rh:
        met_pert[:, 3] = meteo[idx, 3]

    if scale_mean_prec and np.mean(meteo[idx, 0]) > 0:
        mean_p = np.mean(met_pert[:, 0])
        if mean_p != 0:
            met_pert[:, 0] *= np.mean(meteo[idx, 0]) / mean_p

    if pert_asymm_prec and np.mean(meteo[idx, 0]) > 0:
        mask = met_pert[:, 0] > meteo[idx, 0]
        met_pert[:, 0][mask] *= c_asymm_prec

    # --- State perturbation ---
    R_state_p = R_state[(lat, lon)]["R"].values
    eps_state = rng.multivariate_normal(
        np.zeros(state_matrix[idx].shape[0]), R_state_p, N
    )
    state_pert = state_matrix[idx] + eps_state
    np.clip(state_pert, state_limits[0], state_limits[1], out=state_pert)

    return idx, met_pert, state_pert.T, temp_new_i

import multiprocessing as mp

def perturb_meteo_state_pool(
        meteo, m, L_dict, N, L0, L_tilde, s_dict, statistics_matrix,
        inflat_deflat, R_state, state_matrix,
        state_limits, pert_prec, pert_rad,
        pert_temp, pert_rh, scale_mean_prec, c_asymm_prec,
        pert_asymm_prec, temp_old, obs_mask,
        n_processes=None, rng_seed_base=12345):

    num_sites = len(obs_mask)

    # Prepare outputs
    meteo_ensemble = np.zeros((num_sites, meteo.shape[1], N))
    state_matrix_pert = np.zeros((num_sites, state_matrix.shape[1], N))
    temp_new = np.zeros_like(temp_old)

    # Precompute shared quantities
    inflat_deflat_array = np.array(inflat_deflat).reshape(m,1) * np.ones((1,N))
    L_tilde_inv = np.linalg.inv(L_tilde)
    keys = list(statistics_matrix[(0, 0)]['statistics'])

    # Put large arrays in global shared dict
    global_data = dict(
        L_dict=L_dict,
        s_dict=s_dict,
        statistics_matrix=statistics_matrix,
        R_state=R_state,
        temp_old=temp_old,
    )

    # Build task list
    tasks = []
    for i, latlon in enumerate(obs_mask):
        tasks.append((
            i, latlon, meteo,state_matrix, temp_old, m, N, L0, L_tilde_inv, inflat_deflat_array,
            pert_prec, pert_rad, pert_temp, pert_rh,
            scale_mean_prec, c_asymm_prec, pert_asymm_prec,
            state_limits, keys, rng_seed_base
        ))

    # Run parallel pool
    with mp.Pool(processes=n_processes, initializer=_init_worker, initargs=(global_data,)) as pool:
        for idx, met_pert, st_pert, temp_new_i in pool.imap_unordered(_perturb_site_worker, tasks):
            meteo_ensemble[idx] = met_pert
            state_matrix_pert[idx] = st_pert
            temp_new[idx] = temp_new_i

    return meteo_ensemble, state_matrix_pert, temp_new
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
