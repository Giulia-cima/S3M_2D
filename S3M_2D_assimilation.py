from scipy.stats import norm
import numpy as np
from statistics import NormalDist


""" 

                    meteo_assimilation, 6, R, N, L0, L_tilde, perturbations_data, statistics,
                    inflation_deflation, R_state, state_matrix[:, i, :], state_limits,
                    output_matrix[:, i, :], pert_prec, pert_rad, pert_temp, pert_rh,
                    scale_mean_prec, c_asymm_prec, pert_asymm_prec, temporary_val_old[:, i, :, :],obs_mask
                )


"""
def perturb_meteo_state(
        meteo,  m, R_dict, N, L0, L_tilde, quantile_data_dict, statistics_matrix,
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
    prob_cum = np.zeros((m, N))
    inflat_deflat_array = np.array(inflat_deflat).reshape(m,1) * np.ones((1,N))

    i=0
    for (lat,lon) in obs_mask:

        R1 = R_dict[(lat,lon)]
        R= R1["R"].values

        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            R = (R + R.T) / 2
            R += np.eye(R.shape[0]) * 1e-6
            L = np.linalg.cholesky(R)

        quantile_entry = quantile_data_dict [(lat,lon)]
        s = [
            quantile_entry["air_temp_degC"],
            quantile_entry["prc_mm"],
            quantile_entry["swin_wm"],
            quantile_entry["rel_hum_perc"],
            quantile_entry["T_albedo"],
            quantile_entry["T_melting"]
        ]


        temp_new[i, :, :] = temp_old[i, :, :] + np.dot(L0, np.random.normal(0, 0.01, (m, N)))
        L_tilde_inv = np.linalg.inv(L_tilde)
        sampled_val = np.dot(np.dot(L, L_tilde_inv),  temp_new[i, :, :])

        prob_cum[:, :] = [[NormalDist(mu=0, sigma=1).cdf(sampled_val[p, q])
                              for q in range(N)] for p in range(m)]

        val_tilde = np.zeros((m, N))
        for p in range(m):
            s[p] = s[p].sort_values("emp_rip")  # If it's a pandas DataFrame
            val_tilde[p, :] = [np.interp(prob_cum[p,q], s[p].iloc[:, 2], s[p].iloc[:, 0]) for q in range(N)]
            key = list(statistics_matrix[(lat, lon)]['statistics'].keys())[p]
            epsilon[p,:] = (val_tilde[p, :] - statistics_matrix[(lat,lon)]['statistics'][key]["mean"])

        epsilon[:, :] = epsilon[:, :] * inflat_deflat_array

        for v in range(m):
            key = list(statistics_matrix[(lat, lon)]['statistics'].keys())[v]
            limits = statistics_matrix[(lat, lon)]['statistics'][key]
            ep_min = np.min(epsilon[v, :])
            ep_max = np.max(epsilon[v, :])
            met_val_mean = np.mean(meteo[i, v])

            if ep_min + met_val_mean < limits["min"]:
                mask_v = epsilon[v,:] < 0
                epsilon[v, mask_v] *= (met_val_mean - limits["min"]) / (-ep_min)

            if ep_max + met_val_mean > limits["max"]:
                mask_v = epsilon[v, :] > 0
                epsilon[v, mask_v] *= (limits["max"] - met_val_mean) / ep_max

        # Step 6: Apply perturbation
        # sum to each variable the perturbation to get the perturbed meteo ensemble
        met_pert = meteo[i, :, np.newaxis] + epsilon

        # Step 7: Apply flags
        if not pert_prec:
            met_pert[:, 0] = meteo[i, 0]
        if not pert_rad:
            met_pert[:, 1] = meteo[i, 1]
        if not pert_temp:
            met_pert[:, 2] = meteo[i, 2]
            met_pert[:, 4] = meteo[i, 4]
            met_pert[:, 5] = meteo[i, 5]
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

        for v in range(len(state_matrix[i, :])):
            state_pert[:, v] = np.clip(state_pert[:, v], state_limits[0][v], state_limits[1][v])

        state_matrix_pert[i,:] = state_pert.T

        i +=1

    return  meteo_ensemble, state_matrix_pert, temp_new

def enkf_assimilation_step(state_ens, output_ens, meteo_ens, y_obs,
                           parameters, R_measures, state_limits):
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

    for point in range(num_points):
        state_ens_point = state_ens[point, :, :]
        output_ens_point = output_ens[point, :, :]
        meteo_ens_point = meteo_ens[point, :, :]
        Xb_old = state_ens[point, :, :].copy()
        output_matrix_old = output_ens[point, :, :].copy()
        y = y_obs[point, :]
        B = np.cov(state_ens_point)
        std_dev = np.sqrt(np.diag(B))
        flag = np.zeros(state_dim)

        for k in range(state_dim):
            if std_dev[k] == 0:
                flag[k] = 1
                std_dev[k] = [10, 50, 15, 0.01][k]

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

            Ht = H[:,:, n]
            CC = np.matmul(B, Ht)
            HBH = np.matmul(np.matmul(Ht.T, B), Ht)
            K = np.matmul(CC, np.linalg.inv(HBH + R_measures))
            y_mod = np.matmul(Ht.T, state_ens_point[:, n])
            Xa[:,n] = state_ens_point[:, n] + np.matmul(K, (y - y_mod))
            corrections = np.matmul(K, (y - y_mod))


            for k in range(state_dim):
                Xa[k,n] = np.clip(Xa[k,n], state_limits[0][k], state_limits[1][k])

            output_a[0, n] = Xa[0, n] + Xa[0, n]
            output_a[1, n] = 0 if output_a[0, n] == 0 else (Xa[0,n] / 997 + Xa[1,n] / Xa[2, n])


            Xa_all[point,:,:] = Xa[:,:]
            output_a_all[point,:,:]= output_a[:,:]

        Xa_mean[point,:] = np.mean(Xa_all[point,:,:], axis=1)
        output_a_mean[point,:] = np.mean(output_a_all[point,:,:], axis=1)
        Pa[point, :, :] = np.cov(Xa_all[point,:,:], rowvar=True)

    return Xa_mean, output_a_mean, Pa, corrections