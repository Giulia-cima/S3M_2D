# write a script to emulate a stochastic process with a given covariance matrix R
# and plot the result
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

n = 6  # dimensione del vettore di stato
x0 = np.zeros((n, 1))  # condizione iniziale
N = 24*365*20  # numero di passi


# matrice di covarianza di primo tentativo
r = np.random.randn(n, n)
R0 = np.dot(r, r.T)

# Catena di Markov
X = np.zeros((n, N))
x = x0
L0 = np.linalg.cholesky(R0) # use the transpose to match MATLAB's 'lower' option
for t in range(1, N):
    x = x + np.dot(L0, np.random.normal(0, 0.1, (n, 1)))
    X[:, t] = x.flatten()

plt.plot(X.T)
# save figure in the current directory
plt.savefig('/home/giulia/Cima_Giulia_blandini/S3M/stochastic_process.png')


R_tilde = np.cov(X)
# obtain the cholesky decomposition of R_tilde
L_tilde = np.linalg.cholesky(R_tilde)
# save the matrix L_tilde in a pkl file
pkl.dump(L_tilde, open("/home/giulia/Cima_Giulia_blandini/S3M/L_tilde.pkl", "wb"))
# save the matrix L0 in a pkl file
pkl.dump(L0, open("/home/giulia/Cima_Giulia_blandini/S3M/L0.pkl", "wb"))
