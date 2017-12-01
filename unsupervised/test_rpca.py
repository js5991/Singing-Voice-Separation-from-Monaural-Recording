import model.rpca as rpca
import numpy as np

n = 100
r = 3
np.random.seed(123)
base = 100 + np.cumsum(np.random.randn(n, r), axis=0)
scales = np.abs(np.random.randn(n, r))
L = np.dot(base, scales.T)
S = np.round(0.25 * np.random.randn(n, n))
M = L + S

L_hat, S_hat, r, n_iter = rpca.pcp_alm(M, 500)
print(np.max(np.abs(S - S_hat)))
print(np.max(np.abs(L - L_hat)))
print(n_iter)


_, s, _ = np.linalg.svd(L, full_matrices=False)
#print (s[s > 1e-11])

_, s_hat, _ = np.linalg.svd(L_hat, full_matrices=False)
#print (s_hat[s_hat > 1e-11])
