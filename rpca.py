import numpy as np
import scipy as sp

D = np.random.rand(5, 4)
m, n = D.shape


def implement_inexact_alm_rpca(D, Lambda=1 / np.sqrt(m), tol=1e-7, maxIter=1000):
    Y = D
    norm_two = sp.linalg.svd(Y)[1][0]  # largest singular value
    norm_inf = sp.linalg.norm(Y, ord=np.inf, axis=None, keepdims=False)
    dual_norm = max(norm_two, norm_nuclear)
    Y = Y / dual_norm

    A_hat = np.zeros([m, n])
    E_hat = np.zeros([m, n])
    mu = 1.25 / norm_two  # can be tuned
    mu_bar = mu * 1e7
    rho = 1.5  # can be tuned
    d_norm = sp.linalg.norm(Y, ord='fro')

    iter_count = 0
    total_svd = 0
    converged = False
    stopCriterion = 1
    sv = 10

    while not converged:
        iter_count += 1
        temp_T = D - A_hat + (1 / mu) * Y
        E_hat = np.max(temp_T - Lambda / mu, 0)
        E_hat += np.min(temp_T + Lambda / mu, 0)

        U, s, V = sp.linalg.svd(D - E_hat + (1 / mu) * Y, full_matrices=False)
        svp = (s > 1 / mu).sum()
        if svp < sv:
            sv = min(svp + 1, n)
        else:
            sv = min(svp + round(0.05 * n), n)

        A_hat = U[:, :svp] * (np.diag(s[:svp] - 1 / mu)) * V[:, :svp].T

        total_svd += 1

        Z = D - A_hat - E_hat

        Y += mu * Z
        mu = min(mu * rho, mu_bar)

        stopCriterion = sp.linalg.norm(Z, ord='fro') / d_norm
        if stopCriterion < tol:
            converged = True

        if total_svd % 10 == 0:
            print(str(total_svd) + ' r(A) ' +
                  str(np.linalg.matrix_rank(A_hat)))
            print('|E|_0' + str((E_hat > 0).sum()))
            print('stopCriterion:' + str(stopCriterion))

        if not converged & iter_count >= maxIter:
            print('Maximum iterations reached')
            coverged = 1
    return A_hat, E_hat, iter_count
