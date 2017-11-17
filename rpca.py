import bisect
import numpy as np
from numpy.linalg import norm
from numpy.linalg import svd
from scipy.sparse.linalg import svds


def pcp_alm(X, maxiter=500, tol=1e-7):
    """Principal Component Pursuit
    Finds the Principal Component Pursuit solution.
    Solves the optimization problem::
        (L^*,S^*) = argmin || L ||_* + gamma * || S ||_1
                    (L,S)
                    subject to    L + S = X
    where || . ||_* is the nuclear norm.  Uses an augmented Lagrangian approach
    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        Data matrix.
    maxiter : int, 500 by default
        Maximum number of iterations to perform.
    tol : float, 1e-7 by default - in the paper

    Returns
    -------
    L : array of shape [n_samples, n_features]
        The low rank component
    S : array of shape [n_samples, n_features]
        The sparse component
    (u, sig, vt) : tuple of arrays
        SVD of L.
    n_iter : int
        Number of iterations
    Reference
    ---------
       Candes, Li, Ma, and Wright
       Robust Principal Component Analysis?
       Submitted for publication, December 2009.
    """

    def soft_threshold(v=0, tau=0):
        '''
        shrinkiage opterator S_tau[x]
        '''
        tmp = np.abs(v)
        tmp = np.subtract(tmp, tau)
        tmp = np.maximum(tmp, 0.0)
        return np.multiply(np.sign(v), tmp)

    def svt(X, tau, k, svd_function, out=None):
        def svd_reconstruct(u, sig, v, out=None, tmp_out=None):
            tmp = np.multiply(u, sig, out=tmp_out)
            return np.dot(tmp, v, out=out)

        def truncate_top_k_svd(u, sig, v, k):
            return u[:, :k], sig[:k], v[:k, :]

        m, n = X.shape

        u, sig, v = svd_function(X, k)

        sig = soft_threshold(sig, tau)
        r = np.sum(sig > 0)

        u, sig, v = svd_function(X, r)
        sig = soft_threshold(sig, tau)

        if r > 0:
            #print("Z= reconstructed")
            u, sig, v = truncate_top_k_svd(u, sig, v, r)
            #print("reconstructed sig =", sig)
            Z = svd_reconstruct(u, sig, v, out=out)
        else:
            #print("Z= 0")
            out[:] = 0
            Z = out
            u, sig, v = np.empty((m, 0)), np.empty((0, 0)), np.empty((0, n))
        return (Z, r, (u, sig, v))

    def svd_choice(n, d):
        '''
        choose svd depend on the size 

        return 'dense_top_k_svd/sparse_svds
        '''

        ratio = float(d) / float(n)
        vals = [(0, 0.02), (100, 0.06), (200, 0.26),
                (300, 0.28), (400, 0.34), (500, 0.38)]

        i = bisect.bisect_left([r[0] for r in vals], n)
        choice = dense_top_k_svd if ratio > vals[i - 1][1] else svds
        return choice

    def dense_top_k_svd(A, k):
        '''
        A - matrix
        k - Top K components
        '''
        u, sig, v = svd(A, full_matrices=0)
        return u[:, :k], sig[:k], v[:k, :]

    n = X.shape
    frob_norm = np.linalg.norm(X, 'fro')
    two_norm = np.linalg.norm(X, 2)
    one_norm = np.sum(np.abs(X))
    inf_norm = np.max(np.abs(X))

    print("frob_norm", frob_norm)
    print("two_norm", two_norm)
    print("one_norm", one_norm)
    print("info_norm", inf_norm)

    mu_inv = 4 * one_norm / np.prod(n)

    gamma = 1 / np.sqrt(np.max([X.shape[0], X.shape[1]]))
    print("gamma", gamma)
    # Kicking
    k = np.min([
        np.floor(mu_inv / two_norm),
        np.floor(gamma * mu_inv / inf_norm)
    ])
    Y = k * X
    sv = 10

    # Variable init
    S = np.zeros(n)

    # print("k",k)
    #print("mu_inv", mu_inv)

    for i in range(maxiter):

        # Shrink singular values
        l = X - S + mu_inv * Y
        svd_fun = svd_choice(np.min(l.shape), sv)
        # print(svd_fun)
        #print("sv", sv)
        L, r, (u, sig, v) = svt(l, mu_inv, sv, svd_function=svd_fun)
        #print("non-zero sigular value", r)
        if r < sv:
            sv = np.min([r + 1, np.min(n)])
        else:
            sv = np.min([r + int(np.round(0.05 * np.min(n))), np.min(n)])

        # Shrink entries
        s = X - L + mu_inv * Y
        S = soft_threshold(s, gamma * mu_inv)

        # Check convergence
        R = X - L - S
        stopCriterion = np.linalg.norm(R, 'fro') / frob_norm
        #print("stopCriterion", stopCriterion)
        if stopCriterion < tol:
            break

        # Update dual variable
        Y += 1 / mu_inv * (X - L - S)

    return L, S, (u, sig, v), i + 1
