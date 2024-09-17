import numpy as np
import copy

from lmcp import LMCP


def path_solver(f, df, l, u, x0=None, sigma=0., max_iters=100, tol=1e-4, forward=True, linesearch=True):
    # Implementation of the PATH solver using basic monotone linesearching
    # Solves Nonlinear Mixed Complementary Problems:
    #   find z, w, v
    #   s.t. w-v = f(z)
    #        w >= 0, w'*(z-l) = 0
    #        v >= 0, v'*(u-z) = 0
    #        l <= z <= u
    # f is a function which maps from Rn->Rn
    # df is a function which returns the Jacobian of f, hence maps from Rn-> Rn x Rn

    n = l.shape[0]
    if x0 is None:
        x = -np.ones((n,1))
    else:
        x = x0

    def proj(y):
        return np.maximum(np.minimum(y,u),l)

    def fb(y):
        z = proj(y)
        fz = f(z)
        return np.squeeze(fz) + np.squeeze(y) - np.squeeze(z)

    def merit(y):
        fby = fb(y)
        return 0.5*fby.dot(fby)

    def get_vals(zwvt):
        x = zwvt[0:n] - zwvt[n:2 * n] + zwvt[2 * n:3 * n]
        t = zwvt[-1]
        m = merit(x)
        return x,t,m

    def backward_pathsearch(low,high,path,current_merit):
        # recursive binary search on path to determine point of difference for linesearch
        if high >= low:
            mid = low + (high - low) // 2
            zwvt_half = np.copy(path[mid])
            x_half, t_half, m_half = get_vals(zwvt_half)
            if m_half <= (1.01 - sigma * t_half) * current_merit:
                return backward_pathsearch(mid + 1,high,path,current_merit)
            else:
                return backward_pathsearch(low,mid - 1,path,current_merit)
        else:
            zwvt = np.copy(path[low])
            x,t,m = get_vals(zwvt)
            return x,t,m,low

    current_merit = merit(x)
    for iter in range(max_iters):
        print('iter', iter, current_merit)
        if current_merit < tol:
            z = proj(x)
            w = np.maximum(0, z-x)
            v = np.maximum(0, x-z)
            return (z, w, v, True)
        z = proj(x)
        M = df(z)
        q = f(z) - M.dot(z)
        (path, success) = LMCP(M,q,l,u,x)
        x_prev = np.copy(x)
        m_prev = current_merit
        t_prev = 0
        step_prev = 0
        print("path len", len(path))
        if forward:
            # forward pathsearch
            for L in range(1,len(path)):
                zwvt = np.copy(path[L])
                x_cand, t_cand, m_cand = get_vals(zwvt)
                if m_cand <= (1.01 - sigma*t_cand)*current_merit:
                    x_prev = np.copy(x_cand)
                    t_prev = t_cand
                    m_prev = m_cand
                    step_prev = L
                else:
                    break
        else:

            # check last point in path first
            zwvt_end = np.copy(path[-1])
            x_end, t_end, m_end = get_vals(zwvt_end)
            if m_end <= (1.01 - sigma * t_end) * current_merit:
                x_prev = x_end
                t_prev = t_end
                m_prev = m_end
                step_prev = len(path)-1  # prevent linesearch since last point makes sufficient progress
            else:  # o/w # recursive backward pathsearch
                x_cand, t_cand, m_cand, step_cand = backward_pathsearch(0,len(path)-1,path,current_merit)
                step_prev = step_cand - 1
                zwvt_prev = np.copy(path[step_prev])
                x_prev, t_prev, m_prev = get_vals(zwvt_prev)

        # x_prev, t_prev was last point known to satisfy descent condition
        # x_cand, t_cand violate descent condition.
        # Perform backtracking linesearch between these points to find descent
        if step_prev == len(path)-1 or iter < 5:  # all points in the path make sufficient progress or too early
            x = np.copy(x_prev)
            current_merit = merit(x)
        else:
            dt = t_cand - t_prev
            lam = 1.0
            decay = 0.5
            ls_successful = False
            for backtracking_iters in range(10):
                lam *= decay
                t_ls = (1-lam)*t_prev + lam*t_cand
                x_ls = (1-lam)*x_prev + lam*x_cand
                m_ls = merit(x_ls)
                if m_ls <= (1.01 - sigma*t_ls)*current_merit:
                    x = x_ls
                    current_merit = m_ls
                    ls_successful = True
                    break
            if not ls_successful:
                print("ERROR! LS FAILURE")
                break
    z = proj(x)
    w = np.maximum(0, z-x)
    v = np.maximum(0, x-z)
    return (z, w, v, False)
