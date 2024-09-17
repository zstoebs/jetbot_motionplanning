import numpy as np


def pivot(T, row, col):
    pivot = T[row, :] / T[row, col]
    T -= np.outer(T[:, col], pivot)
    T[row, :] = pivot


def LMCP(M,q,l,u,x0,max_iters=100):
    # Function to compute solution to a mixed LCP:
    # Find z, w, v
    # s.t. Mz+q = w-v
    # w>=0
    # v>=0
    # l<=z<=u
    # (z-l)'w = 0
    # (u-z)'v = 0

    # The process for computing solutions to this problem is described in
    # The path solver: a non-monotone stabilization scheme for mixed complementarity problems
    # by Steven Dirske and Michael Ferris

    # This function should return a tuple (path, success)
    # where success is True if a solution is found, and False otherwise (e.g. ray termination)

    # The return variable 'path' is a list of the variables (z,w,v,t) through the stages of this algorithm.
    # if zwvt is the concatenation of z,w,v,t, then path = [zwvt[0], zwvt[1], zwvt[2],...,zwvt[N]]
    # where zwvt[0] is the starting point of the algorithm, and zwvt[n] is after completing the n-th pivot,
    # where after the final N-th pivot, either a solution or a ray termination was found.

    # I provide some starter code here from my own implementation.
    # The main difference between Lemke's method and the algorithm here, is that the non-basic variables might
    # be non-zero. For example, if z_i is a non-basic variable, then it will have value l_i or u_i, depending on
    # which bound it left the basis. Similarly, when a variable is brought into the basis, it doesn't necessarily start
    # at zero, and it might not increase.
    # For example, if z_i is brought into the basis, and it started at u_i, we need to figure out THE SMALLEST value of
    # z_i, which is less than u_i, such that none of the existing basis variables hit their upper or lower bounds. Which ever
    # of those variables hit their bounds first will become non-basic, and will be constrained to have value equal to the bound
    # that was hit.
    # There are various ways to keep tabs on all of this, but I choose to do this by keeping track of a vector of the lower and upper bounds
    # on the current basis variables. Furthermore, I keep track of the direction the entering variable is coming in at (-1 if starting from an upper
    # bound, +1 if starting from a lower bound). Feel free to start from scratch if it makes more sense to you to keep tabs of everything a different way.
    # The starter code might confuse you more than if you write it from scratch.

    # Finally, remember that because the non-basic variables don't necessarily have a value of 0, we can't ignore them when figuring out
    # how the basic variables will change when one of the non-basics leaves it's bound. Failing to account for this is
    # the most common mistake when implementing this algorithm.

    n = M.shape[0]
    # Assert that shapes are compatible
    q = q.reshape((n,1))
    l = l.reshape((n,1))
    u = u.reshape((n,1))
    x0 = x0.reshape((n,1))
    z0 = np.minimum(np.maximum(x0,l),u)  # project x_0 within bounds

    r = M.dot(z0) + q + x0 - z0

    # z  w  v  t
    # M -I  I  r

    zwvt = np.zeros((3*n+1,1))
    T = np.hstack([M,-np.eye(n), np.eye(n), r, r-q])

    zwvt[0:n] = z0

    basis = []
    basis_lb = np.zeros((n,1))
    basis_ub = np.zeros((n,1))
    nonbasis = []
    nonbasis.append(3*n)
    for i in range(n):
        if z0[i] - l[i] < 1e-6:  # z0 at lower bound -> w0 in basis, z0 in nonbasis
            basis.append(n+i)
            basis_lb[i] = 0
            basis_ub[i] = np.inf
            nonbasis.append(i)
            nonbasis.append(2*n+i)
            zwvt[n+i] = z0[i]-x0[i]  # x0 below lower bound -> shift by distance to lower bound
        elif u[i] - z0[i] < 1e-6:  # z0 at upper bound -> v0 in basis, z_i in nonbasis
            basis.append(2*n+i)
            basis_lb[i] = 0
            basis_ub[i] = np.inf
            nonbasis.append(i)
            nonbasis.append(n+i)
            zwvt[2*n+i] = x0[i]-z0[i]  # x0 above upper bound -> shift by distance to upper bound
        else:                       # z0 within bounds -> z0 in basis, w0 and v0 in nonbasis
            basis.append(i)
            basis_lb[i] = l[i]
            basis_ub[i] = u[i]
            nonbasis.append(n+i)
            nonbasis.append(2*n+i)
    path = []
    path.append(np.copy(zwvt))
    B = T[:,basis]
    T = np.linalg.solve(B, T)  # T_(j+1) = Binv_(j+1).dot(T0) -> pivoting accomplishes this without inversion in loop

    entering_ind = 3*n  # t is first entering var
    entering_val = 0.0
    entering_lb = -np.inf
    entering_ub = 1.0
    entering_dir = 1.0

    for iter in range(max_iters):

        ### ratio test
        nonbasis.remove(entering_ind)
        d_t = T[:, entering_ind].reshape(-1, 1)
        b = T[:, -1].reshape(-1, 1)
        D = T[:, nonbasis]
        xd_t = zwvt[nonbasis]
        b_t = b - D.dot(xd_t)

        # solve for bounds
        xs_b = np.zeros((n, 1))
        xs_b[d_t == 0] = np.inf
        up = d_t * entering_dir < 0
        xs_b[up] = (b_t[up] - basis_ub[up]) / d_t[up]
        down = d_t * entering_dir > 0
        xs_b[down] = (b_t[down] - basis_lb[down]) / d_t[down]
        invalid = np.logical_or(xs_b < entering_lb, xs_b > entering_ub)
        xs_b[invalid] = np.inf

        ### update
        i = np.argmin(abs(xs_b - entering_val))
        xs = xs_b[i]
        if abs(xs) != np.inf:  # entering var did not exit at bound
            leaving_ind = basis[i]

            # assumes that one of up[i] or down[i] is True
            zwvt[leaving_ind] = basis_ub[i] if up[i] else basis_lb[i] # set to exiting bound
            exiting_dir = 1.0 if up[i] else -1.0

            # update tableau
            pivot(T, i, entering_ind)

            # update basis
            basis[i] = entering_ind
            basis_lb[i] = entering_lb
            basis_ub[i] = entering_ub

        else:  # entering var does not enter basis -> no pivot
            leaving_ind = entering_ind
            zwvt[leaving_ind] = entering_ub if entering_dir > 0 else entering_lb  # set to opposite bound
            exiting_dir = entering_dir

        nonbasis.append(leaving_ind)  # update nonbasis

        # update current solutions to system (aka the path coords) based on updated tableau
        b = T[:, -1].reshape(-1, 1)
        D = T[:, nonbasis]
        xd = zwvt[nonbasis]
        b_t = b - D.dot(xd)
        zwvt[basis] = b_t  # xb = Binv.dot(b - D.dot(xd)) = b_t
        path.append(np.copy(zwvt))

        # assert that current zwvt solve the current system
        A = T[:,:-1]
        assert A.dot(zwvt).all() == T[:,-1].all()

        ### pivot rules + return cond
        if n <= leaving_ind < 2*n:  # w_j leaves -> z_j enters at lower bound
            entering_ind = leaving_ind - n
            j = entering_ind
            entering_val = l[j]
            entering_lb = l[j]
            entering_ub = u[j]
            entering_dir = 1.0
        elif 2*n <= leaving_ind < 3*n:  # v_j leaves -> z_j enters at upper bound
            entering_ind = leaving_ind - 2*n
            j = entering_ind
            entering_val = u[j]
            entering_lb = l[j]
            entering_ub = u[j]
            entering_dir = -1.0
        elif leaving_ind < n:  # z_j exiting
            j = leaving_ind
            if exiting_dir < 0:  # z_j exits at lb -> w_j enters at 0 = lb
                entering_ind = n+j
                entering_val = 0
                entering_lb = 0
                entering_ub = np.inf
                entering_dir = 1.0
            else:                   # z_j exits at ub -> v_j enters at 0 = lb
                entering_ind = 2*n+j
                entering_val = 0
                entering_lb = 0
                entering_ub = np.inf
                entering_dir = 1.0
        else:  # t leaves basis -> leaving_ind == 3*n
            return (path, True) if abs(zwvt[leaving_ind] - 1.0) <= 1e-6 else (path, False)
