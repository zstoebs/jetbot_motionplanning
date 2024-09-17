import numpy as np
from lmcp import LMCP


# Randomly generate LCPs with guaranteed success
M = np.array([[1, -1, 2],
              [-1, 5, 1],
              [2, 1, 10]])
for i in range(100):
    q = np.random.randn(3, 1)
    l = np.zeros((3,1))
    u = np.inf*np.ones((3,1))
    x0 = -np.ones((3,1))
    (path, success) = LMCP(M,q,l,u,x0,max_iters=100)
    zwvt = path[-1]
    z = zwvt[0:3]
    w = zwvt[3:6]
    v = zwvt[6:9]
    t = zwvt[9]
    assert(success)
    assert(abs(t-1)<=1e-6)
    assert np.all(z >= l-1e-6)
    assert np.all(z <= u+1e-6)
    assert np.all(w >= -1e-6)
    assert np.all(v >= -1e-6)
    assert np.linalg.norm(M.dot(z) + q - w + v) <= 1e-6
    for j in range(3):
        if l[j] > -np.inf:
            assert (z[j]-l[j])*w[j] <= 1e-6
        else:
            assert w[j] <= 1e-6
        if u[j] < np.inf:
            assert (u[j]-z[j])*v[j] <= 1e-6
        else:
            assert v[j] <= 1e-6

# Randomly generate QPs without non-negativity constraints
for i in range(100):
    P = np.random.randn(5,5)
    P = P.T.dot(P)
    p = np.random.randn(5,1)
    A = np.random.randn(2,5)
    b = np.random.randn(2,1)
    M1 = np.hstack((P,-A.T))
    M2 = np.hstack((A, np.zeros((2,2))))
    M = np.vstack((M1,M2))
    q = np.vstack((p, b))
    l = -np.inf*np.ones((7,1))
    l[5:7] = 0
    u = np.inf*np.ones((7,1))
    x0 = -np.ones((7,1))
    (path, success) = LMCP(M,q,l,u,x0,max_iters=100)
    zwvt = path[-1]
    z = zwvt[0:7]
    w = zwvt[7:14]
    v = zwvt[14:21]
    t = zwvt[21]
    assert(success)
    assert(abs(t-1)<=1e-6)
    assert np.all(z >= l-1e-6)
    assert np.all(z <= u+1e-6)
    assert np.all(w >= -1e-6)
    assert np.all(v >= -1e-6)
    assert np.linalg.norm(M.dot(z) + q - w + v) <= 1e-6
    for j in range(7):
        if l[j] > -np.inf:
            assert (z[j]-l[j])*w[j] <= 1e-6
        else:
            assert w[j] <= 1e-6
        if u[j] < np.inf:
            assert (u[j]-z[j])*v[j] <= 1e-6
        else:
            assert v[j] <= 1e-6

print("All tests passed")
