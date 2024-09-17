import numpy as np
import algopy
from algopy import UTPM, exp
import sympy as sp
import matplotlib.pyplot as plt
from time import time

from path import path_solver

start_time = time()
# set forward (True) or backward (False) search
forward = False

# Define a general optimization problem with non-quadratic cost or nonlinear constraints

T = 50
n = 4
m = 2
dt = 0.1

xinit = np.zeros((4,1))
rad = 1
obs = [2,1.]
goal = [3,1]

# Simple car dynamics
def f(xt,ut):
    y = []
    y.append(xt[0]+dt*xt[2])
    y.append(xt[1]+dt*xt[3])
    y.append(xt[2]+dt*ut[0])
    y.append(xt[3]+dt*ut[1])
    return y

dynamics = []

U = []
X = []
# x0 = MX.sym('x0',n)
x0 = sp.MatrixSymbol('x0', n, 1)
X.append(x0)
dynamics.append(x0-xinit)
for t in range(T):
    # ut = MX.sym('u%s'%t,m)
    ut = sp.MatrixSymbol('u%s'%t, m, 1)
    # xtt = MX.sym('x%s'%(t+1),n)
    xtt = sp.MatrixSymbol('x%s'%(t+1), n, 1)
    pred = sp.Matrix(f(X[-1], ut))
    dynamics.append(xtt-pred)
    U.append(ut)
    X.append(xtt)

cost = 10*((X[-1][0]-goal[0])**2 + (X[-1][1]-goal[1])**2)
for t in range(T):
    cost += (1*U[t][0]**2 + 1*U[t][1]**2)

constraints = []
for t in range(T):
    state = X[t+1]
    const = (state[0]-obs[0])**2 + (state[1]-obs[1])**2 - rad*rad # square distance from obs should be rad*rad
    constraints.append(const)

# min_{X,U} cost
#  s.t. dynamics = 0
#       constraints >= 0

# Form KKT conditions

all_dyn = sp.Matrix(dynamics)
all_ineq = sp.Matrix(constraints)
all_primal_vars = sp.Matrix(X+U)


# dyn_mults = MX.sym('dyn_mults', all_dyn.shape[0])
dyn_mults = sp.MatrixSymbol('dyn_mults', all_dyn.shape[0], 1)
#ineq_mults = MX.sym('ineq_mults', all_ineq.shape[0])
ineq_mults = sp.MatrixSymbol('ineq_mults', all_ineq.shape[0], 1)

lag = cost - sp.MatMul(all_dyn,dyn_mults) - sp.MatMul(all_ineq,ineq_mults)

dlag = jacobian(lag, all_primal_vars)
kkt_expr = vcat([dlag.T, all_dyn, all_ineq])
all_vars = vcat([all_primal_vars,dyn_mults,ineq_mults])
jac_kkt = jacobian(kkt_expr, all_vars)
eval_kkt = Function('kkt',[all_vars],[kkt_expr])
eval_kkt_jac = Function('kkt',[all_vars],[jac_kkt])

n = all_vars.shape[0]
nprimal = (T+1)*4 + T*2
ndyn = (T+1)*4
nineq = all_ineq.shape[0]

def feval(y):
    return np.array(eval_kkt(y))

def dfeval(y):
    return np.array(eval_kkt_jac(y))

l = np.vstack((-np.inf*np.ones((nprimal+ndyn,1)),np.zeros((nineq,1))))
u = np.inf*np.ones((n,1))

u_start = np.zeros((T*2,1))
u0 = np.array([0,1]).reshape(2,1)

x_start = np.zeros(((T+1)*4,1))
x_start[0:4] = xinit
for t in range(T):
    u_start[t*2:(t+1)*2] = u0
    x_start[(t+1)*4:(t+2)*4] = f(x_start[(t)*4:(t+1)*4], u_start[(t)*2:(t+1)*2])

x0 = np.vstack((x_start,u_start,np.zeros(((ndyn+nineq,1)))))



[z,w,v,success] = path_solver(feval, dfeval, l, u, x0=x0, sigma=0.1, max_iters=100, tol=1e-4, forward=forward, linesearch=nonlinear_dyn)
print(success)
print("--- %s seconds ---" % (time() - start_time))
traj = z[0:(T+1)*4]
px = traj[0::4]
py = traj[1::4]

# plot a circle
R = np.linspace(0,6.3,100)
plt.plot(rad*cos(R)+obs[0],rad*sin(R)+obs[1])

plt.figure
plt.plot(px,py)
plt.axis([-4,4,-4,4])
plt.axis('square')
plt.show()