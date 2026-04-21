"""
2D Lid Driven Cavity Flow using Lattice Boltzmann Method (LBM)

This code simulates the 2D lid-driven cavity flow problem using the Lattice Boltzmann Method (LBM) with the D2Q9 lattice model, and the Single Relaxation Time (SRT) collision operator. The simulation is performed on a square domain with a moving lid at the top, and stationary walls on the other sides. Zou/He boundary conditions are applied for the moving lid, while bounce-back boundary conditions are used for the stationary walls.

Author: Sandun Darshana
"""
import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx, ny = 126, 126 # no of grid points
Lx, Ly = 1.0, 1.0 # domain size
nt = 1000000 # no of time steps

x = np.linspace(0, Lx, nx) 
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x / Lx, y / Ly, indexing='ij')

"""
nu shall be chosen such that the flow is stable i.e., omega < 2.0 and u_lid / c < 0.3 to avoid numerical instabilities and ensure incompressibility.
"""

# Physical parameters
nu = 0.05 # kinematic viscosity
Re = 400.0 # Reynolds number
u_lid = Re * nu / (ny - 1) # lid velocity

# LBM parameters
c2 = 1.0 / 3.0 # lattice speed of sound squared
omega = 1.0 / (3.0 * nu + 0.5) # relaxation parameter
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]) # weights
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1]) # lattice velocities x
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1]) # lattice velocities y
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6]) # opposite directions

# Initialize distribution functions
f = np.zeros((nx, ny, 9)) # distribution function
feq = np.zeros((nx, ny, 9)) # equilibrium distribution function

# Initial conditions
rho = np.ones((nx, ny)) # density
ux = np.zeros((nx, ny)) # velocity x
uy = np.zeros((nx, ny)) # velocity y
ux[:, -1] = u_lid # lid velocity

# function to compute equilibrium distribution function
def compute_feq(rho, ux, uy):
    feq = np.zeros((nx, ny, 9))
    for i in range(9):
        cu = cx[i] * ux + cy[i] * uy
        feq[:, :, i] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * (ux**2 + uy**2))
    return feq

# function to perform one time step of LBM (collision and streaming)
def lbm_step(f, rho, ux, uy):
    # compute feq
    feq = compute_feq(rho, ux, uy)
    # collision step
    f += -omega * (f - feq)
    # streaming step
    f_streamed = np.copy(f)
    for i in range(9):
        f_streamed[:, :, i] = np.roll(np.roll(f[:, :, i], cx[i], axis=0), cy[i], axis=1)
    return f_streamed

# function to apply boundary conditions
def apply_bc(f):
    for i in range(9):
        # no-slip boundary conditions (bounce-back)
        if cx[i] == -1: # right wall
            f[-1, :, i] = f[-1, :, opp[i]]
        if cx[i] == 1: # left wall
            f[0, :, i] = f[0, :, opp[i]]
        if cy[i] == 1: # bottom wall
            f[:, 0, i] = f[:, 0, opp[i]]
        if cy[i] == -1: # top wall (lid)
            rho_wall = (1.0 / (1.0 - uy[:, -1])) * (f[:, -1, 0] + f[:, -1, 1] + f[:, -1, 3] + 2.0 * (f[:, -1, 2] + f[:, -1, 5] + f[:, -1, 6]))
            f[:, -1, i] = f[:, -1, opp[i]] + 6.0 * w[i] * rho_wall * u_lid * cx[i]
    return f

# function to compute macroscopic variables
def compute_macroscopic(f):
    rho = np.sum(f, axis=2)
    ux = np.sum(f * cx, axis=2) / rho
    uy = np.sum(f * cy, axis=2) / rho
    return rho, ux, uy

# function to compute residual for convergence check
def compute_residual(ux, uy, ux_old, uy_old):
    diff_sq = np.sum((ux - ux_old)**2 + (uy - uy_old)**2)
    mag_sq = np.sum(ux**2 + uy**2)
    return np.sqrt(diff_sq) / np.sqrt(mag_sq) if mag_sq > 0.0 else 0.0


# Simulation 
feq = compute_feq(rho, ux, uy)
f = feq.copy()

# store old velocity for convergence check
ux_old = ux.copy()
uy_old = uy.copy()

# Time-stepping loop
for t in range(nt):
    f = lbm_step(f, rho, ux, uy)
    f = apply_bc(f)
    rho, ux, uy = compute_macroscopic(f)
    if t % 1000 == 0:
        residual = compute_residual(ux, uy, ux_old, uy_old)
        print(f'Time step: {t}, Residual: {residual:.6e}')
        if residual < 1e-6:
            print(f'Simulation converged at time step {t}')
            break
        ux_old = ux.copy()
        uy_old = uy.copy()

# Plotting results
# velocity magnitude contour and streamlines
plt.figure(figsize=(8, 6)) 

# Transpose all inputs to fix the indexing 'ij' in meshgrid
plt.contourf(X.T, Y.T, np.sqrt((ux / u_lid)**2 + (uy / u_lid)**2).T, levels=25, cmap='jet')
plt.colorbar(label='Velocity magnitude')

plt.streamplot(X.T, Y.T, (ux / u_lid).T, (uy / u_lid).T, color='k', density=1.5)

plt.xlabel('x / Lx')
plt.ylabel('y / Ly')
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.title(f'2D Lid Driven Cavity Flow (Re={Re})')
plt.show()

# velocity profiles along the centerlines
# u velocity along vertical centerline
plt.subplot(1, 2, 1) 
plt.plot(ux[nx//2, :] / u_lid, y / Ly, label='u velocity', color='b')
plt.xlabel('u / u_lid')
plt.ylabel('y / Ly')
plt.ylim(0, Ly)
# plot benchmark data from Ghia et al. (1982) 
data_u = np.loadtxt('u_ghia.csv', delimiter=',', skiprows=1)
plt.plot(data_u[:, 2], data_u[:, 0], 'ro', label='Ghia et al. (1982)') # column 0 - position, column 1 - Re=100, column 2 - Re=400, column 3 - Re=1000
plt.legend()
plt.title('u Velocity Profile along Vertical Centerline')

# v velocity along horizontal centerline
plt.subplot(1, 2, 2) 
plt.plot(x / Lx, uy[:, ny//2] / u_lid, label='v velocity', color='r')
plt.xlabel('x / Lx')
plt.ylabel('v / u_lid')
plt.xlim(0, Lx)
# plot benchmark data from Ghia et al. (1982) 
data_v = np.loadtxt('v_ghia.csv', delimiter=',', skiprows=1)
plt.plot(data_v[:, 0], data_v[:, 2], 'bo', label='Ghia et al. (1982)') # column 0 - position, column 1 - Re=100, column 2 - Re=400, column 3 - Re=1000
plt.legend()
plt.title('v Velocity Profile along Horizontal Centerline')

plt.tight_layout()
plt.show()






