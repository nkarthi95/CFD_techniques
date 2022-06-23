import lb_helper
import numpy as np

idxs_grid = np.array([[8, 1, 2], [7, 0, 3], [6, 5, 4]])

NL = 9
idxs = np.arange(NL)
vx_s = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
vy_s = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1 

def assign_wall_velocity(u_w, vel = [], placement = []):
    (Ny, Nx, axis) = u_w.shape
    if len(placement) == 0:
        return u_w
    for side in placement:
        if side == 'top':
            u_w[Ny - 1, :, 0] += vel[side]
        elif side == 'bottom':
            u_w[0, :, 0] += vel[side]
        elif side == 'left':
            u_w[:, 0, 1] += vel[side]
        elif side == 'right':
            u_w[:, Nx - 1, 1] += vel[side]
    return u_w

def object_detection(F, obj):
    if len(obj) == 0:
        return F
    else:
        for j, b in enumerate(obj):
            temp = F[b, :]
            F[b, :] = temp[:, [0,5,6,7,8,1,2,3,4]]
        return F

def wall_velocity(F, u_w = 0):
    (Ny, Nx, NL) = F.shape
    wall_velocity = np.zeros(F.shape)
    if np.unique(u_w).size == 1:
        return wall_velocity
    else:
        rho, u, v = lb_helper.macroscopic(F)
        rho_w = np.average(rho)
        for i in range(0, NL):
            cu = vx_s[i]*u_w[:, :, 0] + vy_s[i]*u_w[:, :, 1]
            wall_velocity[:, :, i] = 2*weights[i]*rho_w*(cu)/(cs**2)
    return wall_velocity