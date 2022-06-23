import boundaries
import numpy as np
# https://stackoverflow.com/questions/12638790/drawing-a-rectangle-inside-a-2d-numpy-array
# https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c
    
idxs_grid = np.array([[8, 1, 2], [7, 0, 3], [6, 5, 4]])

NL = 9
idxs = np.arange(NL)
vx_s = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
vy_s = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1    

def macroscopic(F, rho0 = 1):
#     (Ny, Nx, NL) = F.shape
#     u = np.zeros((Ny, Nx))
#     v = np.zeros((Ny, Nx))
    rho = rho0*np.sum(F, axis = -1)
    u = np.sum(F*vx_s, axis = -1)/rho
    v = np.sum(F*vy_s, axis = -1)/rho
    return rho, u, v

def equilibrium(F, rho, u, v, cs = 1/np.sqrt(3)):
    F_shape = F.shape
    Feq = np.zeros(F_shape)
    usqr = (u**2 + v**2)/(2*(cs**2))
    for i in range(0, F_shape[-1]):
        cu = vx_s[i]*u + vy_s[i]*v
        Feq[..., i] = weights[i]*rho*(1 + cu/(cs**2) + cu**2/(2*(cs**4)) - usqr)
    return Feq

def streaming(F):
    (Ny, Nx, NL) = F.shape
    for i, vx, vy in zip(np.arange(NL), vx_s, vy_s):
        F[:,:,i] = np.roll(F[:,:,i], vx, axis = 1)
        F[:,:,i] = np.roll(F[:,:,i], vy, axis = 0)
    return F
    
# def object_detection(F, F_star, obj):
#     if len(obj) == 0:
#         return F
#     else:
#         for j, b in enumerate(obj):
#             temp = F_star[b, :]
#             F[b, :] = temp[:, [0,5,6,7,8,1,2,3,4]]
#         return F
            
    
# def bounce_back(F, F_star, sides, u_w = 0):
#     (Ny, Nx, NL) = F.shape
#     w_v = wall_velocity(F, u_w)
#     if len(sides) == 0:
#         return F - w_v
#     else:
#         for side in sides:
#             if 'top' in side:
#                 F[Ny-1, :, idxs_grid[2]] = F_star[Ny-1, :, idxs_grid[0]] - w_v[Ny-1, :, idxs_grid[0]]
#             elif 'bottom' in side:
#                 F[0, : , idxs_grid[0]] = F_star[0, :, idxs_grid[2]] - w_v[0, :, idxs_grid[2]]
#             elif 'left' in side:
#                 F[:, 0, idxs_grid[:, 0]] = F_star[:, 0, idxs_grid[:, 2]] - w_v[:, 0, idxs_grid[:, 2]]
#             elif 'right' in side:
#                 F[:, Nx-1, idxs_grid[:, 2]] = F_star[:, Nx-1, idxs_grid[:, 0]] - w_v[0, Nx-1, idxs_grid[:, 0]]
#     return F

def bounce_back(F, F_star, sides, u_w = 0):
    (Ny, Nx, NL) = F.shape
    w_v = boundaries.wall_velocity(F, u_w)
    if len(sides) == 0:
        return F - w_v
    else:
        for side in sides:
            if 'top' in side:
                F[-2, :, idxs_grid[2]] = F_star[-2, :, idxs_grid[0]] - w_v[-2, :, idxs_grid[0]]
            elif 'bottom' in side:
                F[1, : , idxs_grid[0]] = F_star[1, :, idxs_grid[2]] - w_v[1, :, idxs_grid[2]]
            elif 'left' in side:
                F[:, 1, idxs_grid[:, 0]] = F_star[:, 1, idxs_grid[:, 2]] - w_v[:, 1, idxs_grid[:, 2]]
            elif 'right' in side:
                F[:, -2, idxs_grid[:, 2]] = F_star[:, -2, idxs_grid[:, 0]] - w_v[0, -2, idxs_grid[:, 0]]
    return F

def pressure_gradient(F, rho_in, rho_out):
    
    right = F[:,1,:].copy()
    left = F[:,-2,:].copy()
    
    rho, u, v = macroscopic(right)
    right = equilibrium(right,rho_in,u,v) + ( right - equilibrium(right,rho,u,v) )
    
    rho, u, v = macroscopic(left)
    left = equilibrium(left,rho_out,u,v) + ( left - equilibrium(left,rho,u,v) )
    
    F[:,-1,idxs_grid[:, 0]] = right[:,idxs_grid[:, 0]] 
    F[:,0,idxs_grid[:, 2]] = left[:,idxs_grid[:, 2]]
    
#     (Ny, Nx, NL) = F.shape
#     rho, u, v = macroscopic(F)
    
#     F_rho_in = equilibrium(F, rho_in, u, v)
#     F_rho_out = equilibrium(F, rho_out, u, v)
#     F_eq = equilibrium(F, rho, u, v)
    
#     f_0_eq = equilibrium(F, rho_in, u, v)[:, -2, :]
#     f_n_eq = equilibrium(F, rho_out, u, v)[:, 1, :]
        
#     f_0_neq = F[:, -2, :] - F_eq[:, -2, :]
#     f_n_neq = F[:, 1, :] - F_eq[:, 1, :]
    
#     f_0_star = f_0_eq + f_0_neq
#     f_n_star = f_n_eq + f_n_neq
    
#     F[:, 0, idxs_grid[:, 2]] = f_0_star[:, idxs_grid[:, 2]]
#     F[:, -1, idxs_grid[:, 0]] = f_n_star[:, idxs_grid[:, 0]]
    return F

# def pressure_gradient(F, rho_in, rho_out):
    
#     right = F[:,-1,:].copy()
#     left = F[:,0,:].copy()
    
#     rho, u, v = macroscopic(right)
#     right = equilibrium(right,rho_in,u,v) + ( right - equilibrium(right,rho,u,v) )
    
#     rho, u, v = macroscopic(left)
#     left = equilibrium(left,rho_out,u,v) + ( left - equilibrium(left,rho,u,v) )
    
#     # TODO check axis order after slicing
    
#     F[:,-1,idxs_grid[:, 2]] = right[:,idxs_grid[:, 2]] 
#     F[:,0,idxs_grid[:, 0]] = left[:,idxs_grid[:, 0]]

#     return F

def periodicity(F, sides = ['top', 'bottom', 'left', 'right']):
    if len(sides) == 0:
        return F
    else:
        for side in sides:
            if side == 'top':
                F[-1, :, idxs_grid[0, :]] = F[1, :, idxs_grid[0, :]]
            elif side == 'bottom':
                F[0, :, idxs_grid[2, :]] = F[-2, :, idxs_grid[2, :]]
            elif side == 'left':
                F[:, -1, idxs_grid[:, 0]] = F[:, 1, idxs_grid[:, 0]]
            elif side == 'right':
                F[:, 0, idxs_grid[:, 2]] = F[:, -2, idxs_grid[:, 2]]
    
    return F

def time_loop(F, n_iteration, elements = [], bounce_back_walls = []):
    (Ny, Nx, Nl) = F.shape
    
    rho_s = np.zeros((n_iteration, Ny, Nx))
    u_s = np.zeros((n_iteration, Ny, Nx))
    v_s = np.zeros((n_iteration, Ny, Nx))
    
    timesteps = np.arange(n_iteration, 1)
    for t in range(n_iteration):
        
        # Compute macroscopic moments
        rho, u , v = macroscopic(F)
        
        # Obtain equilibrium distribution 
        Feq = equilibrium(F, rho, u, v)
        
        # Output
        rho_s[t] = rho
        u_s[t] = u
        v_s[t] = v
#         print(rho.sum())
        
        # Collision
        F += - (1.0/tau)*(F - Feq)
    
        # Set boundary conditions
        F_star = F
        
        # Periodic Boundaries
        F = periodicity(F)
    
        # Apply boundary conditions
#         F = object_detection(F, F_star, elements)
        F = boundaries.object_detection(F, elements)
        F = bounce_back(F, F_star, bounce_back_walls, u_w)
        if pressure == True:
            F = pressure_gradient(F, rho_in, rho_out)

        # Streaming
        F[1:ny, 1:nx, :] = streaming(F[1:ny, 1:nx, :])
    
        
    return timesteps, rho_s, u_s, v_s