from sympy import *
from sympy.solvers.solveset import linsolve
import numpy as np
import matplotlib.pyplot as plt
from vtk import vtkStructuredPointsReader, vtkXMLImageDataReader
from vtk.util import numpy_support as VN
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

plt.rcParams.update({'font.size': 22})

def convert_lb(nu_sim = 1/6, rho_sim = 1, T_real = 37):
    rho_real = 993.36

    nu_real = 6.969*(10**(-7))

    kbT_real = (1.380649*10**(-23))*(273 + T_real) #4.28*10**(-21)
    kb_sim = 1

    L = 64 # length of each side in simulation
    x = (64/L * 10**(-6))
    
    variables = 't, m, T'
    t, m, T = symbols(variables)
    headers = variables.split(', ')

    eq1 = rho_sim*(m/(x**3)) - rho_real
    eq2 = nu_sim*(x**2/t) - nu_real
    eq3 = kb_sim*T*(m*(x**2)/(t**2)) - kbT_real
    order_sols = (t, m, T)

    sols = solve([eq1, eq2, eq3], order_sols)[0]
    d = {headers[i]:float(sols[i]) for i in range(3)}
    
    return d

def export_map(t_tensor, header, path = None, corr = None):
    #t_tensor is of shape [L, L, L, denom, num]
    shape = t_tensor.shape
    sz = 15
    
    num = shape[-1]
    denom = shape[-2]
    L = shape[0]
    
    if num > denom:
        x = sz*num/denom
        y = sz
    else:
        x = sz*denom/num
        y = sz
    
    fig, axs = plt.subplots(denom, num, figsize = (x, y))
    
    for i in range(denom):
        for j in range(num):

            S = t_tensor[..., i, j]
            to_plot = np.abs(S[L//2])
            if num*denom == 1:
                curr_axs = axs
                im = curr_axs.imshow(to_plot)
            elif num == 1:
                curr_axs = axs[i]
                im = curr_axs.imshow(to_plot)  
            else:
                curr_axs = axs[i, j]
                im = axs[i, j].imshow(to_plot)
            
            #curr_axs.set(title = header[i*denom + j])
                
            fig.colorbar(im, ax = curr_axs, orientation="horizontal")

    fig.tight_layout()
    
    if path != None:
        fig.savefig(path+'/{0}_struct-fact.png'.format(corr))
        
        
def export_plot(t_tensor, header, path = None, corr = None):
    # t_tensor is of shape (t, 1, 1, 1, denom, num)
    shape = t_tensor.shape
    sz = 15
    
    num = shape[-1]
    denom = shape[-2]
    L = shape[1]
    
    if num > denom:
        x = sz*num/denom
        y = sz
    else:
        x = sz*denom/num
        y = sz
    
    fig, axs = plt.subplots(denom, num, figsize = (x, y))
    
    for i in range(denom):
        for j in range(num):

            to_plot = t_tensor[..., i, j]
            if num*denom == 1:
                curr_axs = axs
                im = curr_axs.plot(to_plot)
            elif num == 1:
                curr_axs = axs[i]
                im = curr_axs.plot(to_plot)  
            else:
                curr_axs = axs[i, j]
                im = axs[i, j].plot(to_plot)
            
            #curr_axs.set(title = header[i*denom + j])

    fig.tight_layout()
    
    if path != None:
        fig.savefig(path+'/{0}.png'.format(corr))
    
    
def cmap_ani(data, interval=50):
    def init():
        img.set_data(data[0])
        return (img,)

    def update(i):
        img.set_data(data[i])
        return (img,)

    fig, ax = plt.subplots()
    img = ax.imshow(data[0], cmap = 'bwr', vmin = np.amin(data), vmax = np.amax(data))
#     plt.imshow(rho, vmin = rho0-0.1, vmax = rho0+0.1, cmap = 'bwr')
    fig.colorbar(img, ax = ax, orientation="horizontal", pad=0.2)
    ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, interval=interval, blit=True)
    plt.close()
    return ani
        
def get_autocorr(time_corr, P_arr, ts):

    for i in range(0, time_corr.shape[0]):
        time_corr[i] = (P_arr[-1]*P_arr[-1 - i] + time_corr[i]*(ts - 1))/ts
        
    time_corr = (P_arr[-1]*P_arr + time_corr*(ts - 1))/ts
        
    return time_corr   

def get_point_headers(data):
    point_data_obj = data.GetAttributes(0)
    headers_list = []
    
    check = 0
    i = 0
    
    while i != None:
    	h = point_data_obj.GetArrayName(i)
    	if h == None or i > 27:
    	    break
    	else:
    	    headers_list.append(h)
    	    i += 1
   
    return headers_list

def read_vti(path):
    reader = vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    
    data = reader.GetOutput()
    dim = data.GetDimensions()
    
    p_headers = get_point_headers(data)
    
    out = np.zeros((*dim, len(p_headers)))
    for i in range(len(p_headers)):
    	out[..., i] = VN.vtk_to_numpy(data.GetPointData().GetArray(p_headers[i])).reshape(*dim, order = 'C')
    	
    return out, p_headers
 
def read_vtk(path):
    reader = vtkStructuredPointsReader()
    reader.SetFileName(path)
    reader.Update()
    
    data = reader.GetOutput()
    dim = data.GetDimensions()
    
    p_headers = get_point_headers(data)
    
    dim = [3, *dim]
    
    out = np.zeros(dim)
    for i in range(len(p_headers)):
    	out = VN.vtk_to_numpy(data.GetPointData().GetArray(p_headers[i])).reshape(*dim, order = 'C')
    
    return out