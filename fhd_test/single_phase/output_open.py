from vtk import vtkStructuredPointsReader, vtkXMLImageDataReader
from vtk.util import numpy_support as VN
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
    
    
def export_plot(t_tensor, path = None, corr = None):
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
    
    fig, axs = plt.subplots(num, denom, figsize = (x, y))
    print(x, y, num, denom)
    
    for i in range(num):
        for j in range(denom):

            S = t_tensor[..., i, j]
            to_plot = np.abs(S[L//2])
            if num*denom == 1:
                curr_axs = axs
                im = curr_axs.imshow(to_plot)
            elif num == 1:
                curr_axs = axs[j]
                im = curr_axs.imshow(to_plot)  
            else:
                curr_axs = axs[i, j]
                im = axs[i, j].imshow(to_plot)
                
            fig.colorbar(im, ax = curr_axs, orientation="horizontal")

    fig.tight_layout()
    
    if path != None:
        fig.savefig(path+'/{0}_struct-fact.png'.format(corr))
      
    return fig
