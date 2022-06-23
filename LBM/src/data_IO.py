import pyevtk.hl 

def write_vtk(path, d):
    # https://pubs.rsc.org/en/content/articlelanding/2017/SM/C7SM00317J#!divAbstract
    # https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python
    # https://docs.paraview.org/en/latest/UsersGuide/filteringData.html
    # https://pyscience.wordpress.com/2014/09/06/numpy-to-vtk-converting-your-numpy-arrays-to-vtk-arrays-and-files/
    # https://github.com/pyscience-projects/pyevtk/blob/master/examples/image.py
    # https://docs.paraview.org/en/latest/UsersGuide/filteringData.html#changing-filter-properties-in-paraview
    pyevtk.hl.imageToVTK(path, pointData = d)
    return 1