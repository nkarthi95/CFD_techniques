from PIL import Image, ImageDraw, ImageFont
import numpy as np

def rotate(vertices, theta):
    theta = (theta/180)*np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(vertices, R)
  
def place_object(grid, obj, f):
    img = Image.fromarray(grid.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(p) for p in obj], fill=f)
    new_data = np.asarray(img)
    return new_data, np.where(new_data == f, True, False)        

def place_rect(rho, x, y, width, height, fill = 0, angle = 0):
    rect = np.array([(-width/2, -height/2), (width/2, -height/2), (width/2, height/2), (-width/2, height/2)])
    offset = np.array([x, y])
    rect = rotate(rect, angle) + offset
    rho, obj = place_object(rho, rect, fill)
    return rho, obj

def place_triangle(rho, x, y, width, height, fill = 0, angle = 0):
    triangle = np.array([(width/2, -height/2), (-width/2, -height/2), (0, height/2)])
    offset = np.array([x, y])
    triangle = rotate(triangle, angle) + offset
    rho, obj = place_object(rho, triangle, fill)
    return rho, obj

def place_ellipse(rho, x, y, r1, r2, fill = 0, angle = 0):
    ell = [(-r1, 0), (0, -r2), (r1, 0), (0, r2)]
    offset = np.array([x, y])
    ell = rotate(ell, angle) + offset
    ell = [(ell[0, 0]), (ell[1, 1]), (ell[2, 0]), (ell[3, 1])]
    img = Image.fromarray(rho.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.ellipse(ell, fill = fill)
    new_data = np.asarray(img)
    return new_data, np.where(new_data == fill, True, False)

def place_plate(rho, x, y, width, height, fill = 0):
    line = [(x - width/2, y + height/2), (x + width/2, y - height/2)]
    img = Image.fromarray(rho.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.line(line, fill = fill)
    new_data = np.asarray(img)
    return new_data, np.where(new_data == fill, True, False)