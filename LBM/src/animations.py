import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

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

def plot_ani(x, y, interval = 50):
    fig, ax = plt.subplots()
    ln, = plt.plot(x[0], y[0])
#     plt.ylim(min(y[0]), max(y[0]))
#     plt.xlim(min(x[0]), max(x[0]))

    def init():
        plt.ylim(min(y[0]), max(y[0]))
        plt.xlim(min(x[0]), max(x[0]))
        ln.set_data(x[0], y[0])
        return ln,

    def update(i):
        plt.ylim(min(y[i]), max(y[i]))
        plt.xlim(min(x[i]), max(x[i]))
        ln.set_data(x[i], y[i])
        return ln,

    ani = animation.FuncAnimation(fig, update, frames= len(x), init_func=init, interval=interval, blit=True)
    plt.close()
    return ani