import numpy as np
import matplotlib.pyplot as plt
from streaming_functions import recalculate_functions

def plot_velocity(f, v, return_plot=False):
    """ Plot the averagae velocity of the distribution f.
    
    
    Args:
        f: probability distribution
        
    Optional:
        c: Directions of D2Q9-scheme
        v: velocity for each channel if already computed so we avoid computing it again. Otherwise computed as sum over channels times c
        return_plot: If True, return the axis-Object to plot it or change it in other file
        fix_dims: If True: To keep plots coherent, invert x-axis and change x and y-axis
    
    """
    v = np.swapaxes(v, 1, 2)
        
    fig, axes = plt.subplots(1,2, layout='tight')
    
    y, x = np.meshgrid(np.arange(f.shape[1]), np.arange(f.shape[2]))
    axes[0].quiver(x,y, v[1,...], v[0,...], angles='xy', scale_units='xy', scale=1, color='b', label='Vector Field')
    axes[0].grid()
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('X')
    #axes[0].axis('equal')
    axes[0].set_xlim(-1, np.max(x)+1)
    axes[0].set_ylim(-1, np.max(y)+1)
    axes[0].set_title("Velocity field")
    axes[0].invert_yaxis()
    
    
    #plot the edges
    axes[0].plot(np.arange(-1,x.shape[0]+1), np.ones(x.shape[0]+2)*(-1), 'k')
    axes[0].plot(np.arange(-1,x.shape[0]+1), np.ones(x.shape[0]+2)*(x.shape[1]), 'k')
    
    axes[0].plot(np.ones(x.shape[1]+2)*(-1), np.arange(-1,x.shape[1]+1), 'k')
    axes[0].plot(np.ones(x.shape[1]+2)*(x.shape[0]), np.arange(-1,x.shape[1]+1), 'k')
    
    rho = np.einsum('cij->ij', f)
    axes[1].imshow(rho, aspect='auto')
    axes[1].get_yaxis().tick_right()
    axes[1].set_title("Density Plot")
    #axes[1].invert_yaxis()
    
    
    if return_plot:
        return axes
    else:
        plt.show()
        
def plot_velocity_slice(f, c=None, v=None, return_plot=False, fix_dims=True, avg_vel=True):
    """ Plot the averagae velocity of the distribution f over a slice to the right.
    
    Args:
        f: probability distribution
        c: Directions of D2Q9-scheme
    Optional:
        v: velocity for each channel if already computed so we avoid computing it again. Otherwise computed as sum over channels times c
        return_plot: If True, return the axis-Object to plot it or change it in other file
        fix_dims: If True: To keep plots coherent, invert x-axis and change x and y-axis
    """
    
    if c is None:
        c = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [-1, 1], [-1, -1], [1, -1], [1, 1]])
    if fix_dims:
        c = np.stack([c[:,1], -c[:,0]], axis=1) 
    if v is None:
        v = np.einsum('cij, cd -> dij', f, c)
    
    if avg_vel:
        #print(v.shape)
        v_slice = np.average(v[0,...],axis=1)
    else:
        v_slice = v[1, :, v.shape[2]//2]
        
    ax = plt.subplot()
    ax.plot(v_slice)
    ax.grid()
    ax.set_xlabel('X (Bottom to Top)')
    ax.set_ylabel('Y-Velocity (to the right)')
    ax.set_yticks([0])
    ax.axis('equal')
    ax.set_title("Velocity")
    if return_plot:
        return ax
    else:
        plt.show()