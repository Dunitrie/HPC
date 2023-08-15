import numpy as np
import matplotlib.pyplot as plt
from streaming_scenarios import streaming, recalculate_functions
from plotting_functions import plot_velocity, plot_velocity_slice

# Choose scenario
scenario = "sliding lid"

n_timesteps = 100
n_plots = 10

# Initialize Grid:
nx = 200  # num of rows
ny = 160  # num of columns

# Initialize weights and discrete direction vectors
weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [-1, 1], [-1, -1], [1, -1], [1, 1]])
wall_speed_north = 0

match scenario:
    case "shear wave 1":
        borders = [0, 0, 0, 0]
    case "shear wave 2":
        borders = [0, 0, 0, 0]
    case "couette flow":
        borders = [0, 1, 0, 1]
        wall_speed_north = 1
    case "poiseuille flow":
        borders = [1, 1, 1, 1]
    case "sliding lid":
        borders = [1, 1, 1, 1]
        wall_speed_north = 1

match scenario:
    case "shear wave 1":
        rho = np.ones((nx, ny))
        epsilon = 0.1
        rho[...] = 1 + epsilon * np.sin(2*np.pi*np.arange(0,ny)/nx)
        v = np.zeros((2, nx, ny))
    case "shear wave 2":
        rho = np.ones((nx, ny))
        v = np.zeros((2, nx, ny))
        epsilon = 0.1
        v[0:...] = epsilon * np.sin(2*np.pi*np.arange(0,ny)/nx)
    case _:
        rho = np.ones((nx+borders[1]+borders[3], ny+borders[0]+borders[2]))
        v = np.zeros((2, nx+borders[1]+borders[3], ny+borders[0]+borders[2]))

# Initialize grid (add goast points or dry notes to each edge)
f = np.einsum("i,jk -> ijk", weights, np.ones_like(rho))

# Loop over timesteps
for idx_time in range(n_timesteps):
    # Calculate the streaming step wrt (global) boundary conditions
    f, rho, v = streaming(f, rho, v, c, weights, borders, wall_speed_north, scenario)

    rho, v = recalculate_functions(f, rho, v, c)  # Update values

    # Plot average velocity vectors
    if idx_time % (n_timesteps // n_plots) == 0:

        plot_velocity(f[:, borders[1]:nx-borders[3], borders[2]:ny-borders[0]], v[:, borders[1]:nx-borders[3], borders[2]:ny-borders[0]], return_plot=True)

        plt.show()

if scenario == "shear wave 1":
    pass