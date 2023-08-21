import numpy as np
import matplotlib.pyplot as plt
from streaming_scenarios import streaming, recalculate_functions
from plotting_functions import plot_velocity, plot_velocity_slice

# Choose scenario
# Possibilities: "shear wave 1", "shear wave 2", "couette flow", "poiseuille flow", "sliding lid"
scenario = "sliding lid"

n_timesteps = 1001
n_plots = 2

# Initialize Grid:
nx = 30  # num of rows
ny = 20  # num of columns

# Initialize weights and discrete direction vectors
weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [-1, 1], [-1, -1], [1, -1], [1, 1]])

wall_speed_north = 1

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
        # Reynolds number calculation: Leaving system lenght L = 300 and viscosity of water nhy= 10^-6 m^2/s,
        # modifying the velocity of the moving wall
        # return the velocity of the moving wall with constant Re
        L = ny
        nü = 10**(-6)
        Re = 1000
        def comp_vel_moving_wall(Re, L):
            return (Re * nü) / L
         wall_speed_north = comp_vel_moving_wall(Re, L)

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
        v[0,...] = epsilon * np.sin(2*np.pi*np.arange(0,ny)/nx)
    case _:
        rho = np.ones((nx+borders[1]+borders[3], ny+borders[0]+borders[2]))
        v = np.zeros((2, nx+borders[1]+borders[3], ny+borders[0]+borders[2]))

# Initialize grid (add goast points or dry notes to each edge)
f = np.einsum("i,jk -> ijk", weights, np.ones_like(rho))

# Loop over timesteps
for idx_time in range(n_timesteps):
    # Plot average velocity vectors
    if idx_time % (n_timesteps // n_plots) == 0:

        print(idx_time)

        #plot_velocity(f, v, return_plot=True)
        plot_velocity(f[:, 1:-1, 1:-1], v[:, 1:-1, 1:-1], return_plot=False)
        plt.savefig("slidinglid.png")
        plt.show()

    # Calculate the streaming step wrt (global) boundary conditions
    f, rho, v = streaming(f, rho, v, c, weights, borders, wall_speed_north, scenario)

    rho, v = recalculate_functions(f, rho, v, c)  # Update values

# streamplot for velocity field at end of simulation
Y, X = np.mgrid[0:nx, 0:ny]
plt.streamplot(X, Y, v[0,1:nx+1, 1:ny+1], v[1,1:nx+1, 1:ny+1 ], density=[1,1])
plt.savefig("slidinglidstream.png")
plt.show()