import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from process_coordination import width_height, bool_boundaries, number_of_blocks
from streaming_functions import streaming, recalculate_functions
from plotting_functions import plot_velocity, plot_velocity_slice

# Initialize parallelization 
comm = MPI.COMM_WORLD
size = comm.Get_size() # num of processes
rank = comm.Get_rank() # rank id of this process

n_timesteps = 50
n_plots = 1

# Initialize Grid:
nx_total = 10  # num of rows
ny_total = 8  # num of columns

# Arrange <size> blocks (num processes) as a optimized grid of
# <n_blocks[0]> rows times <n_blocks[1]> columns.
n_blocks = number_of_blocks((nx_total, ny_total), size)

# Initialize local grid parameters (local grid is the one of the block of this process):
# local size
nx, ny = width_height(rank, nx_total, ny_total, n_blocks)

nx_opt = nx_total//n_blocks[0]
ny_opt = ny_total//n_blocks[1]
block_pos = (rank // n_blocks[1], rank % n_blocks[1])

# Initialize weights and discrete direction vectors
weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [-1, 1], [-1, -1], [1, -1], [1, 1]])

# Initialize grid (add goast points or dry notes to each edge)
rho = np.ones((nx+2, ny+2))  # density values
v = np.zeros((2, nx+2, ny+2))  # average viscosity values
f = np.einsum("i,jk -> ijk", weights, np.ones((nx+2, ny+2)))  # probability density function

# Check on which side this block borders another block or the boundary
borders = bool_boundaries(rank, n_blocks)

# Ranks of the processes of the neighboring blocks (only correct and used when theres no boundary on this side)
rank_right = rank + 1
rank_left = rank - 1
rank_up = rank - n_blocks[1]
rank_down = rank + n_blocks[1]

# Loop over timesteps
for idx_time in range(n_timesteps):
    # Calculate the streaming step wrt (global) boundary conditions
    f, rho, v = streaming(f, rho, v, c, weights, borders)

    # Communicate the outermost grid points to neighboring blocks for the next step
    if not borders[0]:
        if not borders[2]:
            comm.Sendrecv(f[:, :, -2].copy(), rank_right, recvbuf=f[:, :, 0].copy(), source = rank_left)
        else:
            comm.Send(f[:, :, -2].copy(), rank_right)
    if not borders[1]:
        if not borders[3]:
            comm.Sendrecv(f[:, 1, :].copy(), rank_up, recvbuf=f[:, -1, :].copy(), source=rank_down)
        else:
            comm.Send(f[:, 1, :].copy(), rank_up)
    if not borders[2]:
        if not borders[0]:
            comm.Sendrecv(f[:, :, 1].copy(), rank_left, recvbuf=f[:, :, -1].copy(), source = rank_right)
        else:
            comm.Send(f[:, :, 1].copy(), rank_left)
    if not borders[3]:
        if not borders[1]:
            comm.Sendrecv(f[:, -2, :].copy(), rank_down, f[:, 0, :].copy(), source = rank_up)
        else:
            comm.Send(f[:, -2, :].copy(), rank_down)

    if rank == 0:
        pass#f = -5 * f

    # Plot average velocity vectors
    if idx_time % (n_timesteps // n_plots) == 0:
        # stack everything in rank 0
        f_full = np.zeros((9, nx_total, ny_total))
        rho_full = np.ones((nx_total, ny_total))
        v_full = np.zeros((2, nx_total, ny_total))
        if rank == 0:
            comm.Gather(f[:,1:-1,1:-1].copy(),
                        f_full[:, (nx_opt * block_pos[0]):(nx_opt * block_pos[0] + nx), (ny_opt * block_pos[1]):(ny_opt * block_pos[1] + ny)].copy(),
                        root=0)
        f_full, rho_full, v_full = recalculate_functions(f_full, rho_full, v_full, c)
        if rank == 0:
            ax = plot_velocity(f_full, v_full, return_plot=True)
            # x_width = nx_total//n_blocks[0]
            # y_width = ny_total//n_blocks[1]
            # for idx in range(1,n_blocks[1]):
            #     ax.plot(np.ones(f_full[0].shape[1]+2)*(idx*x_width-1), np.arange(-1,f_full[0].shape[1]+1), 'k')
            # for idx in range(1, n_blocks[0]):
            #     ax.plot(np.arange(-1,f_full[0].shape[0]+1), np.ones(f_full[0].shape[0]+2)*(idx*y_width-1), 'k')
            plt.show()
            
            # plot in rank 0
    

# plot the ending
