import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from process_coordination import width_height, bool_boundaries, number_of_blocks
from streaming_functions import streaming, recalculate_functions
from plotting_functions import plot_velocity, plot_velocity_slice

import pickle
import warnings

# Initialize parallelization 
comm = MPI.COMM_WORLD
size = comm.Get_size() # num of processes
rank = comm.Get_rank() # rank id of this process

n_timesteps = 20
n_plots = 5

# Initialize Grid:
nx_total = 20  # num of rows
ny_total = 16  # num of columns

# Arrange <size> blocks (num processes) as a optimized grid of
# <n_blocks[0]> rows times <n_blocks[1]> columns.
n_blocks = number_of_blocks((nx_total, ny_total), size)

# Initialize local grid parameters (local grid is the one of the block of this process):
# local size
nx, ny = width_height(rank, nx_total, ny_total, n_blocks)

nx_opt = nx_total//n_blocks[0]
ny_opt = ny_total//n_blocks[1]

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
    f, rho, v = streaming(f, rho, v, c, weights, borders, rank, idx_time)

    # Order of communcations is important in order that all the corner ghost points will get the diagonal adjacent values via two-step-communcation.
    if not borders[0]:
        comm.send(f[:, :, -2].copy(), rank_right)
        data = comm.recv(source=rank_right)
        f[:, :, -1] = data
    if not borders[2]:
        comm.send(f[:, :, 1].copy(), rank_left)
        data = comm.recv(source=rank_left)
        f[:, :, 0] = data
    if not borders[1]:
        comm.send(f[:, 1, :].copy(), rank_up)
        data = comm.recv(source=rank_up)
        f[:, 0, :] = data
    if not borders[3]:
        comm.send(f[:, -2, :].copy(), rank_down)
        data = comm.recv(source=rank_down)
        f[:, -1, :] = data

    rho, v = recalculate_functions(f, rho, v, c, rank, idx_time)  # Update values

    # Plot average velocity vectors
    if idx_time % (n_timesteps // n_plots) == 0:
        # stack everything in rank 0
        f_full = np.zeros((9, nx_total, ny_total))
        rho_full = np.ones((nx_total, ny_total))
        v_full = np.zeros((2, nx_total, ny_total))
        f_list = comm.gather(f[:,1:-1,1:-1].copy(), root=0)
        if rank == 0:       
            for rank_idx, f_block in enumerate(f_list):
                block_pos = (rank_idx // n_blocks[1], rank_idx % n_blocks[1])
                f_full[:, (nx_opt * block_pos[0]):(nx_opt * block_pos[0] + f_block.shape[1]), (ny_opt * block_pos[1]):(ny_opt * block_pos[1] + f_block.shape[2])] = f_block
            rho_full, v_full = recalculate_functions(f_full, rho_full, v_full, c, 5, -1)
        
            axes = plot_velocity(f_full, v_full, return_plot=True)
            # x_width = nx_total//n_blocks[0]
            # y_width = ny_total//n_blocks[1]
            # for idx in range(1,n_blocks[1]):
            #     ax.plot(np.ones(f_full[0].shape[1]+2)*(idx*x_width-1), np.arange(-1,f_full[0].shape[1]+1), 'k')
            # for idx in range(1, n_blocks[0]):
            #     ax.plot(np.arange(-1,f_full[0].shape[0]+1), np.ones(f_full[0].shape[0]+2)*(idx*y_width-1), 'k')
            plt.show()
            
            # plot in rank 0

# plot the ending
