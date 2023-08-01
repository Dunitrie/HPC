import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from process_coordination import width_height, bool_boundaries, number_of_blocks
from streaming_functions import streaming
from plotting_functions import plot_velocity, plot_velocity_slice

# Initialize parallelization 
comm = MPI.COMM_WORLD
size = comm.Get_size() # num of processes
rank = comm.Get_rank() # rank id of this process

n_timesteps = 50
n_plots = 5

# Initialize Grid:
nx_total = 150  # num of rows
ny_total = 100  # num of columns

# Arrange <size> blocks (num processes) as a optimized grid of
# <n_blocks[0]> rows times <n_blocks[1]> columns.
n_blocks = number_of_blocks((nx_total, ny_total), size)

# Initialize local grid parameters (local grid is the one of the block of this process):
# local size
nx, ny = width_height(rank, nx_total, ny_total, n_blocks)

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

print(f"Rank: {rank}, size: {nx, ny}, borders: {borders}")
print(f"Rank: {rank}, neighbors: {rank_right, rank_up, rank_left, rank_down}")

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
            comm.Sendrecv(f[:, 1, :].copy(), rank_left, recvbuf=f[:, -1, :].copy(), source = rank_right)
        else:
            comm.Send(f[:, 1, :].copy(), rank_left)
    if not borders[3]:
        if not borders[1]:
            comm.Sendrecv(f[:, -2, :].copy(), rank_down, f[:,0,:].copy(), source = rank_up)
        else:
            comm.Send(f[:, -2, :].copy(), rank_down)
                
    # Plot ?? TODO
    if idx_time % (n_timesteps // n_plots) == 0:
        # stack everything in rank 0
        f_full = np.zeros((9, nx_total, ny_total))
        comm.Gather(f[:,1:-1,1:-1].copy(), f_full, root=0)
        if rank == 0:
            plot_velocity(f_full)
            # plot in rank 0 

# plot the ending
