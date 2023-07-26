import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from process_coordination import width_height, bool_boundaries, number_of_blocks
from streaming_functions import streaming

# Initialize parallelization 
comm = MPI.COMM_WORLD
size = comm.Get_size() # n_processes
rank = comm.Get_rank()

n_timesteps = 10
n_plots = 5

# Initialize Grid:
nx_total = 300
ny_total = 200

n_blocks = number_of_blocks((nx_total, ny_total), size)

# Initialize local grid parameters:
# local size
nx, ny = width_height(rank, nx_total, ny_total, n_blocks)
# Initialize weights and discrete direction vectors
weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [-1, 1], [-1, -1], [1, -1], [1, 1]])

# initialize grid
rho = np.ones((nx+2, ny+2))
v = np.zeros((2, nx+2, ny+2))

f = np.einsum("i,jk -> ijk", weights, np.ones((nx+2, ny+2)))

borders = bool_boundaries(rank, n_blocks)

# rank of the process, where there's no (bounce-back-)boundary
rank_right = rank + 1
rank_left = rank - 1
rank_up = rank - n_blocks[1]
rank_down = rank + n_blocks[1]

# loop over timesteps
for idx_time in range(n_timesteps):
    # 1. do the flow
    f, rho, v = streaming(f, rho, v, c, weights, borders)

    # 3. do the communications
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
                
    # 4. do the plotting
    if idx_time % (n_timesteps // n_plots) == 0:
        # stack everything in rank 0
        f_full = np.zeros((9, nx_total, ny_total))
        comm.Gather(f[:,1:-1,1:-1].copy(), f_full, root=0)
        if rank == 0:
            pass
            # plot in rank 0 
# plot the ending
