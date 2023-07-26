import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from process_coordination import width_height, bool_boundaries, number_of_blocks

# Initialize parallelization 
comm = MPI.COMM_WORLD
size = comm.Get_size() # n_processes
rank = comm.Get_rank()

n_timesteps = 10000
n_plots = 5

# Initialize Grid:
nx_total = 300
ny_total = 200

n_blocks = number_of_blocks((nx_total, ny_total), size)

# Initialize local grid parameters:
# local size
nx, ny = width_height(rank, nx_total, ny_total, n_blocks)
borders = bool_boundaries(rank, nx, ny, n_blocks)
# rank of the process, where there's no (bounce-back-)boundary
if borders[0]:
    rank_right = rank + 1
elif borders[2]:
    rank_left = rank - 1
if borders[1]:
    rank_up = rank - n_blocks[1]
elif borders[3]:
    rank_down = rank + n_blocks[1]
    
# Initialize grid
f = np.zeros(9, nx+2, ny+2)
... # initialize beginning distribution

# loop over timesteps
for idx_time in ...:
    # 1. do the flow
    ...
    # 2. do the walls:
    # We should probably distinguish between the indices
    for idx in range(4):
        if borders[idx]:
            
    
    # 3. do the communications
    if not borders[0]:
        if not borders[2]:
            comm.Sendrecv(f[:, :, -2], rank_right, recvbuf=f[:, :, 0], source = rank_left)
        else:
            comm.Send(f[:, :, -2], rank_right)
    if not borders[1]:
        if not borders[3]:
            comm.Sendrecv(f[:, 1, :], rank_up, recvbuf=f[:, -1, :], source=rank_down)
        else:
            comm.Sendrecv(f[:, 1, :], rank_up)
    if not borders[2]:
        if not borders[0]:
            comm.Sendrecv(f[:, 1, :], rank_left, recvbuf=f[:, -1, :], source = rank_right)
        else:
            comm.Send(f[:, 1, :], rank_left)
    if not borders[3]:
        if not borders[1]:
            comm.Sendrecv(f[:, -2, :], rank_down, f[:,0,:], source = rank_up)
        else:
            comm.Send(f[:, -2, :], rank_down)
                
    # 4. do the plotting
    if idx_time % (n_timesteps // n_plots) == 0:
        # stack everything in rank 0
        comm.Gather(...)
        if rank == 0:
            # plot in rank 0 
