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
boundaries = bool_boundaries(rank, nx, ny, n_blocks)

# Initialize grid
grid = np.zeros(9, nx+2, ny+2)
... # initialize beginning distribution

# loop over timesteps
for idx_time in ...:
    # 1. do the flow
    ...
    # 2. do the walls:
    # We should probably distinguish between the indices
    for idx in range(4):
        if boundaries[idx]:
    
    
    # 3. do the communications
        else:
            comm.Sendrecv(...)
    
    # 4. do the plotting
    if idx_time % (n_timesteps // n_plots) == 0:
        # stack everything in rank 0
        comm.Gather(...)
        if rank == 0:
            # plot in rank 0 
            
    
        
        

            
    
    


    
