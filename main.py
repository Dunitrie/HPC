import matplotlib.pyplot as plt
from mpi4py import MPI
from process_coordination import width_height, bool_boundaries, number_of_blocks
from streaming_functions import streaming

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

# Set parameters
omega = 1
wall_speed = 1

# Initialize local grid parameters:
# local size
nx, ny = width_height(rank, nx_total, ny_total, n_blocks)
borders = bool_boundaries(rank, nx, ny, n_blocks)

# Initialize weights and discrete direction vectors
weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [-1, 1], [-1, -1], [1, -1], [1, 1]])

# initialize grid
rho = np.ones((nx+2, ny+2))
v = np.zeros((2, nx+2, ny+2))

f = np.einsum("i,jk -> ijk", weights, np.ones((nx+2, ny+2)))


# loop over timesteps
for idx_time in ...:
    # 1. do the flow
    f, rho, v = streaming(f, rho, v, borders, omega)

    # 3. do the communications
        #else:
        #    comm.Sendrecv(...)
    
    # 4. do the plotting
    if idx_time % (n_timesteps // n_plots) == 0:
        # stack everything in rank 0
        comm.Gather(...)
        if rank == 0:
            # plot in rank 0 
            
    
        
