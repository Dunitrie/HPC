import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


nx = 1000
dx = .1
nt = int(1e5)
dt = .001
D = 1

mu = nx*dx/2
sigma = 20*dx #arbitrary


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

ntotal = 1e3


nlocal = int(ntotal//size)
indices = np.arange(nlocal*rank, nlocal*(rank+1))


## ATTENTION: THIS WAS WRONG IN THE VIDEO
nx1 = nlocal*rank
nx2 = nlocal*(rank+1) # not included
if rank == size-1:
    nlocal = int(ntotal - nlocal*(size-1))
    nx2 = nx
x = np.arange(nx1-1, nx2+1)*dx
c = np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

for idx in range(nt):
    # Send to right and receive from left
    # Needs to be an array -> Slices of one element
    comm.Sendrecv(c[-2:-1], (rank+1)%size, recvbuf=c[0:1], source = (rank-1)%size)
    comm.Sendrecv(c[1:2], (rank-1)%size, recvbuf=c[-1:], source = (rank+1)%size)
    if idx % (nt // 5) == 0:
        x_full_range = np.arange(nx)*dx
        c_full_range = np.zeros(nx)
        comm.Gather(c[1:-1], c_full_range, root=0) 
        if rank == 0:
            plt.plot(x_full_range,c_full_range, '-', label=f"t={idx*dt}")
    # This is just the 2nd order finite differences approximation of d2c_dx2
    d2c_d2x = (np.roll(c,-1)-2*c + np.roll(c,1))/dx**2
    c += d2c_d2x * dt * D
            

#plt.plot(x,c)
if rank == 0:
    plt.title(f"Rank: {rank}, Integral: {np.trapz(c_full_range, x_full_range)}")
    plt.show()