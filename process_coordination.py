import numpy as np

def number_to_factors(N: int) -> list:
    """
    Helper function: Returns all possible combinations of grid-split-possibilities
    """
    factors = []
    for x1 in range(1, N+1):
        if N % x1 == 0:
            factors.append((x1, N//x1))
    return factors

def number_of_blocks(dim, n_processes: int) -> list:
    """
    Return the number and arrangement of blocks as [num blocks in x-dir, num blocks in y-dir].
    """
    factor = dim[1]/dim[0]
    
    combinations = number_to_factors(n_processes)
    
    
    score = 1e6
    idx = 0
    for idx1, combination in enumerate(combinations):
        
        new_score = np.linalg.norm(factor - combination[1]/combination[0])
        #print(score, new_score)
        if new_score < score:
            score = new_score
            idx = idx1
        res = (combinations[idx][1], combinations[idx][0])  # Fit it to the coordination system
    return res

def width_height(rank, nx, ny, n_blocks):
    """
    Helper-function: Returns the width and height of a block depending on the grid parameters
    """
    height = nx // n_blocks[0]
    width = ny // n_blocks[1]
    if rank % n_blocks[0] == (n_blocks[0]-1):
        height = nx - height*(n_blocks[0]-1) 
    if rank % n_blocks[1] == (n_blocks[1]-1):
        width = ny - width*(n_blocks[1]-1)
    return height, width


def bool_boundaries(rank, n_blocks):
    """
    Return, whether there is a wall in that direction depending on the rank of the process 
    Entries are 0:right, 1:up, 2:left, 3:down
    """
    bools = np.zeros(4)
    
    if rank % n_blocks[1] == 0:
        bools[2] = 1
    if rank % n_blocks[1] == n_blocks[1]-1:
        bools[0] = 1
        
    if rank // n_blocks[1] == 0:
        bools[1] = 1
    if rank // n_blocks[1] == n_blocks[0]-1:
        bools[3] = 1
        
    return bools