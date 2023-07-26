def number_to_factors(N: int) -> list:
    """Helper function: Returns all possible combinations of grid-split-possibilities
    """
    factors = []
    for x1 in range(1, N+1):
        if N % x1 == 0:
            factors.append((x1, N//x1))
    return factors

def number_of_blocks(dim, n_processes: int) -> list:
    """ Return the number of 
    
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
    return  combinations[idx]

def width_height(rank, nx, ny, n_blocks):
    """Helper-function: Returns the width and height of a block depending on the grid parameters
    """
    width = nx // n_blocks[0]
    height = ny // n_blocks[1]
    if rank % n_blocks[0] == (n_blocks[0]-1):
        width = nx - width*(n_blocks[0]-1) 
    if rank % n_blocks[1] == (n_blocks[1]-1):
        height = ny - height*(n_blocks[1]-1)
    return width, height


def bool_boundaries(rank, n_blocks):
    """
    Return, whether there is a wall in that direction depending on the rank of the process 
    Entries are 0:right, 1:up, 2:left, 3:down
    """
    bools = np.zeros(4)
    
    if rank % n_blocks[0] == 0:
        bools[2] = 1
    elif rank % n_blocks[0] == n_blocks[0]-1:
        bools[0] = 1
        
    if rank % n_blocks[1] == 0:
        bools[1] = 1
    elif rank % n_blocks[1] == n_blocks[1]-1:
        bools[3] = 1
        
    return bools
