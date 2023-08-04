import numpy as np

import matplotlib.pyplot as plt

omega = 1  # Impact parameter of relaxation
wall_speed = 0  # Speed of moving wall

nx_total = 20  # num of rows
ny_total = 16  # num of columns

def recalculate_functions(f, rho, v, c, rank, step):
    """
    Recalculate density and average velocity at each point after probability density function has been updated.
    See Milestone 1.
    """
    rho = np.einsum("cij -> ij", f)  # density field
    v_noscale = np.einsum("ijk, il -> ljk", f, c)  # velocity field

    v = np.einsum("ijk, jk -> ijk", v_noscale, np.reciprocal(rho))  # divide by rho to get average velocity

    return rho, v

def calc_equi(f, rho, v, c, weights):
    """
    Calculate the equilibrium distribution function.
    See Milestone 2.
    """
    f_equi = np.zeros_like(f[:, 1:nx_total+1, 1:ny_total+1])
    v_abs = np.einsum("ijk -> jk", v[:, 1:nx_total+1, 1:ny_total+1])  # May be negative but will be squared anyway
    for channel in range(1, 9):
        scal = np.einsum("i, ijk -> jk", c[channel], v[:, 1:nx_total+1, 1:ny_total+1])
        sum_bracket = np.ones_like(scal) + 3 * scal + 9/2 * scal * scal - 3/2 * v_abs * v_abs

        f_equi[channel, :, :] = weights[channel] * rho[1:nx_total+1, 1:ny_total+1] * sum_bracket
    print(v[0, 1, :])
    return f_equi

def border_control(f, borders):  
    """
    Handle global boundary conditions (bounce-back and moving wall) through using dry notes.
    See Milestone 4.
    """  
    if borders[1]:  # True when theres a boundary to the north.
        # northern boundary
        f[5, 0] = np.roll(f[5, 0], shift=-1)
        f[6, 0] = np.roll(f[6, 0], shift=1)
        
        rho_N = np.zeros(f.shape[2])
        rho_N[:] = f[0, 1, :] + f[1, 1, :] + f[3, 1, :] +\
            2 * (f[2, 1, :] + f[6, 1, :] + f[5, 1, :])
        f[4, 1, :] = f[2, 0, :]
        # set corner points seperately as we need information from the dry nodes before computing the border control
        f[7, 1, 2:-2] = f[5, 0, 2:-2] + 1/2 * (f[1, 1, 2:-2] - f[3, 1, 2:-2]) - 1/2 * rho_N[2:-2] * wall_speed
        # we do not have the correct values for f[1] and f[3] at this stage, so  compute manually corner grid point
        if borders[0]:
            f[7, 1, -2] = f[5, 0 , -2] - 1/2 * rho_N[-2] * wall_speed
            f[8, 1, -2] = f[6, 0, -2]

        f[8, 1, 2:-2] = f[6, 0, 2:-2] + 1/2 * (f[3, 1, 2:-2] - f[1, 1, 2:-2]) + 1/2 * rho_N[2:-2] * wall_speed
        if borders[2]:
            f[7, 1, 1] = f[5, 0, 1]
            f[8, 1, 1] = f[6, 0, 1] + 1/2 * rho_N[1] * wall_speed

    if borders[0]:
        # eastern boundary
        f[5, :, -1] = np.roll(f[5, :, -1], shift=(1, 0))
        f[8, :, -1] = np.roll(f[8, :, -1], shift=(-1, 0))

        f[3, :, -2] = f[1, :, -1]
        f[6, 2:-2, -2] = f[8, 2:-2, -1] + 1/2 * (f[2, 2:-2, -2] - f[4, 2:-2, -2])
        if borders[3]:
            # here only wrt to southern boundary, as we treated northern boundary above
            f[6, -2, -2] = f[8, -2, -1]
            f[7, -2, -2] = f[5, -2, -1]
        if borders[1]:
            f[6, 1, -2] = f[8, 1, -1]
        f[7, 2:-2, -2] = f[5, 2:-2, -1] + 1/2 * (f[4, 2:-2, -2] - f[2, 2:-2, -2])
        #f[7, 1, -2] = f[5, 1, -1]
        
        

    if borders[2]:
        # western boundary
        f[6, :, 0] = np.roll(f[6, :, 0], shift=(1, 0))
        f[7, :, 0] = np.roll(f[7, :, 0], shift=(-1, 0))

        f[1, :, 1] = f[3, :, 0]
        f[5, 2:-2, 1] = f[7, 2:-2, 0] + 1/2 * (f[2, 2:-2, 1] - f[4, 2:-2, 1])
        if borders[3]:
            f[5, -2, 1] = f[7, -2, 0]
            f[8, -2, 1] = f[6, -2, 0]
        if borders[1]:
            f[5, 1, 1] = f[7, 1, 0]
            # could be that its necessary to update f[8, 1, 1] once more, but this was done in borders[1]
            #f[8, 1, 1] = f[6, 1, 0]
        f[8, 2:-2, 1] = f[6, 2:-2, 0] + 1/2 * (f[4, 2:-2, 1] - f[2, 2:-2, 1])
        
        


    if borders[3]:
        # southern boundary
        f[7, -1] = np.roll(f[7, -1], shift=(0,1))
        f[8, -1] = np.roll(f[8, -1], shift=(0,-1))

        f[2, -2, :] = f[4, -1, :]
        f[5, -2, 2:-2] = f[7, -1, 2:-2] + 1/2 * (f[1, -2, 2:-2] - f[3, -2, 2:-2])
        
        if borders[0]:
            f[5, -2, -2] = f[7, -1, -2]
            #f[6, -2, -2] = f[8, -1, -2] done in borders[0]
        f[6, -2, 2:-2] = f[8, -1, 2:-2] + 1/2 * (f[3, -2, 2:-2] - f[1, -2, 2:-2])
        if borders[2]:
            f[6, -2, 1] = f[8, -1, 1]
            #f[5, -2, 1] = f[7, -1, 1] done in borders[2]

    # Set dry notes to 0.
    if borders[0]:
        f[:, :, -1] = 10
    if borders[1]:
        f[:, 0, :] = 10
    if borders[2]:
        f[:, :, 0] = 10
    if borders[3]:
        f[:, -1, :] = 10

    return f

def streaming(f, rho, v, c, weights, borders, rank, step):
    """
    Pipeline of one complete streaming step.
    """
    # if (step < 3) and rank < 2:
    #     print(f"step: {step}, start, rank: {rank}\n {np.round(f[5:9, :, :], 2)}\n")

    f_equi = calc_equi(f, rho, v, c, weights)  # Equlibrium distribution function
    f[:, 1:nx_total+1, 1:ny_total+1] += omega * (f_equi - f[:, 1:nx_total+1, 1:ny_total+1])  # Relaxation
    #print(f"step: {step}, start, rank: {rank}\n {np.round(f[1:5, :, :], 0)}\n")    

    # if (step < 3) and rank < 2:    
    #     print(f"step: {step}, relaxed, rank: {rank}\n {np.round(f[5:9, :, :], 2)}\n")
        #print(f"equi: {np.round(f_equi[1:5, :, :], 0)}")

    for channel in range(9):  # Move channels wrt their direction
        f[channel] = np.roll(f[channel], shift=c[channel], axis=(0,1))

    # if (step < 3) and rank < 2:
    #     print(f"step: {step}, rolled, rank: {rank}\n {np.round(f[5:9, :, :], 2)}\n")

    f = border_control(f, borders)  # Handle (global) boundary conditions

    

    # if (step < 3) and rank < 2:
    #     print(f"step: {step}, end, rank: {rank}\n {np.round(f[5:9, :, :], 2)}\n")

    rho, v = recalculate_functions(f, rho, v, c, rank, step)  # Update values

    return f, rho, v