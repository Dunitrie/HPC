import numpy as np

import matplotlib.pyplot as plt

omega = 0.3  # Impact parameter of relaxation
wall_speed = 0.5  # Speed of moving wall

def recalculate_functions(f, rho, v, c):
    """
    Recalculate density and average viscosity at each point after probability density function has been updated.
    See Milestone 1.
    """
    rho = np.einsum("cij -> ij", f)  # density field
    v_noscale = np.einsum("ijk, il -> ljk", f, c)  # velocity field
    v = np.einsum("ijk, jk -> ijk", v_noscale, np.reciprocal(rho))  # divide by rho to get averange velocity

    return rho, v

def calc_equi(f, rho, v, c, weights):
    """
    Calculate the equilibrium distribution function.
    See Milestone 2.
    """
    f_equi = np.zeros_like(f)
    v_abs = np.einsum("ijk -> jk", v)  # May be negative but will be squared anyway
    for channel in range(9):
        scal = np.einsum("i, ijk -> jk", c[channel], v)
        sum_bracket = np.ones_like(scal) + 3 * scal + 9/2 * scal * scal - 3/2 * v_abs * v_abs
        f_equi[channel, :, :] = weights[channel] * rho * sum_bracket
    return f_equi

def border_control(f, borders):  
    """
    Handle global boundary conditions (bounce-back and moving wall) through using dry notes.
    See Milestone 4.
    """  
    if borders[0]:  # True when theres a boundary to the east.
        # eastern boundary
        f[5, :, -1] = np.roll(f[5, :, -1], shift=(1, 0))
        f[8, :, -1] = np.roll(f[8, :, -1], shift=(-1, 0))

        f[3, :, -2] = f[1, :, -1]
        # set corner points seperately as we need information from the dry nodes before computing the border control
        f[6, 1:-1, -2] = f[8, 1:-1, -1] + 1/2 * (f[2, 1:-1, -2] - f[4, 1:-1, -2])
        f[7, 1:-1, -2] = f[5, 1:-1, -1] + 1/2 * (f[4, 1:-1, -2] - f[2, 1:-1, -2])

        # we do not have the correct values for f[1] and f[3] at this stage, so  compute manually corner grid point
        if borders[3]:
            f[6, -2, -2] = f[8, -2, -1] + 1/2 * (f[4, -1, -2] - f[4, -2, -2])
            f[7, -2, -2] = f[5, -2, -1] + 1/2 * (f[4, -2, -2] - f[4, -1, -2])
        if borders[1]:
            f[6, 1, -2] = f[8, 1, -1] + 1/2 * (f[2, 1, -2] - f[2, 0, -2])
            f[7, 1, -2] = f[5, 1, -1] + 1/2 * (f[2, 0, -2] - f[2, 1, -2])

        # Roll back edges because we need the corner note in the right place again.        
        f[5, :, -1] = np.roll(f[5, :, -1], shift=(-1, 0))
        f[8, :, -1] = np.roll(f[8, :, -1], shift=(1, 0))

    if borders[2]:
        # western boundary
        f[6, :, 0] = np.roll(f[6, :, 0], shift=(1, 0))
        f[7, :, 0] = np.roll(f[7, :, 0], shift=(-1, 0))

        f[1, :, 1] = f[3, :, 0]
        f[5, 1:-1, 1] = f[7, 1:-1, 0] + 1/2 * (f[2, 1:-1, 1] - f[4, 1:-1, 1])
        f[8, 1:-1, 1] = f[6, 1:-1, 0] + 1/2 * (f[4, 1:-1, 1] - f[2, 1:-1, 1])

        if borders[3]:
            f[5, -2, 1] = f[7, -2, 0] + 1/2 * (f[4, -1, 1] - f[4, -2, 1])
            f[8, -2, 1] = f[6, -2, 0] + 1/2 * (f[4, -2, 1] - f[4, -1, 1])
        if borders[1]:
            f[5, 1, 1] = f[7, 1, 0] + 1/2 * (f[2, 1, 1] - f[2, 0, 1])
            f[8, 1, 1] = f[6, 1, 0] + 1/2 * (f[2, 0, 1] - f[2, 1, 1])
        
        f[6, :, 0] = np.roll(f[6, :, 0], shift=(-1, 0))
        f[7, :, 0] = np.roll(f[7, :, 0], shift=(1, 0))

    if borders[3]:
        # southern boundary
        f[7, -1] = np.roll(f[7, -1], shift=(0, 1))
        f[8, -1] = np.roll(f[8, -1], shift=(0, -1))

        f[2, -2, :] = f[4, -1, :]
        f[5, -2, 1:-1] = f[7, -1, 1:-1] + 1/2 * (f[1, -2, 1:-1] - f[3, -2, 1:-1])
        f[6, -2, 1:-1] = f[8, -1, 1:-1] + 1/2 * (f[3, -2, 1:-1] - f[1, -2, 1:-1])

        if borders[0]:
            f[5, -2, -2] = f[7, -1, -2] + 1/2 * (f[1, -2, -2] - f[1, -2, -1])
            f[6, -2, -2] = f[8, -1, -2] + 1/2 * (f[1, -2, -1] - f[1, -2, -2])
        if borders[2]:
            f[6, -2, 1] = f[8, -1, 1] + 1/2 * (f[3, -2, 0] - f[3, -2, 1])
            f[5, -2, 1] = f[7, -1, 1] + 1/2 * (f[3, -2, 1] - f[3, -2, 0])

        f[7, -1] = np.roll(f[7, -1], shift=(0, -1))
        f[8, -1] = np.roll(f[8, -1], shift=(0, 1))

    if borders[1]:
        # northern boundary
        f[5, 0] = np.roll(f[5, 0], shift=(0, -1))
        f[6, 0] = np.roll(f[6, 0], shift=(0, 1))
        
        rho_N = np.zeros(f.shape[2])
        rho_N[:] = f[0, 1, :] + f[1, 1, :] + f[3, 1, :] +\
              2 * (f[2, 1, :] + f[6, 1, :] + f[5, 1, :])
        f[4, 1, :] = f[2, 0, :]
        f[7, 1, 1:-1] = f[5, 0, 1:-1] + 1/2 * (f[1, 1, 1:-1] - f[3, 1, 1:-1]) - 1/2 * rho_N[1:-1] * wall_speed
        f[8, 1, 1:-1] = f[6, 0, 1:-1] + 1/2 * (f[3, 1, 1:-1] - f[1, 1, 1:-1]) + 1/2 * rho_N[1:-1] * wall_speed

        if borders[0]:
            f[7, 1, -2] = f[5, 0 , -2] + 1/2 * (f[1, 1, -2] - f[1, 1, -1]) - 1/2 * rho_N[-2] * wall_speed
            f[8, 1, -2] = f[6, 0, -2] + 1/2 * (f[1, 1, -1] - f[1, 1, -2]) + 1/2 * rho_N[-2] * wall_speed

        if borders[2]:
            f[7, 1, 1] = f[5, 0, 1] + 1/2 * (f[3, 1, 0] - f[3, 1, 1]) - 1/2 * rho_N[1] * wall_speed
            f[8, 1, 1] = f[6, 0, 1] + 1/2 * (f[3, 1, 1] - f[3, 1, 0]) + 1/2 * rho_N[1] * wall_speed

        f[5, 0] = np.roll(f[5, 0], shift=(0, 1))
        f[6, 0] = np.roll(f[6, 0], shift=(0, -1))

    return f

def streaming(f, rho, v, c, weights, borders):
    """
    Pipeline of one complete streaming step.
    """
    f_equi = calc_equi(f, rho, v, c, weights)  # Equlibrium distrubution function

    f += omega * (f_equi - f)  # Relaxation   

    for channel in range(9):  # Move channels wrt their direction
        f[channel] = np.roll(f[channel], shift=c[channel], axis=(0,1))

    f = border_control(f, borders)  # Handle (global) boundary conditions

    return f, rho, v