import numpy as np

import matplotlib.pyplot as plt

omega = 0.3  # Impact parameter of relaxation

cs2 = 1/3
p_in = 3
p_out = 1

rho_out = p_out / cs2
rho_in = (p_out + p_out - p_in) / cs2

def recalculate_functions(f, rho, v, c):
    """
    Recalculate density and average viscosity at each point after probability density function has been updated.
    See Milestone 1.
    """
    rho = np.einsum("cij -> ij", f)  # density field
    v_noscale = np.einsum("ijk, il -> ljk", f, c)  # velocity field
    v = np.einsum("ijk, jk -> ijk", v_noscale, np.reciprocal(rho))  # divide by rho to get averange velocity

    return rho, v

def calc_equi(f, rho, v, c, weights, scenario):
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
        if(scenario == "poiseuille flow"):
            f_equi[channel, :, 0] = weights[channel] * rho_in * sum_bracket[:, -2]
            f_equi[channel, :, -1] = weights[channel] * rho_out * sum_bracket[:, 1]
    return f_equi

def border_control(f, borders, wall_speed, scenario):  
    """
    Handle global boundary conditions (bounce-back and moving wall) through using dry notes.
    See Milestone 4.
    """ 
    if scenario == "couette flow":
        # southern boundary
        f[7, -1] = np.roll(f[7, -1], shift=(0, 1))
        f[8, -1] = np.roll(f[8, -1], shift=(0, -1))

        f[2, -2, :] = f[4, -1, :]
        f[5, -2, :] = f[7, -1, :]
        f[6, -2, :] = f[8, -1, :]

        f[7, -1] = np.roll(f[7, -1], shift=(0, -1))
        f[8, -1] = np.roll(f[8, -1], shift=(0, 1))

        # northern boundary
        f[5, 0] = np.roll(f[5, 0], shift=(0, -1))
        f[6, 0] = np.roll(f[6, 0], shift=(0, 1))
        
        rho_N = np.zeros(f.shape[2])
        rho_N[:] = f[0, 1, :] + f[1, 1, :] + f[3, 1, :] +\
              2 * (f[2, 1, :] + f[6, 1, :] + f[5, 1, :])
        f[4, 1, :] = f[2, 0, :]
        f[7, 1, :] = f[5, 0, :] + 1/2 * (f[1, 1, :] - f[3, 1, :]) - 1/2 * rho_N[:] * wall_speed
        f[8, 1, :] = f[6, 0, :] + 1/2 * (f[3, 1, :] - f[1, 1, :]) + 1/2 * rho_N[:] * wall_speed

        f[5, 0] = np.roll(f[5, 0], shift=(0, 1))
        f[6, 0] = np.roll(f[6, 0], shift=(0, -1))
    
    if scenario == "poiseuille flow":
        # southern boundary
        f[7, -1] = np.roll(f[7, -1], shift=(0, 1))
        f[8, -1] = np.roll(f[8, -1], shift=(0, -1))

        f[2, -2, :] = f[4, -1, :]
        f[5, -2, :] = f[7, -1, :]
        f[6, -2, :] = f[8, -1, :]

        f[7, -1] = np.roll(f[7, -1], shift=(0, -1))
        f[8, -1] = np.roll(f[8, -1], shift=(0, 1))
        
        # northern boundary
        f[5, 0] = np.roll(f[5, 0], shift=(0, -1))
        f[6, 0] = np.roll(f[6, 0], shift=(0, 1))
        
        f[4, 1, :] = f[2, 0, :]
        f[7, 1, :] = f[5, 0, :]
        f[8, 1, :] = f[6, 0, :]

        f[5, 0] = np.roll(f[5, 0], shift=(0, 1))
        f[6, 0] = np.roll(f[6, 0], shift=(0, -1))

    if scenario == "sliding lid":
        # eastern boundary
        f[5, :, -1] = np.roll(f[5, :, -1], shift=(1, 0))
        f[8, :, -1] = np.roll(f[8, :, -1], shift=(-1, 0))

        f[3, 1:-1, -2] = f[1, 1:-1, -1]
        f[6, 1:-1, -2] = f[8, 1:-1, -1]
        f[7, 1:-1, -2] = f[5, 1:-1, -1]
        
        f[5, :, -1] = np.roll(f[5, :, -1], shift=(-1, 0))
        f[8, :, -1] = np.roll(f[8, :, -1], shift=(1, 0))

        # western boundary
        f[6, :, 0] = np.roll(f[6, :, 0], shift=(1, 0))
        f[7, :, 0] = np.roll(f[7, :, 0], shift=(-1, 0))

        f[1, 1:-1, 1] = f[3, 1:-1, 0]
        f[5, 1:-1, 1] = f[7, 1:-1, 0]
        f[8, 1:-1, 1] = f[6, 1:-1, 0]

        f[6, :, 0] = np.roll(f[6, :, 0], shift=(-1, 0))
        f[7, :, 0] = np.roll(f[7, :, 0], shift=(1, 0))

        # southern boundary
        f[7, -1] = np.roll(f[7, -1], shift=(0, 1))
        f[8, -1] = np.roll(f[8, -1], shift=(0, -1))

        f[2, -2, 1:-1] = f[4, -1, 1:-1]
        f[5, -2, 1:-1] = f[7, -1, 1:-1]
        f[6, -2, 1:-1] = f[8, -1, 1:-1]

        f[7, -1] = np.roll(f[7, -1], shift=(0, -1))
        f[8, -1] = np.roll(f[8, -1], shift=(0, 1))

        # northern boundary
        f[5, 0] = np.roll(f[5, 0], shift=(0, -1))
        f[6, 0] = np.roll(f[6, 0], shift=(0, 1))
        
        rho_N = np.zeros(f.shape[2])
        rho_N[:] = f[0, 1, :] + f[1, 1, :] + f[3, 1, :] +\
              2 * (f[2, 1, :] + f[6, 1, :] + f[5, 1, :])
        rho_N = np.ones(f.shape[2])
        f[4, 1, 1:-1] = f[2, 0, 1:-1]
        f[7, 1, 1:-1] = f[5, 0, 1:-1] + 1/2 * (f[1, 1, 1:-1] - f[3, 1, 1:-1]) - 1/2 * rho_N[1:-1] * wall_speed
        f[8, 1, 1:-1] = f[6, 0, 1:-1] + 1/2 * (f[3, 1, 1:-1] - f[1, 1, 1:-1]) + 1/2 * rho_N[1:-1] * wall_speed

        f[5, 0] = np.roll(f[5, 0], shift=(0, 1))
        f[6, 0] = np.roll(f[6, 0], shift=(0, -1))
    return f

def streaming(f, rho, v, c, weights, borders, wall_speed, scenario):
    """
    Pipeline of one complete streaming step.
    """
    f_equi = calc_equi(f, rho, v, c, weights, scenario)  # Equlibrium distrubution function

    f_old_1 = f[:, :, 1]
    f_old_N = f[:, :, -2]

    f += omega * (f_equi - f)  # Relaxation

    if(scenario == "poiseuille flow"):
        f[:, :, 0] = f_equi[:, :, 0] + (f_old_N - f_equi[:, :, -2])
        f[:, :, -1] = f_equi[:, :, -1] + (f_old_1 - f_equi[:, :, 1])

    print(f[:, 0:4, 0])
    print(f[:, 0:4, -1])
    
    for channel in range(9):  # Move channels wrt their direction
        f[channel] = np.roll(f[channel], shift=c[channel], axis=(0,1))

    f = border_control(f, borders, wall_speed, scenario)  # Handle (global) boundary conditions
    
    return f, rho, v