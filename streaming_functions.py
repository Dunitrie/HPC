import numpy as np

omega = 1
wall_speed = 1

def recalculate_functions(f, rho, v, c):

    rho = np.einsum("ijk -> jk", f)  # density field
    v_noscale = np.einsum("ijk, il -> ljk", f, c)  # velocity field
    v = np.einsum("ijk, jk -> ijk", v_noscale, np.reciprocal(rho))  # divide by rho to get averange velocity

    return f, rho, v

def calc_equi(f, rho, v, c, weights):
    # Caluculate the equilibrium distribution function
    f_equi = np.zeros_like(f)
    v_abs = np.einsum("ijk -> jk", v)  # May be negative but will be squared anyway
    for channel in range(9):
        scal = np.einsum("i, ijk -> jk", c[channel], v)
        sum_bracket = np.ones_like(scal) + 3 * scal + 9/2 * scal * scal - 3/2 * v_abs * v_abs
        f_equi[channel, :, :] = weights[channel] * rho * sum_bracket
    return f_equi

def border_control(f, borders):    
    if borders[0]:
        # eastern boundary
        f[5, :, -1] = np.roll(f[5, :, -1], shift=(1, 0))
        f[8, :, -1] = np.roll(f[8, :, -1], shift=(-1, 0))

        f[3, :, -2] = f[1, :, -1]
        f[6, :, -2] = f[8, :, -1] + 1/2 * (f[2, :, -2] - f[4, :, -2])
        f[7, :, -2] = f[5, :, -1] + 1/2 * (f[4, :, -2] - f[2, :, -2])
    
    if borders[1]:
        # northern boundary
        f[5, 0] = np.roll(f[5, 0], shift=-1)
        f[6, 0] = np.roll(f[6, 0], shift=1)
        
        rho_N = np.zeros(f.shape[2])
        rho_N[:] = f[0, 1, :] + f[1, 1, :] + f[3, 1, :] +\
            2 * (f[2, 1, :] + f[6, 1, :] + f[5, 1, :])
        f[4, 1, :] = f[2, 0, :]
        f[7, 1, :] = f[5, 0, :] + 1/2 * (f[1, 1, :] - f[3, 1, :]) - 1/2 * rho_N * wall_speed
        f[8, 1, :] = f[6, 0, :] + 1/2 * (f[3, 1, :] - f[1, 1, :]) + 1/2 * rho_N * wall_speed

    if borders[2]:
        # western boundary
        f[6, :, 0] = np.roll(f[6, :, 0], shift=(1, 0))
        f[7, :, 0] = np.roll(f[7, :, 0], shift=(-1, 0))

        f[1, :, 1] = f[3, :, 0]
        f[5, :, 1] = f[7, :, 0] + 1/2 * (f[2, :, 1] - f[4, :, 1])
        f[8, :, 1] = f[6, :, 0] + 1/2 * (f[4, :, 1] - f[2, :, 1])

    if borders[3]:
        # southern boundary
        f[7, -1] = np.roll(f[7, -1], shift=1)
        f[8, -1] = np.roll(f[8, -1], shift=-1)

        f[2, -2, :] = f[4, -1, :]
        f[5, -2, :] = f[7, -1, :] + 1/2 * (f[1, -2, :] - f[3, -2, :])
        f[6, -2, :] = f[8, -1, :] + 1/2 * (f[3, -2, :] - f[1, -2, :])

    # necessary / correct???
    if borders[0]:
        f[:, :, -1] = 0
    if borders[1]:    
        f[:, 0, :] = 0  # f[:, 1, :]  # restore imaginary row
    if borders[2]:
        f[:, :, 0] = 0
    if borders[3]:
        f[:, -1, :] = 0  # f[:, -2, :]  # restore imaginary row

    return f

def streaming(f, rho, v, c, weights, borders):
    f_equi = calc_equi(f, rho, v, c, weights)

    f += omega * (f_equi - f)

    for channel in range(9):
        f[channel] = np.roll(f[channel], shift=c[channel], axis=(0,1))

    f = border_control(f, borders)
    
    f, rho, v = recalculate_functions(f, rho, v, c)

    return f, rho, v