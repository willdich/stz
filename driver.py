import numpy as np
from os import environ

environ['LDFLAGS'] = '-Lm -lm'

# Automatic Cython file compilation
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
        reload_support = True)

from shear_wave_test import go

if __name__ == '__main__':
    
    # Inputs for the material lame parameters and density
    lambd = 1.
    mu = 1.
    rho = 1.

    # Minimum and maximum coordinates in each dimension
    max_x = 10. 
    min_x = 0.
    max_y = 10.
    min_y = 0.
    max_z = 10. 
    min_z = 0.

    # Number of grid points in each dimension
    N_x = 100 
    N_y = 100
    N_z = 100

    # Grid spacing in each dimension
    dx = (max_x - min_x) / N_x
    dy = (max_y - min_y) / N_y
    dz = (max_z - min_z) / N_z

    # Total simulation time and number of time points
    t_0 = 0.
    t_f = 1.
    N_t = 100 
    dt = (t_f - t_0) / N_t

    # Run the simulation
    go(N_x, N_y, N_z, N_t,
       np.float64(dx), np.float64(dy), np.float64(dz), np.float64(dt),
       np.float64(max_x) - np.float64(min_x),
       np.float64(max_y) - np.float64(min_y),
       np.float64(max_z) - np.float64(min_z),
       np.float64(mu), np.float64(rho), np.float64(lambd),
       np.float64(t_0), np.float64(t_f))
