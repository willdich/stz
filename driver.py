import numpy as np
from fields import Field
from sim import go
from update_fields import *

if name == '__main__':
    
    # Inputs for the material lame parameters and density
    lamd = 0
    mu = 0
    rho = 0

    # Minimum and maximum coordinates in each dimension
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0
    max_z = 0
    min_z = 0

    # Number of grid points in each dimension
    N_x = 0
    N_y = 0
    N_z = 0 

    # Grid spacing in each dimension
    dx = (max_x - min_x) / N_x
    dy = (max_y - min_y) / N_y
    dz = (max_z - min_z) / N_z

    # Total simulation time and number of time points
    t_0 = 0
    t_f = 100
    N_t = 0
    dt = (t_f - t_0) / N_t

    # Number of parameters stored at each grid point 
    num_params = 20 

    # Instantiate the grid
    # +2 for ghost regions at the edge
    grid = np.zeros( (N_x + 2, N_y + 2, N_z + 2, num_params), dtype=np.float64_t )

    # Run the simulation
    go(N_x, N_y, N_z, N_t,
       np.float64_t(dx), np.float64_t(dy), np.float64_t(dz), np.float64_t(dt),
       np.float64_t(mu), np.float64_t(rho), np.float64_t(lambd),
       grid)
