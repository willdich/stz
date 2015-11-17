import numpy as np

# Automatic Cython file compilation
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
        reload_support = True)

from sim import go

if __name__ == '__main__':
    
    # Inputs for the material lame parameters and density
    lambd = 1
    mu = 1
    rho = 1

    # Minimum and maximum coordinates in each dimension
    max_x = 5 
    min_x = 0
    max_y = 5
    min_y = 0
    max_z = 5
    min_z = 0

    # Number of grid points in each dimension
    N_x = 100 
    N_y = 100
    N_z = 100

    # Grid spacing in each dimension
    dx = (max_x - min_x) / N_x
    dy = (max_y - min_y) / N_y
    dz = (max_z - min_z) / N_z

    # Total simulation time and number of time points
    t_0 = 0
    t_f = 1
    N_t = 100 
    ts = np.linspace(t_0, t_f, N_t)
    dt = (t_f - t_0) / N_t

    # Number of parameters stored at each grid point 
    num_params = 20 

    # Instantiate the grid
    # +2 for ghost regions at the edge
#    grid = np.zeros( (N_x + 2, N_y + 2, N_z + 2, num_params), dtype=np.float64_t )

    # Run the simulation
    with nogil:
        go(N_x, N_y, N_z, N_t,
           np.float64(dx), np.float64(dy), np.float64(dz), np.float64(dt),
           np.float64(mu), np.float64(rho), np.float64(lambd),
           np.float64(t_0), np.float64(t_f), ts)
    #       grid)
