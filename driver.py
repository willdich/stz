import numpy as np
from fields import Field
from update_fields import *

if name == '__main__':
    # Set up material parameters
    # Instantiate the grid?
    # Run the simulation

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

    # Instantiate the grid
    grid = np.zeros( (N_x, N_y, N_z) )

    # Now fill it up with the corresponding fields
    for xx in range(N_x):
        for yy in range(N_y):
            for zz in range(N_z):
                # Automatically defaults to 0 at all grid points
                grid[xx, jj, kk] = Field.__new__(Field)

    # Plug in the boundary conditions
    ### grid[boundary locations].(values) = b.c.'s

    # Run the simulation
    for tt in np.linspace(t_0, t_f, N_t):
        # First loop over the grid and calculate all changes
        for xx in range(N_x):
            for yy in range(N_y):
                for zz in range(N_z):
                    # Calculate the changes in stress
                    update_stress(grid, xx, yy, zz
                                  dx, dy, dz,
                                  dt, mu, lamd, rho)

                    # Calculate the changes in velocities
                    update_velocities(grid, xx, yy, zz,
                                      dx, dy, dz,
                                      dt, rho)

        # Now we have to loop again, because we can't add in the changes until
        # we have calculated the change for EVERY grid point (otherwise some grid points
        # will calculate the new values at timestep n+1 using other grid point values at timestep n+1
        for xx in range(N_x):
            for yy in range(N_y):
                for zz in range(N_z):
                    grid[xx, yy, zz].update()
