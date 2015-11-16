from fields cimport Field
cimport numpy as np
from update_fields import *

cpdef go(int N_x, int N_y, int N_z, int N_t,                                            # Number of grid points in each dimension
         np.float64_t dx, np.float64_t dy, np.float64_t dz, np.float64_t dt,            # Time/spatial discretization
         np.float64_t mu, np.float64_t rho, np.float64_t lambd,                         # Material parameters
         np.float64_t t_0, np.float64_t t_f,                                            # Initial and final time
         np.float64_t [:, :, :, :] grid):

    """ Runs the simulation. Boundary conditions need to be put in EXPLICITLY in this file. 
    Grid is assumed to be of size N_x x N_y x N_z. The fourth dimension of the variable grid are all
    the parameters in the Field stored at grid[x, y, z].
    """

    cdef:
        int xx, yy, zz, tt

    # Plug in any relevant boundary conditions (manually)
    ### BC code here ###

    # Update the ghost regions to enforce periodicity
    set_up_ghost_regions(grid, N_x, N_y, N_z)

    # Run the simulation
    for tt in np.linspace(t_0, t_f, N_t):
        # First loop over the grid and calculate all changes
        for xx in range(N_x):
            for yy in range(N_y):
                for zz in range(N_z):
                    # Calculate the changes in stress
                    update_stresses(grid, xx, yy, zz
                                  dx, dy, dz,
                                  dt, mu, lamd, rho)

                    # Calculate the changes in velocities
                    update_velocities(grid, xx, yy, zz,
                                      dx, dy, dz,
                                      dt, rho)

        # Now we have to loop again, because we can't add in the changes until
        # we have calculated the change for EVERY grid point (otherwise some grid points
        # will calculate the new values at timestep n+1 using other grid point values at timestep n+1)
        for xx in range(N_x):
            for yy in range(N_y):
                for zz in range(N_z):
                    (<Field *> &grid[xx, yy, zz, 0]).update()

cdef set_up_ghost_regions(np.float64_t [:, :, :, :] grid,                           # Our grid
                         int N_x, int N_y, int N_z):        # Number of non-ghost points in each dimension

    """ Sets the ghost regions as necessary. In the serial implementation, this is just simple 
    periodic boundary conditions. This function will become more complex when moving to the parallel implementation
    with MPI-based communcation across processors.
    """

    cdef:
        int xx, yy, zz

    # Instantiate periodic boundary conditions
    # First handle the z periodicity
    # Note that these bounds need to be checked... should it be 1 to N_x + 1?
    for xx in range(1, N_x + 1):
        for yy in range(1, N_y + 1):
            # We have 2 + N_z points in the N_z direction
            # zz = 0 corresponds to the "ghost plane" at the base
            # zz = N_z + 1 corresponds to the "ghost plane" at the top
            # So we identify the ghost region at the base with the topmost "non-ghost" point
            # And the ghost region at the top with the bottommost "non-ghost" point
            <Field *> &grid[xx, yy, 0, 0] = <Field *> &grid[xx, yy, N_z, 0]
            <Field *> &grid[xx, yy, N_z + 1, 0] = <Field *> &grid[xx, yy, 1, 0]


    # Now do the same thing for the x periodicity
    for yy in range(1, N_y + 1):
        for zz in range(1, N_z + 1):
            # See comments in the above loop for explanation
            <Field *> &grid[0, yy, zz, 0] = <Field *> &grid[N_x, yy, zz, 0]
            <Field *> &grid[N_x + 1, yy, zz, 0] = <Field *> &grid[1, yy, zz, 0]

    # And finally the y periodicity
    for xx in range(1, N_x + 1):
        for zz in range(1, N_z + 1):
            # See comments in the above loop for explanation
            <Field *> &grid[xx, 0, zz, 0] = <Field *> &grid[xx, N_y, zz, 0]
            <Field *> &grid[xx, N_y + 1, zz, 0] = <Field *> &grid[xx, 1, zz, 0]
