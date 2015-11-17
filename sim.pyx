from fields cimport Field
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport fprintf, fopen, fclose, FILE
from common import *
from update_fields import *

cpdef go(int N_x, int N_y, int N_z, int N_t,                                            # Number of grid points in each dimension
         np.float64_t dx, np.float64_t dy, np.float64_t dz, np.float64_t dt,            # Time/spatial discretization
         np.float64_t mu, np.float64_t rho, np.float64_t lambd,                         # Material parameters
         np.float64_t t_0, np.float64_t t_f, np.float64_t [:] ts):#,                    # Initial and final time, list of time points
         #np.float64_t [:, :, :, :] grid):

    """ Runs the simulation. Boundary conditions need to be put in EXPLICITLY in this file. 
    Grid is assumed to be of size N_x x N_y x N_z. The fourth dimension of the variable grid are all
    the parameters in the Field stored at grid[x, y, z].
    """

    cdef:
        int xx, yy, zz
        float tt
        Field curr_grid_element
        Field *grid = <Field *> malloc(N_x * N_y * N_z * sizeof(Field))
        FILE *fp


    # Set the output file
    fp = fopen("output.txt", "w")

    # Plug in any relevant boundary conditions (manually)
    ### BC code here ###

    # Instantiate the grid
    for xx in range(N_x + 2):
        for yy in range(N_y + 2):
            for zz in range(N_z + 2):
                # grid[xx, yy, zz] = Field.__new__(Field)
                set_val(grid, N_x, N_y, N_z, xx, yy, zz, Field.__new__(Field))

    # Update the ghost regions to enforce periodicity
    set_up_ghost_regions(grid, N_x, N_y, N_z)

    # Run the simulation
    for tt in ts:
        # First loop over the grid and calculate all changes
        for xx in range(1, N_x + 1):
            for yy in range(1, N_y + 1):
                for zz in range(1, N_z + 1):
                    # Calculate the changes in stress
                    update_stresses(grid, xx, yy, zz,
                                    dx, dy, dz,
                                    dt, mu, lamd, rho)

                    # Calculate the changes in velocities
                    update_velocities(grid, xx, yy, zz,
                                      dx, dy, dz,
                                      dt, rho)

        # Now we have to loop again, because we can't add in the changes until
        # we have calculated the change for EVERY grid point (otherwise some grid points
        # will calculate the new values at timestep n+1 using other grid point values at timestep n+1)
        for xx in range(1, N_x + 1):
            for yy in range(1, N_y + 1):
                for zz in range(1, N_z + 1):
                    curr_grid_element = look_up(grid, N_x, N_y, N_z, xx, yy, zz)
                    (curr_grid_element).update()

                    # And print the data to the output file
                    fprintf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f %f",
                                            tt, xx, yy, zz, curr_grid_element.s11,
                                            curr_grid_element.s12, curr_grid_element.s13,
                                            curr_grid_element.s22, curr_grid_element.s23,
                                            curr_grid_element.s33, curr_grid_element.u,
                                            curr_grid_element.v, curr_grid_element.w)

        # And close the output file
        fclose(fp)

cdef set_up_ghost_regions(Field *grid,                                  # Our grid
                          int N_x, int N_y, int N_z):                   # Number of non-ghost points in each dimension

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
            
            # grid[xx, yy, 0] = grid[xx, yy, N_z]
            set_val(grid, N_x, N_y, N_z, xx, yy, 0, look_up(grid, N_x, N_y, N_z, xx, yy, N_z))

            # grid[xx, yy, N_z + 1] = grid[xx, yy, 1]
            set_val(grid, N_x, N_y, N_z, xx, yy, N_z + 1, look_up(grid, N_x, N_y, N_z, xx, yy, 1))


    # Now do the same thing for the x periodicity
    for yy in range(1, N_y + 1):
        for zz in range(1, N_z + 1):
            # See comments in the above loop for explanation

            # grid[0, yy, zz] = grid[N_x, yy, zz]
            set_val(grid, N_x, N_y, N_z, 0, yy, zz, look_up(grid, N_x, N_y, N_z, N_x, yy, zz))

            # grid[N_x + 1, yy, zz] = grid[1, yy, zz]
            set_val(grid, N_x, N_y, N_z, N_x + 1, yy, zz, look_up(grid, N_x, N_y, N_z, 1, yy, zz))

    # And finally the y periodicity
    for xx in range(1, N_x + 1):
        for zz in range(1, N_z + 1):
            # See comments in the above loop for explanation
            # grid[xx, 0, zz] = grid[xx, N_y, zz]
            set_val(grid, N_x, N_y, N_z, xx, 0, zz, look_up(grid, N_x, N_y, N_z, xx, N_y, zz))

            # grid[xx, N_y + 1, zz] = grid[xx, 1, zz]
            set_val(grid, N_x, N_y, N_z, xx, N_y + 1, zz, look_up(grid, N_x, N_y, N_z, xx, 1, zz))
