from fields cimport Field, update 
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf, fprintf, fopen, fclose, FILE
cimport libc.math
from common cimport *
from update_fields cimport *

cpdef void go(int N_x, int N_y, int N_z, int N_t,                                       # Number of grid points in each dimension
         np.float64_t L_x, np.float64_t L_y, np.float64_t L_z,                          # Grid size in each dimension
         np.float64_t dx, np.float64_t dy, np.float64_t dz, np.float64_t dt,            # Time/spatial discretization
         np.float64_t mu, np.float64_t rho, np.float64_t lambd,                         # Material parameters
         np.float64_t t_0, np.float64_t t_f) nogil:                                     # Initial and final time, list of time points

    """ Runs the simulation. Boundary conditions need to be put in EXPLICITLY in this file. 
    Grid is assumed to be of size N_x x N_y x N_z. The fourth dimension of the variable grid are all
    the parameters in the Field stored at grid[x, y, z].
    """

    cdef:
        int xx, yy, zz, t_ind
        float tt
        Field *curr_grid_element
        Field *grid = <Field *> malloc((N_x + 2) * (N_y + 2) * (N_z + 2) * sizeof(Field))
        FILE *fp
        np.float64_t [20] initial_field_values
        np.float64_t curr_sig_shear, curr_v_shear
        np.float64_t pi = 3.14

    fp = fopen("input_params.dat", "w")
    fprintf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f, %d, %d, %d, %d", L_x, L_y, L_z, dx, dy, dz, dt, mu, rho, lambd, t_0, t_f, N_x, N_y, N_z, N_t) 
    fclose(fp)

    # Initialize the values that every Field will start with.
    for xx in range(20):
        initial_field_values[xx] = 0

    # Set the output file
    fp = fopen("output.txt", "w")

    # Instantiate the grid
    for xx in range(N_x + 2):
        for yy in range(N_y + 2):
            for zz in range(N_z + 2):
                set_val(grid, N_x, N_y, N_z, xx, yy, zz, <Field *> initial_field_values)

    # Update the ghost regions to enforce periodicity
    set_up_ghost_regions(grid, N_x, N_y, N_z)

    # Plug in any relevant boundary conditions (manually)
    set_boundary_conditions(grid, N_x, N_y, N_z,
                            L_x, L_y, L_z,
                            mu, rho)

    # Run the simulation
    for t_ind in range(N_t):
        tt = t_0 + dt * t_ind
        # First loop over the grid and calculate all changes
        for xx in range(1, N_x + 1):
            for yy in range(1, N_y + 1):
                for zz in range(1, N_z + 1):
                    # Calculate the changes in stress
                    update_stresses(grid, 
                                    xx, yy, zz,
                                    N_x, N_y, N_z,
                                    dx, dy, dz,
                                    dt, lambd, mu) 

                    # Calculate the changes in velocities
                    update_velocities(grid,
                                      xx, yy, zz,
                                      N_x, N_y, N_z,
                                      dx, dy, dz,
                                      dt, rho)

        # Now we have to loop again, because we can't add in the changes until
        # we have calculated the change for EVERY grid point (otherwise some grid points
        # will calculate the new values at timestep n+1 using other grid point values at timestep n+1)
        for xx in range(1, N_x + 1):
            for yy in range(1, N_y + 1):
                for zz in range(1, N_z + 1):
                    curr_grid_element = look_up(grid, N_x, N_y, N_z, xx, yy, zz)
                    update(curr_grid_element) 

                    if ((yy == 1) and (zz == 1)):
                        #printf("%d %f %d %f %d %f \n", xx, xx * dx, yy, yy * dy, zz, zz * dz)

                        # Calculate the value of the shear waves
                        curr_sig_shear = shear_wave_sig(xx * dx, L_x, tt, mu, rho)
                        curr_v_shear = shear_wave_v(xx * dx, L_x, tt, mu, rho)

                        # And print the data to the output file
                        fprintf(fp, "%f %f %f %f %f %f %f %f %f %f\n",
                                                tt, xx*dx, yy*dy, zz*dz,
                                                curr_grid_element.v, curr_grid_element.s12,
                                                curr_v_shear, curr_sig_shear,
                                                libc.math.pow(libc.math.fabs(curr_grid_element.v - curr_v_shear), 2),
                                                libc.math.pow(libc.math.fabs(curr_grid_element.s12 - curr_sig_shear), 2))

    # And close the output file
    fclose(fp)

    free(grid)
    
cdef void set_up_ghost_regions(Field *grid,                                  # Our grid
                          int N_x, int N_y, int N_z) nogil:             # Number of non-ghost points in each dimension

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

cdef void set_boundary_conditions(Field *grid,                                                  # Grid
                                  int N_x, int N_y, int N_z,                                    # Number of grid points
                                  np.float64_t L_x, np.float64_t L_y, np.float64_t L_z,         # Grid dimensions
                                  np.float64_t mu, np.float64_t rho) nogil:                     # Material density and mu

    """ Instantiates shear wave boundary/initial conditions """
    cdef:
        int xx, yy, zz                                        # Loop indices
        np.float64_t c_s = libc.math.sqrt(mu / rho)                     # Shear wave speed
        np.float64_t pi = 3.14159265359
        Field *curr_field

    for xx in range(1, N_x):
        for yy in range(1, N_y):
            for zz in range(1, N_z):
                curr_field = look_up(grid, N_x, N_y, N_z, xx, yy, zz)
                curr_field.s12 = - rho * c_s * libc.math.sin(xx / (2 * pi * L_x))
                curr_field.v = libc.math.sin(xx / (2 * pi * L_x))

cdef np.float64_t shear_wave_v(np.float64_t xx, np.float64_t L_x, np.float64_t tt,
                               np.float64_t mu, np.float64_t rho) nogil:

    """ Returns the value of a velocity shear wave located in plane x=xx at time tt """

    cdef np.float64_t pi = 3.14159265359
    cdef np.float64_t c_s = libc.math.sqrt(mu / rho)

    return libc.math.sin(1 / (2 * pi * L_x) * (xx - c_s * tt))

cdef np.float64_t shear_wave_sig(np.float64_t xx, np.float64_t L_x, np.float64_t tt,
                                 np.float64_t mu, np.float64_t rho) nogil:

    """ Returns the value of a shear-stress shear wave located in plane x=xx at time tt """

    return -rho * libc.math.sqrt(mu / rho) * shear_wave_v(xx, L_x, tt, mu, rho)
