from fields cimport Field, update 
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf, fprintf, fopen, fclose, FILE
cimport libc.math
from common cimport *
from update_fields cimport *

cpdef int go(int N_x, int N_y, int N_z, int N_t,                                       # Number of grid points in each dimension
         np.float64_t L_x, np.float64_t L_y, np.float64_t L_z,                          # Grid size in each dimension
         np.float64_t dx, np.float64_t dy, np.float64_t dz, np.float64_t dt,            # Time/spatial discretization
         np.float64_t mu, np.float64_t rho, np.float64_t lambd,                         # Material parameters
         np.float64_t t_0, np.float64_t t_f,                                            # Initial and final time, list of time points
         char *outfile) nogil:                                                            # Name of the output file

    """ Runs the simulation. Boundary conditions need to be put in EXPLICITLY in this file. 
    Grid is assumed to be of size N_x x N_y x N_z. The fourth dimension of the variable grid are all
    the parameters in the Field stored at grid[x, y, z].
    """

    cdef:
        int xx, yy, zz, t_ind                                                               # Iterator variables
        float tt                                                                            # Value of the current time
        Field *curr_grid_element                                                            # Current field
        Field *grid  = <Field *> malloc((N_x + 2) * (N_y + 2) * (N_z + 2) * sizeof(Field))  # Physical grid
        FILE *fp                                                                            # Output file
        np.float64_t [20] initial_field_values                                              # Initial values for the grid
        np.float64_t curr_sig_shear, curr_v_shear                                           # Debug variables

    # Initialize the values that every Field will start with.
    for xx in range(20):
        initial_field_values[xx] = 0

    # Set the output file
    fp = fopen(outfile, "w")

    # Fill the grid with zeros to start with
    for xx in range(N_x + 2):
        for yy in range(N_y + 2):
            for zz in range(N_z + 2):
                set_val(grid, N_x, N_y, N_z, xx, yy, zz, <Field *> initial_field_values)


    # Plug in any relevant boundary conditions (manually)
    set_boundary_conditions(grid, N_x, N_y, N_z, dx,
                            L_x, L_y, L_z,
                            mu, rho)

    # Run the simulation for N_t timesteps
    for t_ind in range(N_t):

        # Update the ghost regions to enforce periodicity
        set_up_ghost_regions(grid, N_x, N_y, N_z)

        # Current time
        tt = t_0 + dt * t_ind

        if (t_ind > 0):
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
                    # Note that the first update does nothing at the moment - this should be fixed later
                    curr_grid_element = look_up(grid, N_x, N_y, N_z, xx, yy, zz)
                    update(curr_grid_element) 

                    # Only print every five timesteps
                    # Because the shear wave solution is not y- or z-dependent, we print for all x values
                    # but only one specific y/z value (since all the y & z values are identical).
                    if ((yy == 1) and (zz == 1) and (t_ind % 5 == 0)):

                        # Calculate the value of the shear waves
                        # We use xx - 1 because xx = 1 corresponds to x = 0: xx=0 is the ghost region!
                        curr_sig_shear = shear_wave_sig((xx - 1) * dx, L_x, tt, mu, rho)
                        curr_v_shear = shear_wave_v((xx - 1) * dx, L_x, tt, mu, rho)

                        # And print the data to the output file
                        fprintf(fp, "%6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f\n",
                                                # We again use xx-1, yy-1, and zz-1 by the above logic
                                                tt, (xx-1)*dx, (yy-1)*dy, (zz-1)*dz,
                                                curr_grid_element.v, curr_grid_element.s12,
                                                curr_v_shear, curr_sig_shear,
                                                libc.math.pow(libc.math.fabs(curr_grid_element.v - curr_v_shear), 2),
                                                libc.math.pow(libc.math.fabs(curr_grid_element.s12 - curr_sig_shear), 2))

    # And close the output file
    fclose(fp)

    free(grid)

    
cdef void set_up_ghost_regions(Field *grid,                                  # Our grid
                          int N_x, int N_y, int N_z) nogil:                  # Number of non-ghost points in each dimension

    """
    Sets the ghost regions as necessary, using the MPI Cartesian communicator to pass edge values to the ghost regions of
    adjacent processes and receive corresponding values in return.
    """

    cdef:
        int xx, yy, zz

    # Instantiate periodic boundary conditions
    # First handle the z periodicity
    for xx in range(1, N_x + 1):
        for yy in range(1, N_y + 1):
            # We have 2 + N_z points in the N_z direction
            # zz = 0 corresponds to the "ghost plane" at the base
            # zz = N_z + 1 corresponds to the "ghost plane" at the top
            # So we identify the ghost region at the base with the topmost "non-ghost" point
            # And the ghost region at the top with the bottommost "non-ghost" point
            
            # grid[xx, yy, 0] = grid[xx, yy, N_z]
            set_val(grid, N_x, N_y, N_z, xx, yy, 0,       look_up(grid, N_x, N_y, N_z, xx, yy, N_z))

            # grid[xx, yy, N_z + 1] = grid[xx, yy, 1]
            set_val(grid, N_x, N_y, N_z, xx, yy, N_z + 1, look_up(grid, N_x, N_y, N_z, xx, yy, 1))


    # Now do the same thing for the x periodicity
    for yy in range(1, N_y + 1):
        for zz in range(1, N_z + 1):
            # See comments in the above loop for explanation

            # grid[0, yy, zz] = grid[N_x, yy, zz]
            set_val(grid, N_x, N_y, N_z, 0,       yy, zz, look_up(grid, N_x, N_y, N_z, N_x, yy, zz))

            # grid[N_x + 1, yy, zz] = grid[1, yy, zz]
            set_val(grid, N_x, N_y, N_z, N_x + 1, yy, zz, look_up(grid, N_x, N_y, N_z, 1,   yy, zz))

    # And finally the y periodicity
    for xx in range(1, N_x + 1):
        for zz in range(1, N_z + 1):
            # See comments in the above loop for explanation
            # grid[xx, 0, zz] = grid[xx, N_y, zz]
            set_val(grid, N_x, N_y, N_z, xx, 0,       zz, look_up(grid, N_x, N_y, N_z, xx, N_y, zz))

            # grid[xx, N_y + 1, zz] = grid[xx, 1, zz]
            set_val(grid, N_x, N_y, N_z, xx, N_y + 1, zz, look_up(grid, N_x, N_y, N_z, xx, 1,   zz))

    ## Now we need to handle the corner regions
    set_val(grid, N_x, N_y, N_z, 0,         0,       0,             look_up(grid, N_x, N_y, N_z, N_x, N_y, N_z))
    set_val(grid, N_x, N_y, N_z, N_x + 1,   N_y + 1, N_z + 1,       look_up(grid, N_x, N_y, N_z, 1,   1,   1))

    set_val(grid, N_x, N_y, N_z, N_x + 1,   0,       0,             look_up(grid, N_x, N_y, N_z, 1,   N_y, N_z))
    set_val(grid, N_x, N_y, N_z, 0,         N_y + 1, N_z + 1,       look_up(grid, N_x, N_y, N_z, N_x, 1,   1))

    set_val(grid, N_x, N_y, N_z, 0,         N_y + 1, 0,             look_up(grid, N_x, N_y, N_z, N_x, 1,   N_z))
    set_val(grid, N_x, N_y, N_z, N_x + 1,   0,       N_z + 1,       look_up(grid, N_x, N_y, N_z, 1,   N_y, 1))

    set_val(grid, N_x, N_y, N_z, 0,         0,       N_z + 1,       look_up(grid, N_x, N_y, N_z, N_x, N_y, 1))
    set_val(grid, N_x, N_y, N_z, N_x + 1,   N_y + 1, 0,             look_up(grid, N_x, N_y, N_z, 1,   1,   N_z))

    # And the "corner lines"
    for yy in range(1, N_y + 1):
        set_val(grid, N_x, N_y, N_z, 0,       yy, N_z + 1,          look_up(grid, N_x, N_y, N_z, N_x, yy, 1))
        set_val(grid, N_x, N_y, N_z, 0,       yy, 0,                look_up(grid, N_x, N_y, N_z, N_x, yy, N_z))
        set_val(grid, N_x, N_y, N_z, N_x + 1, yy, N_z + 1,          look_up(grid, N_x, N_y, N_z, 1,   yy, 1))
        set_val(grid, N_x, N_y, N_z, N_x + 1, yy, 0,                look_up(grid, N_x, N_y, N_z, 1,   yy, N_z))

    for xx in range(1, N_x + 1):
        set_val(grid, N_x, N_y, N_z, xx,  0,       N_z + 1,         look_up(grid, N_x, N_y, N_z, xx,  N_y,  1))
        set_val(grid, N_x, N_y, N_z, xx,  0,       0,               look_up(grid, N_x, N_y, N_z, xx,  N_y,  N_z))
        set_val(grid, N_x, N_y, N_z, xx,  N_y + 1, N_z + 1,         look_up(grid, N_x, N_y, N_z, xx,  1,    1))
        set_val(grid, N_x, N_y, N_z, xx,  N_y + 1, 0,               look_up(grid, N_x, N_y, N_z, xx,  1,    N_z))

    for zz in range(1, N_z + 1):
        set_val(grid, N_x, N_y, N_z, N_x + 1, N_y + 1, zz,          look_up(grid, N_x, N_y, N_z, 1,   1,   zz))
        set_val(grid, N_x, N_y, N_z, 0,       0,       zz,          look_up(grid, N_x, N_y, N_z, N_x, N_y, zz))
        set_val(grid, N_x, N_y, N_z, N_x + 1, 0,       zz,          look_up(grid, N_x, N_y, N_z, 1,   N_y, zz))
        set_val(grid, N_x, N_y, N_z, 0,       N_y + 1, zz,          look_up(grid, N_x, N_y, N_z, N_x, 1,   zz))

cdef void set_boundary_conditions(Field *grid,                                                  # Grid
                                  int N_x, int N_y, int N_z,                                    # Number of grid points
                                  np.float64_t dx,                                              # Grid spacing
                                  np.float64_t L_x, np.float64_t L_y, np.float64_t L_z,         # Grid dimensions
                                  np.float64_t mu, np.float64_t rho) nogil:                     # Material density and mu

    """
    Instantiates shear wave boundary/initial conditions.
    """
    cdef:
        int xx, yy, zz                                        # Loop indices
        Field *curr_field

    # We set each position in the grid (not counting the ghost regions at the edges) to the analytical solution for
    # shear wave initial conditions.
    for xx in range(1, N_x + 1):
        for yy in range(1, N_y + 1):
            for zz in range(1, N_z + 1):
                curr_field = look_up(grid, N_x, N_y, N_z, xx, yy, zz)
                curr_field.v = shear_wave_v((xx - 1) * dx, L_x, 0, mu, rho)
                curr_field.s12 = shear_wave_sig((xx - 1) * dx, L_x, 0, mu, rho)


cdef np.float64_t shear_wave_v(np.float64_t xx, np.float64_t L_x, np.float64_t tt,
                               np.float64_t mu, np.float64_t rho) nogil:

    """
    Returns the value of a velocity shear wave located in plane x=xx at time tt.
    """

    cdef np.float64_t pi = 3.14159265359
    cdef np.float64_t c_s = libc.math.sqrt(mu / rho)

    return libc.math.sin(2 * pi / (L_x) * (xx - c_s * tt))


cdef np.float64_t shear_wave_sig(np.float64_t xx, np.float64_t L_x, np.float64_t tt,
                                 np.float64_t mu, np.float64_t rho) nogil:

    """
    Returns the value of a shear-stress shear wave located in plane x=xx at time tt.
    """

    return -rho * libc.math.sqrt(mu / rho) * shear_wave_v(xx, L_x, tt, mu, rho)

