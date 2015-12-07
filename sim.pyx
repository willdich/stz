from fields cimport Field, update 
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf, sprintf, fprintf, fopen, fclose, FILE, putchar
cimport libc.math
from common cimport *
from update_fields cimport *
cimport mpi4py.mpi_c as mpi

cpdef void go(int N_x, int N_y, int N_z, int N_t,                                       # Number of grid points in each dimension
         np.float64_t L_x, np.float64_t L_y, np.float64_t L_z,                          # Grid size in each dimension
         np.float64_t dx, np.float64_t dy, np.float64_t dz, np.float64_t dt,            # Time/spatial discretization
         np.float64_t mu, np.float64_t rho, np.float64_t lambd,                         # Material parameters
         np.float64_t t_0, np.float64_t t_f,                                            # Initial and final time, list of time points
         char *outfile) nogil:                                                            # Name of the output file

    """ Runs the simulation. Boundary conditions need to be put in EXPLICITLY in this file. 
    Grid is assumed to be of size nn_x x nn_y x nn_z on each process (total size is irrelevant to the calculations).
    The fourth dimension of the variable grid are all the parameters in the Field stored at grid[x, y, z].
    """

    cdef:
        int xx, yy, zz, t_ind                                           # Iterator variables
        int nn_x, nn_y, nn_z                                            # Size of the local grid
        float tt                                                        # Value of the current time
        Field *curr_grid_element                                        # Current field
        Field *grid                                                     # Local grid
        FILE *fp                                                        # Output file (only printed to by master process)
        np.float64_t [20] initial_field_values                          # Initial values for the grid
        np.float64_t curr_sig_shear, curr_v_shear                       # Debug variables
        int [3] dims                                                    # Dimensions of the total grid in number of processors
        mpi.MPI_Comm comm                                               # Our communicator
        int [1] rank                                                    # Local process rank
        int [1] size                                                    # Size of the communicator - used to get total # of processors
        int [3] cc                                                      # Process coordinates in the Cartesian communicator
        int xoff, yoff, zoff                                            # Index offsets relative to the whole grid
        int numchars = 100                                              # for printing
        char *printbuf = <char *> malloc(numchars * sizeof(char))       # Printing
        char *allprint                                                  # Printing
        int i                                                           # index for printing
        int *displs                                                     # for printing
        int *recvcounts                                                 # for printing

    # Initialize MPI Cartesian communicator
    # Puts the total number of processors used to run the code into size
    mpi.MPI_Comm_size(mpi.MPI_COMM_WORLD, size)

    displs = <int *> malloc(size[0] * sizeof(int))
    recvcounts = <int *> malloc(size[0] * sizeof(int))
    for i in range(size[0]):
        displs[i] = i * size[0]
        recvcounts[i] = size[0]

    # Calculate the dimensions of the overall grid in terms of number of processors
    get_dims(size[0], dims)

    # Create our 3D Cartesian communicator with the correct dimensions, periodic in all dimensions
    # Store our communicator in the comm variable
    mpi.MPI_Cart_create(mpi.MPI_COMM_WORLD, 3, dims, [1, 1, 1], 1, &comm)

    # Find our rank in the communicator
    mpi.MPI_Comm_rank(comm, rank)

    # Gets the coordinates of the process within the Cartesian communicator
    # Store these in the length 3 array cc
    mpi.MPI_Cart_coords(comm, rank[0], 3, cc)

    
    # If we are the master process..
    if rank[0] == 0:
        # Allocate some space for the print buffer
        allprint = <char *> malloc(size[0] * numchars * sizeof(char))
        fp = fopen("input_params.dat", "w")
        fprintf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f, %d, %d, %d, %d", L_x, L_y, L_z, dx, dy, dz, dt, mu, rho, lambd, t_0, t_f, N_x, N_y, N_z, N_t) 
        fclose(fp)
    else:
        # Otherwise we don't need it
        allprint = NULL

    # Determine grid size per processor
    # Probably need to be more careful to ensure even division
    nn_x = N_x / dims[0]
    nn_y = N_y / dims[1]
    nn_z = N_z / dims[2]

    # Calculate the offsets
    xoff = cc[0] * nn_x
    yoff = cc[1] * nn_y
    zoff = cc[2] * nn_z

    # Allocate the local grid, now including space for the ghost regions
    # Unlike in the serial case, these must now be communicated between processors
    grid = <Field *> malloc((nn_x + 2) * (nn_y + 2) * (nn_z + 2) * sizeof(Field))

    # Initialize the values that every Field will start with.
    for xx in range(20):
        initial_field_values[xx] = 0

    # Set the output file
    fp = fopen(outfile, "w")

    # Instantiate the grid
    for xx in range(nn_x + 2):
        for yy in range(nn_y + 2):
            for zz in range(nn_z + 2):
                set_val(grid, nn_x, nn_y, nn_z, xx, yy, zz, <Field *> initial_field_values)


    # Plug in any relevant boundary conditions (manually)
    # Note that, now unlike serial, we need to take our local coordinates in the Cartesian communicator
    set_boundary_conditions(grid, nn_x, nn_y, nn_z,
                            cc[0], cc[1], cc[2],
                            dims[0], dims[1], dims[2], dx,
                            L_x, L_y, L_z,
                            mu, rho)
    #set_dummy_boundary_conditions(grid, nn_x, nn_y, nn_z,
    #                        cc[0], cc[1], cc[2],
    #                        dims[0], dims[1], dims[2], dx,
    #                        L_x, L_y, L_z,
    #                        mu, rho)

    # Run the simulation
    for t_ind in range(N_t):

        # Update the ghost regions to enforce periodicity
        set_up_ghost_regions(grid, comm, cc[0], cc[1], cc[2], nn_x, nn_y, nn_z)

        # Current time
        tt = t_0 + dt * t_ind

        if (t_ind > 0):
            # First loop over the grid and calculate all changes
            for xx in range(1, nn_x + 1):
                for yy in range(1, nn_y + 1):
                    for zz in range(1, nn_z + 1):
                        # Calculate the changes in stress
                        update_stresses(grid, 
                                        xx, yy, zz,
                                        nn_x, nn_y, nn_z,
                                        dx, dy, dz,
                                        dt, lambd, mu) 

                        # Calculate the changes in velocities
                        update_velocities(grid,
                                          xx, yy, zz,
                                          nn_x, nn_y, nn_z,
                                          dx, dy, dz,
                                          dt, rho)

        # Now we have to loop again, because we can't add in the changes until
        # we have calculated the change for EVERY grid point (otherwise some grid points
        # will calculate the new values at timestep n+1 using other grid point values at timestep n+1)
        for xx in range(1, nn_x + 1):
            for yy in range(1, nn_y + 1):
                for zz in range(1, nn_z + 1):
                    # Note that the first update does nothing at the moment - this should be fixed later
                    curr_grid_element = look_up(grid, nn_x, nn_y, nn_z, xx, yy, zz)
                    update(curr_grid_element) 

                    if ((yy == 1) and (zz == 1) and (t_ind % 5 == 0)):
                        #printf("%d %f %d %f %d %f \n", xx, xx * dx, yy, yy * dy, zz, zz * dz)

                        # Calculate the value of the shear waves
                        # We use xx - 1 because xx = 1 corresponds to x = 0: xx=0 is the ghost region!
                        curr_sig_shear = shear_wave_sig((xx - 1 + xoff) * dx, L_x, tt, mu, rho)
                        curr_v_shear = shear_wave_v((xx - 1 + xoff) * dx, L_x, tt, mu, rho)

                        # And send the data to the root proc to print to output file
                        sprintf(printbuf, "%6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f",
                                # We again use xx-1, yy-1, and zz-1 by the above logic
                                tt, (xx-1 + xoff)*dx, (yy-1 + yoff)*dy, (zz-1 + zoff)*dz,
                                curr_grid_element.v, curr_grid_element.s12,
                                curr_v_shear, curr_sig_shear,
                                libc.math.pow(libc.math.fabs(curr_grid_element.v - curr_v_shear), 2),
                                libc.math.pow(libc.math.fabs(curr_grid_element.s12 - curr_sig_shear), 2))

                        mpi.MPI_Gather(printbuf, numchars, mpi.MPI_CHAR,
                                       allprint, numchars, mpi.MPI_CHAR,
                                       0, comm)
                        if rank[0] == 0:
                            for i in range(size[0]):
                                fprintf(fp, "%s\n", allprint + (i * numchars))

    # And close the output file
    fclose(fp)

    free(printbuf)
    if rank[0] == 0:
        free(allprint)
    free(grid)
    
cdef void set_up_ghost_regions(Field *grid,                                   # Our grid for this process
                               mpi.MPI_Comm comm, int c_x, int c_y, int c_z,  # MPI cartesian communicator & our position in it
                               int nn_x, int nn_y, int nn_z) nogil:           # Number of non-ghost points in each dimension

    """ Sets the ghost regions as necessary. In the serial implementation, this is just simple 
    periodic boundary conditions. This function will become more complex when moving to the parallel implementation
    with MPI-based communcation across processors.
    """

    cdef:
        int xx, yy, zz
        int back, forward                                 # ranks of procs to send/recv ghost regions
        mpi.MPI_Datatype fieldtype
        mpi.MPI_Status status
        Field *buf_plane_z = <Field *> malloc((nn_x + 2) * (nn_y + 2) * sizeof(Field))

    mpi.MPI_Type_contiguous(20, mpi.MPI_FLOAT, &fieldtype)
    mpi.MPI_Type_commit(&fieldtype)

    # Instantiate periodic boundary conditions
    # First handle the z periodicity
    mpi.MPI_Cart_shift(comm, 2, 1, &back, &forward) 
    for xx in range(1, nn_x + 1):
        for yy in range(1, nn_y + 1):
            # We have 2 + nn_z points in the nn_z direction
            # zz = 0 corresponds to the "ghost plane" at the base
            # zz = nn_z + 1 corresponds to the "ghost plane" at the top
            # So we identify the ghost region at the base with the topmost "non-ghost" point
            # And the ghost region at the top with the bottommost "non-ghost" point
            
            # grid[xx, yy, 0] = grid[xx, yy, nn_z] of adjacent proc
            # copy values to send into buffer
            set_val(buf_plane_z, nn_x, 0, 0, xx, yy, 0,       look_up(grid, nn_x, nn_y, nn_z, xx, yy, nn_z))

    # send values & receive corresponding values from the other side
    mpi.MPI_Sendrecv_replace(buf_plane_z, (nn_x + 2) * (nn_y + 2), fieldtype, forward, 2, back, 2, comm, &status)

    for xx in range(1, nn_x + 1):
        for yy in range(1, nn_y + 1):
            # copy received values into grid
            set_val(grid, nn_x, nn_y, nn_z, xx, yy, 0,        look_up(buf_plane_z, nn_x, 0, 0, xx, yy, 0))

            # grid[xx, yy, nn_z + 1] = grid[xx, yy, 1]
            # copy values to send into buffer
            set_val(buf_plane_z, nn_x, 0, 0, xx, yy, 0,       look_up(grid, nn_x, nn_y, nn_z, xx, yy, 1))

    # send values & receive corresponding values from the other side
    mpi.MPI_Sendrecv_replace(buf_plane_z, (nn_x + 2) * (nn_y + 2), fieldtype, back, 2, forward, 2, comm, &status)

    for xx in range(1, nn_x + 1):
        for yy in range(1, nn_y + 1):
            # copy received values into grid
            set_val(grid, nn_x, nn_y, nn_z, xx, yy, nn_z + 1, look_up(buf_plane_z, nn_x, 0, 0, xx, yy, 0))

    free(buf_plane_z)

    cdef:
        Field *buf_plane_x = <Field *> malloc((nn_y + 2) * (nn_z + 2) * sizeof(Field))

    # Now do the same thing for the x periodicity
    mpi.MPI_Cart_shift(comm, 0, 1, &back, &forward)
    for yy in range(1, nn_y + 1):
        for zz in range(1, nn_z + 1):
            # See comments in the above loop for explanation

            # grid[0, yy, zz] = grid[nn_x, yy, zz]
            set_val(buf_plane_x, nn_y, 0, 0, yy, zz, 0,       look_up(grid, nn_x, nn_y, nn_z, nn_x, yy, zz))

    mpi.MPI_Sendrecv_replace(buf_plane_x, (nn_y + 2) * (nn_z + 2), fieldtype, forward, 0, back, 0, comm, &status)

    for yy in range(1, nn_y + 1):
        for zz in range(1, nn_z + 1):
            set_val(grid, nn_x, nn_y, nn_z, 0, yy, zz,        look_up(buf_plane_x, nn_y, 0, 0, yy, zz, 0))

            # grid[nn_x + 1, yy, zz] = grid[1, yy, zz]
            set_val(buf_plane_x, nn_y, 0, 0, yy, zz, 0,       look_up(grid, nn_x, nn_y, nn_z, 1, yy, zz))

    mpi.MPI_Sendrecv_replace(buf_plane_x, (nn_y + 2) * (nn_z + 2), fieldtype, back, 0, forward, 0, comm, &status)

    for yy in range(1, nn_y + 1):
        for zz in range(1, nn_z + 1):
            set_val(grid, nn_x, nn_y, nn_z, nn_x + 1, yy, zz, look_up(buf_plane_x, nn_y, 0, 0, yy, zz, 0))

    free(buf_plane_x)

    cdef:
        Field *buf_plane_y = <Field *> malloc((nn_x + 2) * (nn_z + 2) * sizeof(Field))

    # And finally the y periodicity
    mpi.MPI_Cart_shift(comm, 1, 1, &back, &forward)
    for xx in range(1, nn_x + 1):
        for zz in range(1, nn_z + 1):
            # See comments in the above loop for explanation
            # grid[xx, 0, zz] = grid[xx, nn_y, zz]
            set_val(buf_plane_y, nn_x, 0, 0, xx, zz, 0,       look_up(grid, nn_x, nn_y, nn_z, xx, nn_y, zz))

    mpi.MPI_Sendrecv_replace(buf_plane_y, (nn_x + 2) * (nn_z + 2), fieldtype, forward, 1, back, 1, comm, &status)

    for xx in range(1, nn_x + 1):
        for zz in range(1, nn_z + 1):
            set_val(grid, nn_x, nn_y, nn_z, xx, 0, zz,        look_up(buf_plane_y, nn_x, 0, 0, xx, zz, 0))

            # grid[xx, nn_y + 1, zz] = grid[xx, 1, zz]
            set_val(buf_plane_y, nn_x, 0, 0, xx, zz, 0,       look_up(grid, nn_x, nn_y, nn_z, xx, 1, zz))

    mpi.MPI_Sendrecv_replace(buf_plane_y, (nn_x + 2) * (nn_z + 2), fieldtype, back, 1, forward, 1, comm, &status)

    for xx in range(1, nn_x + 1):
        for zz in range(1, nn_z + 1):
            set_val(grid, nn_x, nn_y, nn_z, xx, nn_y + 1, zz, look_up(buf_plane_y, nn_x, 0, 0, xx, zz, 0))

    free(buf_plane_y)

    cdef:
        Field *buf_corner = <Field *> malloc(sizeof(Field))

    ## Now we need to handle the corner regions
    mpi.MPI_Cart_rank(comm, [c_x + 1, c_y + 1, c_z + 1], &forward)
    mpi.MPI_Cart_rank(comm, [c_x - 1, c_y - 1, c_z - 1], &back)
    set_val(buf_corner, 0, 0, 0, 0, 0, 0,                         look_up(grid, nn_x, nn_y, nn_z, nn_x, nn_y, nn_z))
    mpi.MPI_Sendrecv_replace(buf_corner, 1, fieldtype, forward, 3, back, 3, comm, &status)
    set_val(grid, nn_x, nn_y, nn_z, 0, 0, 0,                      buf_corner)
    set_val(buf_corner, 0, 0, 0, 0, 0, 0,                         look_up(grid, nn_x, nn_y, nn_z, 1, 1, 1))
    mpi.MPI_Sendrecv_replace(buf_corner, 1, fieldtype, back, 4, forward, 4, comm, &status)
    set_val(grid, nn_x, nn_y, nn_z, nn_x + 1, nn_y + 1, nn_z + 1, buf_corner)

    mpi.MPI_Cart_rank(comm, [c_x - 1, c_y + 1, c_z + 1], &forward)
    mpi.MPI_Cart_rank(comm, [c_x + 1, c_y - 1, c_z - 1], &back)
    set_val(buf_corner, 0, 0, 0, 0, 0, 0,                  look_up(grid, nn_x, nn_y, nn_z, 1, nn_y, nn_z))
    mpi.MPI_Sendrecv_replace(buf_corner, 1, fieldtype, forward, 5, back, 5, comm, &status)
    set_val(grid, nn_x, nn_y, nn_z, nn_x + 1, 0, 0,        buf_corner)
    set_val(buf_corner, 0, 0, 0, 0, 0, 0,                  look_up(grid, nn_x, nn_y, nn_z, nn_x, 1, 1))
    mpi.MPI_Sendrecv_replace(buf_corner, 1, fieldtype, back, 6, forward, 6, comm, &status)
    set_val(grid, nn_x, nn_y, nn_z, 0, nn_y + 1, nn_z + 1, buf_corner)

    mpi.MPI_Cart_rank(comm, [c_x + 1, c_y - 1, c_z + 1], &forward)
    mpi.MPI_Cart_rank(comm, [c_x - 1, c_y + 1, c_z - 1], &back)
    set_val(buf_corner, 0, 0, 0, 0, 0, 0,                  look_up(grid, nn_x, nn_y, nn_z, nn_x, 1, nn_z))
    mpi.MPI_Sendrecv_replace(buf_corner, 1, fieldtype, forward, 7, back, 7, comm, &status)
    set_val(grid, nn_x, nn_y, nn_z, 0, nn_y + 1, 0,        buf_corner)
    set_val(buf_corner, 0, 0, 0, 0, 0, 0,                  look_up(grid, nn_x, nn_y, nn_z, 1, nn_y, 1))
    mpi.MPI_Sendrecv_replace(buf_corner, 1, fieldtype, back, 8, forward, 8, comm, &status)
    set_val(grid, nn_x, nn_y, nn_z, nn_x + 1, 0, nn_z + 1, buf_corner)

    mpi.MPI_Cart_rank(comm, [c_x + 1, c_y + 1, c_z - 1], &forward)
    mpi.MPI_Cart_rank(comm, [c_x - 1, c_y - 1, c_z + 1], &back)
    set_val(buf_corner, 0, 0, 0, 0, 0, 0,                  look_up(grid, nn_x, nn_y, nn_z, nn_x, nn_y, 1))
    mpi.MPI_Sendrecv_replace(buf_corner, 1, fieldtype, forward, 9, back, 9, comm, &status)
    set_val(grid, nn_x, nn_y, nn_z, 0, 0, nn_z + 1,        buf_corner)
    set_val(buf_corner, 0, 0, 0, 0, 0, 0,                  look_up(grid, nn_x, nn_y, nn_z, 1, 1, nn_z))
    mpi.MPI_Sendrecv_replace(buf_corner, 1, fieldtype, back, 10, forward, 10, comm, &status)
    set_val(grid, nn_x, nn_y, nn_z, nn_x + 1, nn_y + 1, 0, buf_corner)

    free(buf_corner)

    cdef:
        Field *buf_line_xz = <Field *> malloc((nn_y + 2) * sizeof(Field))

    # And the "corner lines"
    # Get rank of proc in correct grid position
    mpi.MPI_Cart_rank(comm, [c_x + 1, c_y, c_z - 1], &forward)
    mpi.MPI_Cart_rank(comm, [c_x - 1, c_y, c_z + 1], &back)

    # Fill line buffer
    for yy in range(1, nn_y + 1):
        set_val(buf_line_xz, 0, 0, 0, yy, 0, 0,          look_up(grid, nn_x, nn_y, nn_z, nn_x, yy, 1))
    # Send & receive corner line
    mpi.MPI_Sendrecv_replace(buf_line_xz, nn_y, fieldtype, forward, 11, back, 11, comm, &status)
    # Copy values from buffer back into grid & repeat in opposite direction
    for yy in range(1, nn_y + 1):
        set_val(grid, nn_x, nn_y, nn_z, 0, yy, nn_z + 1, look_up(buf_line_xz, 0, 0, 0, yy, 0, 0))
        set_val(buf_line_xz, 0, 0, 0, yy, 0, 0,          look_up(grid, nn_x, nn_y, nn_z, 1, yy, nn_z))
    mpi.MPI_Sendrecv_replace(buf_line_xz, nn_y, fieldtype, back, 12, forward, 12, comm, &status)
    for yy in range(1, nn_y + 1):
        set_val(grid, nn_x, nn_y, nn_z, nn_x + 1, yy, 0,        look_up(buf_line_xz, 0, 0, 0, yy, 0, 0))

    mpi.MPI_Cart_rank(comm, [c_x + 1, c_y, c_z + 1], &forward)
    mpi.MPI_Cart_rank(comm, [c_x - 1, c_y, c_z - 1], &back)
    for yy in range(1, nn_y + 1):
        set_val(buf_line_xz, 0, 0, 0, yy, 0, 0,                 look_up(grid, nn_x, nn_y, nn_z, nn_x, yy, nn_z))
    mpi.MPI_Sendrecv_replace(buf_line_xz, nn_y, fieldtype, forward, 13, back, 13, comm, &status)
    for yy in range(1, nn_y + 1):
        set_val(grid, nn_x, nn_y, nn_z, 0, yy, 0,               look_up(buf_line_xz, 0, 0, 0, yy, 0, 0))
        set_val(buf_line_xz, 0, 0, 0, yy, 0, 0,                 look_up(grid, nn_x, nn_y, nn_z, 1, yy, 1))
    mpi.MPI_Sendrecv_replace(buf_line_xz, nn_y, fieldtype, back, 14, forward, 14, comm, &status)
    for yy in range(1, nn_y + 1):
        set_val(grid, nn_x, nn_y, nn_z, nn_x + 1, yy, nn_z + 1, look_up(buf_line_xz, 0, 0, 0, yy, 0, 0))

    free(buf_line_xz)


    cdef:
        Field *buf_line_yz = <Field *> malloc((nn_x + 2) * sizeof(Field))

    mpi.MPI_Cart_rank(comm, [c_x, c_y + 1, c_z - 1], &forward)
    mpi.MPI_Cart_rank(comm, [c_x, c_y - 1, c_z + 1], &back)
    for xx in range(1, nn_x + 1):
        set_val(buf_line_yz, 0, 0, 0, xx, 0, 0,          look_up(grid, nn_x, nn_y, nn_z, xx, nn_y, 1))
    mpi.MPI_Sendrecv_replace(buf_line_yz, nn_x, fieldtype, forward, 15, back, 15, comm, &status)
    for xx in range(1, nn_x + 1):
        set_val(grid, nn_x, nn_y, nn_z, xx, 0, nn_z + 1, look_up(buf_line_yz, 0, 0, 0, xx, 0, 0))
        set_val(buf_line_yz, 0, 0, 0, xx, 0, 0,          look_up(grid, nn_x, nn_y, nn_z, xx, 1, nn_z))
    mpi.MPI_Sendrecv_replace(buf_line_yz, nn_x, fieldtype, back, 16, forward, 16, comm, &status)
    for xx in range(1, nn_x + 1):
        set_val(grid, nn_x, nn_y, nn_z, xx, nn_y + 1, 0, look_up(buf_line_yz, 0, 0, 0, xx, 0, 0))

    mpi.MPI_Cart_rank(comm, [c_x, c_y - 1, c_z - 1], &forward)
    mpi.MPI_Cart_rank(comm, [c_x, c_y + 1, c_z + 1], &back)
    for xx in range(1, nn_x + 1):
        set_val(buf_line_yz, 0, 0, 0, xx, 0, 0,                 look_up(grid, nn_x, nn_y, nn_z, xx, 1, 1))
    mpi.MPI_Sendrecv_replace(buf_line_yz, nn_x, fieldtype, forward, 17, back, 17, comm, &status)
    for xx in range(1, nn_x + 1):
        set_val(grid, nn_x, nn_y, nn_z, xx, nn_y + 1, nn_z + 1, look_up(buf_line_yz, 0, 0, 0, xx, 0, 0))
        set_val(buf_line_yz, 0, 0, 0, xx, 0, 0,                 look_up(grid, nn_x, nn_y, nn_z, xx, nn_y, nn_z))
    mpi.MPI_Sendrecv_replace(buf_line_yz, nn_x, fieldtype, back, 18, forward, 18, comm, &status)
    for xx in range(1, nn_x + 1):
        set_val(grid, nn_x, nn_y, nn_z, xx, 0, 0,               look_up(buf_line_yz, 0, 0, 0, xx, 0, 0))

    free(buf_line_yz)


    cdef:
        Field *buf_line_xy = <Field *> malloc((nn_z + 2) * sizeof(Field))

    mpi.MPI_Cart_rank(comm, [c_x + 1, c_y + 1, c_z], &forward)
    mpi.MPI_Cart_rank(comm, [c_x - 1, c_y - 1, c_z], &back)
    for zz in range(1, nn_z + 1):
        set_val(buf_line_xy, 0, 0, 0, zz, 0, 0,                 look_up(grid, nn_x, nn_y, nn_z, nn_x, nn_y, zz))
    mpi.MPI_Sendrecv_replace(buf_line_xy, nn_z, fieldtype, forward, 19, back, 19, comm, &status)
    for zz in range(1, nn_z + 1):
        set_val(grid, nn_x, nn_y, nn_z, 0, 0, zz, look_up(buf_line_xy, 0, 0, 0, zz, 0, 0))
        set_val(buf_line_xy, 0, 0, 0, zz, 0, 0,                 look_up(grid, nn_x, nn_y, nn_z, 1, 1, zz))
    mpi.MPI_Sendrecv_replace(buf_line_xy, nn_z, fieldtype, back, 20, forward, 20, comm, &status)
    for zz in range(1, nn_z + 1):
        set_val(grid, nn_x, nn_y, nn_z, nn_x + 1, nn_y + 1, zz, look_up(buf_line_xy, 0, 0, 0, zz, 0, 0))

    mpi.MPI_Cart_rank(comm, [c_x + 1, c_y - 1, c_z], &forward)
    mpi.MPI_Cart_rank(comm, [c_x - 1, c_y + 1, c_z], &back)
    for zz in range(1, nn_z + 1):
        set_val(buf_line_xy, 0, 0, 0, zz, 0, 0,          look_up(grid, nn_x, nn_y, nn_z, nn_x, 1, zz))
    mpi.MPI_Sendrecv_replace(buf_line_xy, nn_z, fieldtype, forward, 21, back, 21, comm, &status)
    for zz in range(1, nn_z + 1):
        set_val(grid, nn_x, nn_y, nn_z, 0, nn_y + 1, zz, look_up(buf_line_xy, 0, 0, 0, zz, 0, 0))
        set_val(buf_line_xy, 0, 0, 0, zz, 0, 0,          look_up(grid, nn_x, nn_y, nn_z, 1, nn_y, zz))
    mpi.MPI_Sendrecv_replace(buf_line_xy, nn_z, fieldtype, back, 22, forward, 22, comm, &status)
    for zz in range(1, nn_z + 1):
        set_val(grid, nn_x, nn_y, nn_z, nn_x + 1, 0, zz, look_up(buf_line_xy, 0, 0, 0, zz, 0, 0))

    free(buf_line_xy)

cdef void set_dummy_boundary_conditions(Field *grid,                                                  # Grid
                                  int nn_x, int nn_y, int nn_z,                                 # Number of grid points not in the ghost region
                                  int cx, int cy, int cz,                                       # Cartesian location of the current processor
                                  int npx, int npy, int npz,                                    # Number of processors in each dimension
                                  np.float64_t dx,                                              # Grid spacing
                                  np.float64_t L_x, np.float64_t L_y, np.float64_t L_z,         # Grid dimensions
                                  np.float64_t mu, np.float64_t rho) nogil:                     # Material density and mu

    """ Instantiates shear wave boundary/initial conditions """
    cdef:
        int xx, yy, zz                                        # Loop indices
        Field *curr_field
       
    # Just set one point on one edge to nonzero
    xx = 1
    yy = 1
    zz = 1
    curr_field = look_up(grid, nn_x, nn_y, nn_z, xx, yy, zz)
    curr_field.v = 0.1
    curr_field.s12 = 0.1

cdef void set_boundary_conditions(Field *grid,                                                  # Grid
                                  int nn_x, int nn_y, int nn_z,                                 # Number of grid points not in the ghost region
                                  int cx, int cy, int cz,                                       # Cartesian location of the current processor
                                  int npx, int npy, int npz,                                    # Number of processors in each dimension
                                  np.float64_t dx,                                              # Grid spacing
                                  np.float64_t L_x, np.float64_t L_y, np.float64_t L_z,         # Grid dimensions
                                  np.float64_t mu, np.float64_t rho) nogil:                     # Material density and mu

    """ Instantiates shear wave boundary/initial conditions """
    cdef:
        int xx, yy, zz                                        # Loop indices
        int phys_p_cx, phys_p_cy, phys_p_cz                   # "Physical" cartesian indices where the processor indexing starts at the bottom left corner
        int x_off, y_off, z_off                               # Offsets due to our processor location in the grid
        Field *curr_field
        
    # Calculate the offsets due to the processor's location
    # We assume that nn_x, nn_y, nn_z are identical for all processors
    # For shear wave boundary conditions, y_off and z_off are not needed - but they will be in general
    # See http://ppomorsk.sharcnet.ca/Lecture_2_b_topologies.pdf for a 2D Diagram
    # Indexing starts at the topleftmost corner with (0, 0, 0) - we have (0, 0, 0) at the bottomleftmost corner
    # Note that x stays the same, but y and z flip
    phys_p_cx = cx
    phys_p_cy = cy 
    phys_p_cz = cz

    # Now calculate the offset relative to the overall grid
    # This uses the physical cartesian index because it makes the most sense
    # For every processor in the x direction in the "physical division", we have an extra nn_x + 2 points counting ghost
    x_off = phys_p_cx * (nn_x)

    # And same goes for the y and z directions
    y_off = phys_p_cy * (nn_y)
    z_off = phys_p_cz * (nn_z)

    # We only set our local boundary conditions
    # Because this is analytical, we COULD set them in the ghost regions as well - this could be a good test
    # for ghost region communcation.
    for xx in range(1, nn_x + 1):
        for yy in range(1, nn_y + 1):
            for zz in range(1, nn_z + 1):
                curr_field = look_up(grid, nn_x, nn_y, nn_z, xx, yy, zz)
                curr_field.v = shear_wave_v((xx - 1 + x_off) * dx, L_x, 0, mu, rho)
                curr_field.s12 = shear_wave_sig((xx - 1 + x_off) * dx, L_x, 0, mu, rho)

cdef np.float64_t shear_wave_v(np.float64_t xx, np.float64_t L_x, np.float64_t tt,
                               np.float64_t mu, np.float64_t rho) nogil:

    """ Returns the value of a velocity shear wave located in plane x=xx at time tt """

    cdef np.float64_t pi = 3.14159265359
    cdef np.float64_t c_s = libc.math.sqrt(mu / rho)

    return libc.math.sin(2 * pi / (L_x) * (xx - c_s * tt))

cdef np.float64_t shear_wave_sig(np.float64_t xx, np.float64_t L_x, np.float64_t tt,
                                 np.float64_t mu, np.float64_t rho) nogil:

    """ Returns the value of a shear-stress shear wave located in plane x=xx at time tt """

    return -rho * libc.math.sqrt(mu / rho) * shear_wave_v(xx, L_x, tt, mu, rho)

cdef void get_dims(int size, int dims[]) nogil:
    '''
    Return integer dimensions in 3D closest to a cube. Assumes number of procs is a power of 2.
    '''

    cdef:
        int log2_size = <int>(libc.math.log(size) / libc.math.log(2))
        int side
        int pwr

    # If number of procs is cubic, make a cube
    if log2_size % 3 == 0:
        side = 2 ** (log2_size / 3)
        for i in range(3):
            dims[i] = side
    # otherwise, one side will be either one factor of two lower or higher than the other two sides
    elif log2_size % 3 == 1:
        pwr = log2_size / 3
        dims[0] = 2 ** (pwr + 1)
        dims[1] = 2 ** pwr
        dims[2] = 2 ** pwr
    else:
        pwr = log2_size / 3 + 1
        dims[0] = 2 ** pwr
        dims[1] = 2 ** pwr
        dims[2] = 2 ** (pwr - 1)


