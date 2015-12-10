from fields cimport Field

cdef Field *look_up(Field *grid, int N_x, int N_y, int N_z,
             int xx, int yy, int zz) nogil:
    """
    Looks up the value at grid location (xx, yy, zz) in a 3-dimensional array grid of dimensions N_x, 
    N_y, and N_z assumed to be stored as a one dimensional array.
    We also assume that there is a ghost region of size one on either edge in all three directions.
    When adding advective terms, this will change. It may be worth modifying this function to allow for
    a ghost region of arbitrary width.
    """

    return &(grid[xx + (N_x + 2) * (yy + zz * (N_y + 2))])


cdef void set_val(Field *grid, int N_x, int N_y, int N_z,
             int xx, int yy, int zz, Field *new_val) nogil:
    """
    Assigns the position in grid with coordinates xx, yy, zz to the value stored at new_val.
    Grid dimensions are N_x, N_y, N_z, as above.
    """

    # Note that [0] is dereferencing in Cython because * is tuple unpacking in Python
    grid[xx + (N_x + 2) * (yy + zz * (N_y + 2))] = new_val[0]

