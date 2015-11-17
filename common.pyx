from fields cimport Field

cdef Field look_up(Field *grid, int N_x, int N_y, int N_z,
             int xx, int yy, int zz):
    """ Looks up the value at grid location (xx, yy, zz) in a 3-dimensional array grid of dimensions N_x, 
    N_y, and N_z assumed to be stored as a one dimensional array. """

    return grid[xx + yy * N_x + zz * N_x * N_y]

cdef void set_val(Field *grid, int N_x, int N_y, int N_z
             int xx, int yy, int zz, Field new_val):
    """ Assigns grid[xx, yy, zz] to new_val """

    grid[xx + yy * N_x + zz * N_x * N_y] = new_val
