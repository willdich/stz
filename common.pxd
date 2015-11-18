from fields cimport Field

cdef:
    Field *look_up(Field *grid, int N_x, int N_y, int N_z, int xx, int yy, int zz) nogil
    void set_val(Field *grid, int N_x, int N_y, int N_z, int xx, int yy, int zz, Field *new_val) nogil
