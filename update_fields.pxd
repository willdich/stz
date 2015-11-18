cimport numpy as np
from common cimport *
from fields cimport Field

cdef void update_stresses(Field *grid,                                         # Grid of field values
                      int x, int y, int z,                                     # Location in the grid
                      int N_x, int N_y, int N_z,                               # Grid sizes (for lookup)
                      np.float64_t dx, np.float64_t dy, np.float64_t dz,       # Spatial discretization
                      np.float64_t dt,                                         # Time discretization
                      np.float64_t lam, np.float64_t mu) nogil                 # Material Parameters

cdef void update_velocities(Field *grid,                                       # Grid of field values
                      int x, int y, int z,                                     # Location in the grid
                      int N_x, int N_y, int N_z,                               # Grid sizes (for lookup)
                      np.float64_t dx, np.float64_t dy, np.float64_t dz,       # Spatial discretization
                      np.float64_t dt,                                         # Time discretization
                      np.float64_t rho) nogil                                  # Material density
