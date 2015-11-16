cimport numpy as np

cdef class Field:

    cdef:
        np.float64_t u                   # x velocity
        np.float64_t v                   # y velocity
        np.float64_t w                   # z velocity
        np.float64_t s11                 # Six components of the stress tensor
        np.float64_t s12
        np.float64_t s13
        np.float64_t s22
        np.float64_t s23
        np.float64_t s33
        np.float64_t chi                 # Effective temperature
        np.float64_t cu                  # Change in the velocity components
        np.float64_t cv
        np.float64_t cw
        np.float64_t cs11                # Change in the stress components
        np.float64_t cs12
        np.float64_t cs13
        np.float64_t cs22
        np.float64_t cs23
        np.float64_t cs33
        np.float64_t cchi                # Change in the effective temperature

    cdef update(self)
