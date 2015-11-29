cimport numpy as np
from mpi4py cimport MPI

cdef struct Field:

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

cdef inline void update(Field *f) nogil:
    f.u += f.cu
    f.v += f.cv
    f.w += f.cw
    f.s11 += f.cs11
    f.s12 += f.cs12
    f.s13 += f.cs13
    f.s22 += f.cs22
    f.s23 += f.cs23
    f.s33 += f.cs33
    f.chi += f.chi

    f.cu = f.cv = f.cw = f.cs11 = f.cs12 = f.cs13 = f.cs22 = f.cs23 = f.cs33 = f.chi = 0
