cimport numpy as np

cdef class Field:
    """ Cython extension class for storing field values at grid points.
    We need to store both the current values, and the change for the next update. We must update
    the entire grid at once (and hence calculate all changes before adding them to the old values)
    so that values at timestep n+1 are only calculated using values at timestep n.
    """

    cdef np.float64_t u                   # x velocity
    cdef np.float64_t v                   # y velocity
    cdef np.float64_t w                   # z velocity
    cdef np.float64_t s11                 # Six components of the stress tensor
    cdef np.float64_t s12
    cdef np.float64_t s13
    cdef np.float64_t s22
    cdef np.float64_t s23
    cdef np.float64_t s33
    cdef np.float64_t chi                 # Effective temperature
    cdef np.float64_t cu                  # Change in the velocity components
    cdef np.float64_t cv
    cdef np.float64_t cw
    cdef np.float64_t cs11                # Change in the stress components
    cdef np.float64_t cs12
    cdef np.float64_t cs13
    cdef np.float64_t cs22
    cdef np.float64_t cs23
    cdef np.float64_t cs33
    cdef np.float64_t cchi                # Change in the effective temperature

    # Initialize the relevant values
    def __cinit__(self, np.float64_t u, np.float64_t v, np.float64_t w,        
                    np.float64_t s11, np.float64_t s12, np.float64_t s13,      
                    np.float64_t s22, np.float64_t s23, np.float64_t s33,      
                    np.float64_t chi, np.float64_t cu, np.float64_t cv, np.float64_t cw,         
                    np.float64_t cs11, np.float64_t cs12, np.float64_t cs13,
                    np.float64_t cs22, np.float64_t cs23, np.float64_t cs33, np.float64_t cchi):              

        self.u = u                  # x velocity
        self.v = v                  # y velocity
        self.w = w                  # z velocity
        self.s11 = s11              # Six components of the Cauchy stress tensor
        self.s12 = s12
        self.s13 = s13
        self.s22 = s22
        self.s23 = s23
        self.s33 = s33
        self.chi = chi              # Effective temperature
        self.cu = cu                # Change in the velocity components
        self.cv = cv
        self.cw = cw
        self.cs11 = cs11            # Change in the stress components
        self.cs12 = cs12
        self.cs13 = cs13
        self.cs22 = cs22
        self.cs23 = cs23
        self.cs33 = cs33
        self.cchi = cchi            # Change in the effective temperature

    # Update the values using the (already calculated) changes.
    cdef update(self):
        self.u += self.cu
        self.v += self.cv
        self.w += self.cw
        self.s11 += self.cs11
        self.s12 += self.cs12
        self.s13 += self.cs13
        self.s22 += self.cs22
        self.s23 += self.cs23
        self.s33 += self.cs33
        self.chi += self.chi
