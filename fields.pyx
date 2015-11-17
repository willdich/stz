cimport numpy as np

cdef class Field:
    """ Cython extension class for storing field values at grid points.
    We need to store both the current values, and the change for the next update. We must update
    the entire grid at once (and hence calculate all changes before adding them to the old values)
    so that values at timestep n+1 are only calculated using values at timestep n.
    """

    # Initialize the relevant values
    def __init__(self, np.float64_t u=0, np.float64_t v=0, np.float64_t w=0,        
                    np.float64_t s11=0, np.float64_t s12=0, np.float64_t s13=0,      
                    np.float64_t s22=0, np.float64_t s23=0, np.float64_t s33=0,      
                    np.float64_t chi=0, np.float64_t cu=0, np.float64_t cv=0, np.float64_t cw=0,         
                    np.float64_t cs11=0, np.float64_t cs12=0, np.float64_t cs13=0,
                    np.float64_t cs22=0, np.float64_t cs23=0, np.float64_t cs33=0, np.float64_t cchi=0):              

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
    cdef void update(self):
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
