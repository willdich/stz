cimport numpy as np
from fields cimport Field

cdef update_stresses(Field [:, :, :] grid,                           # Grid of field values
                      int x, int y, int z,                                     # Location in the grid
                      np.float64_t dx, np.float64_t dy, np.float64_t dz,       # Spatial discretization
                      np.float64_t dt,                                         # Time discretization
                      np.float64_t lam, np.float64_t mu):                      # Material Parameters

    """ Calculates the updates for the stress tensor s_ij of the field value at grid[x, y, z].
    At the moment, the grid spacing is unnecessary because of the lack of advective terms.
    It has been kept in to be most general.
    """

    # Store our needed variables
    cdef:
        # The value at the current grid point: this is what we are updating
        Field curr_field_value

        # The trace of the local stress tensor
        np.float64_t sig_trace

    # Look up the field at the current grid point
    curr_field_value = grid[x, y, z]

    # Calculate the trace because it shows up in a few places
    sig_trace = curr_field_value.s11 + curr_field_value.s22 + curr_field_value.s33

    # Now calculate the corresponding changes in stresses
    # First the diagonal terms, which have a contribution from the trace of the stress tensor
    curr_field_value.cs11 = dt * (lam * sig_trace + 2 * mu * curr_field_value.s11)
    curr_field_value.cs22 = dt * (lam * sig_trace + 2 * mu * curr_field_value.s22)
    curr_field_value.cs33 = dt * (lam * sig_trace + 2 * mu * curr_field_value.s33)

    # And now calculate the updates for the off diagonal elements
    curr_field_value.cs12 = dt * 2 * mu * curr_field_value.cs12
    curr_field_value.cs13 = dt * 2 * mu * curr_field_value.cs13
    curr_field_value.cs23 = dt * 2 * mu * curr_field_value.cs23

cdef update_velocities(Field [:, :, :] grid,                                   # Grid of field values
                      int x, int y, int z,                                     # Location in the grid
                      np.float64_t dx, np.float64_t dy, np.float64_t dz,       # Spatial discretization
                      np.float64_t dt,                                         # Time discretization
                      np.float64_t rho):                                       # Material density

    """ Calculates the updates for the velocity components of the field value at grid[x, y, z].
    Currently serial. When parallelizing with MPI, we will need some communication at the boundaries here.
    We assume that boundary conditions are handled in the driver program. This means we do not need to worry
    about, e.g., x+1 going out of bounds. We have ghost regions to handle this case and we should only be calling
    this update on the internal regions.
    """

    # Store our our needed variables
    cdef:
        Field me                    # Field value at the currrent location (x, y, z)
        Field xp                    # Field value at (x+1, y, z) 
        Field xm                    # Field value to the left (x-1, y, z)
        Field yp                    # Field value at (x, y+1, z)
        Field ym                    # Field value at (x, y-1, z)
        Field zp                    # Field value at (x, y, z+1)
        Field zm                    # Field value at (x, y, z-1)
        Field xm_ym                 # Field value at (x-1, y-1, z)
        Field xm_zm                 # Field value at (x-1, y, z-1)
        Field ym_zm                 # Field value at (x, y-1, z-1)
        Field xm_ym_zm              # Field value at (x-1, y-1, z-1)

        np.float64_t d_s11_dx       # Derivatives of stress components
        np.float64_t d_s12_dy
        np.float64_t d_s13_dz
        np.float64_t d_s12_dx
        np.float64_t d_s22_dy
        np.float64_t d_s23_dz
        np.float64_t d_s13_dx
        np.float64_t d_s23_dy
        np.float64_t d_s33_dz

        np.float64_t rho_inv        # 1/rho
        np.float64_t dx_inv         # 1/dx
        np.float64_t dy_inv         # 1/dy
        np.float64_t dz_inv         # 1/dz
        
    # Get the inverse values for simplicity
    rho_inv = 1. / rho
    dx_inv = 1. / dx
    dy_inv = 1. / dy
    dz_inv = 1. / dz

    # First look up the corresponding grid values 
    # We cast the all the floats at the corresponding grid points to a Field pointer (telling Cython to
    # interpret the following memory as a field) and then dereference it to get the Field back
    me = grid[x, y, z]
    xp = grid[x + 1, y, z]
    xm = grid[x - 1, y, z]
    yp = grid[x, y + 1, z]
    ym = grid[x, y - 1, z]
    zp = grid[x, y, z + 1]
    zm = grid[x, y, z - 1]
    xm_ym = grid[x - 1, y - 1, z]
    xm_zm = grid[x - 1, y, z - 1]
    ym_zm = grid[x, y - 1, z - 1]
    xm_ym_zm = grid[x - 1, y - 1, z - 1]

    # Now calculate the needed derivatives using a central difference scheme
    # Note that this is calculated using the STAGGERED derivative
    # We are looking at grid point (x, y, z). This is grouped with grid point
    # (x+1/2, y+1/2, z + 1/2).
    # We calculate the staggered x derivative at (x, y, z) (with analogous expressions for other derivatives) as:
    # 1 / (4 * dx) * ( (x + 1/2, y + 1/2, z + 1/2) - (x - 1/2, y + 1/2, z + 1/2) )
    # +  1 / (4 * dx) * ( (x + 1/2, y - 1/2, z + 1/2) - (x - 1/2, y - 1/2, z + 1/2) )
    # + 1 / (4 * dx) * (first expression with z -> z - 1)
    # + 1 / (4 * dx) * (second expression with z -> z-1)
    # First handle the x derivatives
    d_s11_dx = .25 * dx_inv * (me.s11 - xm.s11 + ym.s11 - xm_ym.s11 + zm.s11 - xm_zm.s11 + ym_zm.s11 - xm_ym_zm.s11) 
    d_s12_dx = .25 * dx_inv * (me.s12 - xm.s12 + ym.s12 - xm_ym.s12 + zm.s12 - xm_zm.s12 + ym_zm.s12 - xm_ym_zm.s12) 
    d_s13_dx = .25 * dx_inv * (me.s13 - xm.s13 + ym.s13 - xm_ym.s13 + zm.s13 - xm_zm.s13 + ym_zm.s13 - xm_ym_zm.s13) 

    # Now the y derivatives
    d_s12_dy = .25 * dy_inv * (me.s12 - ym.s12 + xm.s12 - xm_ym.s12 + zm.s12 - ym_zm.s12 + xm_zm.s12 - xm_ym_zm.s12) 
    d_s22_dy = .25 * dy_inv * (me.s22 - ym.s22 + xm.s22 - xm_ym.s22 + zm.s22 - ym_zm.s22 + xm_zm.s22 - xm_ym_zm.s22) 
    d_s23_dy = .25 * dy_inv * (me.s23 - ym.s23 + xm.s23 - xm_ym.s23 + zm.s23 - ym_zm.s23 + xm_zm.s23 - xm_ym_zm.s23) 

    # And last the z derivatives
    d_s13_dz = .25 * dz_inv * (me.s13 - zm.s13 + xm.s13 - xm_zm.s13 + ym.s13 - ym_zm.s13 + xm_ym.s13 - xm_ym_zm.s13)
    d_s23_dz = .25 * dz_inv * (me.s23 - zm.s23 + xm.s23 - xm_zm.s23 + ym.s23 - ym_zm.s23 + xm_ym.s23 - xm_ym_zm.s23)
    d_s13_dz = .25 * dz_inv * (me.s33 - zm.s33 + xm.s33 - xm_zm.s33 + ym.s33 - ym_zm.s33 + xm_ym.s33 - xm_ym_zm.s33)

    # Now calculate the updates
    me.cu = rho_inv * dt * (d_s11_dx + d_s12_dy + d_s13_dz)
    me.cv = rho_inv * dt * (d_s12_dx + d_s22_dy + d_s23_dz)
    me.cw = rho_inv * dt * (d_s13_dx + d_s23_dy + d_s33_dz)
