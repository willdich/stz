cimport numpy as np
cimport fields

cpdef update_stress(Field [:, :, :] grid,                                      # Grid of field values
                      np.uint32_t x, np.uint32_t y, np.uint32_t x,             # Location in the grid
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

cpdef update_velocities(Field [:, :, :] grid,                                  # Grid of field values
                      np.uint32_t x, np.uint32_t y, np.uint32_t x,             # Location in the grid
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
        Field my_f_val              # Field value at the currrent location
        Field xp_f_val              # Field value to the right (x+1)
        Field xm_f_val              # Field value to the left (x-1)
        Field yp_f_val              # Field value at y+1
        Field ym_f_val              # Field value at y-1
        Field zp_f_val              # Field value at z+1
        Field zm_f_val              # Field at z-1

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


    # First look up the corresponding grid values
    my_f_val = grid[x, y, z]
    xp_f_val = grid[x + 1, y, z]
    xm_f_val = grid[x - 1, y, z]
    yp_f_val = grid[x, y + 1, z]
    ym_f_val = grid[x, y - 1, z]
    zp_f_val = grid[x, y, z + 1]
    zm_f_val = grid[x, y, z - 1]

    # Now calculate the needed derivatives using a central difference scheme
    d_s11_dx = (xp_f_val.s11 - xm_f_val.s11) / (2 * dx)
    d_s12_dy = (yp_f_val.s12 - ym_f_val.s12) / (2 * dy)
    d_s13_dz = (zp_f_val.s13 - zm_f_val.s13) / (2 * dz)
    d_s12_dx = (xp_f_val.s12 - xm_f_val.s12) / (2 * dx)
    d_s22_dy = (yp_f_val.s22 - ym_f_val.s22) / (2 * dy)
    d_s23_dz = (zp_f_val.s23 - zm_f_val.s23) / (2 * dz)
    d_s13_dz = (xp_f_val.s13 - xm_f_val.s13) / (2 * dx)
    d_s32_dy = (yp_f_val.s32 - ym_f_val.s32) / (2 * dy)
    d_s33_dz = (zp_f_val.s33 - zm_f_val.s33) / (2 * dz)

    # Get the inverse density for simplicity
    rho_inv = 1. / rho

    # Now calculate the updates
    my_f_val.cu = rho_inv * dt * (d_s11_dx + d_s12_dy + d_s13_dz)
    my_f_val.cv = rho_inv * dt * (d_s12_dx + d_s22_dy + d_s23_dz)
    my_f_val.cw = rho_inv * dt * (d_s12_dx + d_s23_dy + d_s33_dz)
