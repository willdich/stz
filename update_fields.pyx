cimport numpy as np
from common cimport *
from fields cimport Field

cdef void update_stresses(Field *grid,                                         # Grid of field values
                      int x, int y, int z,                                     # Location in the grid
                      int N_x, int N_y, int N_z,                               # Grid sizes (for lookup)
                      np.float64_t dx, np.float64_t dy, np.float64_t dz,       # Spatial discretization
                      np.float64_t dt,                                         # Time discretization
                      np.float64_t lam, np.float64_t mu) nogil:                # Material Parameters

    """ Calculates the updates for the stress tensor s_ij of the field value at grid[x, y, z].
    At the moment, the grid spacing is unnecessary because of the lack of advective terms.
    It has been kept in to be most general.
    """

    # Store our needed variables
    cdef:
        Field *me                    # Field value at the currrent location (x, y, z)
        Field *xp                    # Field value at (x+1, y, z) 
        Field *yp                    # Field value at (x, y+1, z)
        Field *zp                    # Field value at (x, y, z+1)
        Field *xp_yp                 # Field value at (x+1, y+1, z)
        Field *xp_zp                 # Field value at (x+1, y, z+1)
        Field *yp_zp                 # Field value at (x, y+1, z+1)
        Field *xp_yp_zp              # Field value at (x+1, y+1, z+1)

        np.float64_t du_dx           # Derivatives of velocity components
        np.float64_t du_dy 
        np.float64_t du_dz 
        np.float64_t dv_dx 
        np.float64_t dv_dy
        np.float64_t dv_dz
        np.float64_t dw_dx
        np.float64_t dw_dy
        np.float64_t dw_dz 

        # The trace of the rate of strain tensor 
        np.float64_t d_trace  
        
        # Convenience quantities
        np.float64_t rho_inv        # 1/rho
        np.float64_t dx_inv         # 1/dx
        np.float64_t dy_inv         # 1/dy
        np.float64_t dz_inv         # 1/dz
        
    # Get the inverse values for simplicity
    dx_inv = 1. / dx
    dy_inv = 1. / dy
    dz_inv = 1. / dz

    # First look up the corresponding grid values 
    me = look_up(grid, N_x, N_y, N_z, x, y, z)
    xp = look_up(grid, N_x, N_y, N_z, x + 1, y, z)
    yp = look_up(grid, N_x, N_y, N_z, x, y + 1, z)
    zp = look_up(grid, N_x, N_y, N_z, x, y, z + 1)
    xp_yp = look_up(grid, N_x, N_y, N_z, x + 1, y + 1, z)
    xp_zp = look_up(grid, N_x, N_y, N_z, x + 1, y, z + 1)
    yp_zp = look_up(grid, N_x, N_y, N_z, x, y + 1, z + 1)
    xp_yp_zp = look_up(grid, N_x, N_y, N_z, x + 1, y + 1, z + 1)

    # First calculate all the (staggered) derivatives
    # See below function for explanation of terms - I wrote that function first
    du_dx = .25 * dx_inv * (xp.u - me.u + xp_yp.u - yp.u + xp_zp.u - zp.u + xp_yp_zp.u - yp_zp.u)
    dv_dx = .25 * dx_inv * (xp.v - me.v + xp_yp.v - yp.v + xp_zp.v - zp.v + xp_yp_zp.v - yp_zp.v)
    dw_dx = .25 * dx_inv * (xp.w - me.w + xp_yp.w - yp.w + xp_zp.w - zp.w + xp_yp_zp.w - yp_zp.w)

    du_dy = .25 * dy_inv * (yp.u - me.u + xp_yp.u - xp.u + yp_zp.u - zp.u + xp_yp_zp.u - xp_zp.u) 
    dv_dy = .25 * dy_inv * (yp.v - me.v + xp_yp.v - xp.v + yp_zp.v - zp.v + xp_yp_zp.v - xp_zp.v) 
    dw_dy = .25 * dy_inv * (yp.w - me.w + xp_yp.w - xp.w + yp_zp.w - zp.w + xp_yp_zp.w - xp_zp.w) 

    du_dz = .25 * dz_inv * (zp.u - me.u + xp_zp.u - xp.u + yp_zp.u - yp.u + xp_yp_zp.u - xp_yp.u) 
    dv_dz = .25 * dz_inv * (zp.v - me.v + xp_zp.v - xp.v + yp_zp.v - yp.v + xp_yp_zp.v - xp_yp.v) 
    dw_dz = .25 * dz_inv * (zp.w - me.w + xp_zp.w - xp.w + yp_zp.w - yp.w + xp_yp_zp.w - xp_yp.w) 

    # Store the trace for simplicity
    d_trace = .5 * (du_dx + dv_dy + dw_dz)

    # Now calculate the corresponding changes in stresses
    # First the diagonal terms, which have a contribution from the trace of D
    me.cs11 = dt * (lam * d_trace + 2 * mu * du_dx)
    me.cs22 = dt * (lam * d_trace + 2 * mu * dv_dy)
    me.cs33 = dt * (lam * d_trace + 2 * mu * dw_dz)

    # And now calculate the updates for the off diagonal elements
    me.cs12 = dt * mu * (du_dy + dv_dx)
    me.cs13 = dt * mu * (du_dz + dw_dx)
    me.cs23 = dt * mu * (dv_dz + dw_dy)

cdef void update_velocities(Field *grid,                                        # Grid of field values
                      int x, int y, int z,                                     # Location in the grid
                      int N_x, int N_y, int N_z,                               # Grid sizes (for lookup)
                      np.float64_t dx, np.float64_t dy, np.float64_t dz,       # Spatial discretization
                      np.float64_t dt,                                         # Time discretization
                      np.float64_t rho) nogil:                                 # Material density

    """ Calculates the updates for the velocity components of the field value at grid[x, y, z].
    Currently serial. When parallelizing with MPI, we will need some communication at the boundaries here.
    We assume that boundary conditions are handled in the driver program. This means we do not need to worry
    about, e.g., x+1 going out of bounds. We have ghost regions to handle this case and we should only be calling
    this update on the internal regions.
    """

    # Store our our needed variables
    cdef:
        Field *me                    # Field value at the currrent location (x, y, z)
        Field *xm                    # Field value to the left (x-1, y, z)
        Field *ym                    # Field value at (x, y-1, z)
        Field *zm                    # Field value at (x, y, z-1)
        Field *xm_ym                 # Field value at (x-1, y-1, z)
        Field *xm_zm                 # Field value at (x-1, y, z-1)
        Field *ym_zm                 # Field value at (x, y-1, z-1)
        Field *xm_ym_zm              # Field value at (x-1, y-1, z-1)
        Field *xp                    # Field value at (x+1, y, z)
        Field *yp                    # Field value at (x, y+1, z)
        Field *zp                    # Field value at (x, y, z+1)

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

        np.float64_t grad_sq_u      # Laplacians of the velocities for use with the viscosity term
        np.float64_t grad_sq_v
        np.float64_t grad_sq_w

        np.float64_t kap = .05      # viscosity fudge factor to smooth out divergences
        
    # Get the inverse values for simplicity
    rho_inv = 1. / rho
    dx_inv = 1. / dx
    dy_inv = 1. / dy
    dz_inv = 1. / dz

    # First look up the corresponding grid values 
    me = look_up(grid, N_x, N_y, N_z, x, y, z)
    xm = look_up(grid, N_x, N_y, N_z, x - 1, y, z)
    ym = look_up(grid, N_x, N_y, N_z, x, y - 1, z)
    zm = look_up(grid, N_x, N_y, N_z, x, y, z - 1)
    xm_ym = look_up(grid, N_x, N_y, N_z, x - 1, y - 1, z)
    xm_zm = look_up(grid, N_x, N_y, N_z, x - 1, y, z - 1)
    ym_zm = look_up(grid, N_x, N_y, N_z, x, y - 1, z - 1)
    xm_ym_zm = look_up(grid, N_x, N_y, N_z, x - 1, y - 1, z - 1)
    xp = look_up(grid, N_x, N_y, N_z, x + 1, y, z)
    yp = look_up(grid, N_x, N_y, N_z, x, y + 1, z)
    zp = look_up(grid, N_x, N_y, N_z, x, y, z + 1)

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

    # Now get the Laplacian terms
    grad_sq_u = dx_inv * dx_inv * (xp.u - 2 * me.u + xm.u) + dy_inv * dy_inv * (yp.u - 2 * me.u + ym.u) + dz_inv * dz_inv * (zp.u - 2 * me.u + zm.u)
    grad_sq_v = dx_inv * dx_inv * (xp.v - 2 * me.v + xm.v) + dy_inv * dy_inv * (yp.v - 2 * me.v + ym.v) + dz_inv * dz_inv * (zp.v - 2 * me.v + zm.v)
    grad_sq_w = dx_inv * dx_inv * (xp.w - 2 * me.w + xm.w) + dy_inv * dy_inv * (yp.w - 2 * me.w + ym.w) + dz_inv * dz_inv * (zp.w - 2 * me.w + zm.w)

    # Now calculate the updates
    me.cu = rho_inv * dt * (d_s11_dx + d_s12_dy + d_s13_dz + kap * grad_sq_u)
    me.cv = rho_inv * dt * (d_s12_dx + d_s22_dy + d_s23_dz + kap * grad_sq_v)
    me.cw = rho_inv * dt * (d_s13_dx + d_s23_dy + d_s33_dz + kap * grad_sq_w)
