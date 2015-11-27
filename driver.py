import numpy as np
from os import environ

# http://stackoverflow.com/questions/2741399/python-pyximporting-a-pyx-that-depends-on-a-native-library
# Link to math library without having to write setup.py
environ['LDFLAGS'] = '-Lm -lm'

# Automatic Cython file compilation
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
    reload_support = True)

from mpi4py import MPI
from multiprocessing.pool import ThreadPool

from parse_input import parse_input
from sim import go

def prepare_input(config_file, dim_x, dim_y, dim_z):

    # get simulation inputs from configuration file
    lambd, mu, rho, min_x, max_x, min_y, max_y, min_z, max_z, N_x, N_y, N_z, t_0, t_f, N_t = parse_input(config_file)

    # Grid spacing in each dimension
    dx = (max_x - min_x) / N_x
    dy = (max_y - min_y) / N_y
    dz = (max_z - min_z) / N_z

    # Timestep
    dt = (t_f - t_0) / N_t

    # Grid size in each direction
    L_x = np.float64(max_x) - np.float64(min_x)
    L_y = np.float64(max_y) - np.float64(min_y)
    L_z = np.float64(max_z) - np.float64(min_z)

    # Number of grid points in each dimension per process
    # Probably need to be a little more careful about this to ensure even division
    nn_x = N_x / dim_x
    nn_y = N_y / dim_y
    nn_z = N_z / dim_z

    params = tuple([nn_x, nn_y, nn_z, L_x, L_y, L_z, dx, dy, dz, dt, mu, rho, lambd, t_0, t_f])

    return params


if __name__ == '__main__':
    import sys

    config_file = sys.argv[1]

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    dims = # some function of size that minimizes size of ghost regions?
    cartcomm = comm.Create_cart(dims, periods=(True, True, True), reorder=True)
    c_x, c_y, c_z = cartcomm.Get_coords(rank)

    go(cartcomm, c_x, c_y, c_z, *(prepare_input(config_file)))

