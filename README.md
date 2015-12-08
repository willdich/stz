# stz

Code for simulating bulk metallic glasses in 3D using the shear transformation zone theory. This is a work in progress, and at the moment, only linear elastic terms have been implemented - thus, it is a glorified solver for linear elasticity. The code is written in Cython and has been parallelized through MPI via mpi4py. The serial version (perhaps useful for testing) can be pulled from the master branch, and the parallel version (naturally) from the parallel_MPI branch. The Odyssey branch contains code suitable for running on Harvard's cluster computer Odyssey, although is likely suitable for clusters in general with slight modifications to ensure proper importing of necessary packages.


## Requirements

This code requires (cython)[http://cython.org/] and (mpi4py)[http://mpi4py.scipy.org/].

## Download and Installation

Grab your copy of the code with

    git@github.com:thereisnoquarter/stz.git

Assuming you have cython installed, you can cause the program to compile to C with:

    python driver.py

## Simulation Paramaters

An example `test.conf` file is provided for configuration options. The file accepts values for the material Lame parameters and density, the size of the grid, the number of points in each dimension, the initial and final time, and an output file. This file can me modified directly or used as a template.To run a simulation with specific parameters, call:

    python driver.py [your configuration file]

simple, right?

## Initial and Boundary Conditions

You will also need to implement the relevant boundary conditions for the problem you wish to solve. This is done in the `set_boundary_conditions()` function in `sim.pyx`.

## Parallel Code

To pull the parallel code, checkout the parallel_MPI branch.
