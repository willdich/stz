# stz
Code for simulating bulk metallic glasses in 3D using the shear transformation zone theory. This is a work in progress, and at the moment, only linear elastic terms have been implemented - thus, it is a glorified solver for linear elasticity. The code is written in Cython and has been parallelized through MPI via mpi4py. The serial version (perhaps useful for testing) can be pulled from the master branch, and the parallel version (naturally) from the parallel_MPI branch. The Odyssey branch contains code suitable for running on Harvard's cluster computer Odyssey, although is likely suitable for clusters in general with slight modifications to ensure proper importing of necessary packages.

To run the code, simply fill in a configuration file (test.conf) is provided as an example. Fill in the set_boundary_conditions() function located in sim.pyx for your desired boundary conditions. The code can then be run in serial with:

python driver.py [your config file]

or:

mpirun -np n python driver.py [your_config_file] 

to run in paralle, where n is the desired number of processors. If running with MPI for the first time, it is suitable to call python driver.py first to cause the Cython code to compile.
