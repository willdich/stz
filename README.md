# stz

stz is a work-in-progress code for simulating bulk metallic glasses in 3D using the shear transformation zone theory of amorphous plasticity. At the moment, only linear elastic terms have been implemented. The code is written in Cython and has been parallelized through MPI via mpi4py. 

## Getting Started

### Requirements

This code requires cython (http://cython.org/) and mpi4py (http://mpi4py.scipy.org/).

### Branches

Pull the `master` branch for a serial implementation, `parallel_MPI` for the parallel implementation, or `odyssey` for code suited for running on Harvard University's Odyssey cluster.

### Download and Installation

Grab your copy of the code with

    git clone git@github.com:thereisnoquarter/stz.git

Assuming you have cython installed, you can cause the program to compile to C with:

    python driver.py

### Simulation Paramaters

An example `test.conf` file is provided for configuration options. The file accepts values for the material Lame parameters and density, the size of the grid, the number of points in each dimension, the initial and final time, and an output file. This file can be modified directly or used as a template.To run a simulation with specific parameters, call:

    python driver.py your_config_file.conf

where `your_config_file.conf` contained the relevant parameters for your problem.

### Initial and Boundary Conditions

You will also need to implement the relevant boundary conditions for the problem you wish to solve. This is done in the `set_boundary_conditions()` function in `sim.pyx`. Shear wave initial conditions have been provided as an example to demonstrate how to implement initial conditions.

### Parallel 

To pull the parallel code, checkout the `parallel_MPI` branch. The code can be run just as in the case of the serial file by calling:

    mpirun -np n driver.py [your configuration file]

where `n` is the number of processors.

## Background Information

Bulk metallic glasses (BMGs) are an alloy whose atoms form an amorphous, random structure, in contrast to most metals. BMGs possess exceptional mechanical properties, such as high tensile strength, excellent wear resistance, and the ability to be efficiently molded and processed. They are under consideration for a wealth of technological applications, such as next-generation smartphone cases and aircraft components. However, their amorphous structure raises fundamental unanswered questions about their mechanical properties, which has hindered their usage in structural applications where they must be guaranteed not to fail.

This code simulates an elastoplastic material model for bulk metallic glasses explicitly. The equations can be found in "C. H. Rycroft, Y. Sui, E. Bouchbinder, An Eulerian projection method for quasi-static elastoplasticity,  J. Comp. Phys. 300 (2015) 136-166". The explicit method implements at the moment only the equations of elasticity; additional terms need to be added to fully simulate hypoelastoplasticity in BMGs, and as such the current code is still a work in progress. The code generalizes what is found in the above reference to three dimensions. Parallelization is achieved by dividing the three dimensional grid into subdomains and using message passing at the boundaries to calculate derivatives.

## Contributors

This code was completed by Anna Whitney and Nicholas Boffi for the final project in CS205. The code will be maintained and updated with the remaining terms defining the STZ model by Nicholas Boffi.
