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

where `your_config_file.conf` contained the relevant parameters for your problem. A picture of this process is also displayed below.

![](https://cloud.githubusercontent.com/assets/2105882/11645662/6ff992e2-9d24-11e5-81af-aa1021f735b6.png)

### Initial and Boundary Conditions

You will also need to implement the relevant boundary conditions for the problem you wish to solve. This is done in the `set_boundary_conditions()` function in `sim.pyx`. Shear wave initial conditions have been provided as an example to demonstrate how to implement initial conditions. Note that in the parallel case, to properly load in initial conditions it is necessary to account for the fact that each processor is shifted in space. These shifts can be computed simply using `cx`, `cy`, and `cz`, the indices of the processor in the Cartesian communicator (described more in detail in the technical details below). Simply multiply each Cartesian index `ci` by the number of subdomain grid elements in that direction, `n_i`.

### Parallel 

To pull the parallel code, checkout the `parallel_MPI` branch:

    git checkout parallel_MPI

 The code can be run just as in the case of the serial file by calling:

    mpirun -np n python driver.py [your configuration file]

where `n` is the number of processors.

## Background Information

Bulk metallic glasses (BMGs) are an alloy whose atoms form an amorphous, random structure, in contrast to most metals. BMGs possess exceptional mechanical properties, such as high tensile strength, excellent wear resistance, and the ability to be efficiently molded and processed. They are under consideration for a wealth of technological applications, such as next-generation smartphone cases and aircraft components. However, their amorphous structure raises fundamental unanswered questions about their mechanical properties, which has hindered their usage in structural applications where they must be guaranteed not to fail.

This code simulates an elastoplastic material model for bulk metallic glasses explicitly. The equations can be found in "C. H. Rycroft, Y. Sui, E. Bouchbinder, An Eulerian projection method for quasi-static elastoplasticity,  J. Comp. Phys. 300 (2015) 136-166". The explicit method implements at the moment only the equations of elasticity; additional terms need to be added to fully simulate hypoelastoplasticity in BMGs, and as such the current code is still a work in progress. The code generalizes what is found in the above reference to three dimensions. Parallelization is achieved by dividing the three dimensional grid into subdomains and using message passing at the boundaries to calculate derivatives.

## Technical Details

### Benchmarking

To be included soon.

### MPI Implementation Details

To achieve parallelization, we divide our spatial grid into `n` subdomains where `n` is the number of processors. This is handled simply using a Cartesian Communicator as provided by `MPI`, where we associate the processor with index `(0, 0, 0)` with the bottom-left corner of our grid, `(1, 0, 0)` with the an equally-sized box whose origin is shifted by `n_x` where `n_x` is the number of x grid points in a subdomain. This generalizes naturally to every other dimension.

For the most part, these subdomains are disconnected, and each processor can solve the equations on their own subdomain. However, because the equations of linear elasticity (and of course hypoelastoplasticity as well, although these are not implemented at the moment) involve spatial derivatives, they are nonlocal. Hence calculating the value of spatial derivatives located at the boundary of a subdomain require field values that are stored in adjacent processors. This requirement is symmetric; for example, if processor `n` requires points from the left-most boundary of processor `m`'s domain, then it necessarily follows that processor `m` requires points from the right-most boundary of processor `n`'s domain. These communications can be handled naturally using the Cartesian communicator's `MPI_Cart_shift` function in conjunction with `MPI_Sendrecv_replace`.

To handle the need of communicating these values, we pad each subdomain with a "ghost region" which surrounds the physical space the processor is solving the equations within. This the ghost region for each process is propulated at the beginning of each timestep with the required adjacent values by the `set_up_ghost_regions()` function. There are three cases that must be considered.

First, if two processors share a face, the two processors must share their (opposite, in the sense as described above) planes with each other.

(image here)

Second, if two processors share an edge, these edges must be communicated in the same way.

(image here)

Last, if two processors share a corner, they must share this individual value with each other.

(image here)

In total, there are six faces, twelve edges, and eight corners, totalling 26 portions of the grid that each processor must send out to adjacent processors, and 26 portions of that grid that each processor must receive from adjacent processors before being capable of computing spatial derivatives in their subdomain. Once a processor's ghost region has been populated, they are free to calculate derivatives at all of their physical grid locations. Clearly derivatives do not need to be calculated in ghost regions as they correspond to points in adjacent processors and are calculated there. Note that the ghost regions must be repopulated at every timestep in accordance with the update of values across the grid.


## Contributors

This code was completed by Anna Whitney and Nicholas Boffi for the final project in CS205. The code will be maintained and updated with the remaining terms defining the STZ model by Nicholas Boffi.
