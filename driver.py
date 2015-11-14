import sys
sys.path.append('../util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

if name == '__main__':
    # Set up material parameters
    # Instantiate the grid?
    # Run the simulation
