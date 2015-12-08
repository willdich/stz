import mpi4py
import os
from os import environ

# http://stackoverflow.com/questions/2741399/python-pyximporting-a-pyx-that-depends-on-a-native-library
# Link to math library without having to write setup.py
environ['LDFLAGS'] = '-Lm -lm'
environ['CFLAGS'] = '-I/n/sw/fasrcsw/apps/Core/Anaconda/1.9.2-fasrc01/x/lib/python-2.7/site-packages/mpi4py/include/'

# Automatic Cython file compilation
import pyximport
pyximport.install(setup_args={"include_dirs":[mpi4py.get_include()]})

from test_cy import go

if __name__ == '__main__':
    go()
