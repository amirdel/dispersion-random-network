from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import sys
import numpy
import os

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

ext_modules = [Extension("trajectory_count_cython", ["trajectory_count_cython.pyx"],
language='c++', extra_compile_args=['-O3', '-march=native'])]
    

setup(
    name='trajectory_count_cython',
    cmdclass={'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    ext_modules=ext_modules,
) 
