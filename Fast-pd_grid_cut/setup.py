from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(Extension(
           "fast_pd",                                           # the extension name
           sources=["fast_pd.pyx", "bc_pd_cache_friendly.cpp"], # the Cython source and
                                                                # additional C++ source files
           include_dirs=[numpy.get_include()],
           language="c++",                                      # generate and compile C++ code
           extra_compile_args=["-O3"],
      )))
