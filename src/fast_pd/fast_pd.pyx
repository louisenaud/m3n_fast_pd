# distutils: language = c++
# distutils: sources = bc_pd_cache_friendly.cpp
from libcpp cimport bool
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

#
cdef extern from "bc_pd_cache_friendly.hpp":
    cdef cppclass BC_PD_CF:

        BC_PD_CF(int R, int C, int L, vector[float] unaries_, vector[float] weights_, vector[float] dist_, vector[float] x_init_) except +

        void optimize(const size_t I_max, bool grow_sink)
        void restore_unaries()

        vector[float] get_solution()


# Create python class
cdef class PyFastPd:
    cdef BC_PD_CF* c_pd      # hold a C++ instance which we're wrapping

    def __cinit__(self, int R, int C, int L, np.ndarray[float, ndim=1] unaries_, np.ndarray[float, ndim=1] weights_, np.ndarray[float, ndim=1] dist_, np.ndarray[float, ndim=1] x_init_):
        self.c_pd = new BC_PD_CF(R, C, L, unaries_, weights_, dist_, x_init_)

    def __dealloc__(self):
        del self.c_pd

    def restore_unaries(self):
        self.c_pd.restore_unaries()

    def optimize(self, int I_max, bool grow_sink):
        self.c_pd.optimize(I_max, grow_sink)

    def get_solution(self):
        return np.asarray(self.c_pd.get_solution(), dtype=np.int32)
