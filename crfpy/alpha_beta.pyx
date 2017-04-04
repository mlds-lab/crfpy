import sys
cimport cython
import numpy as np
from scipy.misc import logsumexp
cimport numpy as np
from libc.math cimport exp, log 
import time
# from rutils import *
# from log_sum_exp import my_logaddexp,my_1d_logsumexp

DTYPE=np.float64
ctypedef np.float64_t DTYPE_t

ITYPE=np.int
ctypedef np.int_t ITYPE_t

cdef DTYPE_t my_logaddexp(DTYPE_t a, DTYPE_t b):
    cdef DTYPE_t t = a - b
    if t > 0:
        return a + log(1 + exp(-t))
    elif t <= 0:
        return b + log(1 + exp(t))
    else:
        return t
        
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef DTYPE_t my_1d_logsumexp(np.ndarray[DTYPE_t,ndim=1] a) except *:
    cdef DTYPE_t a_max = a[0]
    cdef DTYPE_t a_sum = 0.0
    # cdef DTYPE_t out = 0.0
    cdef int i,j
    cdef int a_len = a.shape[0]

    for i in range(a_len):
        if a[i] > a_max:
            a_max = a[i]
    for i in range(a_len):
        a_sum += exp(a[i] - a_max)

    return log(a_sum) + a_max

def cy_alpha(np.ndarray[DTYPE_t,ndim=2] unary_potentials, np.ndarray[DTYPE_t,ndim=2] transition_weights):
    cdef int N = unary_potentials.shape[0]
    cdef int n_classes = unary_potentials.shape[1]
    cdef np.ndarray[DTYPE_t,ndim=2] alpha = np.zeros((N,n_classes),dtype=DTYPE)
    cdef int i,j,k
    
    alpha[0] = unary_potentials[0]
    for i in range(1,N):
        for j in range(n_classes):
            alpha[i,j] = unary_potentials[i,j] + my_1d_logsumexp(transition_weights[:,j] + alpha[i-1])
    
    return alpha
    
def cy_beta(np.ndarray[DTYPE_t,ndim=2] unary_potentials, np.ndarray[DTYPE_t,ndim=2] transition_weights):
    cdef int N = unary_potentials.shape[0]
    cdef int n_classes = unary_potentials.shape[1]
    cdef np.ndarray[DTYPE_t,ndim=2] beta = np.zeros((N,n_classes),dtype=DTYPE)
    cdef int i,j,k

    for i in range(N-1)[::-1]:
        for j in range(n_classes):
            beta[i,j] = my_1d_logsumexp(transition_weights[j] + beta[i+1] + unary_potentials[i+1])

    return beta