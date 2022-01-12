# coding: utf-8
r"""
Some utility functions for sequences.
"""

#############################################################################
# Copyright (C) 2022 Philipp Nuspl, philipp.nuspl@jku.at
#
# This program is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
# more details.
#
# You should have received a copy of the GNU General Public License along 
# with this program. If not, see <https://www.gnu.org/licenses/>. 
#############################################################################

from sage.parallel.decorate import fork
from sage.misc.functional import multiplicative_order
from sage.rings.semirings.non_negative_integer_semiring import NN
from sage.misc.functional import round
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import free_module_element as vector
from sage.rings.integer_ring import ZZ

class TimeoutError(Exception):
    """Exception raised if computations did not finish in given ``time``.

    Attributes:
    
        - ``time`` -- number indicating the seconds used for the computation
        - ``message`` -- given by "Timeout after s seconds" where s is
          ``time``.
    
    """

    def __init__(self, time):
        self.time = time
        self.message = f"Timeout after {round(time,2)} seconds"
        super().__init__(self.message)

def timeout(func, time=10, *args, **kwargs):
    r"""
    Run a method ``func`` in a forked subprocess for the given
    time and with the the given arguments. If ``time``
    is not positive, the method is might run indefinitely if
    the function does not return.
    
    INPUT:
    
    - ``func`` -- a function
    - ``time`` (default: ``10``) -- a number; if positive, it specifies the time
      ``func`` should be run; if not positive it will be ignored
    - ``*args`` -- additional arguments passed to ``func``
    - ``**kwargs`` -- additional arguments passed to ``func``
    
    OUTPUT:
    
    Returns ``func(*args, **kwargs)``. If the function does not finish in the 
    given time, a :class:`TimeoutError` is raised. Exceptions raised by ``func``
    are raised again.
    
    """
    if time > 0 :
        @fork(timeout=time, verbose=False)
        def forked_func_timeout():
            return encode_exceptions(func, *args, **kwargs)
        timeout_return = forked_func_timeout()
        if timeout_return != "NO DATA (timed out)" :
            return decode_exceptions(timeout_return)
        else :
            raise TimeoutError(time) 
    else :
        @fork(verbose=True)
        def forked_func():
            return encode_exceptions(func, *args, **kwargs)
        return decode_exceptions(forked_func())
    
def encode_exceptions(func, *args, **kwargs) :
    r"""
    Encodes the exception raised by a function as an object.
    
    INPUT:
    
    - ``func`` -- a function
    - ``time`` (default: ``10``) -- a number; if positive, it specifies the time
      ``func`` should be run; if not positive it will be ignored
    - ``*args`` -- additional arguments passed to ``func``
    - ``**kwargs`` -- additional arguments passed to ``func``
    
    OUTPUT:
    
    Calls the given function ``func`` with the given parameters.
    If the function raises an exception, this exception is returned
    as an object. Otherwise the usual return value is returned.
    """
    try :
        ret = func(*args, **kwargs)
        return ret
    except Exception as e:
        return e
    
def decode_exceptions(ret) :
    r"""
    Checks whether the given value is an exception. If it is, it raises
    this exception. Otherwise the given value is just returned.
    """
    if isinstance(ret, Exception) :
        raise ret
    else :
        return ret
    
def is_root_of_unity(a) :
    r"""
    INPUT:
    
    - ``a`` -- a number
    
    OUTPUT:
    
    Returns ``True`` if ``a`` is a root of unity and ``False`` otherwise.
    """
    try :
        order = a.multiplicative_order()
        if order in NN :
            return True
    except :
        pass
    return False

### Some methods to deal with algebraic structures over a sequence ring.

def matrix_subsequence(matrix_sequence, u, v = 0):
    r"""
    Returns the matrix subsequence.

    INPUT:

    - ``matrix_sequence`` -- a matrix over a sequence ring
    - ``u`` -- a rational number
    - ``v`` (optional) -- a rational number

    OUTPUT:
    
    If the matrix `(c_{i,j}(n))_{i,j}` is given, the matrix
    `(c_{i,j}(\operatorname{floor}(u n + v)))_{i,j}` is returned.
    """
    n_rows = matrix_sequence.nrows()
    n_cols = matrix_sequence.ncols()
    M = matrix(matrix_sequence.base_ring(), n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            M[i,j] = matrix_sequence[i,j].subsequence(u, v)
    return M

def vector_subsequence(vector_sequence, u, v = 0):
    r"""
    Returns the vector subsequence.

    INPUT:

    - ``vector_sequence`` -- a vector over a sequence ring
    - ``u`` -- a rational number
    - ``v`` (optional) -- a rational number

    OUTPUT:
    
    If the vector `(c_{i}(n))_{i}` is given, the vector
    `(c_{i}(\operatorname{floor}(u n + v)))_{i}` is returned.
    """
    n_rows = len(vector_sequence)
    ret = vector(vector_sequence.base_ring(), n_rows)
    for i in range(n_rows):
        ret[i] = vector_sequence[i].subsequence(u, v)
    return ret

def vector_interlace(vector_sequences, initial_values = []):
    r"""
    Returns the interlacing of the given vectors of sequences.

    INPUT:

    - ``vector_sequences`` -- container of vectors of sequences over a 
      common ring.
    - ``initial_values`` -- list of vectors in the ground field.

    OUTPUT:
    
    Interlacing of ``vector_sequences`` with the prepended values
    ``initial_values`` at the beginning. 
    """
    dim = len(vector_sequences[0])
    R = vector_sequences[0].base_ring()
    ret = vector(R, dim)
    
    for i in range(dim) :
        interlacing_seq = [seq_vec[i] for seq_vec in vector_sequences[1:]]
        interlaced = vector_sequences[0][i].interlace(*interlacing_seq)
        prepend_values_i = [val[i] for val in initial_values]
        ret[i] = interlaced.prepend(prepend_values_i)
        
    return ret
    

def eval_matrix(matrix_sequence, n=1) :
    r"""
    Returns the matrix evaluated at an index.

    INPUT:

    - ``matrix_sequence`` -- a matrix over a sequence ring
    - ``n`` (default: ``1``) -- a natural number

    OUTPUT:
    
    If the matrix `(c_{i,j})_{i,j}` is given, the matrix
    `(c_{i,j}(n))_{i,j}` is returned.
    """
    n_rows = matrix_sequence.nrows()
    n_cols = matrix_sequence.ncols()
    K = matrix_sequence.base_ring().base_ring()
    M = matrix(K, n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            M[i,j] = matrix_sequence[i,j][n]
    return M

def eval_vector(vector_sequence, n) :
    r"""
    Returns the vector evaluated at an index.

    INPUT:

    - ``vector_sequence`` -- a vector over a sequence ring
    - ``n`` (default: ``1``) -- a natural number

    OUTPUT:
    
    If the vector `(c_{i})_{i}` is given, the vector
    `(c_{i}(n))_{i}` is returned.
    """
    n_rows = vector_sequence.length()
    K = vector_sequence.base_ring().base_ring()
    v = vector(K, n_rows)
    for i in range(n_rows):
        v[i] = vector_sequence[i][n]
    return v

def shift_matrix(matrix_sequence, n=1) :
    r"""
    Returns the matrix shifted.

    INPUT:

    - ``matrix_sequence`` -- a matrix over a sequence ring
    - ``n`` (default: ``1``) -- a natural number

    OUTPUT:
    
    If the matrix `(c_{i,j})_{i,j}` is given, the matrix
    `(\sigma^n c_{i,j})_{i,j}` is returned.
    """
    M = matrix(matrix_sequence)
    for i in range(M.nrows()):
        for j in range(M.ncols()):
            M[i,j] = M[i,j].shift(n)
    return M

def shift_vector(vector_sequence, n=1) :
    r"""
    Returns the vector shifted.

    INPUT:

    - ``vector_sequence`` -- a vector over a sequence ring
    - ``n`` (default: ``1``) -- a natural number

    OUTPUT:
    
    If the vector `(c_{i})_{i}` is given, the vector
    `(\sigma^n c_{i})_{i}` is returned.
    """
    v = vector(vector_sequence[0].parent(), len(vector_sequence))
    for i in range(v.length()):
        v[i] = vector_sequence[i].shift(n)
    return v

def shift_rat_vector(vector_sequence, n=1) :
    r"""
    Returns the vector shifted.

    INPUT:

    - ``vector_sequence`` -- a vector over a rational function ring
    - ``n`` (default: ``1``) -- a natural number

    OUTPUT:
    
    If the vector `(c_{i})_{i}` is given, the vector
    `(\sigma^n c_{i})_{i}` is returned.
    """
    v = vector(vector_sequence[0].parent(), len(vector_sequence))
    gen = vector_sequence[0].parent().gen()
    for i in range(v.length()):
        v[i] = vector_sequence[i].subs({gen: gen+n})
    return v

def split_list(l, r) :
    r"""
    Splits a list into lists of equal size.
    
    INPUT:
    
    - ``l`` -- a list
    - ``r`` -- a natural number which divides ``len(l)``.
    
    OUTPUT:
    
    The list ``l`` split into ``r`` lists of equal size
    """
    d = ZZ(len(l)/r)
    return [l[d*i:d*(i+1)] for i in range(r)]

def split_list_rec(l, r) :
    r"""
    Splits a list recursively into lists of equal size.
    
    INPUT:
    
    - ``l`` -- a list
    - ``r`` -- a list of natural numbers such that the product
      of these is precisely ``len(l)``.
    
    OUTPUT:
    
    The list ``l`` split into ``r[0]`` lists of equal size which
    itself are split into ``r[1]`` lists of equal size, etc.
    """
    if not r :
        return l
    l_split = split_list(l, r[0])
    return [split_list_rec(sublist, r[1:]) for sublist in l_split]

def log(cls, msg, level=0, time=-1, time2=-1) :
    r"""
    Assumes that the given class has a logger object
    called ``log``. Tries to output the given message ``msg`` as info using 
    this logger. 
    
    These messages can be displayed using::
    
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
    INPUT:
    
    - ``cls`` -- a class having a logger object called ``log``
    - ``msg`` -- a string; the message that is shown by the logger
    - ``level`` -- a natural number, indicating the indentation of the message
    - ``time`` -- if ``time2`` is not given, this is a specific time
    - ``time2`` -- used to compute the time-span from ``time`` to
      ``time2`` to indicate how long a computation took.
    
    OUTPUT:
    
    A string which contains leading whitespaces (the amount depends on
    ``level``), the specified message ``msg`` and the given time.
    """
    logger = cls.log
    indent = level*"    "
    
    if time == -1 and time2 == -1 :
        logger.info(indent+msg)
    elif time2 == -1 :
        msg_time = f" (at time {time2})"
        logger.info(indent+msg+msg_time)
    else :
        msg_time = f" (taking time {time2-time})"
        logger.info(indent+msg+msg_time)
    