# coding: utf-8
r"""
Difference definable sequence ring.

Sublasses :class:`rec_sequences.RecurrenceSequenceRing` and defines 
sequences satisfying a linear
recurrence equation with coefficients in some sequence
ring. Such a recurrence sequence 
`a(n)` is defined by a recurrence

.. MATH::
    c_0(n) a(n) + \dots + c_r(n) a(n+r) = 0 \text{ for all } n \geq 0

with `c_r(n) \neq 0` for all `n` and initial values `a(0),...,a(r-1)`. 
This is based on the theory and algorithms developed in [JPNP21a]_.

EXAMPLES::

    sage: from rec_sequences.DifferenceDefinableSequenceRing import *
    sage: from rec_sequences.CFiniteSequenceRing import *
    
    sage: C = CFiniteSequenceRing(QQ)
    sage: C2 = DifferenceDefinableSequenceRing(C) # C^2-finite sequence ring
    sage: n = var("n")
    
    sage: print( C2([C((-2)^n), -1], [1]) )
    Difference definable sequence of order 1 and degree 1 with coefficients:
    > c0 (n) : C-finite sequence c0(n): (2)*c0(n) + (1)*c0(n+1) = 0 and c0(0)=1
    > c1 (n) : C-finite sequence c1(n)=-1
    and initial values a(0)=1
    
    sage: c = C(2^n+(-1)^n)
    sage: d = C(2^n+1)
    sage: a = C2([c,-1], [1], name="a")
    sage: b = C2([d,-1], [1], name="b")
    
    sage: a[:10]
    [1, 2, 2, 10, 70, 1190, 36890, 2397850, 304526950, 78263426150]
    sage: b[:10]
    [1, 2, 6, 30, 270, 4590, 151470, 9845550, 1270075950, 326409519150]
    sage: a_plus_b = a+b
    sage: a_plus_b[:10]
    [2, 4, 8, 40, 340, 5780, 188360, 12243400, 1574602900, 404672945300]
    sage: a_plus_b.order(), a_plus_b.degree()
    (3, 8)
    
    sage: a_times_b = a*b
    sage: a_times_b[:7]
    [1, 4, 12, 300, 18900, 5462100, 5587728300]
    
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

from __future__ import absolute_import, division, print_function

from copy import copy
from math import factorial, log2
from operator import pow
from datetime import datetime
import logging

from numpy import random

from sage.repl.rich_output.pretty_print import show
from sage.arith.all import gcd
from sage.arith.functions import lcm
from sage.calculus.var import var
from sage.functions.other import floor, ceil, binomial
from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.matrix.constructor import Matrix
from sage.matrix.special import identity_matrix
from sage.matrix.special import block_diagonal_matrix
from sage.misc.all import prod, randint
from sage.misc.sage_input import sage_input
from sage.rings.all import ZZ, QQ, CC
from sage.rings.qqbar import QQbar
from sage.rings.ring import CommutativeAlgebra
from sage.rings.ring import CommutativeRing
from sage.structure.element import CommutativeAlgebraElement
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.element import RingElement
from sage.categories.fields import Fields
from sage.categories.algebras import Algebras
from sage.categories.commutative_algebras import CommutativeAlgebras
from sage.categories.commutative_rings import CommutativeRings
from sage.categories.rings import Rings
from sage.structure.unique_representation import UniqueRepresentation
from sage.combinat.combinat import stirling_number2
from sage.arith.misc import falling_factorial
from sage.functions.other import binomial
from sage.symbolic.ring import SR


from sage.repl.rich_output.pretty_print import show

from .SequenceRingOfFraction import SequenceRingOfFraction
from .SequenceRingOfFraction import FractionSequence
from .CFiniteSequenceRing import CFiniteSequence
from .FunctionalEquation import FunctionalEquation
from .LinearSolveSequence import LinearSolveSequence
from .utility import eval_matrix, eval_vector, shift_matrix, shift_vector, \
                     log, matrix_subsequence

from .RecurrenceSequenceRing import RecurrenceSequenceElement
from .RecurrenceSequenceRing import RecurrenceSequenceRing
from .RecurrenceSequenceRing import RecurrenceSequenceRingFunctor

####################################################################################################

class DifferenceDefinableSequence(RecurrenceSequenceElement):
    r"""
    A difference definable sequence, i.e. a sequence where every term can be determined by a linear recurrence
    with coefficients coming from a difference ring and finitely many initial values. We assume that this
    recurrence holds for all values and that the leading coefficient is non-zero for all n (this is not checked).
    """

    log = logging.getLogger("DDS")

    def __init__(self, parent, coefficients, initial_values, name = "a", 
                 is_gen = False, construct=False, cache=True):
        r"""
        Construct a difference definable sequence `a(n)` with recurrence

        .. MATH::
            c_0(n) a(n) + \dots + c_r(n) a(n+r) = 0 \text{ for all } n \geq 0

        from given list of coefficients `c_0, ... , c_r` and given list of
        initial values `a(0), ..., a(r-1)`.

        .. NOTE::
        
            We assume that the leading coefficient `c_r` does not contain any 
            zero terms. If it does, this might yield problems in later 
            computations.
        
        INPUT:

        - ``parent`` -- a ``DifferenceDefinableSequenceRing``
        - ``coefficients`` -- the coefficients of the recurrence
        - ``initial_values`` -- a list of initial values, determining the 
          sequence with at least order of the recurrence many values
        - ``name`` (default "a") -- a name for the sequence

        OUTPUT:

        A sequence determined by the given recurrence and 
        initial values.
        
        EXAMPLES::
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C) 
            sage: n = var("n")
            
            sage: a = C2([C((-2)^n), -1], [1]) 
            sage: print(a)
            Difference definable sequence of order 1 and degree 1 with coefficients:
            > c0 (n) : C-finite sequence c0(n): (2)*c0(n) + (1)*c0(n+1) = 0 and c0(0)=1
            > c1 (n) : C-finite sequence c1(n)=-1
            and initial values a(0)=1
            sage: a[:10]
            [1, 1, -2, -8, 64, 1024, -32768, -2097152, 268435456, 68719476736]
            
        """  
        RecurrenceSequenceElement.__init__(self, parent, coefficients, 
                                           initial_values, name, is_gen, 
                                           construct, cache)
        
        base = parent.base()
        coefficients = [base(coeff) for coeff in coefficients]
        coefficients = self._remove_leading_zeros(coefficients)
        self._coefficients = coefficients
        self._degree = max([coeff.order() for coeff in self._coefficients])

        self._create_values(20)

#tests
    def compress(self) :
        r"""
        Tries to compress the recurrence defining the sequence by 
        compressing all coefficients.
        
        OUTPUT:
        
        The same sequence with possible compressed coefficients in the
        recurrence.
        
        """
        coefficients = self.coefficients()
        compressed_coefficients = [coeff.compress() for coeff in coefficients]
        return self.parent()(compressed_coefficients, self.initial_values(), 
                             name=self._name)
        
    def clear_common_factor(self) :
        r"""
        Tries to clear a common factor in the coefficients.
        
        OUTPUT:
        
        The same sequence with possible smaller coefficients in the
        recurrence.
        
        """
        clear_divisor = LinearSolveSequence.clear_divisor
        _, new_coefficients = clear_divisor(self.coefficients())

        return self.parent()(new_coefficients, self.initial_values(), 
                             name=self._name)
        
    def _create_values(self, n) :
        r"""
        Create values [self[0],...,self[n]] in self._values
        """
        pre_computed = len(self._values) - 1

        for i in range(1+pre_computed-self.order(), n+1-self.order()) :
            new_value = sum(coeff[i]*self._values[i+j] for j, coeff in enumerate(self.coefficients()[:-1]))
            self._values.append(-1/(self.coefficients()[-1][i])*new_value)

#conversion

    def _test_conversion_(self):
        r"""
        Test whether a conversion of ``self`` into an int/float/long/... is possible;
        i.e. whether the sequence is constant or not.

        OUTPUT:

        If ``self`` is constant, i.e. there exists a `k` in K, such that self(n) = k for all n in NN,
        then this value `k` is returned. If ``self`` is not constant ``None`` is returned.
        """
        raise NotImplementedError

    def __float__(self):
        r"""
        Tries to convert ``self`` into a float.
        This is possible iff ``self`` represents a constant sequence for some
        constant values in ``QQ``.
        If the conversion is not possible an error message is displayed.
        """
        i = self._test_conversion_()
        if i is not None:
            return float(i)

        raise TypeError("No conversion possible")

    def __int__(self):
        r"""
        Tries to convert ``self`` into an integer.
        This is possible iff ``self`` represents a constant sequence for some
        constant value in ``ZZ``.
        If the conversion is not possible an error message is displayed.
        """
        i = self._test_conversion_()
        if i is not None and i in ZZ:
            return int(i)

        raise TypeError("No conversion possible")

    def _integer_(self, ZZ):
        r"""
        Tries to convert ``self`` into a Sage integer.
        This is possible iff ``self`` represents a constant sequence for some
        constant value in ``ZZ``.
        If the conversion is not possible an error message is displayed.
        """
        return ZZ(int(self))

    def _rational_(self):
        r"""
        Tries to convert ``self`` into a Sage rational.
        This is possible iff ``self`` represents a constant sequence for some
        constant value in ``QQ``.
        If the conversion is not possible an error message is displayed.
        """
        i = self._test_conversion_()
        if i is not None and i in QQ:
            return QQ(i)

        raise TypeError("No conversion possible")

    def __long__(self):
        r"""
        Tries to convert ``self`` into a long integer.
        This is possible iff ``self`` represents a constant sequence for some
        constant value in ``ZZ``.
        If the conversion is not possible an error message is displayed.
        """
        i = self._test_conversion_()
        if i is not None and i in ZZ:
            return long(i)

        raise TypeError("No conversion possible")

    def _remove_trailing_zeros(self, container) :
        r"""
        Removes all the trailing zeros from a given container over some ring
        and returns this container.
        """
        if not container :
            return container

        R = container[0].parent()
        for i in range(len(container)) :
            if container[i] != R(0) :
                return container[i:]

        return container

    def _remove_leading_zeros(self, container) :
        r"""
        Removes all the zeros from the back of a given container over some ring
        and returns this container.
        """
        if not container :
            return container

        new_container = self._remove_trailing_zeros(list(reversed(container)))
        if not new_container :
            return new_container
        else :
            return list(reversed(new_container))

    def _remove_zeros(self, container) :
        r"""
        Removes all the zeros from the back and the front of a given container
        over some ring and returns this container, together with indices i,j
        such that the new container is ``container``[i:j] (j excluded)
        """
        if not container :
            return container, 0, 0

        container_wo_leading_0 = self._remove_leading_zeros(container)
        j = len(container_wo_leading_0)
        container_wo_trailing_0 = self._remove_trailing_zeros(container_wo_leading_0)
        i = len(container_wo_leading_0)-len(container_wo_trailing_0)
        return container_wo_trailing_0, i, j

    def _canonical_recurrence(self, coefficients) :
        r"""
        Removes all the zeros from the back and the front of list of coefficients
        over some ring and returns these coefficients

        PROBLEM: SUPPOSE WE HAVE SEQUENCE a_n=(1,0,0,0,...) AND RECURRENCE
        a_{n+1} = 0 for all n >= 0
        IF WE CHANGE IT TO a_n = 0, IT DOES NOT HOLD FOR INITIAL VALUES
        HENCE WE MIGHT HAVE TO ADD MORE INITIAL VALUES
        """
        new_coefficients, i, j = self._remove_zeros(coefficients)
        leading_coeff = new_coefficients[-1]
        if 0 in leading_coeff[-i:5] :
            raise ValueError("The shifted leading coefficient contains a 0")

        new_coefficients_shifted = [coeff.shift(-i) for coeff in new_coefficients]
        return new_coefficients_shifted

#representation

    def __hash__(self):
        r"""
        """
        return super().__hash__()

    def _repr_(self, name=None):
        r"""
        Produces a string representation of the sequence.
        
        INPUT:
        
        - ``name`` (optional) -- a string used as the name of the sequence;
          if not given, ``self.name()`` is used.
        
        OUTPUT:
        
        A string representation of the sequence consisting of the recurrence
        and enough initial values to uniquely define the sequence.
        
        EXAMPLES::
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C) 
            sage: n = var("n")
            
            sage: a = C2([C((-2)^n), -1], [1]) 
            sage: print(a)
            Difference definable sequence of order 1 and degree 1 with coefficients:
            > c0 (n) : C-finite sequence c0(n): (2)*c0(n) + (1)*c0(n+1) = 0 and c0(0)=1
            > c1 (n) : C-finite sequence c1(n)=-1
            and initial values a(0)=1
            
        """
        if name==None :
            name = self._name
        r = "Difference definable sequence of order {}".format(self.order())
        r += " and degree {} with coefficients:\n".format(self.degree())
        for i in range(self.order()+1) :
            r += f" > c{i} (n) : " + self._coefficients[i]._repr_(name=f"c{i}") + "\n"
        init_repr = [f"{self._name}({i})={val}" for i, val in enumerate(self._initial_values)]
        r += "and initial values " + " , ".join(init_repr)
        return r

    def _latex_(self, name=None):
        r"""
        Creates a latex representation of the sequence.
        This is done by creating the latex representation of the closed form
        of the coefficients.
        
        OUTPUT: 
        
        A latex representation showing the recurrence and the initial values 
        of the sequence.
        
        EXAMPLES::

            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C) 
            sage: n = var("n")
            
            sage: a = C2([C((-2)^n), -1], [1]) 
            sage: print(latex(a))
            \left(\left(-2\right)^{n}\right)\cdot a(n) + \left(-1\right) \cdot a(n+1) = 0 \quad a(0)=1
            
        """
        if name==None :
            name = self._name
        coeffs = [(index, coeff) for index, coeff \
                                 in enumerate(self.coefficients()[1:], 1) \
                                 if not coeff.is_zero()]
        coeffs_repr = [r"\left({}\right) \cdot {}(n+{})".format(coeff._latex_(), name, i) \
                                 for i, coeff in coeffs]
        init_repr = ["{}({})={}".format(name, i, val) \
                                 for i, val in enumerate(self._initial_values)]
        r = r"\left({}\right)\cdot {}(n)".format(self.coefficients()[0]._latex_(), name)
        if self.order() > 0 :
            r += " + " + " + ".join(coeffs_repr) + " = 0"
        elif self.order() == 0 :
            r += " = 0"
        r += r" \quad " + " , ".join(init_repr)

        return r

# helper for arithmetic
    def _companion_sum(self, right) :
        r"""
        Computes the companion matrix of ``self+right``.
        """
        comp_left = self.companion_matrix()
        comp_right = right.companion_matrix()
        return block_diagonal_matrix(comp_left, comp_right)

    def _companion_product(self, right) :
        r"""
        Computes the companion matrix of ``self*right``.
        """
        comp_left = self.companion_matrix()
        comp_right = right.companion_matrix()
        return comp_left.tensor_product(comp_right)
    
    def _companion_subsequence(self, u) :
        r"""
        Computes the companion matrix of ``self`` for the subsequence 
        ``self[n*u]``. This is given by ``Ma[u*n]*...*Ma[u*n+u-1]`` where 
        ``Ma`` denotes the companion matrix of ``seq``.
        """
        comp = self.companion_matrix()
        return prod(matrix_subsequence(comp, u, v) for v in range(u))

# arithmetic
    def _add_(self, right, *args, **kwargs):
        r"""
        Return the termwise sum of ``self`` and ``right``.
        
        INPUTS:

        - ``right`` -- a sequence over the same ring as ``self``.
        
        OUTPUTS: 
        
        The addition of ``self`` with ``right``.
        
        EXAMPLES:: 

            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C)
            sage: n = var("n")

            sage: c = C(2^n+(-1)^n)
            sage: d = C(2^n+1)
            sage: a = C2([c,-1], [1], name="a")
            sage: b = C2([d,-1], [1], name="b")
            
            sage: a[:10]
            [1, 2, 2, 10, 70, 1190, 36890, 2397850, 304526950, 78263426150]
            sage: b[:10]
            [1, 2, 6, 30, 270, 4590, 151470, 9845550, 1270075950, 326409519150]
            sage: a_plus_b = a+b
            sage: a_plus_b[:10]
            [2, 4, 8, 40, 340, 5780, 188360, 12243400, 1574602900, 404672945300]
            
        """
        if self.__is_zero__():
            return right
        if right.__is_zero__():
            return self
                
        time_start = datetime.now()
        
        comp = self._companion_sum(right)
        QR = comp.base_ring()
        v0 = vector(QR, comp.nrows()) # first column of matrix
        v0[0] = QR(1); v0[self.order()] = QR(1)

        rec = self.parent()._compute_recurrence(comp, v0, *args, **kwargs)
        r = len(rec)
        initial_values = [sum(x) for x in zip(self[:r], right[:r])]

        time_end = datetime.now()
        msg = "Addition computed"
        log(self, msg, 0, time_start, time_end)

        return self.parent()(rec, initial_values).clear_common_factor()

    def _mul_(self, right, *args, **kwargs):
        r"""
        Return the product of ``self`` and ``right``. The result is the 
        termwise product (Hadamard product) of ``self`` and 
        ``right``.
        
        INPUTS:

        - ``right`` --  a sequence over the same ring as ``self``.
        
        OUTPUTS: 
        
        The product of ``self`` with ``right``.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C)
            sage: n = var("n")

            sage: c = C(2^n+(-1)^n)
            sage: d = C(2^n+1)
            sage: a = C2([c,-1], [1], name="a")
            sage: b = C2([d,-1], [1], name="b")
            
            sage: a[:10]
            [1, 2, 2, 10, 70, 1190, 36890, 2397850, 304526950, 78263426150]
            sage: b[:10]
            [1, 2, 6, 30, 270, 4590, 151470, 9845550, 1270075950, 326409519150]
            sage: a_times_b = a*b
            sage: a_times_b[:7]
            [1, 4, 12, 300, 18900, 5462100, 5587728300]
            
        """
        if self.__is_zero__() or right.__is_zero__():
            return self.parent().zero()
        
        time_start = datetime.now()
        
        comp = self._companion_product(right)
        QR = comp.base_ring()
        v0 = vector(QR, comp.nrows()) # first column of matrix
        v0[0] = QR(1)

        rec = self.parent()._compute_recurrence(comp, v0, *args, **kwargs)
        r = len(rec)
        initial_values = [prod(x) for x in zip(self[:r], right[:r])]

        time_end = datetime.now()
        msg = "Product computed"
        log(self, msg, 0, time_start, time_end)

        element_constructor = self.parent()._element_constructor_
        return element_constructor(rec, initial_values).clear_common_factor()

    def subsequence(self, u, v=0, *args, **kwargs):
        r"""
        Returns the sequence `c(n u + v)`.

        INPUT:

        - ``u`` -- a natural number
        - ``v`` (default: ``0``) -- a natural number

        OUTPUT:
        
        The sequence `c(n u + v)`.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C)

            sage: f = C([1,1,-1], [0,1]) # Fibonacci numbers
            sage: c0 = f.subsequence(2, 3)*(f.subsequence(2, 1)*f.subsequence(2, 3)-f.subsequence(2, 2)^2)
            sage: c1 = f.subsequence(2, 2)*(f.subsequence(2, 3)+f.subsequence(2, 1))
            sage: c2 = -f.subsequence(2, 1)
            sage: fs = C2([c0,c1,c2],[0, 1]) # f(n^2)
            
            sage: fs_subs = fs.subsequence(2, 1) # long time
            sage: fs_subs.order() # long time
            2
            sage: fs_subs[:5] # long time
            [1, 34, 75025, 7778742049, 37889062373143906]
            
        """
        if v != 0 :
            return self.shift(v).subsequence(u, *args, **kwargs)
        
        time_start = datetime.now()
        
        comp = self._companion_subsequence(u)
        QR = comp.base_ring()
        v0 = vector(QR, comp.nrows()) # first column of matrix
        v0[0] = QR(1)

        rec = self.parent()._compute_recurrence(comp, v0, *args, **kwargs)
        r = len(rec)        
        initial_values = [self[u*n] for n in range(r)]
        
        time_end = datetime.now()
        msg = "Subsequence computed"
        log(self, msg, 0, time_start, time_end)
        
        element_constructor = self.parent()._element_constructor_
        return element_constructor(rec, initial_values).clear_common_factor()
    
    def multiple(self, d) :
        r"""
        Computes the sequence `c(\left \lfloor{n/d}\right \rfloor)`.

        INPUT:

        - ``d`` -- a natural number

        OUTPUT:
        
        The sequence `c(\left \lfloor{n/d}\right \rfloor)`.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C)
            sage: n = var("n")

            sage: a = C2([C(2^n)+1, -1], [1]) 
            sage: a[:10]
            [1, 2, 6, 30, 270, 4590, 151470, 9845550, 1270075950, 326409519150]
            
            sage: a2 = a.multiple(2)
            sage: a2.order()
            2
            sage: a2[:10]
            [1, 1, 2, 2, 6, 6, 30, 30, 270, 270]
            
        """
        coeffs_multiple = []
        for coeff in self.coefficients() :
            coeffs_multiple += [coeff.subsequence(1/d)]
            coeffs_multiple += (d-1)*[self.base().zero()] 
        coeffs_multiple = coeffs_multiple[:-(d-1)]
        
        order = len(coeffs_multiple)
        initial_values = [self[floor(n/d)] for n in range(order)]
        element_constructor = self.parent()._element_constructor_
        return element_constructor(coeffs_multiple, initial_values)

    @staticmethod
    def _split_exp_term(term):
        r"""
            Given a symbolic expression of the form c*n^i*gamma^n
            returns the triple (c, i, gamma).
        """
        #print("term = {}".format(term))
        
        # is constant
        if term.is_constant() :
            return (term, 0, 1)
        
        n = term.variables()[0]
        if term == n :
            return (1, 1, 1)
        operands = term.operands()
        if len(operands) == 2 :
            if operands[0] == n and operands[1].is_constant() :
                # is of form n^i
                return (1, int(operands[1]), 1)
            elif operands[1] == n and operands[0].is_constant() :
                # is of form gamma^n
                return (1, 0, operands[0])

        triple = [1, 0, 1]
        for operand in operands :
            c, i, gamma = DifferenceDefinableSequence._split_exp_term(operand)
            triple[0] = triple[0]*c
            triple[1] = triple[1]+i
            triple[2] = triple[2]*gamma

        return tuple(triple)

#base ring related functions
    def base(self):
        r"""
        OUTPUT:
        
        The base of the parent of ``self``, c.f. 
        :meth:`DifferenceDefinableSequenceRing.base`.

        """
        return self.parent().base()

#part extraction functions

    def coefficients(self):
        r"""
        Returns the list of polynomial coefficients of the recurrence of 
        ``self`` in the format ``[c0,...,cr]`` representing the recurrence 
        
        .. MATH::
            c_0(n) a(n) + \dots + c_r(n) a(n+r) = 0.
            
        OUTPUT:
        
        The coefficients of the recurrence of the sequence.
        
        EXAMPLES::
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C)
            sage: n = var("n")

            sage: a = C2([C(2^n)+1, -1], [1]) 
            sage: a.coefficients()
            [C-finite sequence a(n): (2)*a(n) + (-3)*a(n+1) + (1)*a(n+2) = 0 
            and a(0)=2 , a(1)=3,
            C-finite sequence a(n)=-1]
            
        """
        return self._coefficients

    def degree(self):
        r"""
        OUTPUT:
        
        The degree of the recurrence, i.e. the maximal
        order of the coefficients
        
        EXAMPLES::
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C)
            sage: n = var("n")

            sage: a = C2([C(2^n)+1, -1], [1]) 
            sage: (C(2^n)+1).order()
            2
            sage: a.degree()
            2
            
        """
        return self._degree

    def companion_matrix(self):
        r"""
        Returns the `r \times r` companion matrix 
        
        .. MATH::
            \begin{pmatrix}
                0 & 0 & \dots & 0 & -c_0/c_r \\
                1 & 0 & \dots & 0 & -c_1/c_r \\
                0 & 1 & \dots & 0 & -c_2/c_r \\
                \vdots & \vdots & \ddots & \vdots & \vdots \\
                0 & 0 & \dots & 1 & -c_{r-1}/c_r
            \end{pmatrix} .
            
        of ``self`` with entries in the
        ``SequenceRingOfFraction`` of the base. 
        
        OUTPUT:
        
        The companion matrix.
        
        EXAMPLES::
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C)
            sage: n = var("n")

            sage: a = C2([C(2^n)+1, -1], [1]) 
            sage: a.companion_matrix()
            [Fraction sequence:
            > Numerator: C-finite sequence a(n): (2)*a(n) + (-3)*a(n+1) + (1)*a(n+2) = 0 and a(0)=2 , a(1)=3
            > Denominator: C-finite sequence a(n)=1
            ]
            
        """
        R = SequenceRingOfFraction(self.base())
        leading_coeff = self.leading_coefficient()
        coefficients = self.coefficients()
        M_comp = matrix(R, self.order())
        for i in range(self.order()) :
            for j in range(self.order()) :
                if j == self.order()-1 :
                    M_comp[i,j] = -R(coefficients[i], leading_coeff)
                elif i==j+1 :
                    M_comp[i,j] = R(1)
                else :
                    M_comp[i,j] = R(0)
        return M_comp


####################################################################################################

class DifferenceDefinableSequenceRing(RecurrenceSequenceRing):
    r"""
    A Ring of difference definable sequences over a ring of sequences.
    """

    Element = DifferenceDefinableSequence
    log = logging.getLogger("DDR")

# constructor

    def __init__(self, base, guess=False, verified=True, name=None, 
                 element_class=None, category=None):
        r"""
        Constructor for a difference definable sequence ring.

        INPUT:

        - ``base`` -- a difference ring, e.g. the ring of C-finite sequences
          or D-finite sequences
        - ``guess`` (default: ``False``) -- if ``True``, the linear systems
          that arise in the computations are solved with guessing (if possible)
        - ``verified`` (default: ``True``) -- if ``True``, the solutions of
          the linear system are verified to be correct. If ``False`` there can
          be rare cases, for which the solutions are wrong.

        OUTPUT:

        A ring of difference definable sequences over ``base``.

        EXAMPLES::
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: DifferenceDefinableSequenceRing(C) 
            Ring of difference definable sequences over Ring of C-finite 
            sequences over Rational Field

        """
        if base not in Rings() :
            raise TypeError("Difference definable sequence ring is defined over a ring.")

        self._base_diff_ring = base
        self._guess = guess 
        self._verified = verified
        RecurrenceSequenceRing.__init__(self, base.base_ring())

    def _element_constructor_(self, x, y=None,  name="a", check=True, 
                              is_gen = False, construct=False, **kwds):
        r"""
        Tries to construct a sequence `a(n)`.
        
        This is possible if:

        - ``x`` is already a difference definable sequence.
        - ``x`` is a list of ``base`` elements and ``y`` is a list of field 
          elements. Then ``x`` is interpreted as the coefficients of the 
          sequence and ``y`` as the initial 
          values of the sequence, i.e. `a(0), ..., a(r-1)`.
        - ``x`` is in ``base``.
        - ``x`` can be converted into a field element. Then it is interpreted 
          as the constant sequence `(x)_{n \in \mathbb{N}}`
        
        EXAMPLES::
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C) 
            sage: n = var("n")
            
            sage: c = C(2^n+(-1)^n)
            sage: d = C(2^n+1)
            sage: a = C2([c,-d], [1], name="a")
            sage: print(latex(a))
            \left(2^{n} + \left(-1\right)^{n}\right)\cdot a(n) + \left(-2^{n} - 
            1\right) \cdot a(n+1) = 0 \quad a(0)=1
            sage: a2 = C2(a)
            
            sage: b = C2(c)
            sage: print(b)
            Difference definable sequence of order 2 and degree 1 with 
            coefficients:
            > c0 (n) : C-finite sequence c0(n)=2
            > c1 (n) : C-finite sequence c1(n)=1
            > c2 (n) : C-finite sequence c2(n)=-1
            and initial values a(0)=2 , a(1)=1
            
        """   
        try :
            return super()._element_constructor_(x, y, name=name)
        except NotImplementedError:
            pass
        
        K = self.base_ring()
        R = self.base()
        if isinstance(x, list) and isinstance(y, list) :
            return self.Element(self, x, y)
        elif x in K :
            return self._create_constant_sequence(x)
        elif x in R : # check whether R is sequence ring
            try :
                coeffs_R = [R(coeff) for coeff in x.coefficients()]
                return self.Element(self, coeffs_R, x.initial_values())
            except Exception:
                raise NotImplementedError("Conversions not implemented!")
        else :
            raise NotImplementedError("Conversions not implemented!")

    def _coerce_map_from_(self, P):
        r"""
        """
        if self.base().has_coerce_map_from(P) :
            return True
        else :
            return self.base_ring().has_coerce_map_from(P)

    def _repr_(self):
        r"""
        OUTPUT:
        
        A string representation of the sequence ring.
        
        EXAMPLES::
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: DifferenceDefinableSequenceRing(C) 
            Ring of difference definable sequences over Ring of C-finite 
            sequences over Rational Field

        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        r = self._cached_repr = "Ring of difference definable sequences over " + self.base()._repr_()
        return r

    def _latex_(self):
        r"""
        OUTPUT:
        
        A latex representation of the sequence ring.
        
        EXAMPLES::

            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: print(latex(DifferenceDefinableSequenceRing(C)))
            \mathcal{D_\sigma}(\mathcal{C}(\Bold{Q}))

        """
        return r"\mathcal{D_\sigma}(" + self.base()._latex_() + ")"

    def with_guessing(self) :
        r"""
        OUTPUT:
        
        ``True`` if solutions of linear systems are obtained via
        guessing (if possible). ``False`` if they are 
        always computed using a generalized inverse approach.
        """
        return self._guess

    def verified(self) :
        r"""
        OUTPUT:
        
        ``True`` if solutions of linear systems are verified to be
        correct. ``False`` if they are not verified.
        """
        return self._verified

    def base(self):
        r"""
        OUTPUT:
        
        Return the base over which the sequence ring is defined.
        
        EXAMPLES::
            
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C) 
            sage: C2.base()==C
            True
            
        """
        return self._base_diff_ring

    def random_element(self, order=2, degree=1, *args, **kwds):
        r"""
        Return a random difference definable sequence.

        INPUT:

        -``order`` (default: ``2``) -- the order of the recurrence of the random
        difference definable sequence
        -``degree`` (default ``1``) -- the order of the coefficient sequences

        OUTPUT:

        A difference definable sequence with a random recurrence of order 
        ``order``, random coefficient sequences of order ``degree``
        and random initial values constisting of integers between ``-100``
        and ``100``.
        
        .. NOTE::
        
            It is only guaranteed that the leading coefficient of the recurrence
            has no zeros in the first ``20`` values.
        
        EXAMPLES::
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C) 
            
            sage: C2.random_element(order=2).order() 
            2
            
        """
        coefficients = self._random_coefficients(order+1, degree, *args, **kwds)
        initial_values = [randint(-100, 100) for i in range(order)]
        element_constructor = self._element_constructor_
        return element_constructor(coefficients, initial_values)

    def random_monic_element(self, order=2, degree=1, *args, **kwds):
        r"""
        Return a random difference definable sequence where the leading coefficient
        of the recurrence is ``1``.

        INPUT:

        -``order`` (default: ``2``) -- the order of the recurrence of the random
        difference definable sequence
        -``degree`` (default: ``1``) -- the order of the coefficient sequences

        OUTPUT:

        A difference definable sequence with a random recurrence of order ``order``,
        random coefficient sequences of order ``degree``
        and random initial values constisting of integers between ``-100`` 
        and ``100`` where the leading coefficient of the recurrence is ``1``.
        
        EXAMPLES::
        
            sage: from rec_sequences.DifferenceDefinableSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = DifferenceDefinableSequenceRing(C) 
            
            sage: a = C2.random_monic_element(order=2)
            sage: a.leading_coefficient().is_one() 
            True
            
        """
        coefficients = self._random_coefficients(order, degree, *args, **kwds)
        coefficients = coefficients + [self.base().one()]
        initial_values = [randint(-100, 100) for i in range(order)]
        element_constructor = self._element_constructor_
        return element_constructor(coefficients, initial_values)

    def _random_coefficients(self, number, order, *args, **kwds) :
        r"""
        Return random coefficients from the base.

        INPUT:

        -``number`` -- the number of generated elements
        -``order`` -- the order of the generated elements

        OUTPUT:

        A list of ``number`` many random elements from the base such that
        the first element is not 0 and the last element is a unit.
        """
        if number < 1 :
            raise ValueError("Number needs to be at least 1.")
        if number == 1 :
            return [self._random_unital_base_element(order, *args, **kwds)]
        else :
            trailing = self._random_non_zero_base_element(order, *args, **kwds)
            leading  = self._random_unital_base_element(order, *args, **kwds)
            R = self.base()
            others = [R.random_element(order, *args, **kwds) for i in range(number-2)]
            return [trailing] + others + [leading]

    def _random_non_zero_base_element(self, order, *args, **kwds) :
        r"""
        Return random non-zero ring element from the base of given order.
        """
        R = self.base()
        element = R.random_element(order, *args, **kwds)
        while element == R.zero() :
            element = R.random_element(*args, **kwds)
        return element

    def _random_unital_base_element(self, order, *args, **kwds) :
        r"""
        Return random unital ring element from the base of given order.
        This only uses the heuristic that it checks the first 20 values
        and determines a sequence to be unital if all of those are non-zero.
        """
        R = self.base()
        K = self.base_ring()
        element = R.random_element(order, *args, **kwds)
        while K.zero() in element[:20] :
            element = R.random_element(*args, **kwds)
        return element

    def change_base_ring(self,R):
        r"""
        Return a copy of ``self`` but with the base ring ``R``.
        
        OUTPUT:
        
        A differene definable sequence ring with base ``R``.
        """
        if R is self.base():
            return self
        else:
            D = DifferenceDefinableSequenceRing(R, guess=self.with_guessing(),
                                                verified=self.verified())
            return D
        
# helper for arithmetic
    def _compute_recurrence(self, A, v0, *args, **kwargs):
        r"""
        Computes a recurrence using linear systems which arise by linear system
        which get larger iteratively. The first column of the system is given 
        by v0. The column i+1 is iteratively computed by A*shift(vi). We try to
        solve systems A*x = -vi for increasing i until the system has a 
        solution. Then, this solution is returned.
        Additional optional parameters :
            guess, check_entries, verified
        """
        # so many entries are checked initially to search for right order
        # of the sum 
        check_entries = kwargs.get("check_entries", 100)
        verified = kwargs.get("verified", True)  
        
        DDR = DifferenceDefinableSequenceRing
        check_linear_system = DDR._check_linear_system_numerically

        QR = A.base_ring()

        time_ansatz_size = datetime.now()
        
        # create linear systems until Rouche-Capelli theorem
        # shows that it is solvable for some initial values n
        rhs = A*shift_vector(v0)
        M = matrix(QR, 1, A.nrows(), [v0]).transpose()
        while not check_linear_system(M, -rhs, check_entries) :
            M = M.augment(rhs)
            rhs = A*shift_vector(rhs)

        time_ansatz_size_done = datetime.now()
        order = M.ncols()
        msg = f"order {order} computed"
        log(self, msg, 0, time_ansatz_size, time_ansatz_size_done)

        try :
            rec = LinearSolveSequence.solve(M, -rhs, 
                                            guess = self.with_guessing(),
                                            verified = self.verified())
            time_system_solved = datetime.now()
            log(self, "linear system solved", 0, 
                time_ansatz_size_done, time_system_solved)
            lc, rec_new = LinearSolveSequence.clear_denominators(rec, 100)
            time_denom_cleared = datetime.now()
            log(self, "denominators cleared", 0, 
                time_system_solved, time_denom_cleared)
            return list(rec_new) + [lc]
        except ValueError as e:
            # system was not big enough, have to go bigger
            # do this by checking more initial values in the next step
            new_entries = floor(check_entries*1.1)
            msg = f"error {str(e)} occured while solving the linear system, \
                    try increasing initial values used to {new_entries}"
            log(self, msg)
            return self.compute_recurrence(A, v0, check_entries = new_entries)
            

    @staticmethod
    def _check_linear_system_numerically(M, rhs, n1=100, n0=0) :
        r"""
        Checks whether the system M[n]*x[n]=rhs[n] has 
        a solution x[n] for every n0<=n<n1. If so, return True.
        Otherwise, return False. We use the Rouche-Capelli theorem
        at every term to decide this.
        """
        for n in range(n0,n1) :
            M_eval = eval_matrix(M, n)
            rhs_eval = eval_vector(rhs, n)
            M_aug_eval = M_eval.augment(rhs_eval)
            if M_eval.rank() != M_aug_eval.rank() :
                return False
        return True

class DifferenceDefinableSequenceRingFunctor(RecurrenceSequenceRingFunctor):
    def __init__(self):
        r"""
        Constructs a ``DifferenceDefinableSequenceRingFunctor``.
        """
        RecurrenceSequenceRingFunctor.__init__(self)

    def _apply_functor(self, x):
        return DifferenceDefinableSequenceRing(x)

    def _repr_(self):
        r"""
        Returns a string representation of the functor.
        
        OUTPUT:
        
        The string "DifferenceDefinableSequenceRing(\*)"

        """
        return "DifferenceDefinableSequenceRing(*)"
