# coding: utf-8
r"""
D-finite sequences

Sublasses :class:`rec_sequences.RecurrenceSequenceRing` and defines 
sequences satisfying a linear
recurrence equation with polynomial coefficients. Such a D-finite sequence 
`a(n)` is defined by a recurrence

.. MATH::
    p_0(n) a(n) + \dots + p_r(n) a(n+r) = 0 \text{ for all } n \geq 0

and initial values `a(0),...,a(r-1)`. 
This is just a wrapper of |UnivariateDFiniteSequence|_ from the package 
|ore_algebra|_. 
The only difference is, that we make sure that the leading coefficient
of the recurrences have no zeros by shifting the recurrence appropriately
(which might increase the order of the recurrence).

.. |UnivariateDFiniteSequence| replace:: ``UnivariateDFiniteSequence``
.. _UnivariateDFiniteSequence: http://www.algebra.uni-linz.ac.at/people/mkauers/ore_algebra/generated/ore_algebra.dfinite_function.html#ore_algebra.dfinite_function.UnivariateDFiniteSequence
.. |ore_algebra| replace:: ``ore_algebra``
.. _ore_algebra: https://github.com/mkauers/ore\_algebra

EXAMPLES::

    sage: from rec_sequences.DFiniteSequenceRing import *
    
    sage: R.<n> = PolynomialRing(QQ)
    sage: D = DFiniteSequenceRing(R) # create D-finite sequence ring over QQ
    
    sage: fac = D([n+1,-1],[1]) # define factorials
    sage: fac[:10]
    [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
    
    sage: # define harmonic numbers using guessing
    sage: harm_terms = [sum(1/i for i in range(1,k)) for k in range(1,20)]
    sage: harm = D(harm_terms)
    sage: harm
    D-finite sequence a(n): (-n - 1)*a(n) + (2*n + 3)*a(n+1) + (-n - 2)*a(n+2) 
    = 0 and a(0)=0 , a(1)=1
    
    sage: a = fac+harm
    sage: a.order(), a.degree()
    (3, 4)
    
    sage: harm.is_eventually_positive() # harm. numbers positive from term 1 on
    1
    
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

import pprint
import logging

from copy import copy
from math import factorial
from operator import pow

from numpy import random

from sage.arith.all import gcd
from sage.arith.functions import lcm
from sage.calculus.var import var
from sage.calculus.predefined import x as xSymb
from sage.functions.other import floor, ceil, binomial
from sage.matrix.constructor import matrix
from sage.matrix.special import identity_matrix
from sage.misc.all import prod, randint
from sage.misc.flatten import flatten
from sage.rings.all import ZZ, QQ, CC
from sage.modules.free_module_element import free_module_element as vector
from sage.symbolic.ring import SR
from sage.rings.cfinite_sequence import CFiniteSequences as SageCFiniteSequences
from sage.rings.ring import CommutativeAlgebra
from sage.structure.element import CommutativeAlgebraElement
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.fraction_field import FractionField, FractionField_generic
from sage.structure.element import RingElement
from sage.categories.pushout import ConstructionFunctor
from sage.categories.fields import Fields
from sage.categories.algebras import Algebras
from sage.categories.commutative_algebras import CommutativeAlgebras
from sage.categories.commutative_rings import CommutativeRings
from sage.structure.unique_representation import UniqueRepresentation
from sage.misc.inherit_comparison import InheritComparisonClasscallMetaclass
from sage.interfaces.qepcad import qepcad_formula, qepcad, qformula
from sage.functions.other import floor

from itertools import combinations

from ore_algebra import OreAlgebra
from ore_algebra.guessing import guess
from ore_algebra.dfinite_function import UnivariateDFiniteSequence
from ore_algebra.dfinite_function import DFiniteFunctionRing

from .utility import TimeoutError, timeout, is_root_of_unity, split_list, log, \
                     shift_rat_vector
from .ZeroPattern import ZeroPattern
from .SignPattern import SignPattern
from .RecurrenceSequenceRing import RecurrenceSequenceElement
from .RecurrenceSequenceRing import RecurrenceSequenceRing
from .RecurrenceSequenceRing import RecurrenceSequenceRingFunctor

####################################################################################################


class DFiniteSequence(RecurrenceSequenceElement):
    r"""
    A D-finite sequence, i.e. a sequence where every term can be determined by a linear recurrence
    with polynomial coefficients and finitely many initial values. We assume that this recurrence
    holds for all values.
    """
    log = logging.getLogger("DFin")

    def __init__(self, parent, coefficients, initial_values, name = "a", 
                 is_gen = False, construct=False, cache=True):
        r"""
        Construct a D-finite sequence `a(n)` with recurrence

        .. MATH::
            p_0(n) a(n) + \dots + p_r(n) a(n+r) = 0 \text{ for all } n \geq 0

        from given list of coefficients `p_0, ... , p_r` and given list of
        initial values `a(0), ..., a(r-1)`.

        INPUT:

        - ``parent`` -- a ``DFiniteSequenceRing``
        - ``coefficients`` -- the coefficients of the recurrence
        - ``initial_values`` -- a list of initial values, determining the 
          sequence with at least order of the recurrence many values
        - ``name`` (default "a") -- a name for the sequence

        OUTPUT:

        A D-finite sequence determined by the given recurrence and 
        initial values.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: fac
            D-finite sequence a(n): (n + 1)*a(n) + (-1)*a(n+1) = 0 and a(0)=1
            
        """        
        # shift recurrence such that leading coefficient has
        # no zeros anymore
        find_max_zero = DFiniteSequence._largest_zeros_of_polynomial
        poly_ring = parent.associated_ore_algebra().base_ring()
        max_zero = find_max_zero(poly_ring(coefficients[-1]))
        recurrence = parent.associated_ore_algebra()(coefficients)
        if max_zero >= 0 :
            shift_op = parent.associated_ore_algebra().gen()
            recurrence = shift_op**(max_zero+1)*recurrence
            coefficients = recurrence.list()
            
        parent_ore = parent.get_ore_algebra_ring()
        self._dfinite_seq = UnivariateDFiniteSequence(parent_ore, recurrence, 
                                                      initial_values)
        
        RecurrenceSequenceElement.__init__(self, parent, coefficients, 
                                           initial_values, name, is_gen, 
                                           construct, cache)
        
        # workaround for ticket #31716 in sage; fixed in future versions
        if isinstance(coefficients[0].parent(), FractionField_generic) :
            new_ring = coefficients[0].parent().base()
            coefficients = [new_ring(coeff) for coeff in coefficients]
            
        coefficients = [parent.base()(coeff) for coeff in coefficients]
        if not coefficients :
            raise ValueError("No coefficients given.")

        self._recurrence = self.parent().associated_ore_algebra()(coefficients)
        self._values = self._recurrence.to_list(self.initial_values(), 20)
        
    @staticmethod
    def _largest_zeros_of_polynomial(poly) :
        r"""
        Returns the largest non-negative integer root of ``poly``
        if such a root exists and -1 otherwise.
        """
        # poly is constant
        if poly in poly.parent().base_ring() :
            if poly.is_zero() :
                raise ValueError("Zero polynomial does not have "
                                 "largest integer root")
            else :
                return -1
            
        if poly.parent().base() is QQ : # clear denominators
            common_denom = lcm([QQ(coeff).denominator() 
                                for coeff in poly.coefficients()])
            poly = poly*common_denom
        roots = poly.roots(ring=ZZ, multiplicities=False)
        if not roots or max(roots) < 0 :
            return -1
        else :
            return max(roots)
        
    def compress(self, proof=False):
        r"""
        Tries to compress the sequence ``self`` as much as
        possible by trying to find a smaller recurrence.
        
        INPUT: 
        
        - ``proof`` (default: ``False``) if ``True``, then the result is
          guaranteed to be true. Otherwise it can happen, although unlikely, 
          that the sequences are different.

        OUTPUT:

        A sequence which is equal to ``self`` but may consist of a smaller 
        operator (in terms of the order). In the worst case if no
        compression is possible ``self`` is returned.
        """
        K = self.base_ring()
        order = self.order()
        values = self[0:order+30]
        try :
            new_recurrence = self.parent().guess(values)
        except (TypeError, ValueError, AttributeError) :
            return self

        new_sequence = type(self)(self.parent(), new_recurrence.coefficients(), 
                                  values)
        if new_recurrence.order() < order :
            if proof :
                if (self._add_(-new_sequence)).__is_zero__() :
                    return new_sequence
                else :
                    return self
            return new_sequence

        return self

    def _create_values(self, n) :
        r"""
        Create values [self[0],...,self[n]] in self._values
        """
        self._values = self._dfinite_seq.expand(n+1)

#positivity

    def sign_pattern(self, bound=0, time=-1, data=100) :
        r"""
        Suppose that ``self`` has a sign pattern which is cyclic.
        We try to guess this pattern  and then verify it using
        the Gerhold-Kauers [KP10]_ method.
        
        INPUT:
        
        - ``bound`` (default: ``0``) -- length of induction hypothesis.
        - ``time`` (default: ``-1``) -- if positive, this is the maximal time 
          (in seconds) used to verify the sign patterns.
        - ``data`` (default: ``100``) -- number of terms used to guess the
          sign-pattern.
        
        OUTPUT:
        
        The sign pattern of the sequence as an object of type
        :class:`rec_sequences.SignPattern`. If no pattern could be guessed
        or this pattern could not be verified, a ``ValueError`` is raised.
                  
        EXAMPLES::

            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm.sign_pattern()
            Sign pattern: initial values <0> cycle <+>
            
            sage: n = var("n")
            sage: D(3^n).interlace(D(-2^n)).prepend([1,3,-2]).sign_pattern()
            Sign pattern: initial values <+> cycle <+->

        """
        terms = self[:data]
        pattern = SignPattern.guess(terms)
        # how many inequalities do we have to show?
        if time > 0:
            num_ineq = len(pattern.get_positive_progressions()) 
            num_ineq += len(pattern.get_negative_progressions())
            time = time/num_ineq
        
        # initial values are valid for sure, we try
        # to prove the arithmethic progressions
        pattern_false = ValueError("Guessed sign pattern is false, try to " \
                                   "increase the value of data")
        not_verified = ValueError("Could not verify the sign pattern")
        for prog in pattern.get_positive_progressions() :
            subseq = self.subsequence(prog.get_diff(), prog.get_shift())
            try :
                positive = subseq.is_positive(bound=bound, time=time, 
                                              strict=True)
                if not positive :
                    raise pattern_false
            except (ValueError, TimeoutError):
                raise not_verified
        for prog in pattern.get_negative_progressions() :
            subseq = -self.subsequence(prog.get_diff(), prog.get_shift())
            try :
                negative = subseq.is_positive(bound=bound, time=time, 
                                              strict=True)
                if not negative :
                    raise pattern_false
            except (ValueError, TimeoutError):
                raise not_verified
        for prog in pattern.get_zero_progressions() :
            subseq = self.subsequence(prog.get_diff(), prog.get_shift())
            zero = subseq.is_zero()
            if not zero :
                raise pattern_false
        return pattern
    
    def zeros(self, bound=0, time=-1, data=100) :
        r"""
        Computes the zeros of the sequence provided that the sequence
        satisfies the Skolem-Mahler-Lech-Theorem, i.e., the zeros
        consist of finitely many zeros together with a finite number of
        arithmetic progressions. The method :meth:`sign_pattern` is used
        to derive the sign pattern from which the zeros are extracted.
        All the parameters correspond to the parameters of :meth:`sign_pattern`.
        
        INPUT:
        
        - ``bound`` (default: ``0``) -- length of induction hypothesis.
        - ``time`` (default: ``-1``) -- if positive, this is the maximal time 
          (in seconds) used to verify the sign patterns.
        - ``data`` (default: ``100``) -- number of terms used to guess the
          sign-pattern.
        
        OUTPUT:
        
        The zero pattern of the sequence as an object of type
        :class:`rec_sequences.ZeroPattern`. If no pattern could be guessed
        or this pattern could not be verified, a ``ValueError`` is raised.
        
        EXAMPLES::

            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm.zeros()
            Zero pattern with finite set {0} and no arithmetic progressions
            
            sage: n = var("n")
            sage: D(3^n).interlace(D(-2^n)).prepend([1,3,-2]).zeros()
            Zero pattern with finite set {} and no arithmetic progressions
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C(10*[1,0,-1,0]).prepend([0,0,2]).zeros() # random
            Zero pattern with finite set {0, 1} and arithmetic progressions: 
            - Arithmetic progression (4*n+6)_n
            - Arithmetic progression (4*n+4)_n

        """
        sign_pattern = self.sign_pattern(bound, time, data)
        progressions = sign_pattern.get_zero_progressions()
        init_values = sign_pattern.get_initial_values()
        ex_zeros = set([i for i, val in enumerate(init_values) if val == 0])
        return ZeroPattern(ex_zeros, progressions)
        
    
    def has_no_zeros(self, bound=0, time=-1, bound_n = 5):
        r"""
        Tries to prove that the sequence has no zeros. This is done using
        different algorithms (if ``time`` is specified, ``time/2`` is used
        for each of the algorithm):
        
        1. Determine the sign pattern using :meth:`sign_pattern` and check 
           whether it contains zeros. 
        2. Uses :meth:`is_eventually_positive` to show 
           that the squared sequence is positive.
        
        INPUT:
        
        - ``bound`` (default: ``0``) -- length of induction hypothesis
        - ``time`` (default: ``-1``) -- if positive, this is the maximal time 
          (in seconds) after computations are aborted
        - ``bound_n`` (default: ``5``) -- index up to which it is checked
          whether the sequences is positive from that term on for the
          algorithms using :meth:`is_eventually_positive`.
        
        OUTPUT:
        
        Returns ``True`` if every term of the sequence is not equal to zero
        and ``False`` otherwise. Raises a ``TimeoutError`` if neither could
        be proven. If the formula for CAD is too big a ``RuntimeError``
        might be raised.
                  
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1]) 
            sage: fac.has_no_zeros() # long time
            True
            
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm.has_no_zeros()
            False
            
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ) 
            
            sage: n = var("n")
            sage: C(3^n-n*2^n).has_no_zeros(time=10) # long time
            True
            
            sage: C(10*[1,-1]).has_no_zeros(time=10) # long time
            True
            
            sage: C(n-4).has_no_zeros(time=10) # long time
            False
            
        """
        if 0 in self[:bound_n] :
            return False
        
        time_alg = time/2
        try :
            zero_pattern = self.zeros(bound=bound, time=time_alg)
            DFiniteSequence.log.info(f"Used sign pattern")
            if zero_pattern.non_zero() :
                return True
            else :
                return False
        except (ValueError, TimeoutError) as e:
            pass 
        
        try :
            n0 = (self**2).is_eventually_positive(bound=bound, time=time_alg, 
                                                 bound_n = bound_n, strict=True)
            DFiniteSequence.log.info(f"Used that squared sequence is positive")
            if 0 not in self[:n0] :
                return True
        except TimeoutError:
            raise TimeoutError(time)
        
    def __contains__(self, item) :
        r"""
        Checks whether ``item`` is a term of the sequence.
        
        INPUT:
        
        - ``item`` -- an element in the ground field of the sequence 
        
        OUTPUT:
        
        Returns ``True`` if there is a term ``item`` in the sequence and
        ``False`` otherwise.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1]) 
            sage: factorial(5) in fac # long time
            True
            
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ) 
            
            sage: 2 in C(10*[1,-1]) 
            False
            
            sage: 4 in C.an_element() 
            False
            
            sage: 13 in C.an_element() 
            True
            
        """
        return not (self-item).has_no_zeros()
        
    
    def is_eventually_positive(self, bound_n = 5, bound=0, strict=True, 
                               time=-1) :
        r"""
        Uses the Gerhold-Kauers methods (Algorithm 2 in [KP10]_) 
        to check whether the sequence is eventually positive. 
        This is done by checking whether ``self.shift(n)`` is positive for 
        ``n <= bound_n``. The smallest such index is returned.
        For every ``n`` the given ``time`` (if given) and ``bound`` is used.
        
        INPUT:
        
        - ``bound_n`` (default: ``5``) -- index up to which it is checked
          whether the sequences is positive from that term on.
        - ``bound`` (default: ``0``) -- length of induction hypothesis
        - ``strict`` (default: ``True``) -- if ``False`` non-negativity 
          instead of positivity is checked
        - ``time`` (default: ``-1``) -- if positive, this is the maximal time 
          (in seconds) after computations are aborted
        
        OUTPUT:
        
        Returns ``n`` from where the sequence on is positive.
        If no such index could be found, a ``ValueError`` is raised.
                  
        EXAMPLES::

            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm.is_eventually_positive()
            1
            
            sage: fac = D([n+1,-1],[1])
            sage: fac.is_eventually_positive()
            0
            
        """
        # first, find n0 <= bound_n, s.t. self[n]>0 (or self[n] >=0)
        # for n0 <= n <= bound_n
        n0 = bound_n
        for n in reversed(range(n0+1)) :
            if (self[n] <= 0 and strict) or (self[n] < 0 and not strict) :
                break
            else :
                n0 = n                
        
        for n in range(bound_n+1) :
            shift = self.shift(n)
            try :
                ret = shift.is_positive(bound, strict, time)
                if ret == True :
                    return n0
            except (ValueError, TimeoutError) as e:
                pass 
        raise ValueError("Could not prove whether eventually positive.")
                
    def __lt__(self, other) :
        r"""
        Checks whether ``self`` is less than ``other`` termwise, i.e.,
        checks whether ``self[n] < other[n]`` for all natural numbers
        ``n`` can be proven. Only returns ``True`` if the inequality
        can be proven.
        
        .. NOTE::
        
            Since the inequality is checked termwise, this method is not 
            equivalent to ``!(self >= other)``.
        
        ALGORITHM:
        
        The first 20 initial values are used to check whether the inequality
        can be falsified. Then, :meth:`is_positive` is used using the amount
        of time specified in the initialization of the parent class
        (default: 2 seconds).
        
        INPUT:
        
        - ``other`` -- a D-finite sequence.
        
        OUTPUT:
        
        ``True`` if ``self[n] < other[n]`` for all natural numbers
        ``n`` and ``False`` otherwise. Raises a ``ValueError`` if neither
        could be shown.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            sage: C = CFiniteSequenceRing(QQ, time_limit = 10)
            
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: 0 < harm
            False
            sage: fac = D([n+1,-1],[1])
            sage: fac > 0
            True
            sage: harm < fac
            False
            
            sage: fib = C([1,1,-1], [0,1])
            sage: c = C([2,-1], [1])
            sage: c > fib
            True
            
        """
        if not all([self[n] < other[n] for n in range(20)]) :
            return False
        diff = other-self 
        time = self.parent()._time_limit
        try :
            return diff.is_positive(bound=5, time=time)
        except (TimeoutError, ValueError) :
            raise ValueError("Could not decide inequality")
        
    def __gt__(self, other) :
        r"""
        Equivalent to ``other < self``, cf. :meth:`__lt__`.
        """
        return other < self 
    
    def __le__(self, other) :
        r"""
        Checks whether ``self`` is less than or equal to ``other`` termwise, 
        i.e., checks whether ``self[n] <= other[n]`` for all natural numbers
        ``n`` can be proven. Only returns ``True`` if the inequality
        can be proven.
        
        .. NOTE::
        
            Since the inequality is checked termwise, this method is not 
            equivalent to ``!(self > other)`` or to 
            ``self < other or self==other``.
        
        ALGORITHM:
        
        The first 20 initial values are used to check whether the inequality
        can be falsified. Then, :meth:`is_positive` is used using the amount
        of time specified in the initialization of the parent class
        (default: 2 seconds).
        
        INPUT:
        
        - ``other`` -- a D-finite sequence.
        
        OUTPUT:
        
        ``True`` if ``self[n] <= other[n]`` for all natural numbers
        ``n`` and ``False`` otherwise. Raises a ``ValueError`` if neither
        could be shown.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            sage: C = CFiniteSequenceRing(QQ, time_limit = 10)
            
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm >= 1
            False
            sage: fac = D([n+1,-1],[1])
            sage: 0 <= fac
            True
            
            sage: fib = C([1,1,-1], [1,1])
            sage: luc = C([1,1,-1], [2,1])
            sage: luc >= fib
            True
            
        """
        if not all([self[n] <= other[n] for n in range(20)]) :
            return False
        diff = other-self 
        time = self.parent()._time_limit
        try :
            return diff.is_positive(bound=5, strict=False, time=time)
        except (TimeoutError, ValueError) :
            raise ValueError("Could not decide inequality")
        
    def __ge__(self, other) :
        r"""
        Equivalent to ``other <= self``, cf. :meth:`__le__`.
        """
        return other <= self 
    
    def is_positive(self, bound=0, strict=True, time=-1):
        r"""
        Uses the Gerhold-Kauers methods (Algorithm 2 in [KP10]_) 
        to check whether the sequence is positive. 
        
        INPUT:
        
        - ``bound`` (default: ``0``) -- length of induction hypothesis
        - ``strict`` (default: ``True``) -- if ``False`` non-negativity 
          instead of positivity is checked
        - ``time`` (default: ``-1``) -- if positive, this is the maximal time 
          (in seconds) after computations are aborted
        
        OUTPUT:
        
        Returns ``True`` if it is positive, ``False`` if it is not positive and
        raises a ``ValueError`` exception if it could neither 
        prove or disprove positivity. If the time runs out, a ``TimeoutError``
        is raised.
                  
        EXAMPLES::

            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm.is_positive()
            False
            
            sage: fac = D([n+1,-1],[1])
            sage: fac.is_positive()
            True
            
        """
        return self._is_positive_algo2_timed(bound=bound, time=time,
                                             strict=strict)


    def _is_positive_algo2_timed(self, bound=0, strict=True, time=-1):
        r"""
        Uses the Gerhold-Kauers method (Algorithm 2 in Kauers/Pillwein ISSAC) 
        to check whether the sequence is positive. Returns True if it is,
        False if it is not and throws an exception if it could neither prove or
        disprove positivity. If strict is set to False, instead of
        positivity, it is checked whether the sequence is non-negative.
        
        Uses an induction hypothesis of length `bound` and the given `time` 
        (if positive).
        """
        return timeout(self._is_positive_algo2, time, bound=bound, 
                       strict=strict)

    def _is_positive_algo2(self, bound=0, strict=True):
        r"""
        Uses the Gerhold-Kauers method (Algorithm 2 in Kauers/Pillwein ISSAC) 
        to check whether the sequence is positive. Returns True if it is,
        False if it is not and throws an exception if it could neither prove or
        disprove positivity. If strict is set to False, instead of
        positivity, it is checked whether the sequence is non-negative.
        
        Uses an induction hypothesis of length `bound`.
        """
        r = self.order()
        if r == 0 : # then self is zero sequence as lc has no zeros
            return not strict
        
        with SR.temp_var(n=r) as x_var, SR.temp_var() as mu,\
             SR.temp_var() as n_var, SR.temp_var() as xi_var :
            x_var = vector(SR, [x for x in x_var])
            qf_vars = list(x_var)+[n_var]
            qf_vars_str = frozenset(list(map(str,qf_vars)))
            cad = qepcad_formula
            ineq = self._create_ineq_positive_algo2(mu, n_var, x_var, strict)
            form = cad.forall(qf_vars, cad.implies(n_var>=xi_var, ineq))
            form_exec = qepcad(form)
            for n in range(r+bound) :
                form_exec_n = form_exec.replace(str(xi_var), str(n))
                form_qff = qformula(form_exec_n, qf_vars_str)
                if self[n] < 0 or (self[n] == 0 and strict) :
                    return False
                cond_mu = mu>0 if strict else mu>=0
                if r > 1 :
                    inits = cad.and_([self[n+j] >= mu*self[n+j-1] 
                                    for j in range(1,r)])
                    total = cad.exists(mu, cad.and_(form_qff, inits, cond_mu))
                else :
                    total = cad.exists(mu, cad.and_(form_qff, cond_mu))
                if qepcad(total) == "TRUE" :
                    return True
        raise ValueError("Could not decide whether positive!")
    
    def _create_ineq_positive_algo2(self, mu, n_var, x_var, strict=True) :
        r"""
        Creates the formula 
            (self[n]>0 and self[n+1]>=mu*self[n] and ... and self[n+r-1]>=mu*self[n+r-2]) => self[n+r]>=mu*self[n+r-1]
        if `strict` is True and 
            (self[n]>=0 and self[n+1]>=mu*self[n] and ... and self[n+r-1]>=mu*self[n+r-2]) => self[n+r]>=mu*self[n+r-1]
        if `strict` is False
        using the given variables x for c[n],...,c[n+r-1].
        """
        r = self.order()
        cad = qepcad_formula
        if strict :
            lhs = (x_var[0] > 0)
        else :
            lhs = (x_var[0] >= 0)
        if r > 1 :
            lhs = cad.and_([lhs]+[x_var[j] >= mu*x_var[j-1] 
                                  for j in range(1,r)])
        R = PolynomialRing(self.base_ring(), n_var)
        clear_denom = DFiniteSequence._clear_denominators
        gcd_shifts, shifts = clear_denom(self.get_shift(r), R)
        rhs = shifts*x_var >= mu*x_var[r-1]*gcd_shifts
        return qepcad_formula.implies(lhs, rhs)
    
    @staticmethod
    def _clear_denominators(l, R) :
        r"""
        Given a vector of rational functions l=[l[0],...,l[r]], computes
        the lcm g of the denominators of l[0],...,l[r]. Returns g and 
        a vector of polynomials g*l over the given polynomial ring R.
        """
        denoms = [li.denominator() for li in l]
        g = lcm(denoms)
        return R(g), vector(R, [R(li*g) for li in l])

#conversion
    def _test_conversion_(self):
        r"""
        Test whether a conversion of ``self`` into an int/float/long/... is possible;
        i.e. whether the sequence is constant or not.

        OUTPUT:

        If ``self`` is constant, i.e. there exists a `k` in K, such that self(n) = k for all n in NN,
        then this value `k` is returned. If ``self`` is not constant ``None`` is returned.
        """
        if(self.is_zero()) :
            return self.base_ring().zero()
        initial_values = self.initial_values()
        if len(initial_values) > 0:
            i = self.initial_values()[0]
        else:
            i = 0
        if all(x == i for x in self.initial_values()):
            Sn = self.parent().associated_ore_algebra().gen()
            if self.recurrence().quo_rem(Sn-1)[1].is_zero():
                return i
        return None

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

            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: D([-n-1,2*n+3,-n-2],[0,1],name="harm")
            D-finite sequence harm(n): (-n - 1)*harm(n) + (2*n + 3)*harm(n+1) + 
            (-n - 2)*harm(n+2) = 0 and harm(0)=0 , harm(1)=1       
            
        """
        if name==None :
            name = self._name

        if self._test_conversion_() != None :
            const_repr = str(self._test_conversion_())
            return "D-finite sequence {}(n)={}".format(name, const_repr)

        coeffs = [(index, coeff) for index, coeff \
                                 in enumerate(self.coefficients()[1:], 1) \
                                 if not coeff.is_zero()]
        coeffs_repr = [f"({coeff})*{name}(n+{i})" for i, coeff in coeffs]
        init_repr = [f"{name}({i})={val}" for i, val in enumerate(self._initial_values)]
        r = "D-finite sequence {}(n): ".format(name)
        r += "({})*{}(n)".format(self.coefficients()[0], name)
        if self.order() > 0 :
            r += " + " + " + ".join(coeffs_repr) + " = 0"
        r += " and " + " , ".join(init_repr)

        return r

    def _latex_(self, name=None):
        r"""
        Not yet implemented.
        """
        raise NotImplementedError

# arithmetic
    def shift(self, k=1):
        r"""
        Shifts ``self`` k-times.

        INPUT:

        - ``k`` (default: ``1``) -- an integer

        OUTPUT:

        The sequence `(a(n+k))_{k \in \mathbb{N}}`.
        
        EXAMPLES::

            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: a = D([n+1,-1],[1]).shift(2)
            sage: a[:10]
            [2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800]
            
        """
        if k==0 :
            return self

        gen = self.parent()._poly_ring.gen()
        coeffs_shifted = [coeff.subs({gen : gen+k}) 
                            for coeff in self.coefficients()]
        return type(self)(self.parent(), coeffs_shifted, 
                          self[k:k+self.order()])

    def divides(self, right, bound=100, divisor=False) :
        r"""
        Checks whether ``self`` divides ``right`` in the sequence ring,
        i.e. checks whether there is a sequence ``div`` in this ring such that
        ``right/self == div``. 
    
        .. WARNING::
        
            We assume that ``self`` does not contain any zero terms.
                
        INPUT:
        
        - ``right`` -- a sequence in the same ring as ``self``.
        - ``bound`` (default: ``100``) -- maximal number of terms used
          to guess the divisor
        - ``divisor`` (default: ``False``) -- if ``True``, then
          the divisor is returned instead of ``True`` if it could be found.
          
        OUTPUT:
        
        Returns ``True`` if ``self`` divides ``right`` could be proven. 
        If it could not be proven, ``False`` is returned. 
        If ``divisor`` is ``True``, then ``div`` is returned instead of 
        ``True``.
        
        EXAMPLES::

            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: D([n+1,n^2-7,n+2], [3,1]).divides(D([n^2+3, n+1], [7]))
            False
            
            sage: fac = D([n+1,-1],[1])
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: div = fac.divides(harm, divisor=True)
            sage: div*fac == harm
            True
            
        """
        data = [right[n]/self[n] for n in range(bound)]
        try :
            div = self.parent().guess(data)
            if self*div==right :
                if divisor :
                    return div 
                else :
                    return True
            else :
                return False
        except :
            return False
        

    def __invert__(self):
        r"""
        Tries to compute the multiplicative inverse, if this is possible, by 
        guessing the inverse using ``100`` terms. 
        Such an inverse exists if and only if
        the sequence is an interlacing of hypergeometric sequences.
        
        The method can be called by ~self or self.inverse_of_unit().
        
        OUTPUT: 

        The multiplicative inverse of the sequence if it exists.
        Raises an ``ValueError`` if the sequence is not invertible
        or the inverse could not be found. 
         
        EXAMPLES::

            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: ~D([n+1,-1],[1])
            D-finite sequence a(n): (1)*a(n) + (-n - 1)*a(n+1) = 0 and a(0)=1
            
            sage: ~D([-n-1,2*n+3,-n-2],[0,1])
            Traceback (most recent call last):
            ...
            ValueError: Sequence is not invertible
            
        """
        n = 100
        data = self[:100]
        if 0 in data :
            raise ValueError("Sequence is not invertible")
        
        data_inv = [1/term for term in data]
        try :
            inv = self.parent().guess(data_inv)
        except ValueError:
            raise ValueError("Could not find a recurrence for the inverse")
        if self*inv == self.parent().one() :
            return inv
        else :
            raise ValueError("Guessed recurrence is not the inverse")

    def cauchy(self, right) :
        r"""
        Computes the Cauchy product of two sequences.
        
        INPUT:

        - ``right`` -- other D-finite sequences over the 
          same D-finite sequence ring

        OUTPUT:
        
        The cauchy product of ``self`` with ``right``.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: cauchy = harm.cauchy(fac)
            sage: cauchy.order(), cauchy.degree()
            (16, 22)
            
        """
        left2 = self.get_ore_algebra_sequence()
        right2 = right.get_ore_algebra_sequence()
        result = left2.cauchy_product(right2)
        return self.parent()(result)
    
    def _add_(self, right):
        r"""
        Return the sum of ``self`` and ``right``.
        
        INPUTS:

        - ``right`` -- D-finite sequences over the same D-finite sequence ring
          as ``self``.
        
        OUTPUTS: 
        
        The addition of ``self`` with ``right``.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: (fac+harm)[:8]
            [1, 2, 7/2, 47/6, 313/12, 7337/60, 14449/20, 705963/140]
            
        """

        if self.is_zero() :
            return right
        elif right.is_zero() :
            return self 
        
        left2 = self.get_ore_algebra_sequence()
        right2 = right.get_ore_algebra_sequence()
        result = left2 + right2
        return self.parent()(result)

    def _mul_(self, right):
        r"""
        Return the product of ``self`` and ``right``. The result is the 
        termwise product (Hadamard product) of ``self`` and 
        ``right``. To get the cauchy product use the method :meth:`cauchy`.
        
        INPUTS:

        - ``right`` -- D-finite sequences over the same D-finite sequence ring
          as ``self``
        
        OUTPUTS: 
        
        The product of ``self`` with ``right``.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: (fac*harm)[:10]
            [0, 1, 3, 11, 50, 274, 1764, 13068, 109584, 1026576]
            
        """
        if self.is_zero() or right.is_zero():
            return self.parent().zero()
        
        left2 = self.get_ore_algebra_sequence()
        right2 = right.get_ore_algebra_sequence()
        result = left2*right2
        return self.parent()(result)
    
    def __call__(self, expr) :
        r"""
        Returns the sequence ``self[expr(n)]`` if expr is a symbolic
        expression in one variable representing a linear polynomial.

        INPUT:

        - ``expr`` -- a linear rational univariate polynomial (can be 
          in the symbolic ring)

        OUTPUT:
        
        The sequence ``self[expr(n)]``.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm.subsequence(2) == harm(2*n)
            True
            
        """
        if expr in ZZ :
            return self[expr]
        vars = expr.variables()
        if len(vars) != 1 :
            raise NotImplementedError("Cannot convert expression to "
                                      "linear polynomial")
        with SR.temp_var() as v :
            R = PolynomialRing(QQ, v)
            poly = R(expr.subs({vars[0] : v}))
            if poly.degree() > 1 :
                raise NotImplementedError("Polynomial has to be linear")
            v, u = poly.coefficients(sparse=False)
            return self.subsequence(u, v)
    
    def subsequence(self, u, v=0):
        r"""
        Returns the sequence ``self[floor(u*n+v)]``.

        INPUT:

        - ``u`` -- a rational number
        - ``v`` (optional) -- a rational number

        OUTPUT:
        
        The sequence ``self[floor(u*n+v)]``.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm[:10]
            [0, 1, 3/2, 11/6, 25/12, 137/60, 49/20, 363/140, 761/280, 7129/2520]
            sage: harm.subsequence(2)[:5]
            [0, 3/2, 25/12, 49/20, 761/280]
            
        """
        gen = self.ann().parent().base_ring().gen()
        op = self.ann().annihilator_of_composition(u*gen+v)
        order = op.order()
        find_max_zero = DFiniteSequence._largest_zeros_of_polynomial
        max_initial_value = max(find_max_zero(op.list()[-1])+order+1, order)
        initial_values = [self[floor(u*n+v)] for n in range(max_initial_value)]
        return type(self)(self.parent(), op.list(), initial_values) 
    
    def sum(self):
        r"""
        Returns the sequence `\sum_{i=0}^n c(i)`, the sequence describing
        the partial sums.
        
        OUTPUT: 
        
        The D-finite sequence `\sum_{i=0}^n c(i)`.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm[:10]
            [0, 1, 3/2, 11/6, 25/12, 137/60, 49/20, 363/140, 761/280, 7129/2520]
            sage: harm.sum()[:10]
            [0, 1, 5/2, 13/3, 77/12, 87/10, 223/20, 481/35, 4609/280, 4861/252]
            
        """
        left2 = self.get_ore_algebra_sequence()
        result = left2.sum()
        return self.parent()(result)
    
    def interlace(self, *others):
        r"""
        Returns the interlaced sequence of self with ``others``.

        INPUT:

        - ``others`` -- other D-finite sequences over the same D-finite
          sequence ring

        OUTPUT:
        
        The interlaced sequence of self with ``others``.

        EXAMPLES:: 
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: fac.interlace(harm)[:10]
            [1, 0, 1, 1, 2, 3/2, 6, 11/6, 24, 25/12]
            
        """
        ops = [seq.ann() for seq in others]
        op = self.ann().annihilator_of_interlacing(*ops)
        m = 1 + len(others)
        order = op.order()
        all_seqs = [self]+list(others)
        find_max_zero = DFiniteSequence._largest_zeros_of_polynomial
        max_initial_value = max(find_max_zero(op.list()[-1])+order+1, order)
        initial_values = [all_seqs[i%m][i//m] for i in range(max_initial_value)]
            
        seq = type(self)(self.parent(), op.list(), initial_values) 
        return seq
    
    def prepend(self, values) :
        r"""
        Prepends the given values to the sequence.
        
        Input
        - ``values`` -- list of values in the base ring

        OUTPUT:
        
        A sequence having the same terms with the additional ``values``
        at the beginning.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: fac.prepend([-17,32,5])[:10]
            [-17, 32, 5, 1, 1, 2, 6, 24, 120, 720]
            
        """
        return super().prepend(values).compress(proof=True)

#part extraction functions
    def get_ore_algebra_sequence(self) :
        r"""
        Returns the underlying ``UnivariateDFiniteSequence``.
        
        OUTPUT:
        
        A ``UnivariateDFiniteSequence`` describing the same sequence.
        
        .. seealso:: Check the `documentation <http://www.algebra.uni-linz.ac.at/people/mkauers/ore_algebra/generated/ore_algebra.dfinite_function.html#ore_algebra.dfinite_function.UnivariateDFiniteSequence>`_ of ``UnivariateDFiniteSequence`` in the ``ore_algebra`` package
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: fac.get_ore_algebra_sequence()
            Univariate D-finite sequence defined by the annihilating operator 
            -Sn + n + 1 and the initial conditions {0: 1}
            
        """
        return self._dfinite_seq
    
    def degree(self):
        r"""
        The maximal degree of the coefficient polynomials.
        
        OUTPUT:
        
        Returns the degree of the sequence.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: fac.degree()
            1
            
        """
        return max([poly.degree() for poly in self.coefficients()])
    
    def recurrence(self):
        r"""
        The annihilating operator of ``self`` as an ``OreOperator``.
        
        OUTPUT:
        
        Annihilating ``OreOperator`` of ``self``.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: fac.recurrence()
            -Sn + n + 1
            
        """
        return self._recurrence

    def ann(self):
        r"""
        Alias of :meth:`recurrence`.
        """
        return self.recurrence()
        
    def coefficients(self):
        r"""
        Returns the list of polynomial coefficients of the recurrence of 
        ``self`` in the format ``[p0,...,pr]`` representing the recurrence 
        
        .. MATH::
            p_0(n) a(n) + \dots + p_r(n) a(n+r) = 0.
            
        OUTPUT:
        
        The coefficients of the recurrence of the sequence.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)

            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm.coefficients()
            [-n - 1, 2*n + 3, -n - 2]
            
        """
        R = self.base()
        return [R(coeff) for coeff in self._recurrence.list()]
    
    def companion_matrix(self):
        r"""
        Returns the `r \times r` companion matrix 
        
        .. MATH::
            \begin{pmatrix}
                0 & 0 & \dots & 0 & -p_0/p_r \\
                1 & 0 & \dots & 0 & -p_1/p_r \\
                0 & 1 & \dots & 0 & -p_2/p_r \\
                \vdots & \vdots & \ddots & \vdots & \vdots \\
                0 & 0 & \dots & 1 & -p_{r-1}/p_r
            \end{pmatrix} .
            
        of ``self`` with entries in the
        fraction field of the base. 
        
        OUTPUT:
        
        The companion matrix.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)

            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: harm.companion_matrix()
            [                0  (-n - 1)/(n + 2)]
            [                1 (2*n + 3)/(n + 2)]
            
        """
        R = FractionField(self.base())
        leading_coeff = self.leading_coefficient()
        coefficients = self.coefficients()
        M_comp = matrix(R, self.order())
        for i in range(self.order()) :
            for j in range(self.order()) :
                if j == self.order()-1 :
                    M_comp[i,j] = -R(coefficients[i]/leading_coeff)
                elif i==j+1 :
                    M_comp[i,j] = R(1)
                else :
                    M_comp[i,j] = R(0)
        return M_comp
    
    def get_shift(self, i) :
        r"""
        Returns a vector v, s.t. 
        ``self[n+i] = v[0]*self[n]+...+v[r-1]*self[n+r-1]``
        for all ``n`` if ``self`` has order ``r``.
        
        INPUT:
        - ``i`` -- a non-negative integer
        
        OUTPUT:
        A vector describing the components of ``self[n+i]`` w.r.t. 
        the generating system ``self[n],...,self[n+r-1]``.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)

            sage: harm = D([-n-1,2*n+3,-n-2],[0,1])
            sage: v = harm.get_shift(3)
            sage: v
            ((-2*n^2 - 7*n - 5)/(n^2 + 5*n + 6), 
            (3*n^2 + 12*n + 11)/(n^2 + 5*n + 6))
            sage: all([harm[n+3]==sum(v[i](n)*harm[n+i] for i in [0,1]) 
            ....: for n in range(10)])
            True
                    
        """
        R = FractionField(self.base())
        r = self.order()
        if i < r :
            return identity_matrix(R,r).row(i)
        else :
            v_shifted = shift_rat_vector(self.get_shift(i-1), 1)
            return self.companion_matrix()*v_shifted

####################################################################################################

class DFiniteSequenceRing(RecurrenceSequenceRing):
    r"""
    A Ring of D-finite sequences over a field.
    """

    Element = DFiniteSequence

# constructor

    def __init__(self, ring, time_limit = 2, name=None, element_class=None,
                 category=None):
        r"""
        Constructor for a D-finite sequence ring.

        INPUT:

        - ``ring`` -- a polynomial ring over a field containing the 
          coefficients of the recurrences.
        - ``time_limit`` (default: ``2``) -- a positive number indicating
          the time limit in seconds used to prove inequalities. 

        OUTPUT:

        A ring of D-finite sequences over the given ring.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: DFiniteSequenceRing(R)
            Ring of D-finite sequences over Rational Field
            
            sage: S.<n> = PolynomialRing(NumberField(x^2-2, "z"))
            sage: DFiniteSequenceRing(S)
            Ring of D-finite sequences over Number Field in z with defining 
            polynomial x^2 - 2
            
        """
        if not is_PolynomialRing(ring) :
            raise TypeError("D-finite sequences are defined over a polynomial "
                            "ring")
        
        self._poly_ring = ring
        self._var_poly_ring = self._poly_ring.variable_name()
        self._ore_algebra = OreAlgebra(self._poly_ring, "S"+self._var_poly_ring)
        self._function_ring = DFiniteFunctionRing(self._ore_algebra)
        self._time_limit = time_limit
        
        self._var_ore_algebra = self._ore_algebra.variable_name()
        field = self._ore_algebra.base_ring().base_ring()
        
        RecurrenceSequenceRing.__init__(self, field)

    def _element_constructor_(self, x, y=None, name="a", check=True, 
                              is_gen = False, construct=False, **kwds):
        r"""
        Tries to construct a sequence `a(n)`.
        
        This is possible if:

        - ``x`` is already a sequence in the right ring.
        - ``x`` is a ``UnivariateDFiniteSequence``.
        - ``x`` is a list of field elements and ``y`` is a list of field 
          elements. Then ``x`` is interpreted as the coefficients of the 
          sequence and ``y`` as the initial 
          values of the sequence, i.e. `a(0), ..., a(r-1)`.
        - ``x`` can be converted into a field element. Then it is interpreted 
          as the constant sequence `(x)_{n \in \mathbb{N}}`
        - ``x`` is in the symbolic ring and ``y`` a variable, then values of 
          ``x.subs(y=n)`` for integers ``n`` are created and a recurrence for 
          this sequence is guessed. If ``y`` is not given, we try to extract it 
          from ``x``.
        - ``x`` is a list, then guessing on this list is used to determine a 
          recurrence.
        - ``x`` is a univariate polynomial, then the sequence represents the
          polynomial sequence.
        - ``x`` is a ``RecurrenceSequence``, the first ``y`` terms are used
          to guess a D-finite sequence (if ``y`` is not given, ``100`` terms 
          are used).
        
        EXAMPLES::

            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: fac = D([n+1,-1],[1])
            sage: fac
            D-finite sequence a(n): (n + 1)*a(n) + (-1)*a(n+1) = 0 and a(0)=1
            
            sage: fac==D(fac)
            True
            
            sage: D(1/7)[:5]
            [1/7, 1/7, 1/7, 1/7, 1/7]
            
            sage: n = var("n")
            sage: D(2^n)[:5]
            [1, 2, 4, 8, 16]
            
            sage: D(10*[1,0])
            D-finite sequence a(n): (1)*a(n) + (-1)*a(n+2) = 0 and a(0)=1 , 
            a(1)=0
                        
        """      
        try :
            return super()._element_constructor_(x, y, name=name)
        except NotImplementedError:
            pass
        
        K = self.base_ring()
        R = self.base()
        if isinstance(x, UnivariateDFiniteSequence) :
            coefficients = x.ann().list()
            max_initial_value = max(list(x.singularities()) + [x.ann().order()])
            initial_values = x.expand(max_initial_value+1)
            return DFiniteSequence(self, coefficients, initial_values, 
                                   name=name)
        elif isinstance(x, list) and isinstance(y, list) :
            return self.Element(self, x, y, name=name)
        elif isinstance(x, RecurrenceSequenceElement) :
            y = 100 if y is None else y 
            return self.guess(x[:y], name=name)
        elif x in K:
            return self._create_constant_sequence(x, name=name)
        elif isinstance(x, list) :
            return self.guess(x, name=name)
        elif x in R : # x is polynomial
            return self._polynomial(R(x), name=name)
        elif x in SR :
            return self._create_sequence_from_symbolic(x, y, name=name)
        else :
            raise NotImplementedError("Conversion not implemented!") 
        
    def _polynomial(self, x, name="a") :
        r"""
        Tries to convert a polynomial from the base to this ring.
        """
        max_zero = DFiniteSequence._largest_zeros_of_polynomial(x)
        gen = x.parent().gen()
        initial_values = [x.subs({gen:n}) for n in range(max_zero+2)]
        x_shift = x.subs({gen : gen+1})
        coefficients = [x_shift, -x]
        return DFiniteSequence(self, coefficients, initial_values, name=name)

    def _create_sequence_from_symbolic(self, x, y, k=100, name="a") :
        r"""
        Guesses using ``k`` terms a sequence which represents the symbolic
        expression ``x`` in the variable ``y``. If ``y`` is not given,
        then the variable is extracted from the symbolic expression ``x``.
        """
        if not y :
            y = x.arguments()[0]
            
        data = []
        for i in range(k) :
            data.append(x.subs({y:i}))
        return self.guess(data, name=name) 

    def _coerce_map_from_(self, P):
        r"""
        """
        if isinstance(P, UnivariateDFiniteSequence) :
            return self.base_ring().has_coerce_map_from(P.base_ring())
        elif isinstance(P, DFiniteSequenceRing) :
            return self.base().has_coerce_map_from(P.base())
        elif P is self.base() :
            return True
        else :
            return super()._coerce_map_from_(P)
        
    def base(self) :
        r"""
        The polynomial base ring.
        
        OUTPUT:
        
        The polynomial base ring.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            sage: D.base()
            Univariate Polynomial Ring in n over Rational Field
            
        """
        return self.associated_ore_algebra().base_ring()    
    
    def get_ore_algebra_ring(self) :
        r"""
        Return the associated ``DFiniteFunctionRing``.
        
        OUTPUT:
        
        Associated ``DFiniteFunctionRing``.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            sage: D.get_ore_algebra_ring()
            Ring of D-finite sequences over Univariate Polynomial Ring in n 
            over Rational Field
            
        """
        return self._function_ring
    
    def associated_ore_algebra(self):
        r"""
        Returns the ore algebra associated to the recurrences.
        
        OUTPUT:
        
        The ore algebra containing recurrences of this D-finite
        sequence ring.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            sage: D.associated_ore_algebra()
            Univariate Ore algebra in Sn over Univariate Polynomial Ring in n 
            over Rational Field
            
        """
        return self._ore_algebra

    def change_base_ring(self, R):
        r"""
        Return a copy of ``self`` but with the base ring ``R``.
        
        OUTPUT:
        
        A D-finite sequence ring with base ``R``.
        """
        if R is self.base():
            return self
        else:
            C = self(R)
            return C

    def construction(self):
        r"""
        Shows how the given ring can be constructed using functors.
        
        OUTPUT:
        
        A functor ``F`` and a ring ``R`` such that ``F(R)==self``
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: F, R = D.construction()
            sage: F._apply_functor(R) == D
            True
            
        """
        return DFiniteSequenceRingFunctor(), self.base()

    def _repr_(self):
        r"""
        OUTPUT:
        
        A string representation of the D-finite sequence ring.
        
        EXAMPLES::
        
            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: DFiniteSequenceRing(R)
            Ring of D-finite sequences over Rational Field
            
            sage: S.<n> = PolynomialRing(NumberField(x^2-2, "z"))
            sage: DFiniteSequenceRing(S)
            Ring of D-finite sequences over Number Field in z with defining 
            polynomial x^2 - 2

        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        r = self._cached_repr = "Ring of D-finite sequences over " + self.base_ring()._repr_()
        return r

    def _latex_(self):
        r"""
        OUTPUT:
        
        A latex representation of the D-finite sequence ring.
        
        EXAMPLES::

            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: print(latex(DFiniteSequenceRing(R)))
            \mathcal{D}(\Bold{Q})
            
            sage: S.<n> = PolynomialRing(NumberField(x^2-2, "z"))
            sage: print(latex(DFiniteSequenceRing(S)))
            \mathcal{D}(\Bold{Q}[z]/(z^{2} - 2))

        """
        return r"\mathcal{D}(" + self._base_ring._latex_() + ")"
    
    def _own_guess(self, data, ensure=3, *args, **kwds) :
        r"""
        Tries to guess a minimal order D-finite recurrence for the given data.
        Makes sure that the linear system has at least ``ensure`` more 
        equations than variables.
        We fix the same maximal degree for all coefficients.
        If no recurrence could be found, a ValueError is raised.
        If a recurrence could be found, the associated ore polynomial is 
        returned.
        If ``cut`` is specified and ``True``, then only the minimal number
        of equations necessary to have the given ``ensure`` are used.
        """
        ring = self.associated_ore_algebra().base_ring()
        D = len(data)-1
        max_order = floor((D-ensure)/2)
        for r in range(0, max_order+1) :
            max_degree = floor((D-ensure-2*r)/(r+1))
            if "max_degree" in kwds :
                max_degree = min(kwds["max_degree"], max_degree)
            for d in range(0, max_degree+1) :
                # populate linear system
                N = D-r+1
                if "cut" in kwds and kwds["cut"] :
                    N = (r+1)*(d+1) + ensure
                M = matrix(QQ, N, (r+1)*(d+1))
                #print(f"dimensions of M={M.dimensions()}")
                for n in range(0, N) :
                    M.set_row(n, flatten([[(n**l)*data[n+k] for l in range(d+1)
                                           ] for k in range(r+1)]))
                ker = M.right_kernel()
                if ker.dimension() != 0 :
                    rec = list(ker.basis()[0])
                    split_rec = split_list(rec, r+1)
                    O = self.associated_ore_algebra()
                    op = O([ring(coeffs) for coeffs in split_rec])
                    # might happen that we get lower order recurrence than
                    # expected. this is a false candidate as such a lower
                    # order recurrence would have been found before
                    if op.order() == r :
                        return op
        
        raise ValueError("Could not find recurrence")

    def guess(self, data, name="a", *args, **kwds):
        r"""
        From given values guess a D-finite sequence using the ``ore_algebra``
        package.
        
        INPUT:
        
        - ``data`` -- list of elements in the base field of the C-finite
          sequence ring
        - ``name`` (default: "a") -- a name for the resulting sequence
        - ``algorithm`` (optional) -- if "rec_sequences", then 
          another straightforward implementation for sequences over QQ
          is used.
        - ``operator_only`` (optional) -- if ``True`` only the ore-operator
          of the recurrence is returned.
        
        All additional arguments are passed to the ``ore_algebra`` guessing
        routine.
        
        OUTPUT:
        
        A D-finite sequence with the specified terms as initial values.
        If no such sequence is found, a ``ValueError`` is raised.
        
        EXAMPLES::

            sage: from rec_sequences.DFiniteSequenceRing import *
            sage: R.<n> = PolynomialRing(QQ)
            sage: D = DFiniteSequenceRing(R)
            
            sage: D.guess([factorial(n) for n in range(20)])
            D-finite sequence a(n): (n + 1)*a(n) + (-1)*a(n+1) = 0 and a(0)=1
            sage: D.guess([sum(1/i for i in range(1,n)) for n in range(1,20)], 
            ....:         algorithm="rec_sequences", operator_only = True)
            (1/2*n + 3/2)*Sn^3 + (-n - 5/2)*Sn^2 + (1/2*n + 1)*Sn
            
        """
        if self.base_ring() != QQ :
            kwds["algorithm"] = "rec_sequences"
            
        new_kwds = {key : value for key, value in kwds.items() 
                    if key not in ["algorithm", "operator_only"]}
        
        if "algorithm" in kwds and kwds["algorithm"] == "rec_sequences" :
            # assume that zeros of D-finite sequences are cyclic 
            # remove these zeros from sequence
            
            # create zero-pattern
            pattern = ZeroPattern.guess(data, *args, **new_kwds)
            log(self.Element, 
                f"The following zero-pattern was computed: {pattern}")
            cycle_start = pattern.get_cycle_start()
            cycle_length = pattern.get_cycle_length()
            
            ops = [] # operators for all subsequences
            for i in range(cycle_length) :
                if not pattern[cycle_start+i] : # subsequence is 0
                    ops.append(self.associated_ore_algebra().one())
                else : # subsequence is not 0
                    new_data = data[cycle_start+i::cycle_length]
                    op = self._own_guess(new_data, *args, **new_kwds)
                    ops.append(op)
                      
            # interlace recurrences 
            log(self.Element, f"Interlace operators {ops}")
            rec = ops[0].annihilator_of_interlacing(*(ops[1:]))
            log(self.Element, f"Interlaced operator is: {rec}")                
                           
            # shift recurrence to make sure that it holds for initial
            # terms as well
            Sn = self.associated_ore_algebra().gen()
            recurrence = rec*Sn**cycle_start
            
            # can happen that interlaced recurrences don't hold
            # for initial values, hence might need to shift more 
            applied_data = recurrence(data)
            min_non_zero = max([i+1 for i, el in enumerate(applied_data) 
                                if el != 0] + [0])
            recurrence = recurrence*Sn**min_non_zero
                
        else : # use ore_algebra package
            # using max_degree might not give the C-finite recurrence operator
            # of smallest order, see ex:
            # C-finite sequence a(n): (0)*a(n) + (3)*a(n+1) + (-1)*a(n+3) = 0 
            # and a(0)=1 , a(1)=1 , a(2)=-1
            recurrence = guess(data, self.associated_ore_algebra(),
                                *args, **new_kwds)
        
        if "operator_only" in kwds and kwds["operator_only"]:
            if not all(v == 0 for v in recurrence(data)) :
                print("Something went wrong:")
                print(data)
                raise TypeError("Something went wrong, wrong recurrence")
            return recurrence
        else :
            return self.Element(self, recurrence.list(), data, name = name)

class DFiniteSequenceRingFunctor(RecurrenceSequenceRingFunctor):
    def __init__(self):
        r"""
        Constructs a ``DFiniteSequenceRingFunctor``.
        """
        RecurrenceSequenceRingFunctor.__init__(self)

    def _apply_functor(self, x):
        return DFiniteSequenceRing(x)

    def _repr_(self):
        r"""
        Returns a string representation of the functor.
        
        OUTPUT:
        
        The string "DFiniteSequenceRing(\*)" .        
        """
        return "DFiniteSequenceRing(*)"
