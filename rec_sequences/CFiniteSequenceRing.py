# coding: utf-8
r"""
C-finite sequences

Sublasses :class:`rec_sequences.DFiniteSequenceRing` and defines 
sequences satisfying a linear
recurrence equation with constant coefficients. Such a C-finite sequence `a(n)`
is defined by a recurrence

.. MATH::
    c_0 a(n) + \dots + c_r a(n+r) = 0 \text{ for all } n \geq 0
    
and initial values `a(0),...,a(r-1)`. In the background, we use the ore_algebra 
package to perform all operations.

EXAMPLES::

    sage: from rec_sequences.CFiniteSequenceRing import *
    
    sage: C = CFiniteSequenceRing(QQ) # create C-finite sequence ring over QQ
    
    sage: f = C([1,1,-1], [0,1]) # define Fibonacci numbers
    sage: f[:10] # get first 10 terms
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    
    sage: n = var("n");
    sage: a = C(2^n) # define an exponential sequence
    sage: a.normalized()
    C-finite sequence a(n): (-2)*a(n) + (1)*a(n+1) = 0 and a(0)=1
    sage: a.closed_form()
    2^n
    
    sage: b = a+f
    sage: b.order()
    3
    
    sage: b.is_positive()
    True
    
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
from rec_sequences.RecurrenceSequenceRing import RecurrenceSequenceRing

from numpy import random

from sage.arith.all import gcd
from sage.arith.functions import lcm
from sage.calculus.var import var
from sage.calculus.predefined import x as xSymb
from sage.functions.other import floor, ceil, binomial
from sage.matrix.constructor import matrix
from sage.matrix.special import identity_matrix
from sage.misc.all import prod, randint
from sage.rings.all import ZZ, QQ, CC
from sage.modules.free_module_element import free_module_element as vector
from sage.symbolic.ring import SR
from sage.rings.cfinite_sequence import CFiniteSequences as SageCFiniteSequences
from sage.rings.ring import CommutativeAlgebra
from sage.structure.element import CommutativeAlgebraElement
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
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
from sage.functions.log import log as ln

from itertools import combinations

from ore_algebra import OreAlgebra
from ore_algebra.guessing import guess

from .SequenceRingOfFraction import SequenceRingOfFraction
from .utility import TimeoutError, timeout, is_root_of_unity
from .SignPattern import SignPattern
from .DFiniteSequenceRing import DFiniteSequence
from .DFiniteSequenceRing import DFiniteSequenceRing
from .DFiniteSequenceRing import DFiniteSequenceRingFunctor


####################################################################################################


class CFiniteSequence(DFiniteSequence):
    r"""
    A C-finite sequence, i.e. a sequence where every term can be determined 
    by a linear recurrence with constant coefficients and finitely many 
    initial values. We assume that this recurrence holds for all values.
    """
    log = logging.getLogger("CFin")

    def __init__(self, parent, coefficients, initial_values, name = "a", 
                 is_gen = False, construct=False, cache=True):
        r"""
        Construct a C-finite sequence `a(n)` with recurrence

        .. MATH::
            c_0 a(n) + \dots + c_r a(n+r) = 0 \text{ for all } n \geq 0
            
        from given list of coefficients `c_0, ... , c_r` and given list of
        initial values `a(0), ..., a(r-1)`.

        INPUT:

        - ``parent`` -- a ``CFiniteSequenceRing``
        - ``coefficients`` -- the coefficients ``[c0,...,cr]`` of the recurrence
        - ``initial_values`` -- a list of initial values, determining 
          the sequence with at least ``r`` many values
        - ``name`` (default "a") -- a name for the sequence

        OUTPUT:

        A C-finite sequence determined by the given recurrence and initial values.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: C([1,-2], [1])
            C-finite sequence a(n): (1)*a(n) + (-2)*a(n+1) = 0 and a(0)=1
        """
        DFiniteSequence.__init__(self, parent, coefficients, initial_values, 
                                 name, is_gen, construct, cache)
        
        # contains _negative_values = [self[-1],self[-2],...]
        self._negative_values = [] 

    def dfinite(self, ring) :
        r"""
        Returns ``self`` as D-finite sequence in the given ring 
        with possibly shorter recurrence.
        There is a shorter recurrence if
        
        - ``self`` satisfies a shorter C-finite recurrence or
        - ``self`` has multiple eigenvalues.
        
        INPUT:
        
        - ``ring`` -- a ``DFiniteSequenceRing`` 
        
        OUTPUT:
        
        A sequence in the specified ring which is equal to ``self``.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: from rec_sequences.DFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ) 
            sage: D = DFiniteSequenceRing(QQ['n']) 
            
            sage: c = C([1,-3,3,-1], [1,2,5])
            sage: c.dfinite(D)
            D-finite sequence a(n): (n^2 + 2*n + 2)*a(n) + (-n^2 - 1)*a(n+1) 
            = 0 and a(0)=1
        """
        intial_values = self.order()*4
        try :
            d_fin = ring.guess(self[:intial_values])
        except ValueError:
            return ring(self)
        
        if d_fin == self :
            return d_fin
        else :
            return ring(self)
        
# action

    def normalized(self):
        r"""
        Returns a normalized version of the same sequence, i.e.,
        the leading coefficient of the sequence is 1.
        
        OUTPUT:
        
        A C-finite sequence equal to ``self`` with a recurrence
        with leading coefficient 1.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ) 
            
            sage: a = C([1,2], [1])
            sage: a.normalized().coefficients()
            [1/2, 1]
        """
        lc = self.leading_coefficient()
        coeffs = self.coefficients()
        coeffs_normalized = [coeff/lc for coeff in coeffs]
        initial_values = self.initial_values()
        return CFiniteSequence(self.parent(), coeffs_normalized, initial_values)

#tests

    def _eq_(self, right):
        r"""
        Return whether the two CFiniteSequences ``self`` and ``right`` are 
        equal. This is done by checking if enough initial values agree.
        
        INPUT:
        
        - ``right`` -- a ``CFiniteSequence``
        
        OUTPUT:
        
        True if the sequences are equal at every term and False otherwise.
        
        EXAMPLES::
         
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ) 
            
            sage: a = C([2,-3,1], [1,2])
            sage: b = C([2,-1], [1])
            sage: c = C([2,-1], [2])
            sage: a==b
            True
            sage: a==c
            False
            sage: C(1) == 1
            True
            sage: 2 == C(1)
            False
            
        """
        if right == None :
            return False
            
        ord = self.order()+right.order()
        for n in range(ord) :
            if self[n] != right[n] :
                return False 
        return True
    
    def is_degenerate(self):
        r"""
        Checks whether a sequence is degenerate. This is a necessary but 
        not sufficient condition for a sequence to have infinitely many zeros.
        
        OUTPUT: 
        
        Returns True if the sequence is degenerate, i.e. the ratio of two
        distinct roots is a root of unity. Returns False otherwise.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ) 
            
            sage: a = C([2,-1], [1])
            sage: a.is_degenerate()
            False
            
            sage: b = C(10*[1,0,2])
            sage: b.is_degenerate() 
            True  
        """
        roots = self.roots("gen_splitting_field_var", "gen_charpoly_var",
                           multiplicities=False)
        pairs_root = combinations(roots, 2)
        for pair in pairs_root :
            if not pair[1].is_zero() and is_root_of_unity(pair[0]/pair[1]) :
                return True
        return False
    
    def has_no_zeros(self, bound=0, time=-1, bound_n = 5):
        r"""
        Tries to prove that the sequence has no zeros. This is done using
        several algorithms (if ``time`` is specified, ``time/5`` is used
        for each of the algorithm):
        
        1. Uses the Gerhold-Kauers method [KP10]_ to show that ``self[n]*mu^n``
           is positive for some mu != 0.
        2. Uses the Gerhold-Kauers method [KP10]_ to show that 
           ``(-self)[n]*mu^n`` is positive for some mu != 0.
        3. Uses :meth:`is_eventually_positive` to show 
           positivity.
        4. Uses :meth:`is_eventually_positive` to show 
           negativity.
        5. Uses :meth:`is_eventually_positive` to show 
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
        
        time_alg = time/5
        try :
            non_zero = self._has_no_zeros_algo1(bound=bound, time=time_alg)
            CFiniteSequence.log.info(f"Used Algorithm 1")
            return non_zero
        except (ValueError, TimeoutError) as e:
            pass 
        
        try :
            neg = -self
            non_zero = neg._has_no_zeros_algo1(bound=bound, time=time_alg)
            CFiniteSequence.log.info(f"Used Algorithm 1 with negation")
            return non_zero
        except (ValueError, TimeoutError) as e:
            pass 
        
        bound_n = 5
        try :
            n0 = self.is_eventually_positive(bound=bound, 
                                             time=time_alg/bound_n, 
                                             bound_n = bound_n, strict=True)
            CFiniteSequence.log.info(f"Used eventual positivity from {n0} on")
            if 0 not in self[:n0] :
                return True
            else :
                return False
        except (ValueError, TimeoutError) as e:
            pass 
        
        try :
            neg = -self
            n0 = neg.is_eventually_positive(bound=bound, time=time_alg/bound_n, 
                                            bound_n = bound_n, strict=True)
            CFiniteSequence.log.info(f"Used eventual negativity from {n0} on")
            if 0 not in self[:n0] :
                return True
            else :
                return False
        except (ValueError, TimeoutError) as e:
            pass 
        
        try :
            square = self**2
            n0 = square.is_eventually_positive(bound=bound, 
                                               time=time_alg/bound_n, 
                                               bound_n = bound_n, strict=True)
            CFiniteSequence.log.info(f"Used that squared sequence is positive")
            if 0 not in self[:n0] :
                return True
            else :
                return False
        except TimeoutError:
            raise TimeoutError(time)
    
    def _has_no_zeros_algo1(self, bound=0, time=-1):
        r"""
        Uses the Gerhold-Kauers method to show that self[n]*mu^n is increasing
        for some mu != 0 using at most the given `time` (if positive).
        
        Returns a ValueError if the given `time` or `bound` was not 
        sufficient.
        """
        return timeout(self._has_no_zeros_algo1_not_timed, time, bound=bound)
    
    def _has_no_zeros_algo1_not_timed(self, bound=0):
        r"""
        Uses the Gerhold-Kauers method to show that self[n]*mu^n is positive
        for some mu != 0.
        
        Returns a ValueError if the given `bound` was not 
        sufficient.
        """
        r = self.order()
        x_var = vector(SR, var("xvc", n=r))
        mu = var("mvc")
        cad = qepcad_formula
        # x_var[j] corresponds to c(n+i)*mu^(n+i)
        lhs = cad.and_([0<x_var[i] for i in range(0,r)])
        x_mu_var = vector(SR, [mu**(r-i)*xi for i, xi in enumerate(x_var)])
        rhs = (0 < self.get_shift(r)*x_mu_var)
        formula = cad.forall(list(x_var), qepcad_formula.implies(lhs, rhs))
        #print("with qf: " + str(formula))
        formula_qff = qformula(qepcad(formula),frozenset(list(map(str,x_var))))
        #print("qff-free: "+ str(formula_qff))
        for n in range(r+bound) :
            if self[n] == 0 :
                return False
            cond_mu = (mu != 0) 
            if r > 1 :
                inits = cad.and_([0<mu**(n+i)*self[n+i] for i in range(0,r)])
                total = cad.exists(mu, cad.and_(formula_qff, inits, cond_mu))
            else :
                total = cad.exists(mu, cad.and_(formula_qff, cond_mu))
            #print(f"Start computation for n={n}")
            #print(total)
            if qepcad(total) == "TRUE" :
                return True
            #print(f"Could not decide for n={n}")
        raise ValueError("Could not decide whether nonzero!")
    
    def _has_no_zeros_algo2(self, bound=0, time=-1):
        r"""
        Checks whether self^2[n]>0 for all n.
        """
        squared = self**2
        try :
            return squared.is_positive(bound=bound, strict=True, time=time)
        except ValueError :
            raise ValueError("Could not decide whether nonzero!")
    
    def is_eventually_positive(self, bound_n = 5, bound=0, strict=True, 
                               time=-1) :
        r"""
        Uses the Gerhold-Kauers methods (Algorithm 1 and 2 in [KP10]_) 
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
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ) 
            
            sage: C([1,1,-1], [0,1]).is_eventually_positive()
            1
            sage: C(10*[1,2]).is_eventually_positive(strict=True)
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
                
    
    def is_positive(self, bound=0, strict=True, time=-1):
        r"""
        Uses the Gerhold-Kauers methods (Algorithm 1 and 2 in [KP10]_) 
        to check whether the sequence is positive. 
        If these methods fail, it is checked whether it can be shown that
        the sequence is eventually monotonously increasing (and positive
        up to that term). If for the C-finite
        representation of the sequence positivity could not be shown,
        the same is tried with a (possibly) shorter D-finite representation. 
        
        .. NOTE::
        
            Note, that a time limit has to be given in order to run through
            all possible sub-algorithms. If such a time limit is not given
            it is possible that the method does not terminate even though
            it could show positivity if a time limit were given.
        
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
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ) 
            
            sage: C([1,1,-1], [0,1]).is_positive(strict=False)
            True
            sage: C(10*[1,2]) > 0
            True
            sage: C(10*[1,3,-2]).is_positive()
            False
            sage: n = var("n")
            sage: C(n^2+1).is_positive(time=10) # long time
            True
            sage: n = var("n")
            sage: C(3^n+2^n+1).is_positive(time=2)
            True
            
        """
        try :
            non_zero = self.is_positive_algo2(bound=bound, time=time/3, 
                                                    strict=strict)
            return non_zero
        except (ValueError, TimeoutError) as e:
            pass 
        
        try :
            non_zero = self.is_positive_algo1(bound=bound, time=time/3,
                                                    strict=strict)
            return non_zero
        except (ValueError, TimeoutError) as e:
            pass 
        
        try :
            diff = self.shift() - self
            is_increasing = diff.is_positive_algo2(bound=bound, time=time/3,
                                                   strict=strict)
            cond = (lambda a : a > 0) if strict else (lambda a : a >= 0)
            if is_increasing and cond(self[0]) :
                return True 
            
        except (ValueError, TimeoutError) as e:
            pass
        
        if not self.is_squarefree() :
            R = PolynomialRing(self.base_ring(), "n")
            d_fin_ring = DFiniteSequenceRing(R)
            d_fin = self.dfinite(d_fin_ring)
            return d_fin.is_positive(bound=bound, time=time, strict=strict)
        else :
            raise ValueError("Could not prove positivity")

    def is_positive_algo1(self, bound=0, strict=True, time=-1):
        r"""
        Uses the Gerhold-Kauers methods, in particular Algorithm 1 in [KP10]_, 
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
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ) 
            
            sage: C([1,1,-1], [0,1]).is_positive_algo1(strict=False)
            True
            sage: C(10*[1,2]).is_positive_algo1()
            True
            sage: C(10*[1,3,-2]).is_positive_algo1()
            False
            
        """
        return timeout(self._is_positive_algo1, time, bound=bound,
                       strict=strict)

    def _is_positive_algo1(self, bound=0, strict=True):
        r"""
        Uses the Gerhold-Kauers method (Algorithm 1 in Kauers/Pillwein ISSAC) 
        to check whether the sequence is positive. Returns True if it is,
        False if it is not and throws an exception if it could neither prove or
        disprove positivity. If strict is set to False, instead of
        positivity, it is checked whether the sequence is non-negative.
        
        Uses an induction hypothesis of length `bound`.
        """
        r = self.order()
        if min(self[:r]) < 0 or ( min(self[:r]) == 0 and strict ):
            return False
        
        x_var = vector(SR, var("xvc", n=r))
        for n in range(r,r+bound+1) :
            if self[n] < 0 or ( self[n] == 0 and strict ):
                return False
            cad = qepcad_formula
            formula = cad.forall(list(x_var),
                        self._create_ineq_positive_algo1(n, x_var, strict))
            if qepcad(formula) == "TRUE" :
                return True
        raise ValueError("Could not decide whether positive!")

    def is_positive_algo2(self, bound=0, strict=True, time=-1):
        r"""
        Uses the Gerhold-Kauers methods, in particular Algorithm 2 in [KP10]_, 
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
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ) 
            
            sage: C([1,1,-1], [0,1]).is_positive_algo2(strict=False)
            True
            sage: C(10*[1,3,-2]).is_positive_algo2()
            False
            sage: n = var("n")
            sage: C((1/2)^n+1).is_positive_algo2() # long time
            True
            
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
        x_var = vector(SR, var("xvc", n=r))
        mu = var("mvc")
        cad = qepcad_formula
        formula = cad.forall(list(x_var), 
                             self._create_ineq_positive_algo2(mu, x_var, strict)
                             )
        formula_qff = qformula(qepcad(formula),frozenset(list(map(str,x_var))))
        for n in range(r+bound) :
            if self[n] < 0 or (self[n] == 0 and strict) :
                return False
            cond_mu = mu>0 if strict else mu>=0
            if r > 1 :
                inits = cad.and_([self[n+j] >= mu*self[n+j-1] 
                                  for j in range(1,r)])
                total = cad.exists(mu, cad.and_(formula_qff, inits, cond_mu))
            else :
                total = cad.exists(mu, cad.and_(formula_qff, cond_mu))
            if qepcad(total) == "TRUE" :
                return True
        raise ValueError("Could not decide whether positive!")
        
            
    def _create_ineq_positive_algo1(self, i, x_var, strict=True) :
        r"""
        Creates the formula::
        
            (self[n]>0 and ... and self[n+i-1]>0) => self[n+i]>0
            
        if `strict` is True and::
        
            (self[n]>=0 and ... and self[n+i-1]>=0) => self[n+i]>=0
            
        if `strict` is False
        using the given variables x_var for self[n],...,self[n+r-1].
        """
        cad = qepcad_formula
        if strict :
            lhs = cad.and_([self.get_shift(j)*x_var > 0 for j in range(i)])
            rhs = self.get_shift(i)*x_var > 0
        else :
            lhs = cad.and_([self.get_shift(j)*x_var >= 0 for j in range(i)])
            rhs = self.get_shift(i)*x_var >= 0
        
        return qepcad_formula.implies(lhs, rhs)
    
    def _create_ineq_positive_algo2(self, mu, x_var, strict=True) :
        r"""
        Creates the formula::
        
            (self[n]>0 and self[n+1]>=mu*self[n] and ... and cself[n+r-1]
            >=mu*self[n+r-2]) => self[n+r]>=mu*self[n+r-1]
            
        if `strict` is True and::
        
            (self[n]>=0 and self[n+1]>=mu*self[n] and ... and cself[n+r-1]
            >=mu*self[n+r-2]) => self[n+r]>=mu*self[n+r-1]
            
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
        rhs = self.get_shift(r)*x_var >= mu*x_var[r-1]
        return qepcad_formula.implies(lhs, rhs)

    def is_squarefree(self):
        r"""
        Checks whether the characteristic polynomial is squarefree.
        
        OUTPUT:
        
        ``True`` if the characteristic polynomial of ``self`` is squarefree
        and ``False`` otherwise.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ) 
            
            sage: a = C([1,-2,1], [1,2])
            sage: a.is_squarefree()
            False
            
            sage: b = C([2,1], [1])
            sage: b.is_squarefree()
            True
        """
        R = PolynomialRing(self.base_ring(), "x_is_squarefree_CFin")
        return self.charpoly(R).is_squarefree()

    def __getitem__(self,n):
        r"""
        Return the n-th term of ``self``.

        INPUT:

        - ``n`` -- a natural number or a ``slice`` object

        OUTPUT:

        The ``n``-th sequence term of ``self`` (starting with the ``0``-th,
        i.e. to get the first term one has to call ``self[0]``) if
        ``n`` is a natural number. If ``n`` is a slice object, the corresponding
        section of the sequence is returned.
        
        An error is raised if no upper bound in the ``slice`` object is 
        specified.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: c = C([1,1,-1], [0,1])
            sage: c[-5:5]
            [5, -3, 2, -1, 1, 0, 1, 1, 2, 3]
            
            sage: d = C([2,1],[1])
            sage: d[7]
            -128
        """
        if isinstance(n, slice) :
            if n.stop == None :
                raise ValueError("Sequences are infinite. Need to specify upper bound.")
            elif n.step != None and n.step != 1 :
                raise NotImplementedError
            if len(self._values) < n.stop :
                self._create_values(n.stop+1)
            if n.start == None or n.start >= 0 :
                #if self._values != self.to_sage()[0:len(self._values)] :
                #    print("Something fishy is going on!")
                return self._values[n]
            else :
                if -n.start > len(self._negative_values) :
                    self._create_negative_values(-n.start)
                num_values = n.stop - n.start
                stop_positive = max(0, num_values+n.start)
                start_negative = max(0, -n.stop)
                negative_part = list(reversed(self._negative_values[start_negative:-n.start]))
                return negative_part + self._values[:stop_positive]

        if n >= 0 :
            try:
                return self._values[n]
            except IndexError:
                self._create_values(n+2)
                return self._values[n]
        else : # n < 0
            try:
                return self._negative_values[-n-1]
            except IndexError:
                self._create_negative_values(-n)
                return self._negative_values[-n-1]


    def _create_values(self, n) :
        r"""
        Create values [self[0],...,self[n]] in self._values
        """
        pre_computed = len(self._values) - 1

        for i in range(1+pre_computed-self.order(), n+1-self.order()) :
            new_value = sum(coeff*self._values[i+j] for j, coeff in enumerate(self.coefficients()[:-1]))
            self._values.append(-1/(self.coefficients()[-1])*new_value)

    def _create_negative_values(self, n) :
        r"""
        Create values [self[-1],...,self[-n]] in self._negative_values
        """
        trailing_coeff = self.coefficients()[0]
        if trailing_coeff.is_zero() :
            raise ZeroDivisionError("Cannot compute values at negative indices if trailing coefficient is 0.")
        pre_computed = len(self._negative_values)
        for i in range(-pre_computed-1, -n-1, -1) :
            new_value = 0
            for j, coeff in enumerate(self.coefficients()[1:],1) :
                # self_i_j = self[i+j]
                self_i_j = self._values[i+j] if i+j >= 0 else self._negative_values[-(i+j)-1]
                new_value += coeff*self_i_j
            self._negative_values.append(-1/trailing_coeff*new_value)

#conversion
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
        over some ring and returns this container.
        """
        if not container :
            return container

        container_wo_leading_0 = self._remove_leading_zeros(container)
        container_wo_trailing_0 = self._remove_trailing_zeros(container_wo_leading_0)
        return container_wo_trailing_0

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
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: C([1,1,-1], [0,1], name="f")
            C-finite sequence f(n): (1)*f(n) + (1)*f(n+1) + (-1)*f(n+2) = 0 and 
            f(0)=0 , f(1)=1        
            
        """
        if name==None :
            name = self._name

        if self._test_conversion_() != None :
            const_repr = str(self._test_conversion_())
            return "C-finite sequence {}(n)={}".format(name, const_repr)

        coeffs = [(index, coeff) for index, coeff \
                                 in enumerate(self.coefficients()[1:], 1) \
                                 if not coeff.is_zero()]
        coeffs_repr = [f"({coeff})*{name}(n+{i})" for i, coeff in coeffs]
        init_repr = [f"{name}({i})={val}" for i, val in enumerate(self._initial_values)]
        r = "C-finite sequence {}(n): ".format(name)
        r += "({})*{}(n)".format(self.coefficients()[0], name)
        if self.order() > 0 :
            r += " + " + " + ".join(coeffs_repr) + " = 0"
        r += " and " + " , ".join(init_repr)

        return r

    def closed_form(self) :
        r"""
        Uses the Sage-native CFiniteSequence to compute a closed form, i.e., 
        a polynomial linear combination of exponential sequence and
        returns that closed form.
        
        OUTPUT:
        
        The closed form as an object in the symbolic ring.
        
        EXAMPLES::
                
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: C([2,-1], [1]).closed_form()
            2^n

            sage: C([1,1,-1], [0,1]).closed_form()
            sqrt(1/5)*(1/2*sqrt(5) + 1/2)^n - sqrt(1/5)*(-1/2*sqrt(5) + 1/2)^n
            
        """
        sage_self = self.to_sage()
        return sage_self.closed_form()

    def to_dependencies(self, name=None) :
        r"""
        Returns a string representing the sequence in a form the Mathematica
        package ``Dependencies`` expects.
        """
        if name==None :
            name = self.name()
            
        coeffs = self.normalized().coefficients()[:-1]
        coeffs_string = [f"({-coeff})*{name}[n+{i}]" 
                            for i, coeff in enumerate(coeffs)]
        r = self.order()
        rec_string = f"{name}[n+{r}]==" + " + ".join(coeffs_string)
        initial_value_string = [f"{name}[{i}]=={val}" 
                            for i, val in enumerate(self.initial_values())]
        return " , ".join([rec_string] + initial_value_string) 

    def to_sage(self, var_name="y") :
        r"""
        Returns the sequence as a Sage C-finite sequence.
        
        INPUT: 
        
        - ``var_name`` (default: "y") -- a string representing the variable name
          of the built-in sage C-finite sequence ring
        
        OUTPUT:
        
        The sequence as an element in the ring of C-Finite sequences in 
        ``var_name`` over the field ``self.base_ring()``
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: a = C([2,-1], [1]).to_sage()
            sage: a
            C-finite sequence, generated by -1/2/(y - 1/2)
            sage: a.parent()
            The ring of C-Finite sequences in y over Rational Field
            
        """
        K = self.base_ring()
        coefficients = self.coefficients()
        leading_coeff = K(self.leading_coefficient())
        new_coefficients = [-K(coeff)/leading_coeff for coeff in coefficients]
        new_coefficients = new_coefficients[:-1]

        SageC_fin = SageCFiniteSequences(K, var_name)
        if not new_coefficients :
            return SageC_fin.zero()
        else :
            return SageC_fin.from_recurrence(new_coefficients, 
                                             self.initial_values())

    def _latex_(self, name=None):
        r"""
        Creates a latex representation of the sequence.
        This is done by creating the latex representation of the closed form.
        
        OUTPUT: 
        
        A latex representation showing the closed form of the sequence.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: print(latex(C([1/2,-1], [1])))
            \left(\frac{1}{2}\right)^{n}
            
        """
        return self.closed_form()._latex_()

# arithmetic
    def cauchy(self, right) :
        r"""
        Computes the Cauchy product of two sequences.
        
        INPUT:

        - ``right`` -- C-finite sequences over the same C-finite sequence ring
          as ``self``

        OUTPUT:
        
        The cauchy product of ``self`` with ``right``
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)

            sage: a = C([2,-1], [1])
            sage: b = C([3,-1], [1])
            sage: a.cauchy(b)
            C-finite sequence a(n): (6)*a(n) + (-5)*a(n+1) + (1)*a(n+2) = 0 and 
            a(0)=1 , a(1)=5
            
        """
        op = self.ann()*right.ann()
        order = op.order()
        initial_values = [sum(self[i]*right[n-i] for i in range(n+1)) 
                                      for n in range(order)]
        seq = CFiniteSequence(self.parent(), op.list(), initial_values) 
        return seq.compress(proof=True)
    
    def _add_(self, right, compress=True):
        r"""
        Return the sum of ``self`` and ``right``. We use the method ``lclm`` 
        from the OreAlgebra package to get  the new annihilator. 
        
        INPUTS:

        - ``right`` -- C-finite sequences over the same C-finite sequence ring
          as ``self``
        - ``compress`` (default: ``True``) -- boolean specifying whether the
          resulting sequence should be expressed with a recurrence as small as
          possible
        
        OUTPUTS: 
        
        The addition of ``self`` with ``right``.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)

            sage: a = C([2,-1], [1])
            sage: b = C([3,-1], [1])
            sage: a+b
            C-finite sequence a(n): (6)*a(n) + (-5)*a(n+1) + (1)*a(n+2) = 0 and 
            a(0)=2 , a(1)=5
            
        """
        if self.__is_zero__():
            return right
        if right.__is_zero__():
            return self

        sum_ann = self.recurrence().lclm(right.recurrence())
        # using algorithm=euclid might yield wrong results, see 
        # (Sn-1).lclm(Sn-a, algorithm="euclid") with a=sqrt(5) does not
        # annihilate a^n+1.
        
        order = sum_ann.order()
        intial_values_sum = [self[i]+right[i] for i in range(order)]

        sum = type(self)(self.parent(), sum_ann.list(), intial_values_sum)
        if compress :
            sum_compress = sum.compress(True)
            return sum_compress
        else :
            return sum

    def _mul_(self, right, compress=True):
        r"""
        Return the product of ``self`` and ``right``. The result is the 
        termwise product (Hadamard product) of ``self`` and 
        ``right``. To get the cauchy product use the method :meth:`cauchy`.
        ``_mul_`` uses the method ``symmetric_product`` of the ``ore_algebra`` 
        package to get the new annihilator.
        
        INPUTS:

        - ``right`` -- C-finite sequences over the same C-finite sequence ring
          as ``self``
        - ``compress`` (default: ``True``) -- boolean specifying whether the
          resulting sequence should be expressed with a recurrence as small as
          possible
        
        OUTPUTS: 
        
        The product of ``self`` with ``right``.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)

            sage: a = C([2,-1], [1])
            sage: b = C([3,-1], [1])
            sage: a*b
            C-finite sequence a(n): (-6)*a(n) + (1)*a(n+1) = 0 and a(0)=1
            
        """
        if self.__is_zero__() or right.__is_zero__():
            return self.parent().zero()

        #getting the operator
        prod_ann = self.recurrence().symmetric_product(right.recurrence())
        order = prod_ann.order()
        intial_values_prod = [self[i]*right[i] for i in range(order)]

        prod = type(self)(self.parent(), prod_ann.list(), intial_values_prod)
        if compress :
            prod_compress = prod.compress(True)
            return prod_compress
        else :
            return prod
    
    def _companion_sparse_subsequence(self, u, v) :
        r"""
        Used to construct the coordinate vectors for computing
        a recurrence for the sparse subsequence.
        
        INPUT:
        
        - ``u`` -- a natural number
        - ``v`` -- a natural number
        
        OUTPUT:
        
        A matrix of C-finite sequences `M_c^{u n + v}` where
        `M_c` denotes the companion matrix of ``self``.
        """
        # in fact self.order() terms should suffice
        num_terms = 10*self.order()
        M = self.companion_matrix()
        M_pows = [M**(u*n+v) for n in range(num_terms)]
        M_cfin = matrix(self.parent(), self.order(), self.order())
        for i in range(self.order()) :
            for j in range(self.order()) :
                data = [M_pows_n[i,j] for M_pows_n in M_pows]
                M_cfin[i,j] = self.parent().guess(data)
        return M_cfin
    
    def factorial(self, R, s=None) :
        r"""
        Returns the `C^2`-finite sequence `a(n)=\prod_{i=s}^n c(i)` in the
        given ring. This sequence will satisfy the recurrence
        
        .. MATH::
            d(n) a(n) - a(n+1) = 0 \text{ for all } n \geq 0
        
        where `d(n)=c(n+1)` for `n \geq s-1` and `d(n)=1` for `n < s-1`. 
        If `s` is not given, the smallest `s` such that `c(s) \neq 0` is chosen.
        In the case that `c` is the Fibonacci sequence, this sequence is called
        the Fibonorial numbers.
        
        .. NOTE::
        
            It only makes sense to choose `s` in such a way that `c(n) \neq 0`
            for all `n \geq s`. Otherwise, the sequence will be eventually zero.
        
        INPUT:
        
        - ``R`` -- a ring of type 
          :class:`rec_sequences.C2FiniteSequenceRing`
        - ``s`` (default: ``None``) -- a natural number
        
        OUTPUT:
        
        The `C^2`-finite sequence `a(n)=\prod_{i=s}^n c(i)` as an element of 
        ``R``. 
        
        EXAMPLES::
        
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRing(QQ)
            
            sage: f = C([1,1,-1], [0,1])
            sage: f_fac = f.factorial(C2) # fibonorials A003266
            sage: f_fac[:10]
            [1, 1, 1, 2, 6, 30, 240, 3120, 65520, 2227680]
            
            sage: l = C([1,1,-1], [2,1])
            sage: l_fac = l.factorial(C2) # lucastorials A135407
            sage: l_fac[:10]
            [2, 2, 6, 24, 168, 1848, 33264, 964656, 45338832, 3445751232]
            
        """
        if self.is_zero() :
            return R.zero()
        
        if s == None :
            # set s minimal such that self[s] != 0
            s = 0
            while self[s] == 0 :
                s += 1
                
        d = self.shift(s+1).prepend(s*[1])
        initial_value = self[0] if s == 0 else 1
        return R([d, -1], [initial_value])
    
    def sparse_subsequence(self, R, u=1, v=0, w=0, binomial_basis=False) :
        r"""
        Returns the `C^2`-finite sequence `c(u n^2 + vn + w)` in
        the given ring.
        
        .. NOTE::
        
            The sequence need to be defined in the given range.
            If the sequence cannot be extended to the negative numbers
            (if the trailing coefficient is zero), then the indices 
            `u n^2 + vn +w` all have to be non-negative for every `n`. 
        
        INPUT:
        
        - ``R`` -- a ring of type 
          :class:`rec_sequences.DifferenceDefinableSequenceRing`,
          usually a :class:`rec_sequences.C2FiniteSequenceRing`
        - ``u`` (default: ``1``) -- an integer; if ``u==0`` then
          the method :meth:`subsequence` is used.
        - ``v`` (default: ``0``) -- an integer
        - ``w`` (default: ``0``) -- an integer
        - ``binomial_basis`` (default: ``False``) -- a boolean; if ``True``
          the sequence `c(u \binom{n,2} + vn + w)` is computed instead, i.e.
          the binomial basis is chosen instead of the monomial one.
        
        OUTPUT:
        
        The `C^2`-finite sequence `c(u n^2 + vn + w)` as element in ``R``.
        
        EXAMPLES::

            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRing(QQ)
            
            sage: f = C([1,1,-1], [0,1]) # Fibonacci numbers
            sage: # compute sequence f(n^2), A054783
            sage: f_sq = f.sparse_subsequence(C2)
            sage: f_sq.order()
            2
            sage: f_sq[:10] == [f[n^2] for n in range(10)]
            True
            
            sage: p = C([1,1,0,-1], [3,0,2]) # Perrin numbers
            sage: # compute sequence p(2n^2+3n+1)
            sage: p_sq = p.sparse_subsequence(C2, 2, 3, 1)
            sage: p_sq[:10] == [p[2*n^2+3*n+1] for n in range(10)]
            True
            
            sage: from sage.functions.other import binomial
            sage: # compute sequence f(binomial(n,2)), A081667
            sage: f_binom = f.sparse_subsequence(C2, binomial_basis=True)
            sage: f_binom[:10]
            [0, 0, 1, 2, 8, 55, 610, 10946, 317811, 14930352]
            sage: f_binom[:10] == [f[binomial(n, 2)] for n in range(10)]
            True
            
        """
        if u == 0 :
            return self.subsequence(v, w)
        
        if not binomial_basis :
            # always compute with binomial basis
            return self.sparse_subsequence(R, 2*u, v+u, w, binomial_basis=True)
        
        QR = SequenceRingOfFraction(self.parent())
        A = self._companion_sparse_subsequence(u, 0).change_ring(QR)
        r = self.order()
        v0_matrix = self._companion_sparse_subsequence(v, w-r+1)
        v0 = v0_matrix.column(r-1).change_ring(QR)
        
        rec = R._compute_recurrence(A, v0)    
        r = len(rec)
        initial_values = [self[u*binomial(n,2)+v*n+w] for n in range(r)]

        return R(rec, initial_values).clear_common_factor()
    
    def subsequence(self, u, v=0):
        r"""
        Returns the sequence ``self[floor(u*n+v)]``.

        INPUT:

        - ``u`` -- a rational number
        - ``v`` (optional) -- a rational number

        OUTPUT:
        
        The sequence ``self[floor(u*n+v)]``.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: a = C([2,-1], [1])
            sage: a.subsequence(3, 1)
            C-finite sequence a(n): (-8)*a(n) + (1)*a(n+1) = 0 and a(0)=2
            
            sage: f = C([1,1,-1], [0,1])
            sage: f[:10]
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
            sage: f.subsequence(2)[:5]
            [0, 1, 3, 8, 21]
            
        """
        gen = self.ann().parent().base_ring().gen()
        op = self.ann().annihilator_of_composition(u*gen+v)
        order = op.order()
        initial_values = [self[floor(u*n+v)] for n in range(order)]
        subseq = type(self)(self.parent(), op.list(), initial_values)
        return subseq.compress(True)
    
    def sum(self):
        r"""
        Returns the sequence `\sum_{i=0}^n c(i)`, the sequence describing
        the partial sums.
        
        OUTPUT: 
        
        The C-finite sequence `\sum_{i=0}^n c(i)`.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: a = C([2,-1], [1])
            sage: a.sum()
            C-finite sequence a(n): (-2)*a(n) + (3)*a(n+1) + (-1)*a(n+2) = 0 
            and a(0)=1 , a(1)=3            
            
            sage: f = C([1,1,-1], [0,1])
            sage: f.sum()[:10]
            [0, 1, 2, 4, 7, 12, 20, 33, 54, 88]
            
        """
        op = self.ann().annihilator_of_sum()
        order = op.order()
        initial_values = [sum(self[i] for i in range(n+1)) 
                                      for n in range(order)]
        seq = type(self)(self.parent(), op.list(), initial_values) 
        return seq.compress(proof=True)
    
    def interlace(self, *others):
        r"""
        Returns the interlaced sequence of self with ``others``.

        INPUT:

        - ``others`` -- other C-finite sequences over the same C-finite
          sequence ring

        OUTPUT:
        
        The interlaced sequence of self with ``others``.
        
        EXAMPLES::

            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: a = C([2,-1], [1])
            sage: f = C([1,1,-1], [0,1])
            sage: f.interlace(a)[:10]
            [0, 1, 1, 2, 1, 4, 2, 8, 3, 16]
            
        """
        ops = [seq.ann() for seq in others]
        op = self.ann().annihilator_of_interlacing(*ops)
        m = 1 + len(others)
        order = op.order()
        all_seqs = [self]+list(others)
        initial_values = [all_seqs[i%m][i//m] for i in range(order)]
            
        seq = type(self)(self.parent(), op.list(), initial_values) 
        return seq.compress(proof=True)

#base ring related functions


#part extraction functions

    def charpoly(self, ring=None):
        r"""
        Returns the characteristic polynomial of ``self``.
        If a polynomial ring is given, the polynomial will be in this ring.
        If no ring is given, a symbolic expression in ``x`` will be given.
        
        INPUT:
        
        - ``ring`` (default: ``None``) -- a univariate polynomial ring which 
          should contain the characteristic polynomial
        
        OUTPUT:
        
        The characteristic polynomial of ``self`` as an element in ``ring``
        if given or as a symbolic expression in ``x`` otherwise.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: C([2,-1], [1]).charpoly()
            -x + 2
            
            sage: R.<y> = PolynomialRing(QQ)
            sage: C([1,1,-1], [0,1]).charpoly(R)
            -y^2 + y + 1
            
        """
        if not ring :
            gen = xSymb
            ring = SR
        else :
            gen = ring.gen()

        return sum(ring(coeff)*gen**i 
                   for i,coeff in enumerate(self.coefficients()))
    
    def roots(self, gen="z", gen_char="y", *args, **kwargs):
        r"""
        Returns all roots (also called eigenvalues) of the characteristic
        polynomial of the sequence as elements in the splitting field.
        The generator of the splitting field is ``gen`` and the base polynomial
        is written in the generator ``gen_char``. 
        
        INPUT:

        - ``gen`` (default: "z") -- name for the generator of the underlying 
          splitting field
        - ``gen_char`` (default: "y") -- name for the generator of the 
          polynomial ring containing the characteristic polynomial
        
        Any additional arguments are passed to Sage's ``roots`` method.
        
        OUTPUT:
        
        All roots of the recurrence.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: C([2,-1], [1]).roots(multiplicities=False)
            [2]
            
            sage: C([1,1,-1], [0,1]).roots()
            [(z, 1), (-z + 1, 1)]
             
        """
        charpoly = self.charpoly(PolynomialRing(self.base_ring(),"y"))
        K = charpoly.splitting_field(gen)
        return charpoly.roots(K, *args, **kwargs)
    
    def _closed_form_list(self, field=None) :
        r"""
        Returns a list l of tuples li=(ki,ci,gi) such that:: 
        
            self[n] = sum_i ci*n^ki*gi^n
            
        for all n where ki and gi are in `field` (if it is not given, it
        is chosen automatically) and ki is a natural number.
        """
        """"roots = self.roots()
        if not field :
            K = roots[0][0].parent()
        else :
            K = field
            roots = [(K(root[0]),root[1]) for root in roots]
        r = self.order()"""
        return NotImplementedError

####################################################################################################

class CFiniteSequenceRing(DFiniteSequenceRing):
    r"""
    A Ring of C-finite sequences over a field.
    """

    Element = CFiniteSequence

# constructor

    def __init__(self, field = QQ, name=None, element_class=None, category=None):
        r"""
        Constructor for a C-finite sequence ring.

        INPUT:

        - ``field`` (default: ``QQ``) -- a field of characteristic zero over 
          which the C-finite sequence ring is defined.

        OUTPUT:

        A ring of C-finite sequences

        ..TODO:: 
        
            If field is fraction field, change it to fraction ring of 
            polynomial ring which
            can be handled by the ore_algebra package.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: CFiniteSequenceRing(QQ)
            Ring of C-finite sequences over Rational Field
            sage: CFiniteSequenceRing(NumberField(x^2-2, "z"))
            Ring of C-finite sequences over Number Field in z with defining
            polynomial x^2 - 2
            
        """
        self._poly_ring = PolynomialRing(field, "n")
        
        DFiniteSequenceRing.__init__(self, self._poly_ring)

    def _element_constructor_(self, x, y=None, name="a", check=True, 
                              is_gen = False, construct=False, **kwds):
        r"""
        Tries to construct a sequence `a(n)`.
        
        This is possible if:

        - ``x`` is already a sequence in the right ring.
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
          to guess a C-finite sequence (if ``y`` is not given, ``100`` terms 
          are used).
        
        EXAMPLES::

            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: f = C([1,1,-1],[1,1])
            sage: f
            C-finite sequence a(n): (1)*a(n) + (1)*a(n+1) + (-1)*a(n+2) = 0 and 
            a(0)=1 , a(1)=1
            
            sage: f==C(f)
            True
            
            sage: C(1/7)[:5]
            [1/7, 1/7, 1/7, 1/7, 1/7]
            
            sage: n = var("n")
            sage: C(2^n)[:5]
            [1, 2, 4, 8, 16]
            
            sage: C(10*[1,0])
            C-finite sequence a(n): (1)*a(n) + (-1)*a(n+2) = 0 and a(0)=1 , 
            a(1)=0
            
            sage: R.<x> = PolynomialRing(QQ)
            sage: C(x^2-1)[:5]
            [-1, 0, 3, 8, 15]
            
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: C2 = C2FiniteSequenceRing(QQ)
            sage: a = C2([1, 1], [17])
            sage: a_cfin = C(a)
            sage: a_cfin
            C-finite sequence a(n): (1)*a(n) + (1)*a(n+1) = 0 and a(0)=17
            
        """      
        try :
            try :
                if (not y) and is_PolynomialRing(x.parent()):
                    return self.polynomial(x, name=name)
            except AttributeError :
                pass
            return super()._element_constructor_(x, y, name=name)
        except NotImplementedError :
            raise NotImplementedError("Conversions not implemented!") 
        
    @staticmethod
    def geometric(ratio,  name="a") :
        r"""
        Creates the C-finite sequence ``ratio^n`` over the field
        in which ``ratio`` lives.
        
        INPUT:
        
        - ``ratio`` -- a number
        - ``name`` (default: "a") -- a name for the sequence
        
        OUTPUT:
        
        The geometric C-finite sequence with given ``ratio`` and given 
        ``name``.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: CFiniteSequenceRing.geometric(5)
            C-finite sequence a(n): (5)*a(n) + (-1)*a(n+1) = 0 and a(0)=1
            
        """
        if ratio in ZZ :
            K = QQ
        else :
            K = ratio.parent()
        parent = CFiniteSequenceRing(K)
        rec = [K(ratio), K(-1)]
        return parent._element_constructor_(rec, [K(1)], name=name)
    
    @staticmethod
    def polynomial(poly, name="a") :
        r"""
        Creates the C-finite sequence representing the given polynomial
        over the ground field of the polynomial ring.
        
        A polynomial of degree `d` satisfies a recurrence of order `d+1`
        with coefficients 
        
        .. MATH::
            \binom{d+1}{i} (-1)^{d+i} \text{ for } i=0,\dots,d+1
        
        INPUT:
        
        - ``poly`` -- a polynomial over a univariate polynomial ring
        - ``name`` (default: "a") -- a name for the sequence
        
        OUTPUT:
        
        The polynomial C-finite sequence represented by the given polynomial 
        and with given ``name``.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: R.<y> = PolynomialRing(QQ)
            sage: CFiniteSequenceRing.polynomial(y+1)[:5]
            [1, 2, 3, 4, 5]
            
        """
        K = poly.parent().base_ring()
        d = poly.degree()
        
        parent = CFiniteSequenceRing(K)
        rec = parent._monomial(d, name=name)
        initial_values = [poly(n) for n in range(d+1)]
        
        return parent._element_constructor_(rec.coefficients(), 
                                            initial_values, name=name)
    
    def _monomial(self, d, name="a") :
        r"""
        Creates the C-finite sequence representing the sequence n^d.
        """
        K = self.base_ring()
        rec = [K(binomial(d+1,i)*(-1)**(d+i)) for i in range(d+2)]
        initial_values = [K(n**d) for n in range(d+1)]
        return self(rec, initial_values, name=name)            

    def construction(self):
        r"""
        Shows how the given ring can be constructed using functors.
        
        OUTPUT:
        
        A functor ``F`` and a field ``K`` such that ``F(K)==self``
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: F, K = C.construction()
            sage: F._apply_functor(K) == C
            True
            
        """
        return CFiniteSequenceRingFunctor(), self.base_ring()

    def _repr_(self):
        r"""
        OUTPUT:
        
        A string representation of the C-finite sequence ring.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: CFiniteSequenceRing(QQ)
            Ring of C-finite sequences over Rational Field
            sage: CFiniteSequenceRing(NumberField(x^3-2, "z"))
            Ring of C-finite sequences over Number Field in z with defining
            polynomial x^3 - 2

        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        r = self._cached_repr = "Ring of C-finite sequences over " \
            + self.base_ring()._repr_()
        return r

    def _latex_(self):
        r"""
        OUTPUT:
        
        A latex representation of the C-finite sequence ring.
        
        EXAMPLES::

            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: print(latex(CFiniteSequenceRing(QQ)))
            \mathcal{C}(\Bold{Q})
            sage: print(latex(CFiniteSequenceRing(NumberField(x^3-2, "z"))))
            \mathcal{C}(\Bold{Q}[z]/(z^{3} - 2))

        """
        return r"\mathcal{C}(" + self._base_ring._latex_() + ")"

    def random_element(self, order=2, *args, **kwds):
        r"""
        Return a random C-finite sequence.

        INPUT:

        - ``order`` (default 2) -- the order of the recurrence of the random
          C-finite sequence
        
        Any additional arguments are passed to the method computing the
        random coefficients.

        OUTPUT:

        A C-finite sequence with a random recurrence of order ``order`` and 
        random initial values consisting of integers between -10 and 10.
        
        EXAMPLES::
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: C.random_element(order=1) # random
            C-finite sequence a(n): (-6)*a(n) + (3)*a(n+1) = 0 and a(0)=-8
            
        """
        coefficients = self._random_coefficients(order+1, *args, **kwds)
        initial_values = [randint(-10, 10) for i in range(order)]
        return CFiniteSequence(self, coefficients, initial_values)

    def random_element_from_initial_values(self, initial_values, *args, **kwds):
        r"""
        Return a random C-finite sequence with given initial values.

        INPUT:

        - ``initial_values``-- the initial values of the random recurrence

        OUTPUT:

        A C-finite sequence with a random recurrence and given initial values.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: C.random_element_from_initial_values([1,2])[:5] # random
            [1, 2, 91/3, 1288/3, 54733/9]
            
        """
        order = len(initial_values)
        coefficients = self._random_coefficients(order+1, *args, **kwds)
        return CFiniteSequence(self, coefficients, initial_values)

    def _random_coefficients(self, number, *args, **kwds) :
        r"""
        Return random coefficients from the base field.

        INPUT:

        -``number`` -- the number of generated field elements

        OUTPUT:

        A list of ``number`` many random elements from the field such that
        the first and the last element are not 0.
        """
        if number < 1 :
            raise ValueError("Number needs to be at least 1.")
        if number == 1 :
            return [self._random_non_zero_field_element(*args, **kwds)]
        else :
            trailing = self._random_non_zero_field_element(*args, **kwds)
            leading  = self._random_non_zero_field_element(*args, **kwds)
            K = self.base_ring()
            others   = [K.random_element(*args, **kwds) for i in range(number-2)]
            return [trailing] + others + [leading]

    def base(self) :
        r"""
        OUTPUT:
        
        The base field of the C-finite sequence ring
        
        EXAMPLES:: 
        
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: CFiniteSequenceRing(QQ).base()
            Rational Field
            
        """
        return self.base_ring()    

    def _random_non_zero_field_element(self, *args, **kwds) :
        r"""
        Return random non-zero field element from the base field.
        """
        K = self.base_ring()
        element = K.random_element(*args, **kwds)
        while element == K.zero() :
            element = K.random_element(*args, **kwds)
        return element

    def guess(self, data, name="a", *args, **kwds):
        r"""
        From given values guess a C-finite sequence using the ``ore_algebra``
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
        
        A C-finite sequence with the specified terms as initial values.
        If no such sequence is found, a ``ValueError`` is raised.
        
        EXAMPLES::

            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: C = CFiniteSequenceRing(QQ)
            
            sage: C.guess(10*[1,-1])
            C-finite sequence a(n): (1)*a(n) + (1)*a(n+1) = 0 and a(0)=1
            sage: C.guess([i for i in range(10)])
            C-finite sequence a(n): (-1)*a(n) + (2)*a(n+1) + (-1)*a(n+2) = 0 
            and a(0)=0 , a(1)=1
            
        """
        if "max_degree" not in kwds :
            #return super().guess(data, name=name, max_degree=0, *args, **kwds,
            #                     algorithm="rec_sequences", cut=True)
            return super().guess(data, name=name, max_degree=0, *args, **kwds)
        else :
            #return super().guess(data, name=name, *args, **kwds,
            #                     algorithm="rec_sequences", cut=True)
            return super().guess(data, name=name, *args, **kwds)

class CFiniteSequenceRingFunctor(DFiniteSequenceRingFunctor):
    def __init__(self):
        r"""
        Constructs a ``CFiniteSequenceRingFunctor``.
        """
        DFiniteSequenceRingFunctor.__init__(self)

    def _apply_functor(self, x):
        return CFiniteSequenceRing(x)

    def _repr_(self):
        r"""
        Returns a string representation of the functor.
        
        OUTPUT:
        
        The string "CFiniteSequenceRing(\*)".
        
        """
        return "CFiniteSequenceRing(*)"
