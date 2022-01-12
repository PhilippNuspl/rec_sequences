# coding: utf-8
r"""
Ring of `C^2`-finite sequences 

Sublasses :class:`rec_sequences.DifferenceDefinableSequenceRing` and 
defines sequences satisfying a linear
recurrence equation with coefficients which are C-finite sequence. 
Such a `C^2`-finite sequence `a(n)` is defined by a recurrence

.. MATH::
    c_0(n) a(n) + \dots + c_r(n) a(n+r) = 0 \text{ for all } n \geq 0

with `c_r(n) \neq 0` for all `n` and initial values `a(0),...,a(r-1)`. 
This is based on the theory and algorithms developed in [JPNP21a]_ and
[JPNP21b]_.

EXAMPLES::

    sage: from rec_sequences.C2FiniteSequenceRing import *
    sage: from rec_sequences.CFiniteSequenceRing import *
    
    sage: C = CFiniteSequenceRing(QQ)
    sage: C2 = C2FiniteSequenceRing(QQ)
        
    sage: n = var("n")
    sage: c = C(2^n+1)
    sage: a = C2([c, -1], [3])
    sage: a[:10]
    [3, 6, 18, 90, 810, 13770, 454410, 29536650, 3810227850, 979228557450]
    
    sage: f = C([1,1, -1], [0,1]) # define fibonacci numbers
    sage: l = C([1,1,-1], [2,1]) # define lucas numbers
    sage: f_fac = f.factorial(C2) # define fibonorials
    sage: f_fac
    C^2-finite sequence of order 1 and degree 2 with coefficients:
    > c0 (n) : C-finite sequence c0(n): (1)*c0(n) + (1)*c0(n+1) + (-1)*c0(n+2) = 0 and c0(0)=1 , c0(1)=1
    > c1 (n) : C-finite sequence c1(n)=-1
    and initial values a(0)=1
    sage: f_fac[:10]
    [1, 1, 1, 2, 6, 30, 240, 3120, 65520, 2227680]
    
    sage: a_plus_f = a+f_fac
    sage: a_plus_f.order(), a_plus_f.degree()
    (2, 25)
    sage: a_plus_f[:10]
    [4, 7, 19, 92, 816, 13800, 454650, 29539770, 3810293370, 979230785130]
    
    sage: l_sparse = l.sparse_subsequence(C2) # define l(n^2)
    sage: l_sparse[:8]
    [2, 1, 7, 76, 2207, 167761, 33385282, 17393796001]
    
    sage: f_times_l = f_fac * l_sparse
    sage: f_times_l.order(), f_times_l.degree()
    (2, 14)
    sage: f_times_l[:5]
    [2, 1, 7, 152, 13242]
    
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

from .SequenceRingOfFraction import SequenceRingOfFraction
from .SequenceRingOfFraction import FractionSequence
from .CFiniteSequenceRing import CFiniteSequence
from .CFiniteSequenceRing import CFiniteSequenceRing
from .FunctionalEquation import FunctionalEquation
from .DifferenceDefinableSequenceRing import DifferenceDefinableSequence
from .DifferenceDefinableSequenceRing import DifferenceDefinableSequenceRing
from .utility import log, split_list, split_list_rec

####################################################################################################

class C2FiniteSequence(DifferenceDefinableSequence):
    r"""
    A C^2-finite sequence, i.e. a sequence where every term can be determined by a linear recurrence
    with coefficients coming from a C-finite sequence ring and finitely many initial values. We assume that this
    recurrence holds for all values and that the leading coefficient is non-zero for all n (this is not checked).
    """

    log = logging.getLogger("C2Fin")

    def __init__(self, parent, coefficients, initial_values, name = "a", 
                 is_gen = False, construct=False, cache=True, *args, **kwds):
        r"""
        Construct a `C^2`-finite sequence `a(n)` with recurrence

        .. MATH::
            c_0(n) a(n) + \dots + c_r(n) a(n+r) = 0 \text{ for all } n \geq 0

        from given list of coefficients `c_0, ... , c_r` and given list of
        initial values `a(0), ..., a(r-1)`.

        .. NOTE::
        
            We assume that the leading coefficient `c_r` does not contain any 
            zero terms. If it does, this might yield problems in later 
            computations.
        
        INPUT:

        - ``parent`` -- a ``C2FiniteSequenceRing``
        - ``coefficients`` -- the coefficients of the recurrence
        - ``initial_values`` -- a list of initial values, determining the 
          sequence with at least order of the recurrence many values
        - ``name`` (default "a") -- a name for the sequence

        OUTPUT:

        A sequence determined by the given recurrence and 
        initial values.
        
        EXAMPLES::

            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRing(QQ)
                
            sage: n = var("n")
            sage: c = C(2^n+1)
            sage: a = C2([c, -1], [3])
            sage: a
            C^2-finite sequence of order 1 and degree 2 with coefficients:
            > c0 (n) : C-finite sequence c0(n): (2)*c0(n) + (-3)*c0(n+1) + (1)*c0(n+2) = 0 and c0(0)=2 , c0(1)=3
            > c1 (n) : C-finite sequence c1(n)=-1
            and initial values a(0)=3
            
        """  
        if not isinstance(parent, C2FiniteSequenceRing) :
            raise TypeError("Parent has to be a C^2-finite sequence ring.")

        DifferenceDefinableSequence.__init__(self, parent, coefficients,
                                             initial_values, name, is_gen,
                                             construct, cache, *args, **kwds)

#tests


#conversion

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
        
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRing(QQ)
                
            sage: n = var("n")
            sage: c = C(2^n)
            sage: a = C2([c, -1], [1])
            sage: a
            C^2-finite sequence of order 1 and degree 1 with coefficients:
            > c0 (n) : C-finite sequence c0(n): (2)*c0(n) + (-1)*c0(n+1) = 0 and c0(0)=1
            > c1 (n) : C-finite sequence c1(n)=-1
            and initial values a(0)=1  
            
        """
        if name==None :
            name = self._name
        r = "C^2-finite sequence of order {}".format(self.order())
        r += " and degree {} with coefficients:\n".format(self.degree())
        for i in range(self.order()+1) :
            r += f" > c{i} (n) : " + self._coefficients[i]._repr_(name=f"c{i}") + "\n"
        init_repr = [f"{self._name}({i})={val}" for i, val in enumerate(self._initial_values)]
        r += "and initial values " + " , ".join(init_repr)
        return r

    def _latex_(self, name=None):
        r"""
        Creates a latex representation of the sequence.
        This is done by creating the latex representation of the closed forms
        of the coefficients and showing the recurrence and the initial values.
        
        OUTPUT: 
        
        A latex representation showing the closed form of the sequence.
        
        EXAMPLES::
        
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRing(QQ)
            
            sage: n = var("n")
            sage: c = C(2^n+1)
            sage: d = C(3^n)
            sage: a = C2([c, d], [1])
            
            sage: print(latex(a))
            \left(2^{n} + 1\right)\cdot a(n) + \left(3^{n}\right) \cdot a(n+1) = 0 \quad a(0)=1
            
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

    @staticmethod
    def _split_exp_term(term):
        r"""
        Given a symbolic expression of the form c*n^i*gamma^n
        returns the triple (c, i, gamma).
        """
        # print("term = {}".format(term))
        
        # is constant
        try:
            if term in QQbar :
                return (term, 0, 1)
        except NotImplementedError: 
            pass
        if term.is_constant() :
            return (term, 0, 1)
        
        n = term.variables()[0]
        if term == n :
            return (1, 1, 1)
        operands = term.operands()
        if len(operands) == 2 :
            if (term/n).is_constant() :
                # is of form c*n
                return (term/n, 1, 1)
            elif operands[0] == n and operands[1].is_constant() :
                # is of form n^i
                return (1, int(operands[1]), 1)
            elif operands[1] == n and operands[0].is_constant() :
                # is of form gamma^n
                return (1, 0, operands[0])

        triple = [1, 0, 1]
        for operand in operands :
            c, i, gamma = C2FiniteSequence._split_exp_term(operand)
            triple[0] = triple[0]*c
            triple[1] = triple[1]+i
            triple[2] = triple[2]*gamma

        return tuple(triple)


    @staticmethod
    def _split_cfinite(seq):
        r"""
        Given a C-finite sequence ``seq``, computes a list of triples 
            (c, i, gamma)
        representing the factors c*n^i*gamma^n in the closed form of 
        `´seq``.
        """
        split_exp_term = C2FiniteSequence._split_exp_term
        factors = list()
        closed_form = seq.closed_form().expand()
        w0 = SR.wild(0); w1 = SR.wild(1)
        if closed_form.is_constant() :
            return [(closed_form, 0, 1)]
        elif not closed_form.find(w0+w1) :
            # print("Only simple factor {}".format(closed_form)) 
            # have only gamma^n or n^i
            factors.append(split_exp_term(closed_form))
        else :
            for factor in closed_form.operands() :
                factors.append(split_exp_term(factor))
        return factors

    def functional_equation(self):
        r"""
        Computes the functional equation corresponding to the sequence.
        This functional equation is given as object of type
        :class:`rec_sequences.FunctionalEquation` and represents a
        functional equation of the form 
 
        .. MATH::
            \sum_{k=1}^m \alpha_k x^{j_k} g^{(d_k)} (\lambda_k x) = p(x)
            
        for constants `\alpha_k, \lambda_k`, natural numbers `j_k, d_k` and
        a polynomial `p(x)` where `g` denotes the generating function of the
        sequence.
        
        OUTPUT:
        
        A functional equation for the generating function of the sequence.

        EXAMPLES::

            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRing(QQ)
            
            sage: n = var("n")
            sage: c = C(2^n+1)
            sage: d = C(3^n)
            sage: a = C2([c, d], [1])
            
            sage: print(a.functional_equation())
            (x)g(2x) + (x)g(x) + (1/3)g(3x) = 1/3

        """
        R = PolynomialRing(QQbar, "x")
        gen = R.gen()
        equation = list()
        rhs = R(0)
        r = self.order()

        for j, coeff in enumerate(self.coefficients()) :
            factors = C2FiniteSequence._split_cfinite(coeff)
            for c, i, gamma in factors :
                c = QQbar(c)
                gamma = QQbar(gamma)
                # print(f"c={c}, i={i}, j={j}, gamma={gamma}")
                for k in range(i+1) :
                    for l in range(k+1) :
                        constant_coeff = c*binomial(i,k)*(-j)**(i-k)*stirling_number2(k,l)*gamma**(-j)
                        coeff_fe = constant_coeff*gen**(l-j+r)
                        if j > l :
                            p_kl = - sum(self[n]*falling_factorial(n, l)*gamma**n*gen**(n+r-j) for n in range(l,j))
                            rhs += constant_coeff*p_kl

                        equation.append((l, gamma, coeff_fe))

        return FunctionalEquation(R, equation, -rhs)


####################################################################################################

class C2FiniteSequenceRing(DifferenceDefinableSequenceRing):
    r"""
    A Ring of C^2-finite sequences.
    """

    Element = C2FiniteSequence

# constructor

    def __init__(self, base_ring, name=None, element_class=None, category=None, 
                 *args, **kwds):
        r"""
        Constructor for a `C^2`-finite sequence ring.

        INPUT:

        - ``field`` (default: ``QQ``) -- a field of characteristic zero over 
          which the `C^2`-finite sequence ring is defined.

        OUTPUT:

        A ring of `C^2`-finite sequences
        
        EXAMPLES::
        
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: C2FiniteSequenceRing(QQ)
            Ring of C^2-finite sequences with base field Rational Field
            
        """
        if base_ring not in Fields() :
            raise TypeError("The base ring of a C^2-finite sequence ring is a field")

        C = CFiniteSequenceRing(base_ring)

        DifferenceDefinableSequenceRing.__init__(self, C, *args, **kwds)

    def _element_constructor_(self, x, y=None, name = "a", check=True, 
                              is_gen = False, construct=False, *args, **kwds):
        r"""
        Tries to construct a sequence `a(n)`.
        
        This is possible if:

        - ``x`` is already a `C^2`-finite sequence.
        - ``x`` is a list of C-finite sequences and ``y`` is a list of field 
          elements. Then ``x`` is interpreted as the coefficients of the 
          recurrence and ``y`` as the initial 
          values of the sequence, i.e. `a(0), ..., a(r-1)`.
        - ``x`` is a C-finite sequence.
        - ``x`` can be converted into a field element. Then it is interpreted 
          as the constant sequence `(x)_{n \in \mathbb{N}}`
        
        EXAMPLES::
        
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRing(QQ) 
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
            C^2-finite sequence of order 2 and degree 1 with coefficients:
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
            return C2FiniteSequence(self, x, y, name=name, *args, **kwds)
        elif x in R and isinstance(x, CFiniteSequence) : # check whether R is sequence ring
            try :
                coeffs_R = [R(coeff) for coeff in x.coefficients()]
                return C2FiniteSequence(self, coeffs_R, x.initial_values(), 
                                        name=name, *args, **kwds)
            except Exception:
                pass
        elif x in K :
            return self._create_constant_sequence(x, *args, **kwds, name=name)
        else :
            raise NotImplementedError("Conversions not implemented!")

    def _repr_(self):
        r"""
        OUTPUT:
        
        A string representation of the sequence ring.
        
        EXAMPLES::
        
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: C2FiniteSequenceRing(QQ)
            Ring of C^2-finite sequences with base field Rational Field

        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        r = "Ring of C^2-finite sequences with base field " \
            + self.base_ring()._repr_()
        self._cached_repr = r
        return r

    def _latex_(self):
        r"""
        OUTPUT:
        
        A latex representation of the sequence ring.
        
        EXAMPLES::

            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: print(latex(C2FiniteSequenceRing(QQ)))
            \mathcal{C^2}(\Bold{Q})

        """
        return r"\mathcal{C^2}(" + self._base_ring._latex_() + ")"

    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        return sib.name("C2FiniteSequenceRing")(sib(self.base_ring()))

    def guess(self, data, zeros, order_bound = 2, degree_bound = 0, 
              ensure = 1, simple = False) :
        r"""
        Given a list of terms `a(0), a(1), a(2), \dots`, try to guess
        a `C^2`-finite recurrence 
        
        .. MATH::
            c_0(n) a(n) + \dots + c_r(n) a(n+r) = 0 \text{ for all } n \geq 0

        satisfied by these terms. The eigenvalues of the coefficient sequences 
        `c_i` are chosen among the given values ``zeros``.
        
        .. NOTE::
        
            We assume that the base field is big enough to also contain the 
            zeros of the characteristic polynomials of the coefficient 
            sequences.
        
        INPUT:
        
        - ``data`` -- a list of elements in the base field of the ring used for 
          guessing
        - ``zeros`` -- a list of elements in the base field of the ring used for
          the roots of the coefficient sequences
        - ``order_bound`` (default: ``2``) -- a natural number, 
          the bound on the order `r` of the recurrence
        - ``degree_bound`` (default: ``0``) -- a natural number, the maximal
          multiplicity of the given roots in the coefficient polynomials.
        - ``ensure`` (default: ``1``) -- a natural number, the number of excess
          equations used for the linear system; the larger this value the more 
          confident one can be about the result.
        - ``simple`` (default: ``False``) -- a boolean; if ``True`` a simple
          `C^2`-finite recurrence (i.e., a recurrence with leading coefficient
          ``1``) is guessed. 
        
        OUTPUT:
        
        A `C^2`-finite sequence with the given initial values `a(0),a(1),
        \dots`. If no `C^2`-finite recurrence could be found, a ``ValueError``
        is raised.
        
        EXAMPLES::
            
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: C2 = C2FiniteSequenceRing(QQ)
            
            sage: zeros = [1,2,4]
            sage: data = [2^(n^2) for n in range(100)]
            sage: a = C2.guess(data, zeros)
            sage: a[:100] == data
            True
            
            sage: data2 = [n^2*2^(n^2) for n in range(100)]
            sage: C2.guess(data2, zeros)
            Traceback (most recent call last):
            ...
            ValueError: No recurrence found.
            
            sage: a2 = C2.guess(data2, zeros, degree_bound=2)
            sage: a2.degree()
            3
            
            sage: # guess recurrence for f(n^2)
            sage: from itertools import *
            sage: K.<a> = NumberField(x^2-5)
            sage: CK = CFiniteSequenceRing(K)
            sage: CCK = C2FiniteSequenceRing(K)
            sage: zeros = set([1,(1+a)/2,(1-a)/2])
            sage: all_zeros = zeros
            sage: for comb in combinations_with_replacement(zeros, 4) :
            ....:     all_zeros.add(mul(comb))
            sage: f = CK([1,1,-1],[0,1])
            sage: data = [f[n^2] for n in range(100)]
            sage: f2 = CCK.guess(data, all_zeros) # long time
            sage: f2[:100] == data # long time
            True
            
            sage: f2_simple = CCK.guess(data, all_zeros, 3, 0, True) # long time
            sage: f2_simple[:100] == data # long time
            True
            
        """
        if simple :
            return self._guess_simple(data, zeros, order_bound, degree_bound, 
                                      ensure)
            
        # sum_{i=0..r} sum_{j=0..s} sum_{k=0..d} c_ijk n^k zj^n a[n+i] = 0
        m = len(data)-1
        r = order_bound 
        s = len(zeros)-1
        d = degree_bound
        K = self.base_ring()
        cls = self.Element

        num_variables = (r+1)*(s+1)*(d+1)

        # get m-order_bound+1 many equations
        excess_equations = m-order_bound+1 - num_variables 
        if excess_equations < ensure :
            raise ValueError("Not enough data to get a suitable linear system.")
        elif excess_equations > ensure :
            new_bound = ensure+order_bound+num_variables 
            # -> ensure more eq than vars
            return self.guess(data[:new_bound], zeros, order_bound,
                              degree_bound)

        log(cls, "Set up system", 0)
        # set up linear system
        M = matrix(K, m-order_bound+1, num_variables)
        for n in range(m-order_bound+1) :
            # extract coeffs
            for i in range(order_bound+1) :
                for j, zero in enumerate(zeros) :
                    for k in range(d+1) :
                        coeff = (n**k)*(K(zero)**n)*K(data[n+i])
                        col = i*(s+1)*(d+1)+j*(d+1)+k
                        M[n, col] = coeff

        msg = f"Try to solve system with {M.ncols()} variables and " \
              + f"{M.nrows()} equations."
        log(cls, msg, 0)

        sol = M.right_kernel()
        if sol.dimension() == 0 :
            raise ValueError("No recurrence found.")
        log(cls, f"System solved, kernel has dimension {sol.dimension()}", 0)
        log(cls, f"Try if one solution yields valid recurrence", 0)
        for sol in sol.gens() :
            try :                
                seq = self._create_sequence_for_guess(sol, data, zeros, r, d)
                return seq
            except ZeroDivisionError :
                pass 

        raise ValueError("No recurrence found.") 
    
    def _create_sequence_for_guess(self, sol, data, zeros, order_bound,
                                   degree_bound) :
        r"""
        Creates the sequence for the guessed recurrence.
        """
        r = order_bound 
        s = len(zeros)-1
        d = degree_bound
        
        R = self.base()

        sol_split = split_list_rec(sol, [r+1,s+1])
        rec = []
        for i in range(r+1) :
            coeff = self.base()(0)
            for k in range(d+1) :
                for j, zero in enumerate(zeros) :
                    geom = CFiniteSequenceRing.geometric(zero)
                    monom = R._monomial(k)
                    coeff += sol_split[i][j][k]*monom*geom
            rec.append(coeff)
        
        return self._element_constructor_(rec, data)

    def _guess_simple(self, data, zeros, order_bound = 2, degree_bound = 0, 
                      ensure = 1) :
        r"""
        Guess a simple `C^2`-finite recurrence
        """
        m = len(data)-1
        r = order_bound 
        s = len(zeros)-1
        d = degree_bound
        K = self.base_ring()
        cls = self.Element

        num_variables = (r)*(s+1)*(d+1)

        # get m-order_bound+1 many equations
        excess_equations = m-r+1 - num_variables 
        if excess_equations < ensure :
            raise ValueError("Not enough data to get a suitable linear system.")
        elif excess_equations > ensure :
            new_bound = ensure+r+num_variables 
            # -> ensure more eq than vars
            return self._guess_simple(data[:new_bound], zeros, r,
                                      degree_bound)

        log(cls, "Set up system", 0)
        # set up linear system
        M = matrix(K, m-r+1, num_variables)
        for n in range(m-r+1) :
            # extract coeffs
            for i in range(r) :
                for j, zero in enumerate(zeros) :
                    for k in range(d+1) :
                        coeff = (n**k)*(K(zero)**n)*K(data[n+i])
                        col = i*(s+1)*(d+1)+j*(d+1)+k
                        M[n, col] = coeff
                        
        rhs = vector(K, [-data[n+r] for n in range(m-r+1)])

        msg = f"Try to solve system with {M.ncols()} variables and " \
              + f"{M.nrows()} equations."
        log(cls, msg, 0)

        try :
            sol = M.solve_right(rhs)
        except ValueError :
            raise ValueError("No recurrence found.")
                       
        seq = self._create_simple_sequence_for_guess(sol, data, zeros, r, d)
        return seq
    
    def _create_simple_sequence_for_guess(self, sol, data, zeros, order_bound,
                                   degree_bound) :
        r"""
        Creates the simple sequence for the guessed recurrence.
        """
        r = order_bound 
        s = len(zeros)-1
        d = degree_bound
        
        R = self.base()

        sol_split = split_list_rec(sol, [r,s+1])
        rec = []
        for i in range(r) :
            coeff = self.base()(0)
            for k in range(d+1) :
                for j, zero in enumerate(zeros) :
                    geom = CFiniteSequenceRing.geometric(zero)
                    monom = R._monomial(k)
                    coeff += sol_split[i][j][k]*monom*geom
            rec.append(coeff)
        rec += [self.base().one()]
        
        return self._element_constructor_(rec, data)