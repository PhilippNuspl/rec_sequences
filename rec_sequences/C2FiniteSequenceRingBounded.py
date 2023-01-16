# coding: utf-8
r"""
    
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

from .SequenceFieldOfFraction import SequenceFieldOfFraction
from .SequenceFieldOfFraction import FractionFieldSequence
from .CFiniteSequenceRing import CFiniteSequence
from .CFiniteSequenceRing import CFiniteSequenceRing
from .DifferenceDefinableSequenceRing import DifferenceDefinableSequence
from .DifferenceDefinableSequenceRing import DifferenceDefinableSequenceRing
from .utility import log, shift_vector
from .IntegerRelations import IntegerRelations

####################################################################################################

class C2FiniteSequenceBounded(DifferenceDefinableSequence):
    r"""
    A C^2-finite sequence, i.e. a sequence where every term can be determined by a linear recurrence
    with coefficients coming from a C-finite sequence ring and finitely many initial values. We assume that this
    recurrence holds for all values and that the leading coefficient only
    contains finitely many zeros (which is not checked).
    """

    log = logging.getLogger("C2FinBound")

    def __init__(self, parent, coefficients, initial_values, name = "a", 
                 is_gen = False, construct=False, cache=True, *args, **kwds):
        r"""
        Construct a `C^2`-finite sequence `a(n)` with recurrence

        .. MATH::
            c_0(n) a(n) + \dots + c_r(n) a(n+r) = 0 \text{ for all } n \geq 0

        from given list of coefficients `c_0, ... , c_r` and given list of
        initial values `a(0), ..., a(r-1)`.

        .. NOTE::
        
            We assume that the leading coefficient `c_r` only contains finitely 
            many zero terms.
        
        INPUT:

        - ``parent`` -- a ``C2FiniteSequenceRingBounded``
        - ``coefficients`` -- the coefficients of the recurrence
        - ``initial_values`` -- a list of initial values, determining the 
          sequence with at least order of the recurrence many values
        - ``name`` (default "a") -- a name for the sequence

        OUTPUT:

        A sequence determined by the given recurrence and 
        initial values.
        
        EXAMPLES::

            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRingBounded(QQ)
                
            sage: n = var("n")
            sage: c = C(2^n+1)
            sage: a = C2([c, -1], [3])
            sage: a
            C^2-finite sequence of order 1 and degree 2 with coefficients:
            > c0 (n) : C-finite sequence c0(n): (2)*c0(n) + (-3)*c0(n+1) + (1)*c0(n+2) = 0 and c0(0)=2 , c0(1)=3
            > c1 (n) : C-finite sequence c1(n)=-1
            and initial values a(0)=3
            
        """  
        if not isinstance(parent, C2FiniteSequenceRingBounded) :
            raise TypeError("Parent has to be a bounded C^2-finite sequence "
                            "ring.")

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
        
            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRingBounded(QQ)
                
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
        
            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRingBounded(QQ)
            
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
    def eigenvalues(self) :
        r"""
        Returns a list of the eigenvalues that appear in the coefficients
        of the defining recurrence.
        
        OUTPUT:
        
        A list of the pairwise distinct eigenvalues of the coefficients.
        
        EXAMPLES::
        
            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRingBounded(QQ)
            
            sage: var("n");
            n
            sage: c = C(2^n+3^n)
            sage: d = C((-1)^n)
            sage: e = C(1)
            
            sage: sorted(C2([c,d*c,d,c,e], [1,2,3,5]).eigenvalues())
            [-3, -2, -1, 1, 2, 3]
        
        """
        all_eigenvalues = []
        for coefficient in self.coefficients() :
            all_eigenvalues += coefficient.roots(multiplicities=False)
            
        return list(set(all_eigenvalues))
    
    def torsion_number(self, *others) :
        r"""
        Compute the torsion number of some C^2-finite sequences.
        
        INPUT:
        
        - ``others`` -- C^2-finite sequences
        
        OUTPUT:
        
        The torsion number of ``self`` and the other given C^2-finite sequences.
        
        EXAMPLES::
        
            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRingBounded(QQ)
            
            sage: alt = C(10*[1,-1])
            sage: C2([alt, 1], [1]).torsion_number()
            2
        
        """
        eigenvalues = self.eigenvalues()
        for other in others :
            eigenvalues += other.eigenvalues()
            
        eigenvalues = list(set(eigenvalues))
        basis_relations = IntegerRelations.integer_relations(eigenvalues)
        
        divisors = basis_relations.elementary_divisors()
        return sorted(divisors)[-1]
    
    # arithmetic operations 
    
    def subsequence(self, u, v=0, check_torsion_number = True, *args, **kwargs):
        r"""
        Returns the sequence `c(n u + v)`.

        INPUT:

        - ``u`` -- a natural number
        - ``v`` (default: ``0``) -- a natural number

        OUTPUT:
        
        The sequence `c(n u + v)`.
            
        """
        if v != 0 :
            return self.shift(v).subsequence(u, *args, **kwargs)
        
        # can assume v=0
        
        if u == 1 :
            return self
        
        if check_torsion_number :
            torsion_number = self.torsion_number()
        QC = SequenceFieldOfFraction(self.base())
        
        log(self, f"Compute subsequence at {u}", 0)
        if not check_torsion_number or torsion_number.divides(u) :
            # compute subsequence directly
            M = self._companion_subsequence(u, field=True)
            w0 = identity_matrix(QC, M.ncols()).column(0)
            
            recurrence = self.parent()._compute_recurrence(M, w0)
            # TODO: make sure to compute sufficiently many initial values
            initial_values = [self[u*n] for n in range(len(recurrence))]
            
            return self.parent()(recurrence, initial_values)
            
        else :
            # compute subsequence as interlacings
            subsequences = [self.subsequence(torsion_number*u, i*u, False)
                            for i in range(torsion_number)]
            return subsequences[0].interlace(*subsequences[1:])
            
    def _add_(self, right, *args, **kwargs):
        r"""
        Return the termwise sum of ``self`` and ``right``.
        
        INPUTS:

        - ``right`` -- a sequence over the same ring as ``self``.
        
        OUTPUTS: 
        
        The addition of ``self`` with ``right``.
        
        EXAMPLES::
        
            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRingBounded(QQ)
            
            sage: var("n");
            n
            sage: c = C(2^n+1)
            sage: a = C2([c, -1], [3])
            sage: f = C(3^n+5^n)
            sage: a_plus_f = a+f
            sage: a_plus_f.order() == a.order()+f.order()
            True
            sage: a_plus_f[:10] == [ai+fi for ai, fi in zip(a[:10],f[:10])]
            True
            
        """
        if self.__is_zero__():
            return right
        if right.__is_zero__():
            return self
        
        QC = SequenceFieldOfFraction(self.base())
                
        torsion_number = self.torsion_number(right)
        log(self, f"Addition: Torsion number {torsion_number} computed", 0)
        subsequences = []
        for i in range(torsion_number) :
            log(self, f"Addition: Subsequences for i={i}", 0)
            a_subs = self.subsequence(torsion_number, i, False)
            b_subs = right.subsequence(torsion_number, i, False)
            M = a_subs._companion_sum(b_subs, True)
            w0 = vector(QC, M.nrows()) # first column of matrix
            w0[0] = QC(1); w0[a_subs.order()] = QC(1)

            recurrence = self.parent()._compute_recurrence(M, w0)
            # TODO: make sure to compute sufficiently many initial values
            r = len(recurrence)
            initial_values = [sum(x) for x in zip(a_subs[:r], b_subs[:r])]
            
            subsequences.append(self.parent()(recurrence, initial_values))
            
        return subsequences[0].interlace(*subsequences[1:])
        
    def _mul_(self, right, *args, **kwargs):
        r"""
        Return the termwise multiplication of ``self`` and ``right``.
        
        INPUTS:

        - ``right`` -- a sequence over the same ring as ``self``.
        
        OUTPUTS: 
        
        The product of ``self`` with ``right``.
        
        EXAMPLES::
        
            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRingBounded(QQ)
            
            sage: var("n");
            n
            sage: c = C(2^n+1)
            sage: a = C2([c, -1], [3])
            sage: f = C(3^n+5^n)
            sage: a_tim_f = a*f
            sage: a_tim_f.order() == a.order()*f.order()
            True
            sage: a_tim_f[:10] == [ai*fi for ai, fi in zip(a[:10],f[:10])]
            True
            
        """
        if self.__is_zero__():
            return right
        if right.__is_zero__():
            return self
        
        QC = SequenceFieldOfFraction(self.base())
                
        torsion_number = self.torsion_number(right)
        log(self, f"Product: Torsion number {torsion_number} computed", 0)
        subsequences = []
        for i in range(torsion_number) :
            log(self, f"Product: Subsequences for i={i}", 0)
            a_subs = self.subsequence(torsion_number, i, False)
            b_subs = right.subsequence(torsion_number, i, False)
            M = a_subs._companion_product(b_subs, True)
            w0 = vector(QC, M.nrows()) # first column of matrix
            w0[0] = QC(1)

            recurrence = self.parent()._compute_recurrence(M, w0)
            # TODO: make sure to compute sufficiently many initial values
            r = len(recurrence)
            initial_values = [prod(x) for x in zip(a_subs[:r], b_subs[:r])]
            
            subsequences.append(self.parent()(recurrence, initial_values))
            
        return subsequences[0].interlace(*subsequences[1:])


####################################################################################################

class C2FiniteSequenceRingBounded(DifferenceDefinableSequenceRing):
    r"""
    A Ring of C^2-finite sequences where the orders for closure properties
    are bounded.
    """

    Element = C2FiniteSequenceBounded
    log = logging.getLogger("C2FinRingBound")

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
        
            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            sage: C2FiniteSequenceRingBounded(QQ)
            Ring of bounded C^2-finite sequences with base field Rational Field
            
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

        - ``x`` is already a bounded `C^2`-finite sequence.
        - ``x`` is a list of C-finite sequences and ``y`` is a list of field 
          elements. Then ``x`` is interpreted as the coefficients of the 
          recurrence and ``y`` as the initial 
          values of the sequence, i.e. `a(0), ..., a(r-1)`.
        - ``x`` is a C-finite sequence.
        - ``x`` can be converted into a field element. Then it is interpreted 
          as the constant sequence `(x)_{n \in \mathbb{N}}`
        
        EXAMPLES::
        
            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRingBounded(QQ) 
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
            return self.Element(self, x, y, name=name, *args, **kwds)
        elif x in R and isinstance(x, CFiniteSequence) : # check whether R is sequence ring
            try :
                coeffs_R = [R(coeff) for coeff in x.coefficients()]
                return self.Element(self, coeffs_R, x.initial_values(), 
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
        
            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            sage: C2FiniteSequenceRingBounded(QQ)
            Ring of bounded C^2-finite sequences with base field Rational Field

        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        r = "Ring of bounded C^2-finite sequences with base field " \
            + self.base_ring()._repr_()
        self._cached_repr = r
        return r

    def _latex_(self):
        r"""
        OUTPUT:
        
        A latex representation of the sequence ring.
        
        EXAMPLES::

            sage: from rec_sequences.C2FiniteSequenceRingBounded import *
            sage: print(latex(C2FiniteSequenceRingBounded(QQ)))
            \mathcal{C^2}(\Bold{Q})

        """
        return r"\mathcal{C^2}(" + self._base_ring._latex_() + ")"

    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        return sib.name("C2FiniteSequenceRingBounded")(sib(self.base_ring()))

# helper for arithmetic

    def _compute_recurrence(self, M, w0):
        r"""
        """
        QC = SequenceFieldOfFraction(self.base())
        
        rows = M.nrows()
        system = matrix(QC, rows, rows+1)
        
        # set up linear system
        log(self, "Set up linear system", 1)
        w = w0
        system.set_column(0, w)
        for i in range(1, rows+1) :
            w = M*shift_vector(w)
            system.set_column(i, w)
            
        # solve linear system, certainly has an element in the kernel
        # as it is underdetermined
        log(self, f"Solve linear system of size {system.dimensions()}", 1)
        solution = system.right_kernel()
        log(self, f"Kernel has dimension {solution.dimension()}", 1)
        solution = solution.basis()[0]
        # strip zeros from end 
        for i in range(len(solution)-1,-1,-1) :
            if not solution[i].is_zero() :
                solution = solution[:i+1]
                log(self, f"Solution stripped off {len(solution)-1-i} zeros", 1)
                break 
            
        # clear denominators, use naively product of denominators to clear them
        log(self, "Clear denominators of solution", 1)

        denominators_cleared = [prod(solution[j].d() \
                                       for j in range(len(solution)) if j != i)\
                                    for i in range(len(solution))]
        solution_cleared = [sol.n()*denom \
                        for sol, denom in zip(solution, denominators_cleared)]
        
        return solution_cleared
        
        