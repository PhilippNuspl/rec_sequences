# coding: utf-8
r"""
Class which represents functional equations of a certain type.
These are formal power series `g(x)` that satisfy an equation
of the form
 
.. MATH::
    \sum_{k=1}^m p_k(x) g^{(d_k)} (\lambda_k x) = p(x)
    
for constants `\lambda_k`, natural numbers `d_k` and
polynomials `p_k(x), p(x)`. The generating functions of `C^2`-finite sequences
satisfy such equations (:meth:`rec_sequences.C2FiniteSequenceRing.C2FiniteSequence.functional_equation`). 
The left-hand side of the functional equation is constructed by triples 
`(d_k, \lambda_k, p_k)`.

EXAMPLES::

    sage: from rec_sequences.FunctionalEquation import *
    
    sage: R.<x> = PolynomialRing(QQ)
    sage: eq = [(0, -1, 3), (1, 2, x+2)]
    sage: print(FunctionalEquation(R, eq, x^2)) 
    (3)g(-x) + (x + 2)g'(2x) = x^2
    
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

from sage.arith.misc import falling_factorial
from sage.symbolic.ring import SR
from sage.calculus.var import var
from sage.structure.sage_object import SageObject
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing

class FunctionalEquation(SageObject):
    def __init__(self, base, equation, rhs = 0, name = "g"):
        r"""
        Constructs a functional equation of the form
        
        .. MATH::
            \sum_{k=1}^m p_k(x) g^{(d_k)} (\lambda_k x) = p(x)
            
        for constants `\lambda_k`, natural numbers `d_k` and
        polynomials `p_k(x), p(x)`


        INPUT:

        - ``base`` -- the base ring which contains the polynomial coefficients 
          `p_k,p` and the constants `\lambda_k`
        - ``equation`` -- a list of triples `(d_k, \lambda_k, p_k)`
        - ``rhs`` -- the right-hand side `p`, a polynomial in ``base``
        - ``name`` (default: "g") -- a name which is displayed if the equation 
          is displayed
        
        OUTPUT:

        A functional equation of the described form.
        
        EXAMPLES::
        
            sage: from rec_sequences.FunctionalEquation import *
            
            sage: R.<x> = PolynomialRing(QQ)
            sage: eq = [(0, -1, 3), (1, 2, x+2)]
            sage: print(FunctionalEquation(R, eq, x^2)) 
            (3)g(-x) + (x + 2)g'(2x) = x^2
            
        """
        SageObject.__init__(self)
        
        if not is_PolynomialRing(base) :
            raise ValueError("Base has to be a polynomial ring")
            
        self._base = base 
        self._base_ring = base.base()
        self._name = name

        self._equation = {}
        for triple in equation :
            key = (triple[0], self._base_ring(triple[1]))
            value = self._base(triple[2])
            if key not in self._equation :
                self._equation[key] = value 
            else :
                self._equation[key] += value
        self._rhs = base(rhs)

    @staticmethod
    def _simplify_print(el, latex=True) :
        if el == 1 :
            return ""
        elif el == -1 :
            return "-"
        else :
            if latex :
                return el._latex_()
            else :
                return str(el)

    def _repr_(self, name=None):
        r"""
        Produces a string representation of the equation where the
        derivative of a function `g(x)` is denoted by `g'(x)`
        where `x` denotes the generator of the base ring.
        
        INPUT:
        
        - ``name`` (optional) -- a string used as the name of the sequence;
          if not given, ``self.name()`` is used.
        
        OUTPUT:
        
        A string representation of the functional equation.
        
        EXAMPLES::
        
            sage: from rec_sequences.FunctionalEquation import *
            
            sage: R.<x> = PolynomialRing(QQ)
            sage: eq = [(0, 1, x), (1, -1, 3), (2, 2, x+2)]
            sage: print(FunctionalEquation(R, eq, x^2)) 
            (x)g(x) + (3)g'(-x) + (x + 2)g''(2x) = x^2
            
        """
        if name==None :
            name = self._name
            
        var_name = str(self._base.gen())
        terms = ["({}){}{}({}{})".format(
                        str(self._equation[key]), 
                        name, 
                        key[0]*"'", 
                        FunctionalEquation._simplify_print(key[1], False),
                        var_name) 
                    for key in self._equation 
                    if not self._equation[key].is_zero()]
        
        r = " + ".join(terms) + " = " + str(self._rhs)

        return r

    def _latex_(self, name=None):
        r"""
        Creates a latex representation of the functional equation where the
        derivative of a function `g(x)` is denoted by `g'(x)`
        where `x` denotes the generator of the base ring.
        
        OUTPUT: 
        
        A latex representation of the functional equation.
        
        EXAMPLES::
        
            sage: from rec_sequences.FunctionalEquation import *
            
            sage: R.<x> = PolynomialRing(QQ)
            sage: eq = [(2, 2, x+2)]
            sage: print(latex(FunctionalEquation(R, eq, 0)))
            \left(x + 2\right)g''\left(2x\right) = 0
            
        """
        if name==None :
            name = self._name
            
        var_name = str(self._base.gen())
        terms = ["\\left({}\\right){}{}\\left({}{}\\right)".format(
                        self._equation[key]._latex_(), 
                        name, 
                        key[0]*"'", 
                        FunctionalEquation._simplify_print(key[1]),
                        var_name) 
                    for key in self._equation 
                    if not self._equation[key].is_zero()]
        
        r = " + ".join(terms) + " = " + self._rhs._latex_()

        return r

    def recurrence(self, ring) :
        r"""
        Creates a recurrence with C-finite coefficients for the coefficient
        sequence of the function.
        
        .. NOTE::

            Every coefficient sequence of a function satisfying a 
            functional equation of the given type satisfy a linear
            recurrence with C-finite coefficients. However, not all
            of those coefficient sequences are `C^2`-finite. 
        
        INPUT:
        
        - ``ring`` -- a :class:`rec_sequences.CFiniteSequenceRing`
          which contains the coefficients of the recurrences.
          
        OUTPUT:
        
        A list of C-finite sequences `c_0,\dots,c_r` such that the coefficient
        sequence `a(n)` of the function satisfies the recurrence 
        `\sum_{i=0}^r c_i(n) a(n+i)` for all `n`.
                
        EXAMPLES::

            sage: from rec_sequences.FunctionalEquation import *
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: R.<x> = PolynomialRing(QQ)
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRing(QQ)

            sage: n = var("n")
            sage: c = C(2^n+1)
            sage: d = C(3^n)
            sage: a = C2([c, d], [1])

            sage: func_eq = a.functional_equation()
            sage: print(func_eq)
            (x)g(2x) + (x)g(x) + (1/3)g(3x) = 1/3
            
            sage: func_eq.recurrence(C) == a.coefficients()
            True
            
            sage: eq = [(0, 1, x), (1, -1, 3*x), (1, 2, 2)]
            sage: func_eq_2 = FunctionalEquation(R, eq, x^2)
            sage: func_eq_2.recurrence(C)
            [C-finite sequence a(n)=1,
            C-finite sequence a(n): (1)*a(n) + (2)*a(n+1) + (1)*a(n+2) = 0 
            and a(0)=-3 , a(1)=6,
            C-finite sequence a(n): (4)*a(n) + (-4)*a(n+1) + (1)*a(n+2) = 0 
            and a(0)=16 , a(1)=48]
            
        """
        n = var("n")

        coeffs = dict()
        valid_from = 0
        # get all coefficients for C2-finite recurrence
        for i, gamma in self._equation :
            p = self._equation[(i, gamma)]
            for j in p.dict() :
                factor = p[j]*falling_factorial(n+i-j, i)*gamma**(n+i-j)
                valid_from = max(valid_from, j)
                if i-j in coeffs :
                    coeffs[i-j] += SR(factor)
                else :
                    coeffs[i-j] = SR(factor)

        shift = max(-min(coeffs.keys()), 0)
        order = max(coeffs.keys()) + shift
        # shift recurrence so that we start at a(n)
        coeffs_new = [coeffs.get(i-shift, SR(0)) for i in range(order+1)]
        coeffs_cfin = [ring(c, n).shift(shift) for c in coeffs_new]

        return coeffs_cfin

    def sequence(self, ring, init_values) :
        r"""
        Creates a `C^2`-finite sequence for the coefficient
        sequence of the function.
        
        .. NOTE::

            Every coefficient sequence of a function satisfying a 
            functional equation of the given type satisfy a linear
            recurrence with C-finite coefficients. However, not all
            of those coefficient sequences are `C^2`-finite. 
        
        INPUT:
        
        - ``ring`` -- a :class:`rec_sequences.C2FiniteSequenceRing`
          which contains the coefficients of the recurrences.
        - ``init_values`` -- initial values used to construct the `C^2`-finite
          sequence.
          
        OUTPUT:
        
        A `C^2`-finite sequence representing the coefficient sequence of the 
        function.
                
        EXAMPLES::

            sage: from rec_sequences.FunctionalEquation import *
            sage: from rec_sequences.C2FiniteSequenceRing import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: R.<x> = PolynomialRing(QQ)
            sage: C = CFiniteSequenceRing(QQ)
            sage: C2 = C2FiniteSequenceRing(QQ)

            sage: n = var("n")
            sage: c = C(2^n+1)
            sage: d = C(3^n)
            sage: a = C2([c, d], [1])

            sage: func_eq = a.functional_equation()
            sage: print(func_eq)
            (x)g(2x) + (x)g(x) + (1/3)g(3x) = 1/3
            
            sage: func_eq.sequence(C2, a[:5]) == a
            True
            
        """
        return ring(self.recurrence(ring.base()), init_values)
                
            

