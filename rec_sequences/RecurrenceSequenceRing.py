# coding: utf-8
r"""
This class provides common functionalities for rings of sequences
defined by linear recurrence relations. The structure of the
rings is given by:

- :class:`CommutativeAlgebra`
    - :class:`rec_sequences.RecurrenceSequenceRing`
        - :class:`rec_sequences.DFiniteSequenceRing`
            - :class:`rec_sequences.CFiniteSequenceRing`
        - :class:`rec_sequences.DifferenceDefinableSequenceRing`
            - :class:`rec_sequences.C2FiniteSequenceRing`
            
So all rings are subclasses of a commutative algebra over a base field
which is the field where all terms of the sequences live.
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

from sage.arith.all import gcd
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
from sage.categories.rings import Rings
from sage.categories.commutative_algebras import CommutativeAlgebras
from sage.categories.commutative_rings import CommutativeRings
from sage.structure.unique_representation import UniqueRepresentation
from sage.misc.inherit_comparison import InheritComparisonClasscallMetaclass
from sage.functions.other import floor



####################################################################################################


class RecurrenceSequenceElement(CommutativeAlgebraElement):
    r"""
    A sequence defined by a linear recurrence equation with some initial values.
    """
    def __init__(self, parent, coefficients, initial_values, name = "a", 
                 is_gen = False, construct=False, cache=True):
        r"""
        Construct a linear recurrence sequence `a(n)` with given coefficients
        of the recurrence and given initial values.

        INPUT:

        - ``parent`` -- a ``RecurrenceSequenceRing``
        - ``coefficients`` -- the coefficients of the recurrence
        - ``initial_values`` -- a list of initial values, determining the 
          sequence with at least order of the recurrence many values
        - ``name`` (default "a") -- a name for the sequence

        OUTPUT:

        A recurrence sequence determined by the given recurrence and initial 
        values.
        """
        try :
            if parent.Element != self.__class__ :
                raise TypeError
        except :
            raise TypeError(f"Parent has to be {parent.Element}.")
        
        if not coefficients :
            raise ValueError("No coefficients given.")

        if len(initial_values) < len(coefficients)-1 :
            raise ValueError("Not enough initial values given.")

        CommutativeAlgebraElement.__init__(self, parent)
        
        K = parent.base_ring()
        order = len(coefficients)-1
        self._initial_values = [K(i) for i in initial_values[:order]]
        self._values = list(self._initial_values)
        self._name = name
        self._is_gen = is_gen

    def __copy__(self):
        r"""
        Return a "copy" of ``self``. This is just ``self``, since recurrence
        sequence elements are immutable.
        """
        return self
#tests

    def __is_zero__(self):
        r"""
        Return whether ``self`` is the zero sequence `0,0,0,\dots`.
        This is the case iff all the initial values are `0` or ``None``.
        
        OUTPUT:
        
        ``True`` if the sequence is constantly zero, ``False`` otherwise.
        
        """
        for x in self.initial_values():
            if x != self.base_ring().zero():
                return False
        return True

    def is_zero(self):
        r"""
        Return whether ``self`` is the zero sequence `0,0,0,\dots`.
        This is the case iff all the initial values are `0` or ``None``.
        
        .. note::
        
            We circumvent Sage's own ``is_zero`` implementation. This is 
            because it calls ``__eq__`` which calls ``__add__`` which might 
            itself check whether sequences are zero.
        
        OUTPUT:
        
        ``True`` if the sequence is constantly zero, ``False`` otherwise.
        
        """
        return self.__is_zero__()

    def __eq__(self, right):
        r"""
        Returns whether the two sequences ``self`` and ``right`` are equal.
        More precisely it is tested if the difference of ``self`` and ``right`` 
        equals `0`.
        """
        if right == None :
            return False
        if self.parent() != right.parent():
            right = self.parent()(right)
        return (self.__add__(-right)).__is_zero__()

    def __ne__(self,right):
        r"""
        Return ``True`` if the sequences ``self`` and ``right`` are NOT equal 
        and ``False`` otherwise
        """
        return not self.__eq__(right)

    def _is_atomic(self):
        r"""
        """
        raise NotImplementedError

    def is_unit(self):
        r"""
        Return ``True`` if ``self`` is a unit. We try to invert the element.
        If this is possible, the element is a unit, otherwise we 
        return ``False``.
        
        OUTPUT:
        
        ``True`` if the sequence is invertible and ``False`` otherwise.
        """
        try :
            self.__invert__()
            return True
        except ValueError:
            return False

    def is_gen(self):
        r"""
        OUTPUT:
        
        Return ``False``; the parent ring is not finitely generated.
        """
        return False

    def prec(self):
        r"""
        OUTPUT:
        
        The precision of these objects is infinite.
        """
        return Infinity

    def __setitem__(self, n, value):
        r"""
        """
        raise IndexError("Sequences are immutable")

    def __iter__(self):
        raise NotImplementedError

#representation

    def __hash__(self):
        r"""
        Hash sequence by hasing the initial values.
        """
        try:
            return self._cached_hash
        except AttributeError:
            pass
        h = self._cached_hash = hash(tuple(self[:self.order()]))
        return h
    
    def name(self):
        r"""
        OUTPUT:
        
        The name of the sequence as a string.
        """
        return self._name

# arithmetic
    def shift(self, k=1):
        r"""
        Shifts ``self`` ``k``-times.

        INPUT:

        - ``k`` (default: ``1``) -- an integer

        OUTPUT:

        The sequence `(a(n+k))_{k \in \mathbb{N}}`.
        """
        if k==0 :
            return self

        coeffs_shifted = [coeff.shift(k) for coeff in self.coefficients()]
        return type(self)(self.parent(), coeffs_shifted, 
                          self[k:k+self.order()])
        
    def sum(self) :
        r"""
        Returns the sequence `\sum_{i=0}^n c(i)`, the sequence describing
        the partial sums.
        
        OUTPUT: 
        
        The sequence `\sum_{i=0}^n c(i)`.
        """
        coeffs = self.coefficients()
        r = self.order()
        others = [coeffs[i-1].shift()-coeffs[i].shift() for i in range(1,r+1)]
        coeffs_sum = [-coeffs[0].shift()] + others + [coeffs[r].shift()]
        initial_values = [sum(self[j] for j in range(i+1)) for i in range(r+1)]
        return type(self)(self.parent(), coeffs_sum, initial_values)

    def __div__(self, right):
        r"""
        Try to divide by mutliplying with the inverse.
        """
        return self*right.__invert__()

    def _add_(self, right):
        r"""
        Return the sum of ``self`` and ``right``.
        """
        raise NotImplementedError

    def _neg_(self):
        r"""
        Return the negative of ``self``.
        """
        neg_intial_values = [-val for val in self.initial_values()]
        return type(self)(self.parent(), self.coefficients(), neg_intial_values)
    
    def difference(self, i=1) :
        r"""
        Returns the forward difference `\Delta c = c(n+1)-c(n)` of the 
        sequence `c`.
        
        INPUT:
        
        - ``i`` (default: ``1``) -- a natural number
        
        OUTPUT:
        
        The iterated forward difference sequence `\Delta^i c`.
        
        """
        if i == 1 :
            return self.shift() - self
        elif i == 0 :
            return self
        else : # assume i is natural and i > 0
            return (self.shift() - self).difference(i-1)

    def _mul_(self, right):
        r"""
        Return the product of ``self`` and ``right``.
        """
        raise NotImplementedError

    def __pow__(self, n, modulus = None):
        r"""
        """
        return self._pow(n)

    def _pow(self, n):
        r"""
        Return ``self`` to the n-th power using repeated squaring.

        INPUT:

        - ``n`` -- a non-negative integer

        OUTPUT:

        The sequence to the power of ``n``.
        """
        if n == 0:
            return self.parent().one()
        if n == 1:
            return self

        #for small n the traditional method is faster
        if n <= 10:
            return self * (self._pow(n-1))

        #for larger n we use repeated squaring
        else:
            result = self.parent().one()
            bit = bin(n)[2:] #binary representation of n
            for i in range(len(bit)):
                result = result * result
                if bit[i] == '1':
                    result = result * self
            return result

    def __floordiv__(self,right):
        r"""
        """
        raise NotImplementedError

    def __mod__(self, other):
        r"""
        """
        raise NotImplementedError
        
    def prepend(self, values) :
        r"""
        Prepends the given values to the sequence.
        
        Input
        - ``values`` -- list of values in the base ring

        OUTPUT:
        A sequence having the same terms with the additional ``values``
        at the beginning.
        """
        K = self.parent().base_ring()
        rec = len(values)*[K(0)]+self.coefficients()
        initial_values = values + self[:self.order()]
        ret = type(self)(self.parent(), rec, initial_values)
        return ret
    
    def nomial_coefficient(self, n, k, m=1) :
        r"""
        Computes the term
        
        .. MATH::
            \prod_{i=1}^k \frac{c(m(n-i+1))}{c(mi)}.
            
        If `c` is the Fibonacci sequence, this is known as the
        fibonomial coefficient.
        
        INPUT:
        
        - ``n`` -- a positive integer
        - ``k`` -- a positive integer
        - ``m`` (default: ``1``) -- a positive integer
        
        OUTPUT:
        
        The coefficient of the sequence at the given term ``n``.
        """
        if k > n or k < 0 :
            return 0
        else :
            return prod(self[m*(n-i+1)]/self[m*i] for i in range(1,k+1))

#base ring related functions

    def base_ring(self):
        r"""
        Return the base field of the parent of ``self``.
        """
        return self.parent().base_ring()
    
    def base(self):
        r"""
        Return the base of the parent of ``self``.
        """
        return self.parent().base()

#part extraction functions

    def __getitem__(self,n):
        r"""
        Return the n-th term of ``self``.

        INPUT:

        - ``n`` -- a natural number

        OUTPUT:

        The n-th sequence term of ``self`` (starting with the 0-th,
        i.e. to get the first term one has to call ``self[0]``)
        """
        if isinstance(n, slice) :
            if n.stop == None :
                raise ValueError("Sequences are infinite. Need to specify"
                                 " upper bound.")
            elif n.step != None and n.step != 1 :
                raise NotImplementedError
            elif n.start != None and n.start < 0 :
                raise NotImplementedError("Cannot evaluate at negative indices")
            if len(self._values) < n.stop :
                self._create_values(n.stop+1)
            if n.start == None or n.start >= 0 :
                return self._values[n]
            else :
                return self._values[:n.stop]

        if n >= 0 :
            try:
                return self._values[n]
            except IndexError:
                self._create_values(n+2)
                return self._values[n]
        else : # n < 0
            raise NotImplementedError("Cannot evaluate at negative indices")


    def _create_values(self, n) :
        r"""
        Create values [self[0],...,self[n]] in self._values
        """
        raise NotImplementedError
    
    def coefficients(self):
        r"""
        OUTPUT:
        
        The coefficients of the recurrence of ``self`` as a list.
        """
        raise NotImplementedError

    def leading_coefficient(self):
        r"""
        OUTPUT:
        
        The leading coefficient of the recurrence.
        """
        return self.coefficients()[-1]

    def order(self):
        r"""
        OUTPUT:
        
        The order of the recurrence of ``self``.
        """
        return len(self.coefficients())-1

    def initial_values(self):
        r"""
        OUTPUT:
        
        The initial values of ``self`` in form of a list.
        """
        return self._initial_values
    
    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        sib_ring = sib(self.parent())
        seq = sib_ring(sib(self.coefficients()), sib(self.initial_values()),
                       name=self._name)
        return seq

####################################################################################################

class RecurrenceSequenceRing(UniqueRepresentation, CommutativeAlgebra):
    r"""
    A Ring of linear recurrence sequences over a field.
    """

    Element = RecurrenceSequenceElement

# constructor
    def __init__(self, field = QQ, name=None, element_class=None, 
                 category=None):
        r"""
        Constructor for a recurrence sequence sequence ring.

        INPUT:

        - ``field`` -- a field of characteristic zero over which the 
            sequence ring is defined.

        OUTPUT:

        A ring of recurrence sequences.
        """
        if field not in Fields() :
            raise TypeError("Recurrence sequences need to be defined "          
                            "over a field")

        self._base_ring = field

        CommutativeAlgebra.__init__(self, field, 
                                    category=CommutativeAlgebras(field))
        
    def _element_constructor_(self, x, y=None, name="a", check=True, 
                              is_gen = False, construct=False, **kwds):
        r"""
        Tries to construct a sequence `a(n)`.
        
        This is possible if:

        - ``x`` is already a sequence in the right ring.
               
        """
        if isinstance(x, self.Element) :
            return self._conversion(x, name=name)
        else :
            raise NotImplementedError("Conversion not implemented!") 
        
        
    def _conversion(self, x, name=None) :
        r"""
        Tries to convert a sequence ``x`` to this ring.
        """
        if not name :
            name = x._name
        K = self.base_ring()
        new_initial_values = [K(val) for val in x.initial_values()]
        R = self.base()
        new_rec = [R(coeff) for coeff in x.coefficients()]
        return self.Element(self, new_rec, new_initial_values, 
                            name = name)

    def _coerce_map_from_(self, P):
        r"""
        """
        if self.base_ring().has_coerce_map_from(P) :
            return True
        elif isinstance(P, type(self)) :
            return self.base_ring().has_coerce_map_from(P.base_ring())
        else :
            return False
            
    def one(self) :
        r"""
        OUTPUT:
        
        The constant sequence `1,1,\dots`.
        """
        return self._create_constant_sequence(self.base_ring().one())

    def zero(self) :
        r"""
        OUTPUT:
        
        The constant sequences `0,0,\dots`.
        """
        return self._create_constant_sequence(self.base_ring().zero())

    def _create_constant_sequence(self, x, name="a") :
        r"""
        Creates a constant sequences `x,x,...` if `x` is in the base field.
        """
        K = self.base_ring()
        if x in K :
            return self.Element(self, [K.one(),-K.one()], [x], name=name)
        
    def _eq_(self,right):
        r"""
        Tests if the two rings ``self``and ``right`` are equal.
        This is the case if they have the same class and are defined
        over the same base ring.
        """
        if not isinstance(right, type(self)) :
            return False
        try:
            return self.base() == right.base()
        except:
            return False

    def is_integral_domain(self, proof = True):
        r"""
        Returns whether ``self`` is an integral domain.
        This is not the case.
        
        OUTPUT:
        
        ``False``
        """
        return False

    def is_noetherian(self):
        r"""
        Returns whether ``self`` is a Noetherian ring.
        This is not the case.
        
        OUTPUT:
        
        ``False``
        """
        raise False

    def is_commutative(self):
        r"""
        Returns whether ``self`` is a commutative ring.
        This is the case.
        
        OUTPUT:
        
        ``True``
        """
        return True

    def _is_valid_homomorphism_(self, domain, im_gens):
        r"""
        """
        raise NotImplementedError

    def __hash__(self):
        r"""
        Create a hash using the string representation of the ring.
        """
        try:
            return self._cached_hash
        except AttributeError:
            pass
        h = self._cached_hash = hash(str(self))
        return h

    def construction(self):
        r"""
        Shows how the given ring can be constructed using functors.
        
        OUTPUT:
        
        A functor ``F`` and a ring ``R`` such that ``F(R)==self``
        """
        return RecurrenceSequenceRingFunctor(), self.base()

    def _repr_(self):
        r"""
        A string representation of the ring.
        """
        raise NotImplementedError

    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        class_name = [class_name for class_name in type(self).__bases__ 
                        if "rec_sequences" in str(class_name)][0].__name__
        return sib.name(class_name)(sib(self.base()))

    def base_ring(self):
        r"""
        OUTPUT: 
        
        The base field over which the sequence ring is defined
        """
        return self._base_ring
    
    def base(self):
        r"""
        OUTPUT:
        
        The base field over which the sequence ring is defined
        """
        return self.base_ring()

    def is_finite(self):
        r"""
        Returns whether ``self`` is a finite ring.
        This is not the case.
        
        OUTPUT:
        
        ``False``
        """
        return False

    def is_exact(self):
        r"""
        Returns whether ``self`` is an exact ring.
        This is the case.
        
        OUTPUT:
        
        ``True``        
        """
        return self.base_ring().is_exact()

    def is_field(self):
        r"""
        Returns whether ``self`` is a field.
        This is not the case.
        
        OUTPUT:
        
        ``False``
        """
        return False

    def _an_element_(self, *args, **kwds):
        r"""
        OUTPUT:
        
        The Fibonacci sequence.
        """
        K = self.base_ring()
        return self.Element(self, [K(1),K(1),-K(1)], [K(0), K(1)])

class RecurrenceSequenceRingFunctor(ConstructionFunctor):
    def __init__(self):
        r"""
        Constructs a ``RecurrenceSequenceRingFunctor``.
        """
        ConstructionFunctor.__init__(self, Fields(), 
                                     Rings())

    ### Methods to implement
    def _coerce_into_domain(self, x):
        if x not in self.domain():
            raise TypeError("The object {} is not an element of {}".format(x, self.domain()))
        return x

    def _apply_functor(self, x):
        raise NotImplementedError

    def _repr_(self):
        raise NotImplementedError

    def __eq__(self, other):
        if(other.__class__ == self.__class__):
            return self.base_ring() == other.base_ring()
        return False

    def merge(self, other):
        if(other.__class__ == self.__class__):
            return (self.type())(pushout(self.base_ring(), other.base_ring()))

        return None

