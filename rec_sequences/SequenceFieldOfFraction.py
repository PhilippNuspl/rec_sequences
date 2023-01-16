# coding: utf-8
r"""
Field of fractions of a sequence ring.

Defines a ring which is the fraction field of a sequence ring where all
sequences contain at most finitely many zeros (or are the zero sequence).
A fraction of two such sequences can therefore be evaluated at almost
all points.

EXAMPLES::

    sage: from rec_sequences.SequenceFieldOfFraction import *
    sage: from rec_sequences.CFiniteSequenceRing import *
    
    sage: C = CFiniteSequenceRing(QQ)
    sage: QC = SequenceFieldOfFraction(C)
    
    sage: a = QC(C([3,0,-1], [1,3]), C([2,-1], [1]))
    sage: a
    Fraction sequence:
    > Numerator: C-finite sequence a(n): (3)*a(n) + (-1)*a(n+2) = 0 
    and a(0)=1 , a(1)=3
    > Denominator: C-finite sequence a(n): (2)*a(n) + (-1)*a(n+1) = 0 and a(0)=1
    sage: b = QC(C([1,-1], [2]))
    
    sage: c = a*b
    sage: c[:10]
    [2, 3, 3/2, 9/4, 9/8, 27/16, 27/32, 81/64, 81/128, 243/256]
    sage: a_inv = ~a
    sage: a_inv*a == 1
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

from copy import copy
from math import factorial
from operator import pow

from sage.repl.rich_output.pretty_print import show
from sage.arith.all import gcd
from sage.calculus.var import var
from sage.functions.other import floor, ceil, binomial
from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.matrix.constructor import Matrix
from sage.matrix.special import identity_matrix
from sage.misc.all import prod, randint
from sage.rings.all import ZZ, QQ, CC
from sage.rings.ring import CommutativeAlgebra
from sage.rings.ring import CommutativeRing
from sage.structure.element import FieldElement
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.element import RingElement
from sage.categories.pushout import ConstructionFunctor
from sage.categories.fields import Fields
from sage.categories.algebras import Algebras
from sage.categories.commutative_algebras import CommutativeAlgebras
from sage.categories.commutative_rings import CommutativeRings
from sage.categories.rings import Rings
from sage.structure.unique_representation import UniqueRepresentation

from .DFiniteSequenceRing import DFiniteSequenceRing
from .SequenceRingOfFraction import FractionSequence, SequenceRingOfFraction, SequenceRingOfFractionFunctor

####################################################################################################

class FractionFieldSequence(FractionSequence, FieldElement):
    r"""
    A fraction sequence, i.e. a fraction where both numerator and denominator
    are coming from a ring of sequences.
    """

    def __init__(self, parent, numerator, denominator, name = "a", 
                 is_gen = False, construct=False, cache=True):
        r"""
        Construct a fraction sequence ``numerator/denominator``.

        .. NOTE::
            
            It is not checked whether numerator and denominator only contain
            finitely many zeros.

        INPUT:

        - ``parent`` -- a :class:`SequenceRingOfFraction`
        - ``numerator`` -- the numerator of the fraction which is either the
          zero sequence or only contains finitely many zeros.
        - ``denominator`` -- the denominator of the fraction which only contains
          finitely many zeros.
        - ``name`` (default "a") -- a name for the sequence

        OUTPUT:

        A fraction sequence ``numerator/denominator``.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceFieldOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceFieldOfFraction(C)
            
            sage: QC(C([3,-1], [1]), C([2,-1], [1]))
            Fraction sequence:
            > Numerator: C-finite sequence a(n): (3)*a(n) + (-1)*a(n+1) = 0 
            and a(0)=1
            > Denominator: C-finite sequence a(n): (2)*a(n) + (-1)*a(n+1) = 0
            and a(0)=1
            
        """

        if not isinstance(parent, SequenceFieldOfFraction) :
            raise TypeError("Parent has to be a SequenceRingOfFraction.")
        if numerator.parent() != parent.base() :
            raise TypeError("Numerator has to be in {}, but is in {}.".format(parent.base(),numerator.parent()))
        if denominator.parent() != parent.base() :
            raise TypeError("Denominator has to be in {}, but is in {}.".format(parent.base(),denominator.parent()))

        FieldElement.__init__(self, parent)
        FractionSequence.__init__(self, parent, numerator, denominator,
                                  name, is_gen, construct, cache)
        
# arithmetic

    def is_unit(self) :
        r"""
        Check whether sequence is multiplicative unit.
        
        OUTPUT:
        
        Return ``True`` if ``self`` is not zero.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceFieldOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceFieldOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: b = QC(c, C(1))
            sage: a.is_unit()
            True
            sage: b.is_unit()
            True
            sage: (a+1).is_unit()
            True

        """
        return not self.is_zero()

    def __invert__(self):
        r"""
        Computes the inverse of the sequence.
                    
        OUTPUT: 

        The multiplicative inverse of the sequence.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceFieldOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceFieldOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: b = QC(c.shift(), d)
            sage: ~b*b == 1
            True
            
        """
        if not self.is_zero() :
            return self.parent()(self.denominator(), self.numerator())
        else :
            raise ValueError("Zero sequence is not invertible")
 

####################################################################################################

class SequenceFieldOfFraction(SequenceRingOfFraction):
    r"""
    The field of fractions over a ring of sequences.
    """

    Element = FractionFieldSequence

# constructor

    def __init__(self, base, name=None, element_class=None, category=None):
        r"""
        Constructor for a sequence field of fractions.

        INPUT:

        - ``base`` -- a base ring which represents a sequence ring

        OUTPUT:

        A sequence field of fraction over the given ``base``.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceFieldOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: SequenceFieldOfFraction(C)
            Sequence field of fractions over Ring of C-finite sequences over 
            Rational Field
            
        """
        SequenceRingOfFraction.__init__(self, base, name, element_class, 
                                        category)

    def _eq_(self, right):
        r"""
        Tests if the two rings ``self``and ``right`` are equal.
        This is the case if they have the same class and are defined
        over the same base ring.
        """
        if not isinstance(right, SequenceFieldOfFraction) :
            return False
        try:
            return self.base() == right.base()
        except:
            return False

    def is_integral_domain(self, proof = True):
        r"""
        Returns whether ``self`` is an integral domain.
        This is the case.
        
        OUTPUT:
        
        ``True``
        """
        return True

    def is_noetherian(self):
        r"""
        Returns whether ``self`` is a Noetherian ring.
        This is the case.
        
        OUTPUT:
        
        ``True``
        """
        return True

    def is_commutative(self):
        r"""
        Returns whether ``self`` is a commutative ring.
        This is the case.
        
        OUTPUT:
        
        ``True``
        """
        return True

    def construction(self):
        r"""
        Shows how the given ring can be constructed using functors.
        
        OUTPUT:
        
        A functor ``F`` and a ring ``R`` such that ``F(R)==self``
        """
        return SequenceFieldOfFractionFunctor(), self.base()

    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        return sib.name("SequenceFieldOfFraction")(sib(self.base()))

    def _repr_(self):
        r"""
        OUTPUT:
        
        A string representation of the sequence field of fractions.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceFieldOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: SequenceFieldOfFraction(C)
            Sequence field of fractions over Ring of C-finite sequences over 
            Rational Field

        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        r = self._cached_repr = "Sequence field of fractions over " \
                                + self.base()._repr_()
        return r

    def is_field(self, proof = True):
        r"""
        Returns whether ``self`` is a field.
        This is the case.
        
        OUTPUT:
        
        ``True``
        """
        return True

    def random_element(self, *args, **kwds):
        r"""
        OUTPUT:
        
        Return a random fraction sequence. This is done by creating a
        random numerator and random denominator. Any additional arguments
        are passed to the ``random_element`` method of the base ring.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceFieldOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            sage: QC.random_element()[:5] # random
            [-8/3, 2, -81/28, 99/112, -1269/896]

        """
        numerator = self.base().random_element(*args, **kwds)
        denominator = self.base().random_element(*args, **kwds)
        while denominator.is_zero() :
            denominator = self.base().random_element(*args, **kwds)
        return self._element_constructor_(numerator, denominator)

    def change_base_ring(self,R):
        r"""
        OUTPUT:
        
        Return a copy of ``self`` but with the base `R`
        """
        if R is self._base:
            return self
        else:
            Q = SequenceFieldOfFraction(R)
            return Q



class SequenceFieldOfFractionFunctor(SequenceRingOfFractionFunctor):
    def __init__(self):
        r"""
        Constructs a ``SequenceFieldOfFractionFunctor``.
        """
        SequenceRingOfFractionFunctor.__init__(self)

    ### Methods to implement

    def _apply_functor(self, x):
        return SequenceFieldOfFraction(x)

    def _repr_(self):
        r"""
        Returns a string representation of the functor.
        
        OUTPUT:
        
        The string "SequenceFieldOfFraction(\*)" .        
        """
        return "SequenceFieldOfFraction(*)"

    def __eq__(self, other):
        if(other.__class__ == self.__class__):
            return self.base() == other.base()
        return False

    def merge(self, other):
        if(other.__class__ == self.__class__):
            return SequenceFieldOfFractionFunctor(pushout(self.base(), 
                                                          other.base()))

        return None

