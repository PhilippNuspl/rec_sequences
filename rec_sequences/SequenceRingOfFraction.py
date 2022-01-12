# coding: utf-8
r"""
Ring of fractions of a sequence ring.

Defines a ring which is the total ring of fractions of a sequence ring.
I.e., elements in this ring are fractions of two sequences where the
sequence in the denominator has no zero terms.

EXAMPLES::

    sage: from rec_sequences.SequenceRingOfFraction import *
    sage: from rec_sequences.CFiniteSequenceRing import *
    
    sage: C = CFiniteSequenceRing(QQ)
    sage: QC = SequenceRingOfFraction(C)
    
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

from numpy import random

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
from sage.structure.element import CommutativeAlgebraElement
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.element import RingElement
from sage.structure.element import RingElement
from sage.categories.pushout import ConstructionFunctor
from sage.categories.fields import Fields
from sage.categories.algebras import Algebras
from sage.categories.commutative_algebras import CommutativeAlgebras
from sage.categories.commutative_rings import CommutativeRings
from sage.categories.rings import Rings
from sage.structure.unique_representation import UniqueRepresentation

from .DFiniteSequenceRing import DFiniteSequenceRing

####################################################################################################

class FractionSequence(CommutativeAlgebraElement):
    r"""
    A fraction sequence, i.e. a fraction where both numerator and denominator
    are coming from a ring of sequences.
    """

    def __init__(self, parent, numerator, denominator, name = "a", is_gen = False, construct=False, cache=True):
        r"""
        Construct a fraction sequence ``numerator/denominator``.

        .. NOTE::
            
            It is not checked whether the denominator contains any zero terms.

        INPUT:

        - ``parent`` -- a :class:`SequenceRingOfFraction`
        - ``numerator`` -- the numerator of the fraction
        - ``denominator`` -- the denominator of the fraction which is non-zero
          at every term
        - ``name`` (default "a") -- a name for the sequence

        OUTPUT:

        A fraction sequence ``numerator/denominator``.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: QC(C([3,-1], [1]), C([2,-1], [1]))
            Fraction sequence:
            > Numerator: C-finite sequence a(n): (3)*a(n) + (-1)*a(n+1) = 0 
            and a(0)=1
            > Denominator: C-finite sequence a(n): (2)*a(n) + (-1)*a(n+1) = 0
            and a(0)=1
            
        """

        if not isinstance(parent, SequenceRingOfFraction) :
            raise TypeError("Parent has to be a SequenceRingOfFraction.")
        if numerator.parent() != parent.base() :
            raise TypeError("Numerator has to be in {}, but is in {}.".format(parent.base(),numerator.parent()))
        if denominator.parent() != parent.base() :
            raise TypeError("Denominator has to be in {}, but is in {}.".format(parent.base(),denominator.parent()))

        CommutativeAlgebraElement.__init__(self, parent)

        # if denominator is constant, we make it 1 and adjust the numerator
        try :
            denom = self.base_ring()(denominator)
            numerator = 1/denom*numerator
            denominator = self.base().one()
        except :
            pass

        self._is_gen = is_gen
        self._numerator = numerator
        self._denominator = denominator


    def __copy__(self):
        r"""
        Return a "copy" of ``self``. This is just ``self``, since C-finite sequences are immutable.
        """
        return self
    
    def simplified(self):
        r"""
        Computes terms of the sequence and tries to guess a sequence 
        in the base ring from these
        that is equal to ``self`` and tries to prove this.
        
        OUTPUT:
        
        An identical sequence with possible trivial denominator if possible.
        
        EXAMPLES::
 
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c*d, d)
            sage: a.denominator() == 1
            False
            sage: a.simplified().denominator() == 1
            True
            
        """
        data = self[:100]
        try :
            new_seq = self.parent().base().guess(data)
            new_seq = self.parent()(new_seq)
            if new_seq == self :
                return new_seq
        except :
            pass
        return self

#tests

    def __is_zero__(self):
        r"""
        Return whether ``self`` is the zero sequence 0,0,0,\dots .
        This is the case iff the numerator is zero.
        
        OUTPUT:
        
        ``True`` if the sequence is constantly zero, ``False`` otherwise.
        
        EXAMPLES::
 
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: d = C([3,1], [1])
            sage: QC(C(0), d).is_zero()
            True
            sage: QC(C(1), d).is_zero()
            False
            
        """
        return self.numerator().is_zero()


    def __eq__(self, right):
        r"""
        Return whether the two fractions ``self`` and ``right`` are equal.
        More precisely it is tested if the difference of ``self`` and ``right`` 
        equals 0.

        INPUT:
        
        - ``right`` - a ``FractionSequence``
        
        OUTPUT:
        
        ``True`` if the sequences are equal and ``False`` otherwise.

        EXAMPLES::
 
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c*d, d)
            sage: b = QC(c, C(1))
            sage: a == b
            True
            sage: b == c
            True
            sage: a+1 == b
            False
            
        """
        if self.parent() != right.parent():
            right = self.parent()(right)
            
        # if base ring is C-finite sequence ring just check equality
        # by checking enough initial values
        if isinstance(self.base(), DFiniteSequenceRing) :
            ord = self.n().order()*right.d().order()
            ord += self.d().order()*right.n().order()
            
            for n in range(ord) :
                if self.n()[n]*right.d()[n] != self.d()[n]*right.n()[n] :
                    return False 
            return True
            
        
        return (self.__add__(-right)).__is_zero__()


    def __ne__(self,right):
        r"""
        INPUT:
        
        - ``right`` - a ``FractionSequence``
        
        OUTPUT:
        
        Return ``True`` if the sequences ``self`` and ``right`` are NOT equal; ``False`` otherwise
        """
        return not self.__eq__(right)

    def _is_atomic(self):
        r"""
        """
        raise NotImplementedError

    def is_unit(self):
        r"""
        Check whether sequence is multiplicative unit.
        
        OUTPUT:
        
        Return ``True`` if ``self`` is a unit.
        This is the case if the numerator has no zeros.
        We check this empirically by checking the first ``100`` values.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: b = QC(c, C(1))
            sage: a.is_unit()
            False
            sage: b.is_unit()
            False
            sage: (a+1).is_unit()
            True
            
        """
        zero = self.base_ring().zero()
        return (zero not in self.numerator()[:100])

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
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: b = QC(c, C(1))
            sage: a[5]
            -5/243
            sage: b[:10] == c[:10]
            True
            
        """
        if isinstance(n, slice) :
            if n.stop == None :
                raise ValueError("Sequences are infinite. Need to specify upper bound.")
            elif n.step != None and n.step != 1 :
                raise NotImplementedError
            else :
                return [nu/de for nu, de in zip(self.numerator()[n], self.denominator()[n])]

        return self.numerator()[n]/self.denominator()[n]

    def __setitem__(self, n, value):
        r"""
        """
        raise IndexError("Fraction sequences are immutable")

    def __iter__(self):
        raise NotImplementedError

#conversion

    def _test_conversion_(self):
        r"""
        Test whether a conversion of ``self`` into an int/float/long/... is possible;
        i.e. whether the sequence is constant or not.

        OUTPUT:

        If ``self`` is constant, i.e. there exists a `k` in K, such that self(n) = k for all n in NN,
        then this value `k` is returned. If ``self`` is not constant ``None`` is returned.
        """
        num_test = self.numerator()._test_conversion_()
        denom_test = self.denominator()._test_conversion_()
        if num_test == None or denom_test == None :
            return None
        else :
            return num_test/denom_test

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

    def _repr_(self):
        r"""
        Produces a string representation of the sequence.
        
        OUTPUT:
        
        A string representation of the sequence consisting of the numerator
        and the denominator of the sequence.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: QC(C([2,1], [1]), C([2,-1], [1]))
            Fraction sequence:
            > Numerator: C-finite sequence a(n): (2)*a(n) + (1)*a(n+1) = 0 
            and a(0)=1
            > Denominator: C-finite sequence a(n): (2)*a(n) + (-1)*a(n+1) = 0
            and a(0)=1
            
        """
        r = "Fraction sequence:\n"
        r += "> Numerator: " + self.numerator()._repr_() + "\n"
        r += "> Denominator: " + self.denominator()._repr_() + "\n"

        return r

    def _latex_(self):
        r"""
        Creates a latex representation of the sequence.
        This is done by using the fraction of the latex representation of the 
        numerator and denominator.
        
        OUTPUT: 
        
        A latex representation showing the closed form of the sequences.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: print(latex(QC(C([2,1], [1]), C([2,-1], [1]))))
            \frac{\left(-2\right)^{n}}{2^{n}}
            
        """
        if self.denominator().is_one() :
            return self.numerator()._latex_()
        else :
            return r"\frac{"+self.numerator()._latex_()+"}{"+self.denominator()._latex_()+"}"

# arithmetic
    def shift(self, k=1):
        r"""
        Shifts ``self`` ``k``-times.

        INPUT:

        - ``k`` (default: ``1``) -- an integer

        OUTPUT:

        The sequence `(a(n+k))_{k \in \mathbb{N}}`.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: a[1:11] == a.shift()[:10]
            True
            
        """
        if k==0 :
            return self

        return FractionSequence(self.parent(), self.numerator().shift(k), 
                                self.denominator().shift(k))

    def __invert__(self):
        r"""
        Tries to compute the multiplicative inverse, if this is possible.
        Such an inverse exists if and only if
        the numerator does not contain any zero terms.
        
        The method can be called by ~self or self.inverse_of_unit().
        
        .. NOTE::
            
            Only some initial terms are checked to see whether a sequence is 
            invertible, cf. :meth:`is_unit`.
                    
        OUTPUT: 

        The multiplicative inverse of the sequence if it exists.
        Raises an ``ValueError`` if the sequence is not invertible.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: b = QC(c.shift(), d)
            sage: ~a
            Traceback (most recent call last):
            ...
            ValueError: Sequence is not invertible
            sage: ~b*b == 1
            True
            
        """
        if self.is_unit() :
            return FractionSequence(self.parent(), self.denominator(), 
                                    self.numerator())
        else :
            raise ValueError("Sequence is not invertible")

    def _add_(self, right):
        r"""
        Return the sum of ``self`` and ``right``. 
        
        INPUTS:

        - ``right`` -- a ``FractionSequence``
        
        OUTPUTS: 
        
        The addition of ``self`` with ``right``.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: b = QC(c.shift(), d)
            sage: s = a+b
            sage: s[:10] == [a[i]+b[i] for i in range(10)]
            True
            
        """
        if self.__is_zero__():
            return right
        if right.__is_zero__():
            return self

        n1 = self.numerator()
        d1 = self.denominator()
        n2 = right.numerator()
        d2 = right.denominator()

        return FractionSequence(self.parent(), n1*d2+d1*n2, d1*d2)

    def _neg_(self):
        r"""
        OUTPUT:
        
        Return the negative of ``self``.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: b = -a
            sage: a+b == 0
            True
            
        """
        return FractionSequence(self.parent(), 
                                -self.numerator(), self.denominator())

    def _mul_(self, right):
        r"""
        Return the product of ``self`` and ``right``. 
        
        INPUTS:

        - ``right`` -- a ``FractionSequence``
        
        OUTPUTS: 
        
        The product of ``self`` with ``right``.
        
        EXAMPLES:: 
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: b = QC(c.shift(), d)
            sage: p = a*b
            sage: p[:10] == [a[i]*b[i] for i in range(10)]
            True
            
        """
        if self.__is_zero__() or right.__is_zero__():
            return self.parent().zero()

        n1 = self.numerator()
        d1 = self.denominator()
        n2 = right.numerator()
        d2 = right.denominator()

        return FractionSequence(self.parent(), n1*n2, d1*d2)

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
    
    def subsequence(self, u, v=0):
        r"""
        Returns the sequence ``self[floor(u*n+v)]``.

        INPUT:

        - ``u`` -- a rational number
        - ``v`` (optional) -- a rational number

        OUTPUT:
        
        The sequence ``self[floor(u*n+v)]``.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: b = a.subsequence(2, 3)
            sage: b[:10] == [a[2*i+3] for i in range(10)]
            True
            
        """
        num = self.numerator().subsequence(u, v)
        denom = self.denominator().subsequence(u, v)
        return FractionSequence(self.parent(), num, denom)
    
    def interlace(self, *others):
        r"""
        Returns the interlaced sequence of self with ``others``.

        INPUT:

        - ``others`` -- other sequences over the same 
          ``SequenceRingOfFraction``

        OUTPUT:
        
        The interlaced sequence of self with ``others``.
        
        EXAMPLES::

            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: e = C([2,-3],[2])
            sage: a = QC(c, d)
            sage: a[:5]
            [0, -1/3, 1/9, -2/27, 1/27]
            sage: b = QC(e, C(1))
            sage: b[:5]
            [2, 4/3, 8/9, 16/27, 32/81]
            sage: a.interlace(b)[:6]
            [0, 2, -1/3, 4/3, 1/9, 8/9]
            
        """
        others_num = [seq.numerator() for seq in others]
        others_denom = [seq.denominator() for seq in others]
        
        interlaced_num = self.numerator().interlace(*others_num)
        interlaced_denom = self.denominator().interlace(*others_denom)
            
        return FractionSequence(self.parent(), interlaced_num, interlaced_denom)
    
    def prepend(self, values) :
        r"""
        Prepends the given values to the sequence.
        
        Input:
        
        - ``values`` -- list of values in the base ring

        OUTPUT:
        
        A sequence having the same terms with the additional ``values``
        at the beginning.
        
        EXAMPLES::

            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: a[:5]
            [0, -1/3, 1/9, -2/27, 1/27]
            sage: a.prepend([1,3])[:7]
            [1, 3, 0, -1/3, 1/9, -2/27, 1/27]
            
        """
        num = self.numerator().prepend(values)
        K = self.parent().base_ring()
        denom = self.denominator().prepend(len(values)*[K(1)])
        return FractionSequence(self.parent(), num, denom)
    
#base ring related functions

    def base_ring(self):
        r"""
        OUTPUT:
        
        Return the base field of the parent of ``self``.
        
        EXAMPLES::

            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: a.base_ring() == QQ
            True
            
        """
        return self.parent().base_ring()

    def base(self):
        r"""
        OUTPUT:
        
        Return the base of the parent of ``self``.

        EXAMPLES::

            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: a.base() == C
            True
            
        """
        return self.parent().base()

#part extraction functions

    def n(self):
        r"""
        Alias of :meth:`numerator`.
        """
        return self.numerator()

    def numerator(self):
        r"""
        OUTPUT:
        
        Return the numerator.
        
        EXAMPLES::

            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: a.numerator() == c
            True
            
        """
        return self._numerator

    def d(self):
        r"""
        Alias of :meth:`denominator`.
        """
        return self.denominator()

    def denominator(self):
        r"""
        OUTPUT:
        
        Return the denominator
        
        EXAMPLES::

            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: a = QC(c, d)
            sage: a.denominator() == d
            True
            
        """
        return self._denominator

    def order(self):
        r"""
        OUTPUT:
        
        Return the order of the recurrence of ``self`` which is the maximum of
        the orders of the numerator and denominator.
        
        EXAMPLES::

            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: c = C([1,1,-1], [0,1])
            sage: d = C([3,1], [1])
            sage: QC(c, d).order()
            2
            sage: QC(d, c).order()
            2
            sage: QC(c, c).order()
            2
            
        """
        return max(self.numerator().order(), self.denominator().order())
    
    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        sib_ring = sib(self.parent())
        seq = sib_ring(sib(self.numerator()), sib(self.denominator()))
        return seq
    

####################################################################################################

class SequenceRingOfFraction(UniqueRepresentation, CommutativeAlgebra):
    r"""
    The ring of fractions over a ring of sequences.
    """

    Element = FractionSequence

# constructor

    def __init__(self, base, name=None, element_class=None, category=None):
        r"""
        Constructor for a sequence ring of fractions.

        INPUT:

        - ``base`` -- a base ring which represents a sequence ring

        OUTPUT:

        A sequence ring of fraction over the given ``base``.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: SequenceRingOfFraction(C)
            Sequence ring of fractions over Ring of C-finite sequences over 
            Rational Field
            
        """
        if base not in Rings() :
            raise TypeError("Sequence ring fractions is defined over a ring.")

        self._base_ring = base.base_ring()

        CommutativeAlgebra.__init__(self, base.base_ring(), category=CommutativeAlgebras(base.base_ring()))

        self._base = base


    def _element_constructor_(self, x, y=None, check=True, is_gen = False, construct=False, **kwds):
        r"""
        Tries to construct a fraction sequence element.

        This is possible if:

        - ``x`` is already a fraction sequence element
        - ``x`` is an element in the base and ``y`` is an element in the base,
          then ``x/y`` is constructed
        - ``x`` is an element in the base and ``y`` is None, then ``x/1`` is 
          constructed
        - ``x`` and ``y`` can be used to construct a sequence element, then 
          this element over 1 is constructed
          
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: fib = C([1,1,-1], [0,1])
            sage: luc = C([1,1,-1], [2,1])
            sage: c = QC(fib, luc)
            sage: c*luc == fib
            True
            
            sage: d = QC(fib)
            sage: d == fib
            True
            
        """
        K = self.base_ring()
        R = self.base()
        if isinstance(x, SequenceRingOfFraction) :
            return x
        elif x!=None and y==None :
            return FractionSequence(self, R(x), R.one())
        elif x!=None and y!=None and not isinstance(x, list) and not isinstance(y, list):
            return FractionSequence(self, R(x), R(y))
        else :
            return FractionSequence(self, R(x,y), R.one())

    def one(self) :
        r"""
        OUTPUT:
        
        The constant sequence `1,1,\dots`.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: QC.one() == C.one()
            True
            
            sage: QC.one()[:10]
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            
        """
        return self._element_constructor_(self.base().one())

    def zero(self) :
        r"""
        OUTPUT:
        
        The constant sequence `0,0,\dots`.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: QC.zero() == C.zero()
            True
            
            sage: QC.zero()[:10]
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
        """
        return self._element_constructor_(self.base().zero())

    def _eq_(self, right):
        r"""
        Tests if the two rings ``self``and ``right`` are equal.
        This is the case if they have the same class and are defined
        over the same base ring.
        """
        if not isinstance(right, SequenceRingOfFraction) :
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
        return self.base().is_integral_domain()

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

    def construction(self):
        r"""
        Shows how the given ring can be constructed using functors.
        
        OUTPUT:
        
        A functor ``F`` and a ring ``R`` such that ``F(R)==self``
        """
        return SequenceRingOfFractionFunctor(), self.base()

    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        return sib.name("SequenceRingOfFraction")(sib(self.base()))
    
    def _coerce_map_from_(self, P):
        r"""
        """
        return self.base().has_coerce_map_from(P)

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

    def _repr_(self):
        r"""
        OUTPUT:
        
        A string representation of the sequence ring of fractions.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: SequenceRingOfFraction(C)
            Sequence ring of fractions over Ring of C-finite sequences over 
            Rational Field

        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        r = self._cached_repr = "Sequence ring of fractions over " + self.base()._repr_()
        return r

    def _latex_(self):
        r"""
        OUTPUT:
        
        A latex representation of the sequence ring of fractions.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            sage: print(latex(QC))
            Q(\mathcal{C}(\Bold{Q}))

        """
        return r"Q(" + self._base._latex_() + ")"

    def base(self):
        r"""
        OUTPUT:
        
        Return the base over which the ring of fractions is defined.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            sage: QC.base() == C
            True

        """
        return self._base

    def base_ring(self):
        r"""
        OUTPUT:
        
        Return the base field over which the ring of fractions is defined
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            sage: QC.base_ring() == QQ
            True

        """
        return self._base_ring

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
        return self.base().is_exact()

    def is_field(self, proof = True):
        r"""
        Returns whether ``self`` is a field.
        This is not the case.
        
        OUTPUT:
        
        ``False``
        """
        return False

    def random_element(self, *args, **kwds):
        r"""
        OUTPUT:
        
        Return a random fraction sequence. This is done by creating a
        random numerator and random denominator (iteratively until 
        a denominator which is a unit is found). Any additional arguments
        are passed to the ``random_element`` method of the base ring.
        
        EXAMPLES::
        
            sage: from rec_sequences.SequenceRingOfFraction import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            sage: QC.random_element()[:5] # random
            [-8/3, 2, -81/28, 99/112, -1269/896]

        """
        numerator = self.base().random_element(*args, **kwds)
        denominator = self.base().random_element(*args, **kwds)
        while not denominator.is_unit() :
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
            Q = SequenceRingOfFraction(R)
            return Q



class SequenceRingOfFractionFunctor(ConstructionFunctor):
    def __init__(self):
        r"""
        Constructs a ``RecurrenceSequenceRingFunctor``.
        """
        ConstructionFunctor.__init__(self, Fields(), Rings())

    ### Methods to implement
    def _coerce_into_domain(self, x):
        if x not in self.domain():
            raise TypeError("The object {} is not an element of {}".format(x, self.domain()))
        return x

    def _apply_functor(self, x):
        return SequenceRingOfFraction(x)

    def _repr_(self):
        r"""
        Returns a string representation of the functor.
        
        OUTPUT:
        
        The string "SequenceRingOfFraction(\*)" .        
        """
        return "SequenceRingOfFraction(*)"

    def __eq__(self, other):
        if(other.__class__ == self.__class__):
            return self.base() == other.base()
        return False

    def merge(self, other):
        if(other.__class__ == self.__class__):
            return SequenceRingOfFractionFunctor(pushout(self.base(), other.base()))

        return None

