# coding: utf-8
r"""
The class describes a cyclic sign pattern from some term on.
I.e., the class describes a finite set of initial values where
the sign need not be cyclic and than a description of the signs 
from the point on where they are cyclic. However, there are also C-finite
sequences which do not have cyclic sign pattern, cf. Example 2.3 in
[AKKOW21]_.

The initial values and elements from the cycle are saved as
1,-1 and 0 describing positive, negative and zero values, 
respectively.

EXAMPLES::

    sage: from rec_sequences.SignPattern import *
    sage: pattern = SignPattern([0,1,-1], [1,0,-1])
    sage: pattern
    Sign pattern: initial values <0+-> cycle <+0->
    sage: pattern[:10]
    [0, 1, -1, 1, 0, -1, 1, 0, -1, 1]
    sage: pattern.get_initial_values()
    [0, 1, -1]
    sage: pattern.get_cycle()
    [1, 0, -1]
    sage: pattern.get_positive_progressions()
    {Arithmetic progression (3*n+3)_n}

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

from sage.arith.functions import lcm
from sage.functions.other import floor
from sage.structure.sage_object import SageObject

from .ArithmeticProgression import ArithmeticProgression

class SignPattern(SageObject):
    r"""
    Describes the cyclic sign pattern of a sequence.
    """

    def __init__(self, initial_values=list(), cycle=list()):
        r"""
        Creates a sign pattern with the given data.
        
        INPUT:

        - ``initial_values`` -- a list of exceptional signs not covered by 
          the cycle.
        - ``cycle`` -- the list of cyclic signs which holds after the 
          ``initial_values`` (has to be non-empty)
        
        OUTPUT:
        
        The described sign pattern.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: pattern = SignPattern([0,1,-1], [1,0,-1])
            sage: print(pattern)
            Sign pattern: initial values <0+-> cycle <+0->
            
        """
        SageObject.__init__(self)
        self._initial_values = initial_values
        self._cycle = cycle

    def __eq__(self, pattern) :
        r"""
        Checks whether two zero patterns are equal.
        
        INPUT:
        
        - ``pattern`` -- a sign pattern
        
        OUTPUT:
        
        ``True`` if the two patterns are equal and ``False`` 
        otherwise.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: pattern1 = SignPattern([1,0,-1], [1,0,-1])
            sage: pattern2 = SignPattern([], [1,0,-1])
            sage: pattern3 = SignPattern([1,0,1], [1,0,-1])
            sage: pattern1 == pattern2 
            True
            sage: pattern2 == pattern3
            False
        
        """
        cycle_top_self = self.get_cycle_start() + self.get_cycle_length()
        cycle_top_pat = pattern.get_cycle_start() + pattern.get_cycle_length()
        check_bound = max(cycle_top_self, cycle_top_pat) + 1
        for n in range(check_bound) :
            if self[n] != pattern[n] :
                return False 
        return True

    @classmethod
    def guess(cls, data, cycles=3):
        r"""
        Computes the sign pattern of the given data.

        INPUT:

        - ``data`` -- a list of numbers
        - ``cycles`` -- the number of cycles that need to
          be detected to conclude whether we indeed have a period. 
          Assume ``cycles``>1.

        OUTPUT:

        The sign-pattern satisfied by ``data``. 
        If no pattern could be found, a ValueError is raised.

        ALGORITHM:
        
        We assume that the periodic behavior starts at least at
        the half-point of the given data. The algorithm is not 
        particularly efficient and runs in ``O(len(data)^3)`` in the worst case.
        However, in the usual case, if data is cyclic almost from the beginning
        it is essentially ``O(len(data))``
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            
            sage: SignPattern.guess([0,2] + 10*[1,-1])
            Sign pattern: initial values <0+> cycle <+->
            sage: SignPattern.guess([1,0,0,1,0,-2,0,3,0,-4,0,5,0,-6,0])
            Traceback (most recent call last):
            ...
            ValueError: No sign pattern found, you can try with more data.
            sage: SignPattern.guess([1,0,0,1,0,-2,0,3,0,-4,0,5,0,-6,0], \
            ....:                   cycles=2)
            Sign pattern: initial values <+0> cycle <0+0->
            
        """
        new_data = []
        for el in data :
            if el < 0 :
                new_data.append(-1)
            elif el == 0 :
                new_data.append(0)
            else :
                new_data.append(1)
        data = new_data

        # check data is cyclic
        # idea: for every i<len(data)/2 we check if data[i] is the start of the
        # cycle-period by checking whether there exists a j>i such that
        # data[i]=data[j] and we have cyclic zero-pattern of length j-i
        # for k >= j.
        m = len(data)
        for i in range(floor(m/2)):
            max_point = floor((m-i)/cycles+i)
            for j in range(i+1, max_point):
                if data[i] == data[j]:  # candidate found
                    d = j-i
                    if ArithmeticProgression._check_cycle_(i, d, data):
                        # cycle was really found
                        cycle = data[i:i+d]
                        return SignPattern(data[:i], cycle)

        raise ValueError("No sign pattern found, you can try with more data.")

    def get_initial_values(self):
        r"""
        OUTPUT:
        
        Returns the initial values of the pattern.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).get_initial_values()
            [0, 1, -1]
            sage: SignPattern([], [1,-1]).get_initial_values()
            []
            
        """
        return self._initial_values

    def get_cycle(self):
        r"""
        OUTPUT:
        
        Returns the cycle of the pattern.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).get_cycle()
            [1, 0, -1]
            sage: SignPattern([0,1,-1], [1]).get_cycle()
            [1]
            
        """
        return self._cycle
    
    def get_positive_progressions(self) :
        r"""
        OUTPUT:
        
        Returns a set of the arithmetic progressions 
        (:class:`rec_sequences.ArithmeticProgression`) 
        where the pattern is positive.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).get_positive_progressions()
            {Arithmetic progression (3*n+3)_n}
        
        """
        prog = set()
        start = len(self.get_initial_values())
        length = len(self.get_cycle())
        for i, el in enumerate(self.get_cycle()) :
            if el > 0 :
                prog.add(ArithmeticProgression(length, start+i))
        return prog
    
    def get_negative_progressions(self) :
        r"""
        OUTPUT:
        
        Returns a set of the arithmetic progressions 
        (:class:`rec_sequences.ArithmeticProgression`) 
        where the pattern is negative.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).get_negative_progressions()
            {Arithmetic progression (3*n+5)_n}
        
        """
        prog = set()
        start = len(self.get_initial_values())
        length = len(self.get_cycle())
        for i, el in enumerate(self.get_cycle()) :
            if el < 0 :
                prog.add(ArithmeticProgression(length, start+i))
        return prog

    def get_zero_progressions(self) :
        r"""
        OUTPUT:
        
        Returns a set of the arithmetic progressions 
        (:class:`rec_sequences.ArithmeticProgression`) 
        where the pattern is zero.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).get_zero_progressions()
            {Arithmetic progression (3*n+4)_n}
        
        """
        prog = set()
        start = len(self.get_initial_values())
        length = len(self.get_cycle())
        for i, el in enumerate(self.get_cycle()) :
            if el == 0 :
                prog.add(ArithmeticProgression(length, start+i))
        return prog

    def repr_initial_values(self):
        r"""
        OUTPUT:
        
        Returns a string representation of the initial values only containing
        +, -, 0 for positive, negative and zero entries.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: print(SignPattern([0,1,-1], [1,0,-1]).repr_initial_values())
            0+-
        
        """
        s = ""
        for c in self._initial_values :
            if c < 0 :
                s += "-"
            elif c==0 :
                s += "0"
            else :
                s += "+"
        return s

    def repr_cycle(self):
        r"""
        OUTPUT:
        
        Returns a string representation of the cycle only containing
        +, -, 0 for positive, negative and zero entries.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: print(SignPattern([0,1,-1], [1,0,-1]).repr_cycle())
            +0-
        
        """
        s = ""
        for c in self._cycle :
            if c < 0 :
                s += "-"
            elif c==0 :
                s += "0"
            else :
                s += "+"
        return s

    def __repr__(self):
        r"""
        OUTPUT:
        
        A string representation of the form 
        "Sign pattern: initial values <a> cycle <b>" where ``a``
        is a string representation of the signs of the initial values
        and ``b`` is a string representation of the signs of the cycle
        (built by the characters +,0,-).
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1])
            Sign pattern: initial values <0+-> cycle <+0->
            sage: SignPattern([], [1,-1])
            Sign pattern: cycle <+->
            
        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        if len(self.get_initial_values()) == 0:
            r = "Sign pattern:"
        else:
            string_initial_values = "<"+self.repr_initial_values()+">"
            r = "Sign pattern: initial values " + string_initial_values
        
        string_cycle = self.repr_cycle()
        r+= " cycle <" + string_cycle + ">"
        self._cached_repr = r
        return r

    def _latex_(self):
        r"""
        OUTPUT:
        
        A latex representation of the form
        
        .. MATH::
            \langle a \underbrace{b}_{} \cdots \rangle
            
        where ``a`` is a string representation of the signs of the initial 
        values and ``b`` is a string representation of the signs of the cycle
        (built by the characters +,0,-).
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: print(latex(SignPattern([0,1,-1], [1,0,-1])))
            \langle0+-\underbrace{+0-}_{}\cdots\rangle
            sage: print(latex(SignPattern([], [1,-1])))
            \langle\underbrace{+-}_{}\cdots\rangle
            
        """
        s = r"\langle"
        s += self.repr_initial_values()
        cycle = self.repr_cycle()
        s += r"\underbrace{" + cycle + "}_{}"
        s += r"\cdots\rangle"
        return s

    def __getitem__(self, n):
        r"""
        INPUT:

        - ``n`` -- a natural number or a slice object (specifying only finitely
          many values)

        OUTPUT:

        Returns -1 if the ``n``-th value is negative +1 if it is positive and
        0 if it is zero. If ``n`` is a slice
        object, a list with these values is returned.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: pattern = SignPattern([0,1,-1], [1,0,-1])
            sage: pattern[3]
            1
            sage: pattern[:10]
            [0, 1, -1, 1, 0, -1, 1, 0, -1, 1]
            
        """
        if isinstance(n, slice) :
            if n.stop == None :
                raise ValueError("Zero patterns are infinite. Need to specify"
                                 " upper bound.")
            else :
                l = list(range(n.stop)[n])
                return [self[i] for i in l]
            
        if n < len(self.get_initial_values()):
            return self.get_initial_values()[n]
        for prog in self.get_positive_progressions():
            if n in prog:
                return 1
        for prog in self.get_negative_progressions():
            if n in prog:
                return -1
        for prog in self.get_zero_progressions():
            if n in prog:
                return 0

    def get_cycle_start(self):
        r"""
        OUTPUT:
        
        Returns the index from which the sign-pattern is guaranteed to
        be cyclic. This index might not be minimal.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).get_cycle_start()
            3
            sage: SignPattern([1,0,-1], [1,0,-1]).get_cycle_start()
            3
            sage: SignPattern([], [1,0,-1]).get_cycle_start()
            0
            
        """
        return len(self.get_initial_values())

    def get_cycle_length(self):
        r"""
        OUTPUT:
        
        Returns the length of the cycle. This might be a multiple
        of the minimal length.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).get_cycle_length()
            3
            sage: SignPattern([], [1,0,-1]).get_cycle_length()
            3
            sage: SignPattern([], [1,0]).get_cycle_length()
            2
            sage: SignPattern([], [1,0,1,0]).get_cycle_length()
            4
        
        """
        return len(self.get_cycle())

    def is_zero(self):
        r"""
        OUTPUT:
        
        Returns ``True`` if the pattern is constantly zero.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).is_zero()
            False
            sage: SignPattern([1,-1], [1,-1]).is_zero()
            False
            sage: SignPattern([0], [0]).is_zero()
            True
            sage: SignPattern([], [0]).is_zero()
            True
        
        """
        if all([val==0 for val in self.get_initial_values()]) and \
           all([val==0 for val in self.get_cycle()]) :
            return True
        else :
            return False

    def non_zero(self):
        r"""
        OUTPUT:
        
        Returns ``True`` if the pattern has no zeros at all.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).non_zero()
            False
            sage: SignPattern([1,-1], [1,-1]).non_zero()
            True
            sage: SignPattern([0,1], [0]).non_zero()
            False
        
        """
        if len(self.get_zero_progressions()) == 0 and \
           0 not in self.get_initial_values():
            return True
        else:
            return False
        
    def is_positive(self):
        r"""
        OUTPUT:
        
        Returns ``True`` if the pattern has only positive values,
        i.e. no zeros or negative values.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).is_positive()
            False
            sage: SignPattern([1], [1]).is_positive()
            True
        
        """
        for val in self.get_initial_values() :
            if val <= 0 :
                return False
        if len(self.get_zero_progressions()) != 0 or \
           len(self.get_negative_progressions()) != 0 :
               return False 
        return True
    
    def is_negative(self):
        r"""
        OUTPUT:
        
        Returns ``True`` if the pattern has only negative values,
        i.e. no zeros or positive values.
        
        EXAMPLES::
        
            sage: from rec_sequences.SignPattern import *
            sage: SignPattern([0,1,-1], [1,0,-1]).is_negative()
            False
            sage: SignPattern([-1], [-1]).is_negative()
            True
        
        """
        for val in self.get_initial_values() :
            if val >= 0 :
                return False
        if len(self.get_zero_progressions()) != 0 or \
           len(self.get_positive_progressions()) != 0 :
               return False 
        return True
    
    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        return sib.name("SignPattern")(sib(self.get_initial_values()),
                                       sib(self.get_cycle()))
