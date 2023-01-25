# coding: utf-8
r"""
Describes the zero pattern of a sequence comprised of a finite
set together with finitely many arithmetic progressions.
Such a zero pattern is represented by a sequence of boolean values
where ``False`` represents a zero and ``True`` a different value.

EXAMPLES::

    sage: from rec_sequences.ZeroPattern import *
    
    sage: progressions = set([ArithmeticProgression(3, 3), \
    ....:                     ArithmeticProgression(5, 2)])
    sage: pattern = ZeroPattern(set([0]), progressions)
    sage: pattern[0] # check that 0-th term is zero
    False
    sage: pattern[1] # check that 1-st term is non-zero
    True
    sage: pattern.get_cycle_start(), pattern.get_cycle_length()
    (3, 15)
    sage: print(pattern.pattern()) # first terms of pattern
    0*00**00*0**0**0*00*
    
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

from builtins import max
from sage.arith.functions import lcm
from sage.functions.other import floor
from sage.structure.sage_object import SageObject

from .ArithmeticProgression import ArithmeticProgression

class ZeroPattern(SageObject):
    r"""
    The class describes the zero pattern of a sequence which
    does not violate the Skolem-Mahler-Lech-Theorem.
    I.e., the class describes a finite set of zeros and some
    finitely many arithmetic progessions where the sequence
    is zero as well.

    This zero pattern might not be unique, i.e. there might
    be several different objects describing the same pattern.
    """

    def __init__(self, finite_set=set(), progressions=set()):
        r"""
        Creates a zero pattern with the given set ``finite_set`` and the
        given arithmetic ``progressions``. 

        INPUT:

        - ``finite_set`` -- a set of exceptional zeros not covered by the
          arithmetic progressions.
        - ``progressions`` -- a set of arithmetic progressions of the type
          :class:`rec_sequences.ArithmeticProgression`.

        OUTPUT:

        The described zero pattern.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3), \
            ....:                     ArithmeticProgression(5, 2)])
            sage: pattern = ZeroPattern(set([0]), progressions)
            sage: print(pattern) # random
            Zero pattern with finite set {0} and arithmetic progressions: 
            - Arithmetic progression (3*n+3)_n
            - Arithmetic progression (5*n+2)_n
        
        """
        SageObject.__init__(self)
        
        # make sure that progressions are not single terms
        cleared_progressions = set()
        for progression in progressions :
            if progression.get_diff() != 0 :
                cleared_progressions.add(progression)
            else :
                finite_set.add(progression.get_shift())
        
        self._finite_set = finite_set
        self._progressions = progressions

    @classmethod
    def from_zero_pattern(cls, finite_set=set(),
                          cycle_start=0, cycle=[]):
        r"""
        Creates a zero pattern with the given set ``finite_set`` and the
        given arithmetic cycle. 

        INPUT:

        - ``finite_set`` -- a set of exceptional zeros not covered by the
          arithmetic progressions.
        - ``cycle_start`` -- a natural number indicating where the periodic
          behavior of the zeros start.
        - ``cycle`` -- the zero-cycle which holds from ``cycle_start`` on. 
          It should be given as list of boolean values where ``False``
          indicates a zero value and ``True`` a non-zero value (or use
          any values that can be casted like this). 

        OUTPUT:

        The described zero pattern.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
            
            sage: pattern = ZeroPattern.from_zero_pattern([0], 3, [0,1,1,0,1])
            sage: print(pattern) # random
            Zero pattern with finite set [0] and arithmetic progressions: 
            - Arithmetic progression (5*n+3)_n
            - Arithmetic progression (5*n+6)_n
            sage: print(pattern.pattern())
            00*0**0*0**0*0**0*0*
            
        """
        # create arithmetic progressions from cycle
        progressions = set()
        cycle_length = len(cycle)
        for i, not_zero in enumerate(cycle):
            if not not_zero:
                shift = cycle_start + i
                progressions.add(ArithmeticProgression(cycle_length, shift))

        return ZeroPattern(finite_set, progressions)

    @classmethod
    def guess(cls, data, cycles=3, *args, **kwds):
        r"""
        Computes the zero pattern of the given data.

        INPUT:

        - ``data`` -- a list of numbers
        - ``cycles`` -- the number of cycles that need to
          be detected to conclude whether we indeed have a period. 
          Assume ``cycles>1``.

        OUTPUT:

        The zero-pattern satisfied by ``data``. 
        If no pattern could be found, a ``ValueError`` is raised.
        
        ALGORITHM:
        
        We assume that the periodic behavior starts at least at
        the half-point of the given data. The algorithm is not 
        particularly efficient and runs in ``O(len(data)^3)`` in the worst case.
        However, in the usual case, if data is cyclic almost from the beginning
        it is essentially ``O(len(data))``
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
            
            sage: ZeroPattern.guess([1,0,0,1,0,2,0,3,0,4,0,5])
            Zero pattern with finite set [1] and arithmetic progressions: 
            - Arithmetic progression (2*n+2)_n
            sage: ZeroPattern.guess(10*[1,0,0,1]) # random
            Zero pattern with finite set {} and arithmetic progressions: 
            - Arithmetic progression (4*n+1)_n
            - Arithmetic progression (4*n+2)_n
        
        """
        data = [0 if el == 0 else 1 for el in data]
        if 0 not in data:
            return ZeroPattern()

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
                    #print(f"candidate found {i}, {j}")
                    d = j-i
                    if ArithmeticProgression._check_cycle_(i, d, data):
                        #print(f"cycle was found, {i}, {d}")
                        # cycle was really found
                        cycle = data[i:i+d]
                        finite_set = [k for k, el in enumerate(data[:i])
                                      if el == 0]
                        from_pattern = ZeroPattern.from_zero_pattern
                        return from_pattern(finite_set, i, cycle)

        raise ValueError("No zero pattern found, you can try with more data.")

    def get_finite_set(self):
        r"""
        OUTPUT:
        
        Returns the finite zero-set of the pattern.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3), \
            ....:                     ArithmeticProgression(5, 2)])
            sage: pattern = ZeroPattern(set([0]), progressions)
            sage: print(pattern.get_finite_set())
            {0}

        """
        return self._finite_set

    def get_progressions(self):
        r"""
        OUTPUT:
        
        Returns the arithmetic progressions of the pattern.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3), \
            ....:                     ArithmeticProgression(5, 2)])
            sage: pattern = ZeroPattern(set([0]), progressions)
            sage: print(pattern.get_progressions()) # random
            {Arithmetic progression (3*n+3)_n, Arithmetic progression (5*n+2)_n}
        
        """
        return self._progressions

    def _repr_(self):
        r"""
        OUTPUT:
        
        A string representation of the zero pattern.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3), \
            ....:                     ArithmeticProgression(5, 2)])
            sage: print(latex(ZeroPattern(set([0]), progressions))) # random
            Zero pattern with finite set {0} and arithmetic progressions: 
            - Arithmetic progression (3*n+3)_n
            - Arithmetic progression (5*n+2)_n
            sage: print(ZeroPattern(set([0,2]), set()))
            Zero pattern with finite set {0, 2} and no arithmetic progressions
            
        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        if len(self.get_finite_set()) == 0:
            string_finite_set = r"{}"
        else:
            string_finite_set = str(self.get_finite_set())
        r = "Zero pattern with finite set " + string_finite_set
        if len(self.get_progressions()) == 0:
            r += " and no arithmetic progressions"
        else:
            r += " and arithmetic progressions: \n- "
            prog_strings = [str(prog) for prog in self.get_progressions()]
            r += "\n- ".join(prog_strings)
            
        self._cached_repr = r
        return r

    def _latex_(self):
        r"""
        OUTPUT:
        
        A latex representation of the zero pattern.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3)])
            sage: print(latex(ZeroPattern(set([0]), progressions))) 
            \{0\}\cup\{3 \cdot n + 3 : n \in \mathbb{N}\}
            
        """
        progressions = [prog._latex_() for prog in self.get_progressions()]
        progressions_latex = r"\cup".join(progressions)
        finite_set_str = [str(el) for el in self.get_finite_set()]
        finite_set_latex = r"\{" + ",".join(finite_set_str) + r"\}"
        if len(self.get_progressions()) == 0 and \
           len(self.get_finite_set()) == 0:
            return r"\{ \}"
        elif len(self.get_progressions()) == 0:
            return finite_set_latex
        elif len(self.get_finite_set()) == 0:
            return progressions_latex
        else:
            return finite_set_latex + r"\cup" + progressions_latex

    def __getitem__(self, n):
        r"""
        INPUT:

        - ``n`` -- a natural number or a slice object (specifying only finitely
          many values)

        OUTPUT:

        Returns ``True`` if the ``n``-th term is non-zero and ``False`` if it 
        is zero provided that ``n`` is a natural number. If ``n`` is a slice
        object, a list with these boolean values is returned.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
            
            sage: progressions = set([ArithmeticProgression(3, 3), \
            ....:                     ArithmeticProgression(5, 2)])
            sage: pattern = ZeroPattern(set([0]), progressions)
            sage: pattern[:5] 
            [False, True, False, False, True]
            
        """
        if isinstance(n, slice) :
            if n.stop == None :
                raise ValueError("Zero patterns are infinite. Need to specify"
                                 " upper bound.")
            else :
                l = list(range(n.stop)[n])
                return [self[i] for i in l]
        
        if n in self.get_finite_set():
            return False
        for prog in self.get_progressions():
            if n in prog:
                return False
        return True
    
    def __eq__(self, pattern) :
        r"""
        Checks whether two zero patterns are equal.
        
        INPUT:
        
        - ``pattern`` -- a zero pattern
        
        OUTPUT:
        
        ``True`` if the two patterns are equal and ``False`` 
        otherwise.
        """
        cycle_top_self = self.get_cycle_start() + self.get_cycle_length()
        cycle_top_pat = pattern.get_cycle_start() + pattern.get_cycle_length()
        check_bound = max(cycle_top_self, cycle_top_pat) + 1
        for n in range(check_bound) :
            if self[n] != pattern[n] :
                return False 
        return True

    def pattern(self, n=20):
        r"""
        INPUT:
        
        - ``n`` -- a natural number
        
        OUTPUT:
        
        Returns a string ``s`` of length ``n`` with characters ``0`` 
        (representing a zero at the corresponding position) and ``*``
        (representing a non-zero at the corresponding position).
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
            
            sage: progressions = set([ArithmeticProgression(3, 3), \
            ....:                     ArithmeticProgression(5, 2)])
            sage: pattern = ZeroPattern(set([0]), progressions)
            sage: print(pattern.pattern(5))
            0*00*
            
        """
        r = ""
        for i in range(n):
            if self[i]:
                r += "*"
            else:
                r += "0"
        return r

    def get_cycle_start(self):
        r"""
        OUTPUT:
        
        Returns the index from which the zero-pattern is guaranteed to
        be cyclic. This index might not be minimal.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3), \
            ....:                     ArithmeticProgression(5, 2)])
            sage: pattern = ZeroPattern(set([0]), progressions)
            sage: print(pattern.get_cycle_start())
            3
        
        """
        finite_set = [el+1 for el in self.get_finite_set()]
        progressions_shifts = [pro.get_shift() for
                               pro in self.get_progressions()]
        return max([0]+finite_set+progressions_shifts)

    def get_cycle_length(self):
        r"""
        OUTPUT:
        
        Returns the length of the cycle. 
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3), \
            ....:                     ArithmeticProgression(5, 2)])
            sage: pattern = ZeroPattern(set([0]), progressions)
            sage: print(pattern.get_cycle_length())
            15
        
        """
        progressions_diffs = [pro.get_diff() for
                              pro in self.get_progressions()]
        return lcm(progressions_diffs)

    def non_zero(self):
        r"""
        OUTPUT:
        
        Returns ``True`` if the pattern has no zeros at all and
        ``False`` otherwise.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3)])
            sage: ZeroPattern(set([0]), progressions).non_zero()
            False
            sage: ZeroPattern(set([3])).non_zero()
            False
            sage: ZeroPattern().non_zero()
            True
        
        """
        if len(self.get_progressions()) == 0 and \
           len(self.get_finite_set()) == 0:
            return True
        else:
            return False
        
    def is_zero(self):
        r"""
        OUTPUT:
        
        Returns ``True`` if the pattern is constantly zero.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3)])
            sage: ZeroPattern(set([0]), progressions).is_zero()
            False
            sage: ZeroPattern(set([3])).is_zero()
            False
            sage: ZeroPattern().is_zero()
            False
            sage: progressions2 = set([ArithmeticProgression(1, 1)])
            sage: ZeroPattern([0], progressions2).is_zero()
            True
            sage: ZeroPattern([], progressions2).is_zero()
            True
            
        """
        check = self.get_cycle_start() + self.get_cycle_length()
        for n in range(check) :
            if self[n] != False :
                return False 
        return True
        
    def eventually_non_zero(self):
        r"""
        OUTPUT:
        
        Returns ``True`` if the pattern is eventually non-zero, i.e.,
        the pattern only consists of finitely many zeros and no 
        arithmetic progressions.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3)])
            sage: ZeroPattern(set([0]), progressions).eventually_non_zero()
            False
            sage: ZeroPattern(set([3])).eventually_non_zero()
            True
            sage: ZeroPattern().eventually_non_zero()
            True
        
        """
        if len(self.get_progressions()) > 0 :
            return False
        else :
            return True
        
    def non_zero_start(self):
        r"""
        OUTPUT:
        
        Returns the index from which on the pattern has no zeros at all.
        If such an index does not exist, a ``ValueError`` is raised.
        
        EXAMPLES::
        
            sage: from rec_sequences.ZeroPattern import *
        
            sage: progressions = set([ArithmeticProgression(3, 3)])
            sage: ZeroPattern(set([0]), progressions).non_zero_start()
            Traceback (most recent call last):
            ...
            ValueError: Pattern contains infinitely many zeros
            sage: ZeroPattern(set([3])).non_zero_start()
            4
            sage: ZeroPattern().non_zero_start()
            0
        
        """
        if len(self.get_progressions()) > 0 :
            raise ValueError("Pattern contains infinitely many zeros")
        elif len(self.get_finite_set()) == 0 :
            return 0
        else :
            return max(self.get_finite_set())+1

    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        return sib.name("ZeroPattern")(sib(self.get_finite_set()),
                                       sib(self.get_progressions()))
