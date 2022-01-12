# coding: utf-8
r"""
Describes an arithmetic progession of the form `(d n + s)_{n \in \mathbb{N}}`.
Such an arithmetic progressions is defined by a shift `s` and the 
difference of two consecutive terms `d`.

EXAMPLES::

    sage: from rec_sequences.ArithmeticProgression import *
    sage: prog = ArithmeticProgression(2, 3)
    sage: print(prog)
    Arithmetic progression (2*n+3)_n
    sage: prog.get_diff(), prog.get_shift()
    (2, 3)
    sage: 5 in prog
    True
    sage: 6 in prog
    False
    sage: prog[2]
    7
    sage: prog[:5]
    [3, 5, 7, 9, 11]
    
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

class ArithmeticProgression(SageObject):
    r"""
    Describes an arithmetic progession (diff*n+shift)_n for natural
    numbers diff, shift.
    """

    def __init__(self, diff=0, shift=0):
        r"""
        Creates the arithmetic progression `(d n+s)_n` where
        `d` is the given difference ``diff`` and `s` is the
        given shift ``shift``.

        INPUT:

        - ``diff`` -- the difference of two consecutive values in the 
          progression.
        - ``shift`` -- the shift of the arithmetic progression.

        OUTPUT:

        The described arithmetic progression.
        """
        SageObject.__init__(self)
        
        self._diff = diff
        self._shift = shift
        
    def __eq__(self, prog) :
        r"""
        Checks whether two arithmetic progressions are equal.
        
        INPUT:
        
        - ``prog`` -- an arithmetic progression
        
        OUTPUT:
        
        ``True`` if the two progressions are equal and ``False`` 
        otherwise.
        
        EXAMPLES::
        
            sage: from rec_sequences.ArithmeticProgression import *
            sage: prog1 = ArithmeticProgression(2, 3)
            sage: prog2 = ArithmeticProgression(2, 3)
            sage: prog3 = ArithmeticProgression(2, 1)
            sage: prog1 == prog2
            True
            sage: prog1 == prog3
            False
            
        """
        return (self.get_diff() == prog.get_diff() and \
                self.get_shift() == prog.get_shift())

    def get_diff(self):
        r"""
        Returns the difference of the arithmetic progression.
        """
        return self._diff

    def get_shift(self):
        r"""
        Returns the shift of the arithmetic progression.
        """
        return self._shift

    def is_zero(self):
        r"""
        True if the arithmetic progression is
        constantly zero.
        
        EXAMPLES::

            sage: from rec_sequences.ArithmeticProgression import *
            sage: ArithmeticProgression(2, 3).is_zero()
            False
            sage: ArithmeticProgression(0, 0).is_zero()
            True

        """
        return (self.get_shift() == 0 and self.get_diff() == 0)

    def __hash__(self):
        r"""
        Creates a hash using the difference and the shift of the progression.
        """
        try:
            return self._cached_hash
        except AttributeError:
            pass
        h = self._cached_hash = hash((self._diff, self._shift, "ArithProgr"))
        return h

    def _repr_(self):
        r"""
        OUTPUT:
        
        A string representation of the form "Arithmetic progression 
        (d*n+s)_n" 
        where ``d`` is the difference of the progression and ``s`` is
        the shift of the progression.
        
        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        diff = self.get_diff()
        shift = self.get_shift()
        r = self._cached_repr = f"Arithmetic progression ({diff}*n+{shift})_n"
        return r

    def _latex_(self):
        r"""
        OUTPUT:
        
        A latex representation of the form 
        
        .. MATH:: 
            \{ d n + s : n \in \mathbb{N} \}
        
        where `d` is the difference of the progression and `s` is
        the shift of the progression.
        
        """
        diff = str(self.get_diff())
        shift = str(self.get_shift())
        return r"\{" + diff + r" \cdot n + " + shift + r" : n \in \mathbb{N}" \
                     + r"\}"

    def __contains__(self, item):
        r"""
        Checks whether the arithmetic progression contains ``item`` 
        provided it is a natural number.
        
        INPUT:
        
        - ``item`` -- a natural number
        
        OUTPUT:
        
        ``True`` if ``item`` is in the progression and ``False`` otherwise.
        """
        return ((item-self.get_shift()) % self.get_diff()) == 0
    
    def __getitem__(self, n):
        r"""
        INPUT:

        - ``n`` -- a natural number or a slice object

        OUTPUT:

        Returns the ``n``-th term of the arithmetic progression.
        """
        if isinstance(n, slice) :
            if n.stop == None :
                raise ValueError("Arithmetic progressions are infinite. "
                                 "Need to specify upper bound.")
            else :
                l = list(range(n.stop)[n])
                return [self[i] for i in l]
        
        return self.get_shift() + n*self.get_diff()

    @classmethod
    def _check_cycle_(cls, i, d, data):
        r"""
        Returns true if ``data`` is cyclic from i on with
        cycle-length d. 
        """
        j = i+d
        m = len(data)
        for k in range(j, m):
            ref = (k-i) % d + i
            if data[k] != data[ref]:
                return False
        return True
    
    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        return sib.name("ArithmeticProgression")(sib(self.get_diff()),
                                                 sib(self.get_shift()))