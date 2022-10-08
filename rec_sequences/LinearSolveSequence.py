# coding: utf-8
r"""
Provides methods to solve linear systems over sequence rings.
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

import logging
from datetime import datetime

from sage.misc.sage_input import sage_input
from sage.modules.free_module import VectorSpace
from sage.modules.free_module_element import free_module_element as vector
from sage.matrix.constructor import matrix
from sage.arith.functions import lcm
from sage.arith.misc import gcd
from sage.misc.misc_c import prod
from sage.repl.rich_output.pretty_print import show

from .ZeroPattern import ZeroPattern
from .utility import eval_matrix, eval_vector, matrix_subsequence, \
                     vector_subsequence, vector_interlace, log
from .SequenceRingOfFraction import SequenceRingOfFraction

class LinearSolveSequence(object):
    r"""
    Provides methods to solve linear systems over sequence rings.
    It is always assumed that the base ring satisfies the Skolem-Mahler-Lech
    Theorem.
    """
    
    log = logging.getLogger("LSS")
    
    @classmethod
    def solve(cls, M, b, guess = True, verified = False) :
        r"""
        Tries to compute a solution `x` of the linear system
        
        .. MATH::
            Mx=b.
            
        Let ``R`` be the base ring. We assume that a sequence in ``R`` 
        is invertible iff it does not have any zeros. If ``R`` is not 
        a :class:`rec_sequences.SequenceRingOfFraction`, 
        the sequences are casted there and the system solved there.
        
        INPUT:

        - ``M`` -- the matrix of the linear system
        - ``b`` -- the right-hand-side of the system
        - ``guess`` (default: ``True``) -- use guessing for solving the system
        - ``verified`` (default: ``False``) -- verify the result formally. If 
          ``False`` the result might not (although unlikely) be a solution to the equation if guessing is used.

        OUTPUT:

        One possible solution `x` as a list if there is one. Otherwise,
        a ``ValueError`` is raised. 
        
        ALGORITHM:
        
        The method implements the algorithm from
        Lemma 4.5 from [JPNP21b]_.
                    
        EXAMPLES::
        
            sage: from rec_sequences.LinearSolveSequence import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: from rec_sequences.SequenceRingOfFraction import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            
            sage: a = QC(C([3,0,-1], [1,3]), C([2,-1], [1]))
            sage: b = QC(C([1,-1], [2]))
            sage: c = QC(C([2,1], [3]))
            sage: d = QC(C([1,1,-1], [1,2]), C([-2,1], [3]))
            sage: v1 = vector(QC, [a,c])
            sage: v2 = vector(QC, [b,d])
            sage: M = matrix(QC, [v1, v2]).transpose()
            sage: rhs = QC(C([3,1], [1]))*v1 \
            ....:       + QC(C([1,2,-1],[0,1]), C([2,-1], [2]))*v2
            sage: x = vector(QC, LinearSolveSequence.solve(M, rhs))
            sage: M*x==rhs
            True

            sage: a2 = C([3,0,-1], [1,3])
            sage: b2 = C([1,-1], [2])
            sage: c2 = C([2,1], [3])
            sage: d2 = C([1,1,-1], [1,2])
            sage: v12 = vector(C, [a2,c2])
            sage: v22 = vector(C, [b2,d2])
            sage: M2 = matrix(C, [v12, v22]).transpose()
            sage: rhs2 = C([3,1], [1])*v12 + C([1,2,-1],[0,1])*v22
            sage: x2 = vector(QC, LinearSolveSequence.solve(M2, rhs2, \
            ....:                                           guess=False))
            sage: M2*x2==rhs2
            True
            
            sage: x3 = vector(QC, LinearSolveSequence.solve(M2, rhs2, \
            ....:                 guess=True, verified=True))
            sage: M2*x3==rhs2
            True
            
        """
        # show(M, "x=", b)
        #print(sage_input(M))
        #print(sage_input(b))
        solve_unique = LinearSolveSequence._solve_linear_system_unique
        
        rows = M.nrows()
        cols = M.ncols()
        R = M.base_ring()
        # make sure elements are in SequenceRingOfFraction
        if not isinstance(R, SequenceRingOfFraction) :
            QR = SequenceRingOfFraction(R)
            M_new = M.change_ring(QR)
            b_new = b.change_ring(QR)
            return LinearSolveSequence.solve(M_new, b_new, guess, verified)
        
        if verified :
            n0, p = LinearSolveSequence._zero_pattern_minors_verified(M) 
        else :
            n0, p =  LinearSolveSequence._zero_pattern_minors(M)
        log(cls, f"Zero pattern of minors cyclic of length {p} from {n0} on", 1)
        
        # find maximally linearly independent vectors 
        # for every m=n0,...,n0+p-1
        j = dict()
        for m in range(n0, n0+p) :
            Mm = eval_matrix(M, m)
            j[m] = LinearSolveSequence._max_cols(Mm)
            
        # solve for initial values
        solutions_x_init = []
        for m in range(0, n0) :
            M_eval = eval_matrix(M, m)
            b_eval = eval_vector(b, m)
            try :
                x_eval = M_eval.solve_right(b_eval)
                solutions_x_init.append(x_eval)
            except :
                raise ValueError(f"System not solvable at term {m}")
            
        # solve system, for all subsequences
        solutions_x = []
        for m in range(n0, n0+p) :
            M_part = M.matrix_from_columns(j[m])
            M_subs = matrix_subsequence(M_part, p, m)
            b_subs = vector_subsequence(b, p, m)
            log(cls, 
                f"Solve system at progession ({p}n+{m}) using columns {j[m]}",
                1)
            x_subs_wo_zeros = solve_unique(M_subs, b_subs, guess)
            
            # add zeros at columns which were removed
            x_subs = vector(R, cols)
            for l, jl in enumerate(j[m]) :
                x_subs[jl] = x_subs_wo_zeros[l]
                
            solutions_x.append(x_subs)
            
        # interlace solutions
        time_pre_interlace = datetime.now()
        x_interlaced = vector_interlace(solutions_x, solutions_x_init)
        time_post_interlace = datetime.now()
        msg = f"Solutions interlaced in time"
        log(cls, msg, 1, time_pre_interlace, time_post_interlace)
            
        # try to simplify solution
        x_interlaced = [seq.simplified() for seq in x_interlaced]
        time_post_simpl = datetime.now()
        msg = f"Simplified solutions"
        log(cls, msg, 1, time_post_interlace, time_post_simpl)
            
        if verified and guess:
            #check_system = LinearSolveSequence._check_linear_system_explicit
            check_system = LinearSolveSequence._check_linear_system
            checked = check_system(M, x_interlaced, b)
            if not isinstance(checked, bool) :
                if guess == True :
                    return LinearSolveSequence.solve(M, b, False, verified)
                else :
                    raise ValueError(f"Computed wrong solution at term {checked}")
        
        return x_interlaced
            

    @classmethod
    def _solve_linear_system_unique(cls, M, rhs, guess=True) :
        r"""
        Return the unique solution x, leading of the system M*x=rhs 
        assuming that such a solution exists
        If guessing is used, but failed, then the explicit algorithm is used.
        """
        if guess :
            ret = LinearSolveSequence._solve_linear_system_guess(M, rhs)
            if not ret :
                return LinearSolveSequence._solve_linear_system_explicit(M, rhs)
            else :
                return ret
        else :
            return LinearSolveSequence._solve_linear_system_explicit(M, rhs)

    @classmethod
    def _solve_linear_system_explicit(cls, M, rhs) :
        r"""
        Return a solution x of the system M*x=rhs 
        assuming that such a solution exists explicitly.
        """
        log(cls, f"Solve {M.nrows()}x{M.ncols()} system explicitly: ", 2)
        
        QR = M.base_ring()

        time_before_setting_up = datetime.now()
        
        M_gram = M.transpose()*M
        M_gram_det = M_gram.determinant()
        
        time_after_setting_up = datetime.now()
        msg = f"compute gramiam and determinant"
        log(cls, msg, 2, time_before_setting_up, time_after_setting_up)
        
        #M_gram_det_inv = QR(M_gram_det).inverse()
        M_inverse = M_gram.adjugate()*M.transpose()
                
        time_inverse_computed = datetime.now()
        msg = f"inverse computed"
        log(cls, msg, 2, time_after_setting_up, time_inverse_computed)
                
        x = M_inverse*rhs
        for i in range(len(x)) :
            x[i] = x[i]*M_gram_det.inverse_of_unit()
                        
        time_solution_computed = datetime.now()
        msg = f"solution computed"
        log(cls, msg, 2, time_inverse_computed, time_solution_computed)
        
        return x

    @classmethod
    def _solve_linear_system_guess(cls, M, rhs) :
        r"""
        Return a solution x, leading of the system M*x=rhs 
        assuming that such a solution exists.
        Uses guessing to compute a solution. This assumes that a unique solution exists for every n.
        If this is not the case, False is returned.
        """
        log(cls, f"Solve {M.nrows()}x{M.ncols()} system using guessing: ", 2)
        
        check_linear_system = LinearSolveSequence._check_linear_system
        guess_det = LinearSolveSequence._guess_det_gram

        QR = M.base_ring()
        R = QR.base()

        # First clear all denominators by multiplying all different
        # denominators of rhs (which should also contain all the ones from the matrix)
        # probably does not help much: Ex: 1*x=1/(2^n+3^n)
        # has solution in localisation, but (2^n+3^n)*y = 1
        # does not have a solution in the original ring
        time_before_clearing_denom = datetime.now()
        denominators = []
        common_denom = R.one()
        for frac in rhs :
            denominators.append(QR(frac).denominator())
        denominators = list(set(denominators))
        common_denom = prod(denominators)

        M_cleared = common_denom*M
        rhs_cleared = common_denom*rhs
        time_after_clearing_denom = datetime.now()
        msg = "denominators cleared"
        log(cls, msg, 2, time_before_clearing_denom, time_after_clearing_denom)

        # guess determinant of gramiam
        # order of determinant of gramiam can be very massive, might need far too many data points
        num_data_points = 100
        det_result = guess_det(M_cleared, R, num_data_points)
        time_after_det_guessed = datetime.now()
        msg = "tried to guess determinant"
        log(cls, msg, 2, time_after_clearing_denom, time_after_det_guessed)
        if not det_result :
            log(cls, "cannot guess determinant", 2)
            return False
        else :
            M_gram_det, M_cleared_n = det_result

        # compute solutions for fixed n in ground field
        x = []
        for n in range(num_data_points) :
            rhs_n = eval_vector(rhs_cleared, n)
            rhs_n = rhs_n*M_gram_det[n]
            try :
                x.append(M_cleared_n[n].solve_right(rhs_n))
            except ValueError:
                log(cls, f"no unique solution for n={n}", 2)
                return False
        time_ind_solutions = datetime.now()
        msg = f"individual solutions computed"
        log(cls, msg, 2, time_after_det_guessed, time_ind_solutions)

        # guess solution in sequence ring
        x_guessed = []
        for i in range(M.ncols()) :
            data_guessing_i = [x[n][i] for n in range(len(x))]
            try :
                x_guessed.append(R.guess(data_guessing_i))
            except ValueError:
                log(cls, "cannot guess solution", 2)
                return False
        x_guessed = vector(QR, x_guessed)
        for i in range(len(x_guessed)) :
            x_guessed[i] = x_guessed[i]*QR(M_gram_det).inverse_of_unit()
        time_guessed_solutions = datetime.now()
        msg = "solution guessed"
        log(cls, msg, 2, time_ind_solutions, time_guessed_solutions)

        return x_guessed

    @classmethod
    def _guess_det_gram(cls, M, R, number_entries) :
        r"""
        Guesses the determinant in R of the gramiam of M using ``number_entries``.
        Returns the guessed determinant and the evaluated matrices if the determinant
        could be guessed and False otherwise.
        """
        det_data = []
        M_n = []
        for n in range(number_entries) :
            M_n.append(eval_matrix(M, n))

        for n in range(number_entries) :
            M_gram_n = M_n[n].transpose()*M_n[n]
            det_data.append(M_gram_n.determinant())
        try :
            M_gram_det = R.guess(det_data)
        except ValueError:
            return False

        return M_gram_det, M_n


    @classmethod
    def _check_linear_system_explicit(cls, M, x, rhs) :
        r"""
        Checks whether ``x`` is a solution of ``M``*``x`` = ``rhs``.
        If this is not the case, False is returned.
        """
        if M*x != rhs :
            log(cls, "Solution of linear system is wrong", 1)
            return False
        return True
    
    @classmethod
    def _check_linear_system(cls, M, x, rhs) :
        r"""
        Checks whether ``x`` is a solution of ``M``*``x`` = ``rhs``.
        If this is not the case, an index is returned at which ``x``
        does not solve the system.
        This check is done by only checking enough initial values.
        """
        log(cls, "Check linear system using initial values: ", 1)
        start_check_solution = datetime.now()
        
        m = M.ncols()
        for l, row in enumerate(M.rows()) :
            # assume row*x = num/denom
            
            # write row[i].numerator() = a[i]
            # write row[i].denominator() = b[i]
            a = [row[i].n() for i in range(m)]
            b = [row[i].d() for i in range(m)]
            ord_num = sum(a[i].order()*x[i].n().order()* \
                          prod(b[j].order()*x[j].d().order() 
                                  for j in range(m) if j != i)
                          for i in range(m))
            ord_denom = prod(b[i].order()*x[i].d().order()
                                for i in range(m))
            bound_check = ord_num*rhs[l].d().order() + \
                          ord_denom*rhs[l].n().order()
                     
            log(cls, f"Check {bound_check} values", 2)     
            for k in range(bound_check) :
                eval_num = sum(a[i][k]*x[i].n()[k]* \
                               prod(b[j][k]*x[j].d()[k]
                                       for j in range(m) if j != i)
                               for i in range(m))
                eval_denom = prod(b[i][k]*x[i].d()[k] for i in range(m))
                if eval_num*rhs[l].d()[k] != eval_denom*rhs[l].n()[k] :
                    log(cls, f"Solution of linear system is wrong", 2)
                    return k
                
        end_check_solution = datetime.now()
        msg = "solution checked"
        log(cls, msg, 2, start_check_solution, end_check_solution)
        
        return True
        
    @classmethod
    def _max_cols(cls, M) :
        r"""
        Given a matrix ``M`` over a field, return a list of
        indices such that the corresponding columns are a basis
        of the image of ``M``.
        
        The algorithm is very inefficient and faster methods exist.
        It removes linearly dependent columns until the columns
        are linearly independent.
        """
        if M.is_zero() :
            return []
        
        K = M.base_ring()
        VS = VectorSpace(K, M.nrows())
        cols = M.columns()
        indices = list(range(len(cols)))
        # remove linearly dependent vectors until
        # they are linearly independent
        while VS.are_linearly_dependent([cols[i] for i in indices]) and \
              len(indices) > 0:
            for index in indices :
                try :
                    M_red = matrix(K, [cols[i] for i in indices if i != index])
                    M_red.transpose().solve_right(cols[index])
                    indices.remove(index) 
                    break                  
                except ValueError:
                    pass
        return indices
        
        
    @classmethod
    def _zero_pattern_minors(cls, M, guess = 100) :
        r"""
        Computes the zero pattern of the minors by guessing it.
        
        INPUT:

        - ``M`` -- the matrix 
        - ``guess`` (default: ``100``) -- the number of terms used to determine
          the 0 pattern

        OUTPUT:

        start_cycle, cycle_length such that from ``start_cycle`` on, all zeros
        in all minors are cyclic with the common ``cycle_length``
        """
        evaluations_M = [eval_matrix(M, n) for n in range(guess)]
        evaluations_minors = [LinearSolveSequence._get_minors(M_eval)
                                for M_eval in evaluations_M]
        start_cycle = 0
        cycle_length = 1
        num_minors = len(evaluations_minors[0])
        for i in range(num_minors) :
            minor_seq = [evaluations_minors[j][i] for j in range(guess)]  
            pattern = ZeroPattern.guess(minor_seq)
            start_cycle = max([start_cycle, pattern.get_cycle_start()])
            cycle_length = lcm([cycle_length, pattern.get_cycle_length()])
            
        return start_cycle, cycle_length
    
    @classmethod
    def _zero_pattern_minors_verified(cls, M) :
        r"""
        Computes the zero pattern of the minors using exact methods.
        
        INPUT:

        - ``M`` -- the matrix 

        OUTPUT:

        start_cycle, cycle_length such that from ``start_cycle`` on, all zeros
        in all minors are cyclic with the common ``cycle_length``
        """
        minors = LinearSolveSequence._get_minors(M)
        
        start_cycle = 0
        cycle_length = 1
        for minor in minors :
            pattern_found = False
            for guess_length in [20, 50, 100, 300, 1000] :
                try :
                    pattern = minor.zeros(data=guess_length)
                    pattern_found = True
                    break
                except ValueError :
                    pass
            if not pattern_found :
                log(cls, f"Zero pattern of {sage_input(minor)} could not " 
                    "be found", 2)
                guess_length = 2000
                while not pattern_found :
                    try :
                        pattern = minor.zeros(data=guess_length)
                        pattern_found = True
                        break
                    except ValueError :
                        pass
                    guess_length = guess_length*2
                    
            start_cycle = max([start_cycle, pattern.get_cycle_start()])
            cycle_length = lcm([cycle_length, pattern.get_cycle_length()])
            
        return start_cycle, cycle_length
        
    @classmethod
    def _get_minors(cls, M) :
        r"""
        Returns a list of all minors of a matrix ``M``.
        """
        min_dim = min(M.nrows(), M.ncols())
        minors = []
        for k in range(1,min_dim+1) :
            minors += M.minors(k)
        return minors
   
    @classmethod
    def clear_divisor(cls, v, bound=50) :
        r"""
        Given a vector of sequences `v=(v_0,\dots,v_m)`
        computes a common divisor `d` of `v_0,\dots,v_m`.
        Tries guessing to find the gcd using ``bound`` number of values.
        If this fails, the common divisor `1` is returned. 
        Assumes that `m \geq 0`, i.e., `v` is non-empty.
        
        INPUT:
        
        - ``v`` -- a vector of sequences
        - ``bound`` (default: ``50``) -- a natural number
        
        OUTPUT:
        
        A pair `d`, ``divisors`` where `d` is a common divisor
        of all `v_i` and ``divisors`` is a list of sequences
        `d_0,\dots,d_m` with `d_i d = v_i` for all `i`.
        
        EXAMPLES::
        
            sage: from rec_sequences.LinearSolveSequence import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: c1 = C([3,0,-1], [1,3])
            sage: c2 = C([-1,1,1], [1,2])
            sage: c3 = C([2,-1,1], [2,1])
            
            sage: c = [c1*c2, c1*c3]
            sage: d, div = LinearSolveSequence.clear_divisor(c, 10)
            sage: d == c1
            True
            sage: d*div[0] == c[0], d*div[1] == c[1]
            (True, True)
            
        """
        R = v[0].parent()
        
        log(cls, f"Clear divisor: ", 1)
        
        log(cls, f"use {bound} values to guess gcd", 2)
        # try guessing to find the lcm
        data = [gcd([seq[n] for seq in v]) for n in range(bound)]
        try :
            time_pre = datetime.now()
            guessed_gcd = R.guess(data)
            
            # is it really a gcd?
            divisors = [guessed_gcd.divides(seq, bound, True) for seq in v]
                        
            time_post = datetime.now()
            if all([not isinstance(div, bool) for div in divisors]) :
                ord = guessed_gcd.order()
                log(cls, f"gcd of order {ord} computed using guessing", 2, 
                    time_pre, time_post)
                return guessed_gcd, divisors
        except (ValueError, ZeroDivisionError) :
            pass 
        
        log(cls, f"Use 1 as common divisor", 2)
        return R.one(), v

    
    @classmethod
    def common_multiple(cls, v, bound=100) :
        r"""
        Given a vector of sequences `v=(v_0,\dots,v_m)`
        computes a common multiple `d` of `v_0,\dots,v_m`.
        Tries guessing to find the lcm using ``bound`` number of values.
        If this fails, the product `v_0 \cdots v_m` is returned. 
        Assumes that `m \geq 0`, i.e., `v` is non-empty.
        
        INPUT:
        
        - ``v`` -- a vector of sequences
        - ``bound`` (default: ``100``) -- a natural number
        
        OUTPUT:
        
        A pair `d`, ``divisors`` where `d` is a common multiple
        of all `v_i` and ``divisors`` is a list of sequences
        `d_0,\dots,d_m` with `d_i v_i = d` for all `i`.
        
        EXAMPLES::
        
            sage: from rec_sequences.LinearSolveSequence import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: c1 = C([3,0,-1], [1,3])
            sage: c2 = C([1,1,-1], [1,2])
            
            sage: d, div = LinearSolveSequence.common_multiple([c1*c2, c1], 10)
            sage: d == c1*c2
            True
            sage: div[0]*c1*c2 == d, div[1]*c1 == d
            (True, True)
            
        """
        R = v[0].parent()
        # makes no sens to make bound bigger than the order of 
        # the product of the sequence
        bound = max(min(prod([vi.order() for vi in v]), bound), 10)
        log(cls, f"use {bound} values to guess lcm", 2)
        # try guessing to find the lcm
        data = [lcm([seq[n] for seq in v]) for n in range(bound)]
        try :
            guessed_lcm = R.guess(data)
            # is it really an lcm?
            divisors = [seq.divides(guessed_lcm, bound, True) for seq in v]
            if all([not isinstance(div, bool) for div in divisors]) :
                log(cls, f"lcm of sequences computed using guessing", 2)
                return guessed_lcm, divisors
        except (ValueError, ZeroDivisionError) :
            pass
        
        # guessing did not work, so just return product
        log(cls, f"use product as lcm", 2)
        divisors = [prod(v[j] for j in range(len(v)) if j != i) 
                              for i in range(len(v))]
        return prod(v), divisors
    
    @classmethod
    def clear_denominators(cls, v, bound=100) :
        r"""
        Clears the denominators of fractional sequences.
        
        INPUT:
        
        - ``v`` -- a (non-empty) vector or list of sequences in a
          :class:`rec_sequences.SequenceRingOfFraction`.
        - ``bound`` (default: ``100``) -- a natural number, used as the
          number of terms to guess a common multiple of the denominators
          of the sequences in ``v``.
        
        OUTPUT:
        
        ``d`` and a list ``v*d`` such that ``d`` is a common multiple of the
        denominators of ``v``.
        
        EXAMPLES::
        
            sage: from rec_sequences.LinearSolveSequence import *
            sage: from rec_sequences.CFiniteSequenceRing import *
            sage: from rec_sequences.SequenceRingOfFraction import *
            
            sage: C = CFiniteSequenceRing(QQ)
            sage: QC = SequenceRingOfFraction(C)
            sage: d0 = C([2,0,-1], [1,-2])
            sage: d1 = C([1,-2,1], [3,1])
            sage: c0 = C([3,0,-1], [1,3])
            sage: c1 = C([1,1,-1], [1,2])
            sage: a = QC(c0, d0)
            sage: b = QC(c1, d1)
            
            sage: v = vector(QC, [a, b])
            sage: d, cleared = LinearSolveSequence.clear_denominators(v, 10)
            sage: d == d0*d1
            True
            sage: cleared_v = vector(C, cleared)
            sage: v*d == cleared_v
            True
            
        """
        log(cls, f"Clear denominators: ", 1)

        time_pre_d = datetime.now()
        denoms = [frac.d() for frac in v]
        d, divisors = LinearSolveSequence.common_multiple(denoms, bound)
        time_post_d = datetime.now()
        log(cls, f"Common multiple of order {d.order()} computed", 2, 
            time_pre_d, time_post_d)
        cleared = [frac.n()*div for frac, div in zip(v, divisors)]

        time_post_cleared = datetime.now()
        log(cls, f"Common multiple cleared", 2, time_post_d, time_post_cleared)
        
        return d, cleared

        
        
            
