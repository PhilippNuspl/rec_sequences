===================================
rec_sequences: A Sage package
===================================

Introduction
=============

This `SageMath <https://www.sagemath.org/>`_ package
provides a framework to work with sequences satisfying linear recurrence
equations. 

- In the simple case, these are sequences satisfying recurrences
  with constant coefficients (so called `C`-finite sequences) or polynomial
  coefficients (so called `D`-finite sequences). The implementation of these
  sequences is based on the 
  |ore_algebra|_ package (cf. [KJJ15]_). 
- From these simple sequences
  one can create more complicated sequences, e.g. sequences satisfying a linear
  recurrence with `C`-finite coefficients, so called `C^2`-finite sequences.
- The package furthermore provides methods to show inequalities of `C`-finite 
  and `D`-finite sequences based on the Gerhold-Kauers method 
  (cf. [GK05]_, [KP10]_).

The theory of `C^2`-finite sequences which serves as the theoretic backbone of 
the package is developed in:

    - Antonio Jiménez-Pastor, Philipp Nuspl, Veronika Pillwein: 
      An extension of holonomic sequences: `C^2`-finite sequences (2021), 
      `10.35011/risc.21-20 <https://epub.jku.at/obvulioa/download/pdf/6880353?
      originalFilename=true>`_

The package is developed by `Philipp Nuspl <mailto:philipp.nuspl@jku.at>`_ and
published under the GNU General Public License v3.0.
The research is funded by the 
Austrian Science Fund (FWF): W1214-N15, project DK15. 

.. |ore_algebra| replace:: ``ore_algebra`` 
.. _ore_algebra: https://github.com/mkauers/ore\_algebra

Installation
=============

The package depends on the |ore_algebra|_ package. 
Several methods are available to install the ``rec_sequences`` package:

1. Using ``sage pip`` one can install the package using ::
       
       sage --pip install git+https://github.com/PhilippNuspl/rec_sequences.git

   This assumes that Sage was built from sources or installed from official
   packages. The command will also automatically install the ``ore_algebra``
   package. The flag ``--user`` can be used for a user local installation. 
2. One can clone the repository and run ``make install`` in the main directory
   of the repository. Again, this will automatically install the ``ore_algebra``
   package.
3. One can move the folder ``rec_sequences`` containing the sources to a 
   directory where Sage can find it.
   Then, ``ore_algebra`` has to be installed separately.

In any case, for showing inequalities the optional Sage package ``qepcad`` is used.
This can usually be installed using :: 

    sage -i qepcad

The ``rec_sequences`` package was developed under Sage 9.4. However, the 
package should also run under older Sage versions (Sage 8.7 or more recent). 

Examples
=========

After the installation, the modules can be imported.
Here, we want to compute with `C`-finite and `C^2`-finite sequences and
import the corresponding modules::

    sage: from rec_sequences.CFiniteSequenceRing import *
    sage: from rec_sequences.C2FiniteSequenceRing import *

Before we can specify specific sequences, we have to create the rings where
these sequences will live::

    sage: C = CFiniteSequenceRing(QQ)
    sage: C2 = C2FiniteSequenceRing(QQ)

Now, we can already specify some `C`-finite sequences, e.g. by giving the 
recurrence or by using some symbolic expression::

    sage: fib = C([1,1,-1], [0,1], name="f")
    sage: fib
    C-finite sequence f(n): (1)*f(n) + (1)*f(n+1) + (-1)*f(n+2) = 0 and f(0)=0 , f(1)=1
    sage: var("n")
    sage: alt = C((-1)^n)
    sage: alt
    C-finite sequence a(n): (1)*a(n) + (1)*a(n+1) = 0 and a(0)=1

Using these sequences, we can already extract terms, perform computations and 
check identities (like Cassini\'s identity here)::

    sage: fib[:10]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    sage: alt[5]
    -1
    sage: c = fib+alt
    sage: c.order()
    3
    sage: fib.shift(-1)*fib.shift(1) - fib^2 == alt
    True

Furthermore, using the Gerhold-Kauers method ([GK05]_, [KP10]_) 
we can check termwise inequalities for `C`-finite 
(or `D`-finite) sequences::

    sage: fib > 0
    False 
    sage: fib.sum() <= 2*fib
    False 
    sage: fib.sum() <= 3*fib
    True

For more information one can check out the documentation 
:class:`rec_sequences.CFiniteSequenceRing` and 
:class:`rec_sequences.DFiniteSequenceRing` (its superclass).

Now, we can also define `C^2`-finite sequences. Using closure properties
we can for instance check whether a certain recurrence is really satisfied
by the subsequence `f(n^2)` of the Fibonacci sequence `f(n)`::

    sage: sparse_fib = fib.sparse_subsequence(C2)
    sage: sparse_fib[:10]
    [0, 1, 3, 34, 987, 75025, 14930352, 7778742049]
    sage: sparse_fib.order(), sparse_fib.degree()
    (2, 2)
    sage: d = C([1, -54, 331, -54, 1], [136, 6710, 317434, 14927768])
    sage: sparse_fib2 = C2([-fib.subsequence(6,11),-d,fib.subsequence(6,9), 1], 
    ....:                  [0, 1, 3])
    sage: sparse_fib == sparse_fib2
    True

For more information one can check out the documentation 
:class:`rec_sequences.C2FiniteSequenceRing` and 
:class:`rec_sequences.DifferenceDefinableSequenceRing` 
(its superclass).

Module documentations
=====================

Here, the detailed documentation of all modules can be found.

Base rings 
----------

`C`-finite and `D`-finite sequence rings, based on the ``ore_algebra``, 
package are implemented. They can serve as the base rings for creating more
complicated rings. All sequence rings are derived from 
``RecurrenceSequenceRing`` which provides common functionality for all 
sequence rings where a sequence is defined by a linear recurrence equation.

.. toctree::
    :maxdepth: 1

    modules_docs/RecurrenceSequenceRing
    modules_docs/CFiniteSequenceRing
    modules_docs/DFiniteSequenceRing

Difference definable rings
---------------------------

Using the `C`-finite and `D`-finite sequence base rings we can create more 
complicated rings. Namely, rings of sequences which are defined by linear 
recurrences with coefficients in one of these base rings.
The class ``DifferenceDefinableSequenceRing`` allows to create these rings.
A special kind of such a ring, is the ring of `C^2`-finite sequences.
Such a ring contains sequences satisfying a linear recurrence with 
`C`-finite coefficients. The special class ``C2FiniteSequenceRing`` contains 
several additional functionalities for this ring, including the computation
of generating functions and a basic guessing routine.

.. toctree::
    :maxdepth: 1

    modules_docs/DifferenceDefinableSequenceRing
    modules_docs/C2FiniteSequenceRing
    
Arithmetic Progressions and Zero Patterns
--------------------------------------------------

The package provides functionalities to compute zeros and sign patterns of
sequences. The class ``ArithmeticProgression`` encapsulates functionality
to work with arithmetic progressions. By the Skolem-Mahler-Lech theorem,
the zeros of a `C`-finite sequence is a finite set together with a finite
number of arithmetic progressions. Such sets of zeros can be handled using
the class ``ZeroPattern``. Similarly, ``SignPattern`` provides a framework
to work with a sign pattern of a sequence which is cyclic from some term on.

.. toctree::
    :maxdepth: 1

    modules_docs/ArithmeticProgression
    modules_docs/ZeroPattern
    modules_docs/SignPattern

Additional Modules 
--------------------

The following modules contain auxiliary methods. The class 
``LinearSolveSequence`` provides methods to solve linear systems of equations
over sequence rings. The class ``SequenceRingOfFraction`` provides a ring 
structure for sequences which can be defined as the quotient of two sequences.
The class ``FunctionalEquation`` represents equations satisfied by the 
generating functions of `C^2`-finite sequences.
Finally, ``utility`` contains various other auxiliary methods.

.. toctree::
    :maxdepth: 1

    modules_docs/LinearSolveSequence
    modules_docs/FunctionalEquation
    modules_docs/SequenceRingOfFraction
    modules_docs/utility

References
------------

.. [GK05] Stefan Gerhold, Manuel Kauers: A Procedure for Proving Special 
    Function Inequalities Involving a Discrete Parameter. In: Proceedings of 
    ISSAC'05, pp. 156–162. 2005. 

.. [KJJ15] Manuel Kauers, Maximilian Jaroschek, Frederik Johansson: Ore 
   Polynomials in Sage. In: Computer Algebra and Polynomials. Lecture Notes in 
   Computer Science. 2015

.. [KP10] Manuel Kauers, Veronika Pillwein: When can we detect that
    a P-finite sequence is positive? In: Proceedings of 
    ISSAC'10, pp. 195–202. 2010. 

.. [JPNP21a] Antonio Jiménez-Pastor, Philipp Nuspl, Veronika Pillwein: 
   An extension of holonomic sequences: `C^2`-finite sequences (2021), doi:
   `10.35011/risc.21-20 <https://epub.jku.at/obvulioa/download/pdf/6880353?
   originalFilename=true>`_

.. [JPNP21b] Antonio Jiménez-Pastor, Philipp Nuspl, Veronika Pillwein: 
   On `C^2`-finite sequences. In: 
   Proceedings of ISSAC'21, pp. 217–224. 2021. `preprint <https://www.
   dk-compmath.jku.at/publications/dk-reports/2021-02-08jp-n-p/at_download/
   file>`_

.. [AKKOW21] Shaull Almagor, Toghrul Karimov, Edon Kelmendi, Joël Ouaknine, and 
   James Worrell. 2021. Deciding ω-regular properties on linear recurrence 
   sequences. Proc. ACM Program. Lang. 5, POPL, Article 48 (January 2021)
   
.. [OW14] Joël Ouaknine and James Worrell. Positivity problems for low-order linear 
   recurrence sequences. In: Proceedings of SODA 14. 2014
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
