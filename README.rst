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
  recurrence with `C`-finite coefficients, so called `C^2`-finite sequences
  (cf. [JPNP21a]_, [JPNP21b]_).
- The package furthermore provides methods to show inequalities of `C`-finite 
  and `D`-finite sequences based on the Gerhold-Kauers method 
  (cf. [GK05]_, [KP10]_).

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


Documentation
--------------

The documentation, including examples on how to use the package can
be found at:

- `TBA <https://TBA>`_

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
   
