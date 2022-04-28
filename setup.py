## -*- encoding: utf-8 -*-
import os
import sys
from setuptools import setup
from codecs import open # To open the README file with proper encoding

# Get information from separate files (README, VERSION)
def readfile(filename):
    with open(filename,  encoding='utf-8') as f:
        return f.read()
    
setup(
    name = "rec_sequences",
    version = readfile("VERSION").strip(), # the VERSION file is shared with the documentation
    description='A Sage package for sequences defined by linear recurrence equations',
    #long_description = readfile("README.md"),
    #long_description_content_type="text/markdown",
    #url='TBA',
    author = "Philipp Nuspl",
    author_email = "philipp.nuspl@jku.at",
    license = "GPLv3+", # See LICENCE file
    classifiers=[
      # How mature is this project? Common values are
      #   3 - Alpha
      #   4 - Beta
      #   5 - Production/Stable
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Topic :: Software Development :: Build Tools',
      'Topic :: Scientific/Engineering :: Mathematics',
      'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
      'Programming Language :: Python :: 3.9.9',
    ], # classifiers list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    keywords = "holonomic SageMath",
    packages = ["rec_sequences"],
    setup_requires   = [],
    install_requires = ['ore_algebra @ git+https://github.com/mkauers/ore_algebra.git','sphinx', 'sphinx-rtd-theme'],
)
