SHELL:=/bin/bash
VERSION=$(shell cat ./VERSION)

# change to your sage command if needed
SAGE=sage

# Package folder
PACKAGE=rec_sequences

all: install doc test
	
# Installing commands
install:
	$(SAGE) setup.py install


# Test using 8 threads
test: install
	$(SAGE) -tp 8 --force-lib $(PACKAGE)

# Documentation commands
doc: install
	cd docs && $(SAGE) -sh -c "make html"

	
