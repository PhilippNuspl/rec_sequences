SHELL:=/bin/bash
VERSION=$(shell cat ./VERSION)

# change to your sage command if needed
SAGE=sage

# Package folder
PACKAGE=rec_sequences

all: install doc docmv test
	
# Installing commands
install:
	$(SAGE) setup.py install


# Test using 8 threads
test: install
	$(SAGE) -tp 8 --force-lib $(PACKAGE)

# Documentation commands
doc: install
	cd docssrc && $(SAGE) -sh -c "make html"
	
docmv: doc
	mv docssrc/build/html .
	rm -r docs
	mv html docs
	touch docs/.nojekyll # add .nojekyll file to make sure that css in docs is loaded properly
