# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = highway-env
SOURCEDIR     = docs
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

apidoc:
	# sphinx-apidoc -o docs -e highway_env
	cd highway_env ; mv -vf __init__.py __init__.py.old ; sphinx-apidoc -f -o ../docs -e -M . ; mv -vf __init__.py.old __init__.py; cd ..

http:
	python -mwebbrowser "http://localhost:8000/_build/html/"
	python -m SimpleHTTPServer


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)