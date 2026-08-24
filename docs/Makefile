SPHINXBUILD = sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html clean

help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR)

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html -W --keep-going

clean:
	rm -rf $(BUILDDIR)
