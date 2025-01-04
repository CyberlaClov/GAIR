# Variables
MD_FILES := $(wildcard *.md)
PDF_FILES := $(MD_FILES:.md=.pdf)
PANDOC = pandoc
PANDOC_OPTIONS = --from markdown \
					--to pdf \
					--pdf-engine=xelatex \
					-V links-as-notes=true

# Default target
all: $(PDF_FILES)

# Pattern rule to convert markdown to PDF
%.pdf: %.md
	$(PANDOC) $(PANDOC_OPTIONS) -o $@ $<

# Clean generated files
clean:
	rm -f $(PDF_FILES)

# Declare phony targets
.PHONY: all clean
