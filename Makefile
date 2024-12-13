# Variables
REPORT_MD = report.md
OUTPUT_PDF = report.pdf
PANDOC = pandoc
PANDOC_OPTIONS = --from markdown \
					--to pdf \
					--pdf-engine=xelatex \
					-V links-as-notes=true \

# Default target
all: $(OUTPUT_PDF)

# Convert markdown to PDF
$(OUTPUT_PDF): $(REPORT_MD)
	$(PANDOC) $(PANDOC_OPTIONS) -o $@ $<

# Clean generated files
clean:
	rm -f $(OUTPUT_PDF)

# Declare phony targets
.PHONY: all clean
