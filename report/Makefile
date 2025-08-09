PDF=usenix2019_v3.1

all: $(PDF).pdf

$(PDF).pdf: $(PDF).tex citations.bib
	pdflatex -interaction=nonstopmode $(PDF).tex
	biber $(PDF)
	pdflatex -interaction=nonstopmode $(PDF).tex
	pdflatex -interaction=nonstopmode $(PDF).tex

clean:
	rm -f $(PDF).aux $(PDF).bbl $(PDF).bcf $(PDF).blg $(PDF).log $(PDF).out $(PDF).run.xml $(PDF).toc

.PHONY: all clean
