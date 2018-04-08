#!/bin/bash

xelatex -shell-escape zhq_paper.tex
bibtex zhq_paper.aux
xelatex -shell-escape zhq_paper.tex
xelatex -shell-escape zhq_paper.tex
rm *.aux *.log *.bbl *.blg *.toc 
evince zhq_paper.pdf
