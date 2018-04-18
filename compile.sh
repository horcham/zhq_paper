#!/bin/bash

xelatex -shell-escape thesis.tex
bibtex thesis.aux
xelatex -shell-escape thesis.tex
xelatex -shell-escape thesis.tex
rm *.aux *.log *.bbl *.blg *.toc 
evince thesis.pdf
