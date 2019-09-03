default:  latex 

latex:
	pdflatex -synctex=1 -interaction=nonstopmode -enable-write18 sawtooth.tex
	bibtex sawtooth
	pdflatex -synctex=1 -interaction=nonstopmode  -enable-write18 sawtooth.tex
	pdflatex -synctex=1 -interaction=nonstopmode -enable-write18 sawtooth.tex

coverletter:
	pdflatex -synctex=1 -interaction=nonstopmode -enable-write18 coverletter.tex
