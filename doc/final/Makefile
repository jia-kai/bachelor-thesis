# $File: Makefile
# $Date: Mon Jun 22 14:49:04 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

TARGET = nasmia
PDFDEPS = build $(addprefix build/,$(wildcard *.tex)) \
		  $(addprefix build/,$(wildcard *.bib)) build/res

all: view

build/$(TARGET).pdf: $(PDFDEPS)
	cd build && latexmk -xelatex -8bit -shell-escape $(TARGET).tex
	ps2pdf14 $@ nasmia.pdf

build/%: %
	[ -h $@ ] || ln -s ../$< build/

build:
	mkdir $@

view: build/$(TARGET).pdf
	okular build/$(TARGET).pdf

rebuild: clean build/$(TARGET).pdf

impressive: build/$(TARGET).pdf
	impressive -ff -x nasmia.pdf -d 6m

clean:
	rm -rf build

.PHONY: all view clean rebuild impressive

# vim: ft=make

