# $File: Makefile
# $Date: Mon Jun 15 22:59:46 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

TARGET = trans
PDFDEPS = build $(addprefix build/,$(wildcard *.tex)) \
		  $(addprefix build/,$(wildcard *.bib)) build/res

all: pdf view

pdf: build/$(TARGET).pdf

build/$(TARGET).pdf: $(PDFDEPS)
	cd build && latexmk -xelatex -8bit $(TARGET).tex

build/%: %
	[ -h $@ ] || ln -s ../$< build/

build:
	mkdir $@

view: build/$(TARGET).pdf
	evince build/$(TARGET).pdf

rebuild: clean build/$(TARGET).pdf

clean:
	rm -rf build

.PHONY: all view clean rebuild pdf

# vim: ft=make

