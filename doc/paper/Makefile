# $File: Makefile
# $Date: Mon Jun 15 22:59:59 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

TARGET = main
PDFDEPS = build $(addprefix build/,$(wildcard *.tex)) \
		  $(addprefix build/,$(wildcard *.bib)) \
		  build/res

all: view

build/$(TARGET).pdf: $(PDFDEPS)
	make -C trans pdf
	cd build && latexmk -xelatex -8bit -shell-escape $(TARGET).tex

build/%: %
	[ -h $@ ] || ln -s ../$< build/

build:
	mkdir $@
	cd $@ && ln -sv ../thuthesis/thu* . && latex thuthesis.ins

view: build/$(TARGET).pdf
	okular build/$(TARGET).pdf

rebuild: clean build/$(TARGET).pdf

clean:
	rm -rf build

.PHONY: all view clean rebuild

# vim: ft=make

