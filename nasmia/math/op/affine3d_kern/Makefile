# $File: Makefile
# $Date: Sun May 03 01:49:05 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

BUILD_DIR = build
TARGET = libaffine3d.so

CXX = g++ -std=c++11
NVCC = nvcc
NVCC_GENCODE = -arch=compute_35 -code=sm_35 -ccbin=g++-4.8

override CPPFLAGS ?= -I. -I/usr/local/cuda/include
override OPTFLAG ?= -O2

override CXXFLAGS += \
	-ggdb \
	-Wall -Wextra -Wnon-virtual-dtor -Wno-unused-parameter -Winvalid-pch \
	-Wno-unused-local-typedefs -fPIC \
	$(CPPFLAGS) $(OPTFLAG)

NVCCFLAGS = $(NVCC_GENCODE) $(OPTFLAG) $(CPPFLAGS) \
			-g --use_fast_math -Xcompiler=-fPIC  \
			# -G #--ptxas-options=-v

override LDFLAGS += $(NVCC_GENCODE) -lcudart -lcuda -shared
override V ?= @

CXXSOURCES = $(shell find -L src -name "*.cpp")
CUDASOURCES = $(shell find -L src -name "*.cu")
CXXOBJS = $(addprefix $(BUILD_DIR)/cxx/,$(CXXSOURCES:.cpp=.o))
CUDAOBJS = $(addprefix $(BUILD_DIR)/cuda/,$(CUDASOURCES:.cu=.o))
OBJS = $(CXXOBJS) $(CUDAOBJS)
DEPFILES = $(OBJS:.o=.d)

all: $(TARGET)

-include $(DEPFILES)

$(BUILD_DIR)/cxx/%.o: %.cpp
	@echo "[cxx] $< ..."
	@mkdir -pv $(dir $@)
	@$(CXX) $(CPPFLAGS) -MM -MT "$@" "$<"  > "$(@:.o=.d)"
	$(V)$(CXX) -c $< -o $@ $(CXXFLAGS)

$(BUILD_DIR)/cuda/%.o: %.cu
	@echo "[nvcc] $< ..."
	@mkdir -pv $(dir $@)
	@$(CXX) $(CPPFLAGS) -x c++-header -MM -MT "$@" "$<"  > "$(@:.o=.d)"
	$(V)$(NVCC) $(NVCCFLAGS) -dc $< -o $@
	# @echo > /dev/stderr "debug enabled!!"

$(TARGET): $(OBJS)
	@echo "Linking ..."
	$(V)$(NVCC) $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean

# vim: ft=make

