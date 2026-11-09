# make: compila build/motor con los backends serial y omp.
# make CUDA=1: añade el backend cuda (requiere nvcc).
# CUDA_ARCH: arquitectura de la GPU destino (sm_52 para Titan X y M40, sm_75 para T4, sm_80 para A100, sm_89 para L4).
# NVCC: compilador de CUDA con sus opciones, p. ej. "nvcc -ccbin /usr/bin/g++" si el g++ por defecto es más nuevo de lo que admite esa versión de CUDA.

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra
CUDA_ARCH ?= sm_75
NVCC ?= nvcc
# Directorio de compilación: cada job de SLURM usa el suyo para no borrar el binario que otro está midiendo.
BUILD ?= build

# Apple clang no trae OpenMP: se usa el libomp de Homebrew.
ifeq ($(shell uname),Darwin)
  OMP := -Xpreprocessor -fopenmp -I$(shell brew --prefix libomp)/include
  OMP_LIB := -L$(shell brew --prefix libomp)/lib -lomp
  CXXFLAGS += -mcpu=native
else
  OMP := -fopenmp
  OMP_LIB := -fopenmp
  CXXFLAGS += -march=native
endif

FUENTES := src/main.cpp src/datos.cpp src/consultas.cpp src/serial.cpp src/omp.cpp
OBJETOS := $(FUENTES:src/%.cpp=$(BUILD)/%.o)

ifdef CUDA
  CXXFLAGS += -DCON_CUDA
  OBJETOS += $(BUILD)/cuda.o
  ENLAZADOR := $(NVCC) -arch=$(CUDA_ARCH) -Xcompiler "$(OMP_LIB)"
else
  ENLAZADOR := $(CXX) $(OMP_LIB)
endif

$(BUILD)/motor: $(OBJETOS)
	$(ENLAZADOR) -o $@ $^

$(BUILD)/%.o: src/%.cpp src/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMP) -c -o $@ $<

$(BUILD)/cuda.o: src/cuda.cu src/*.hpp | $(BUILD)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -DCON_CUDA -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: clean
