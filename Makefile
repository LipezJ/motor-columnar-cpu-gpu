# make          → build/motor con los backends serial y omp
# make CUDA=1   → además el backend cuda (requiere nvcc)
# CUDA_ARCH     → arquitectura de la GPU destino: sm_75 T4, sm_80 A100, sm_89 L4

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra
CUDA_ARCH ?= sm_75

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
OBJETOS := $(FUENTES:src/%.cpp=build/%.o)

ifdef CUDA
  CXXFLAGS += -DCON_CUDA
  OBJETOS += build/cuda.o
  ENLAZADOR := nvcc -arch=$(CUDA_ARCH) -Xcompiler "$(OMP_LIB)"
else
  ENLAZADOR := $(CXX) $(OMP_LIB)
endif

build/motor: $(OBJETOS)
	$(ENLAZADOR) -o $@ $^

build/%.o: src/%.cpp src/*.hpp | build
	$(CXX) $(CXXFLAGS) $(OMP) -c -o $@ $<

build/cuda.o: src/cuda.cu src/*.hpp | build
	nvcc -O3 -std=c++17 -arch=$(CUDA_ARCH) -DCON_CUDA -c -o $@ $<

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: clean
