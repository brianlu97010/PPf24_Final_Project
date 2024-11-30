# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3
NVCCFLAGS = -std=c++17 -O3 $(CUDA_FLAGS)
CUDA_FLAGS = -arch=sm_60  # Adjust based on your GPU architecture

# Target executable name
TARGET = jpeg_encoder

# Source files
CPP_SRC = main.cpp
CUDA_SRC = jpeg_encoder_cuda.cu

# Object files
CPP_OBJ = $(CPP_SRC:.cpp=.o)
CUDA_OBJ = $(CUDA_SRC:.cu=.o)
OBJ = $(CPP_OBJ) $(CUDA_OBJ)

# Header files
HEADERS = jpeg_encoder_cuda.h

# Default target
all: $(TARGET)

# Linking rule
$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compilation rules
%.o: %.cpp $(HEADERS)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f *.o $(TARGET)

.PHONY: all clean
