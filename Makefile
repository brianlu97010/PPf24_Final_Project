# CUDA path
CUDA_PATH ?= /usr/local/cuda

# Compiler settings
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# Include and library paths
CUDA_INCLUDES = -I$(CUDA_PATH)/include
CUDA_LIBS = -L$(CUDA_PATH)/lib64

# Compiler flags
CXXFLAGS = -O3 -Wall $(CUDA_INCLUDES)
NVCCFLAGS = -O3 -arch=sm_60 $(CUDA_INCLUDES)
LDFLAGS = $(CUDA_LIBS) -lcudart

# Source files
CU_SRCS = jpeg_encoder_cuda.cu
CPP_SRCS = main.cpp
HEADERS = jpeg_encoder_cuda.h

# Object files
OBJS = $(CU_SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)

# Output executable
TARGET = jpeg_encoder

# Default target
all: $(TARGET)

# Link the program
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LDFLAGS) -o $(TARGET)

# Compile CUDA source files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ source files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean