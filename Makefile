CXX = nvcc
CXXFLAGS = -std=c++17 -O3

TARGET = jpeg

SRC = test.cu jpeg_encoder.cu

OBJ = $(SRC:.cu=.o)

all: $(TARGET)

$(TARGET) : $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

%.o: %.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
