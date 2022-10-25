TARGET := sudokuSolver
SRC_FILES := $(wildcard src/*.cu)
OBJ_FILES := $(patsubst src%,obj%, $(patsubst %.cu,%.o,$(SRC_FILES)))

INCLUDE := -I./src
LIBPATH :=
LIBS :=

FLAGS :=
CXXFLAGS := $(FLAGS)

CXX := nvcc

all: $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OBJ_FILES) -o $(TARGET) $(LIBPATH) $(LIBS)
	rm -r obj

$(OBJ_FILES): $(SRC_FILES)
	mkdir obj
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

.PHONY: clean
clean:
	rm $(TARGET)