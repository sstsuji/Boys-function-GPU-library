# Makefile

# Directories
SRC_DIR := src
INC_DIR := include
OBJ_DIR := obj
BIN_DIR := bin
GMP_DIR := ${GMP_ROOT}

# Compiler and flags
CXX := g++
NVCC := nvcc
CXXFLAGS := -I$(INC_DIR) -I$(GMP_DIR)/include -std=c++17 -O2 -fopenmp
NVCCFLAGS := -I$(INC_DIR) -I$(GMP_DIR)/include -std=c++17 -arch=sm_80 \
			 -Xcompiler -fopenmp
LDFLAGS := -L$(GMP_DIR)/lib -lgmp

# Source files
CU_SRCS := main.cu mp.cu h_bulk.cu d_bulk.cu d_key.cu \
		   d_single.cu d_incremental.cu
CPP_SRCS := h_single.cpp h_incremental.cpp

# Add directory prefix to source files
CU_SRSC := $(addprefix $(SRC_DIR)/, $(CU_SRCS))
CPP_SRCS := $(addprefix $(SRC_DIR)/, $(CPP_SRCS))

# Object files
CU_OBJS := $(addprefix $(OBJ_DIR)/, $(notdir $(CU_SRCS:.cu=.o)))
CPP_OBJS := $(addprefix $(OBJ_DIR)/, $(notdir $(CPP_SRCS:.cpp=.o)))

# Target executable
TARGET := $(BIN_DIR)/bboys

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# Compile CUDA source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Compile C++ source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean up build artifacts
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

# Phony targets
.PHONY: all clean