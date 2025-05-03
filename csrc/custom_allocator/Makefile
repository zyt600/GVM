all: uvm_allocator

uvm_allocator: uvm_allocator.cpp
	nvcc -std=c++17 -Xcompiler -fPIC -shared $@.cpp -o $@.so -lcuda
