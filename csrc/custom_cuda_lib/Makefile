all: custom_cuda

custom_cuda: custom_cuda.cpp
	g++ -shared -fPIC -o libcustom_cuda.so custom_cuda.cpp -ldl
