CXX=g++

CUDA_INSTALL_PATH=/usr/local/cuda-6.5
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart `pkg-config --libs opencv`
COMPILE_FLAGS= -mcmodel=large -fPIC -g -Wall

#Uncomment the line below if you dont have CUDA enabled GPU
#EMU=-deviceemu

ifdef EMU
CUDAFLAGS+=-deviceemu
endif

all:
	$(CXX) $(COMPILE_FLAGS) -c main.cpp -o Debug/main.o $(CFLAGS)
	nvcc -c dehazing.cu -o Debug/kernel_gpu.o $(CUDAFLAGS) 
	$(CXX) $(COMPILE_FLAGS) Debug/main.o Debug/kernel_gpu.o -o Debug/dehazing $(LDFLAGS)

clean:
	rm -f Debug/*.o Debug/dehazing

