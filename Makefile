CC = g++
CXXFLAGS += -std=c++14 -march=native -fopt-info-vec -fopenmp -O3 -DUSE_GPU -DOUTPUT #-DCPU_DEBUG -DOUTPUT 
LDFLAGS += -std=c++14 -march=native -fopenmp -O3

#CC = icpc
#CXXFLAGS += -std=c++14 -xHost -qopt-report=5 -qopenmp -O3 -DOUTPUT
#LDFLAGS += -std=c++14 -xHost -qopenmp -O3

NVCC = nvcc
CUDAFLAGS += -std=c++14 -O3 -I/home/beiwang/clustering/cub-1.8.0 -gencode=arch=compute_60,code=\"sm_60,compute_60\" --ptxas-options=-v -DUSE_TEXTURE #-DGPU_DEBUG
# -arch=sm_60 is equavalent to -gencode=arch=compute_60,code=\"sm_60,compute_60\"

CUDALDFLAGS += -lcudart 

strip-cluster : strip-cluster.o cluster.o clusterGPU.o
	$(CC) $(LDFLAGS) $(CUDALDFLAGS) -o strip-cluster strip-cluster.o cluster.o clusterGPU.o
cluster.o: cluster.cc cluster.h
	$(CC) $(CXXFLAGS) -o cluster.o -c cluster.cc
clusterGPU.o: clusterGPU.cu 
	$(NVCC) $(CUDAFLAGS) -o clusterGPU.o -c clusterGPU.cu
strip-cluster.o: strip-cluster.cc cluster.h
	$(CC) $(CXXFLAGS) -o strip-cluster.o -c strip-cluster.cc

clean:
	rm -rf strip-cluster *.o *.optrpt
