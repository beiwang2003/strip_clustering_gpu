#CC = g++
#CXXFLAGS += -std=c++14 -O3 -fopenmp -fopt-info-vec -march=native \
# -I${CUDA_PATH}/include -DOUTPUT #-DUSE_GPU #-DCPU_DEBUG
#LDFLAGS += -std=c++14 -O3 -fopenmp -march=native

CC = icpc
CXXFLAGS += -std=c++14 -O3 -qopenmp -qopt-report=5 -xHost \
 -I${CUDA_PATH}/include -DOUTPUT #-DUSE_GPU
LDFLAGS += -std=c++14 -O3 -fopenmp -xHost

NVCC = nvcc
#CUBROOT=/home/beiwang/clustering/cub-1.8.0
CUDAFLAGS += -std=c++14 -O3 --default-stream per-thread --ptxas-options=-v \
 -gencode=arch=compute_60,code=\"sm_60,compute_60\"  \
 -I${CUBROOT} -DGPU_TIMER #-DUSE_TEXTURE -DGPU_DEBUG
# Note: -arch=sm_60 == -gencode=arch=compute_60,code=\"sm_60,compute_60\"
CUDALDFLAGS += -lcudart -L${CUDALIBDIR}

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
