SYSTEMS = $(shell hostname)
COMPILER = gnu

#tigergpu at princeton
ifneq (,$(findstring tigergpu, $(SYSTEMS)))
#git clone https://github.com/cms-patatrack/cmssw.git
	CMSSW_CUDAUTILS_PATH=/home/beiwang/clustering/cmssw
#git clone https://github.com/NVlabs/cub.git
	CUBROOT=/home/beiwang/clustering/cub-1.8.0
#git clone https://github.com/cms-externals/cuda-api-wrappers.git
	CUDA_API_PATH=/home/beiwang/clustering/cuda-api-wrappers/src
endif 

#lnx7188 at cornell
ifneq (,$(findstring lnx7188, $(SYSTEMS)))
	CMSSW_CUDAUTILS_PATH=../cmssw
	CUBROOT=../cub-1.8.0
	CUDA_API_PATH=../cuda-api-wrappers/src
endif

EXTERNAL_SOURCE = ${CMSSW_CUDAUTILS_PATH}/HeterogeneousCore/CUDAUtilities/src

ifeq ($(COMPILER), gnu)
	CC = g++
	CXXFLAGS += -std=c++17 -O3 -fopenmp -fopt-info-vec -march=native \
	-I${CUDA_PATH}/include -I${CMSSW_CUDAUTILS_PATH} -I${CUDA_API_PATH} \
	-mprefer-vector-width=512 -DUSE_GPU -DCACHE_ALLOC #-DOUTPUT #-DCPU_DEBUG
	LDFLAGS += -std=c++17 -O3 -fopenmp -march=native -mprefer-vector-width=512
endif

ifeq ($(COMPILER), intel)
	CC = icpc
	CXXFLAGS += -std=c++14 -O3 -qopenmp -qopt-report=5 -xHost \
	 -I${CUDA_PATH}/include -I${CMSSW_CUDAUTILS_PATH} -I${CUDA_API_PATH} \
	 -qopt-zmm-usage=high #-DOUTPUT #-DUSE_GPU
	LDFLAGS += -std=c++14 -O3 -fopenmp -xHost -qopt-zmm-usage=high
endif

NVCC = nvcc
CUDAFLAGS += -std=c++14 -O3 --default-stream per-thread --ptxas-options=-v \
 -gencode=arch=compute_60,code=\"sm_60,compute_60\"  \
 -I${CUBROOT} -I${CMSSW_CUDAUTILS_PATH} -I${CUDA_API_PATH} -DCUB_STDERR \
 -DCACHE_ALLOC #-DGPU_TIMER -DCACHE_ALLOC #-DUSE_TEXTURE -DGPU_DEBUG
# Note: -arch=sm_60 == -gencode=arch=compute_60,code=\"sm_60,compute_60\"
CUDALDFLAGS += -lcudart -L${CUDALIBDIR} \
-L${CMSSW_CUDAUTILS_PATH}/HeterogeneousCore/CUDAUtilities/src

strip-cluster : strip-cluster.o cluster.o clusterGPU.o allocate_host.o allocate_device.o
	$(CC) $(LDFLAGS) $(CUDALDFLAGS) -o strip-cluster strip-cluster.o cluster.o \
	clusterGPU.o allocate_host.o allocate_device.o
cluster.o: cluster.cc cluster.h
	$(CC) $(CXXFLAGS) -o cluster.o -c cluster.cc
clusterGPU.o: clusterGPU.cu 
	$(NVCC) $(CUDAFLAGS) -o clusterGPU.o -c clusterGPU.cu
allocate_host.o: ${EXTERNAL_SOURCE}/allocate_host.cc
	$(NVCC) $(CUDAFLAGS) -o allocate_host.o -c ${EXTERNAL_SOURCE}/allocate_host.cc
allocate_device.o: ${EXTERNAL_SOURCE}/allocate_device.cc
	$(NVCC) $(CUDAFLAGS) -o allocate_device.o -c ${EXTERNAL_SOURCE}/allocate_device.cc
strip-cluster.o: strip-cluster.cc cluster.h
	$(CC) $(CXXFLAGS) -o strip-cluster.o -c strip-cluster.cc


clean:
	rm -rf strip-cluster *.o *.optrpt
