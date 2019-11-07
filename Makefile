CMSSW_CUDAUTILS_PATH=../cmssw
#git clone https://github.com/cms-patatrack/cmssw.git
EXTERNAL_SOURCE = ${CMSSW_CUDAUTILS_PATH}/HeterogeneousCore/CUDAUtilities/src
CUDA_API_PATH=../cuda-api-wrappers/src
#git clone https://github.com/cms-externals/cuda-api-wrappers.git

CC = g++
CXXFLAGS += -std=c++14 -O3 -fopenmp -fopt-info-vec -march=native \
 -I${CUDA_PATH}/include -I${CMSSW_CUDAUTILS_PATH} -I${CUDA_API_PATH} \
 -mprefer-vector-width=512 #-DUSE_GPU -DCACHE_ALLOC #-DOUTPUT #-DCPU_DEBUG
LDFLAGS += -std=c++14 -O3 -fopenmp -march=native -mprefer-vector-width=512

#CC = icpc
#CXXFLAGS += -std=c++14 -O3 -qopenmp -qopt-report=5 -xHost \
# -I${CUDA_PATH}/include -I${CMSSW_CUDAUTILS_PATH} -I${CUDA_API_PATH} \
# -qopt-zmm-usage=high #-DOUTPUT #-DUSE_GPU
#LDFLAGS += -std=c++14 -O3 -fopenmp -xHost -qopt-zmm-usage=high

NVCC = nvcc
CUBROOT=../cub-1.8.0
#git clone https://github.com/NVlabs/cub.git
CUDAFLAGS += -std=c++14 -O3 --default-stream per-thread --ptxas-options=-v \
 -gencode=arch=compute_60,code=\"sm_60,compute_60\"  \
 -I${CUBROOT} -I${CMSSW_CUDAUTILS_PATH} -I${CUDA_API_PATH} -DCUB_STDERR \
 -DGPU_TIMER #-DCACHE_ALLOC #-DUSE_TEXTURE -DGPU_DEBUG
# Note: -arch=sm_60 == -gencode=arch=compute_60,code=\"sm_60,compute_60\"
CUDALDFLAGS += -lcudart -L${CUDALIBDIR}

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
