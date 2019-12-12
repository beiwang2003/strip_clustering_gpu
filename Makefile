SYSTEMS = $(shell hostname)
COMPILER = gnu

#tigergpu at princeton
ifneq (,$(findstring tigergpu, $(SYSTEMS)))
#git clone https://github.com/NVlabs/cub.git
	CUBROOT=/home/beiwang/clustering/cub-1.8.0
	GPUARCH=sm_60
endif 

#lnx7188 at cornell
ifneq (,$(findstring lnx7188, $(SYSTEMS)))
	CUBROOT=../cub-1.8.0
	GPUARCH=sm_70
endif

# phi3 at UCSD
ifneq (,$(findstring phi3, $(SYSTEMS)))
	GPUARCH=sm_70
endif

# compilers, CUDA, and cub from CMSSW
ifneq (,$(CMSSW_BASE))
	CUBROOT := $(shell cd $(CMSSW_BASE) && scram tool tag cub INCLUDE)
	CUDA_PATH := $(shell cd $(CMSSW_BASE) && scram tool tag cuda CUDA_BASE)
	CUDALIBDIR := $(shell cd $(CMSSW_BASE) && scram tool tag cuda LIBDIR)
endif


ifeq ($(COMPILER), gnu)
	CC = g++
	CXXFLAGS += -std=c++17 -O3 -fopenmp -fopt-info-vec -march=native \
	-I$(CUDA_PATH)/include -I$(CUBROOT) \
	-DUSE_GPU -DCACHE_ALLOC #-mprefer-vector-width=512 -DNUMA_FT -DOUTPUT -DCPU_DEBUG
	LDFLAGS += -std=c++17 -O3 -fopenmp -march=native #-mprefer-vector-width=512
endif

ifeq ($(COMPILER), intel)
	CC = icpc
	CXXFLAGS += -std=c++17 -O3 -qopenmp -qopt-report=5 -xHost \
	 -I$(CUDA_PATH)/include -I$(CUBROOT) \
	 -DNUMA_FT #-qopt-zmm-usage=high -DNUMA_FT -DOUTPUT -DCPU_DEBUG
	LDFLAGS += -std=c++17 -O3 -fopenmp -xHost -qopt-zmm-usage=high
endif

NVCC = nvcc
CUDAFLAGS += -std=c++14 -O3 --default-stream per-thread --ptxas-options=-v -lineinfo \
 -arch=$(GPUARCH) -I$(CUBROOT) \
 -DCACHE_ALLOC #-DCOPY_ADC -DGPU_TIMER #-DUSE_TEXTURE -DGPU_DEBUG -DCUB_STDERR
 # Note: -arch=sm_60 == -gencode=arch=compute_60,code=\"sm_60,compute_60\"
CUDALDFLAGS += -lcudart -L$(CUDALIBDIR)

ifeq ($(COMPILER), intel)
        CUDAFLAGS += -ccbin=icpc #specify intel for nvcc host compiler 
endif

strip-cluster : strip-cluster.o cluster.o clusterGPU.o allocate_host.o allocate_device.o
	$(CC) $(LDFLAGS) $(CUDALDFLAGS) -o strip-cluster strip-cluster.o cluster.o \
	clusterGPU.o allocate_host.o allocate_device.o
cluster.o: cluster.cc cluster.h
	$(CC) $(CXXFLAGS) -o cluster.o -c cluster.cc
clusterGPU.o: clusterGPU.cu 
	$(NVCC) $(CUDAFLAGS) -o clusterGPU.o -c clusterGPU.cu
allocate_host.o: allocate_host.cc
	$(CC) $(CXXFLAGS) -o allocate_host.o -c allocate_host.cc
allocate_device.o: allocate_device.cc
	$(CC) $(CXXFLAGS) -o allocate_device.o -c allocate_device.cc
strip-cluster.o: strip-cluster.cc cluster.h
	$(CC) $(CXXFLAGS) -o strip-cluster.o -c strip-cluster.cc


clean:
	rm -rf strip-cluster *.o *.optrpt
