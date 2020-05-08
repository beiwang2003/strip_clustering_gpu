SYSTEMS = $(shell hostname)
COMPILER = gnu
CUDA_PATH = /usr/local/cuda-10.2

#CUDA_PATH should set in the calling shell if CMSSW tools are not used

#tigergpu at pinceton
ifneq (,$(findstring tigergpu, $(SYSTEMS)))
#git clone https://github.com/NVlabs/cub.git
	CUBROOT=/home/beiwang/clustering/cub-1.8.0
	GPUARCH=sm_60
endif 

#lnx7188 and steve-t4 at cornell
ifneq (,$(filter lnx7188 steve-t4, $(subst ., ,$(SYSTEMS))))
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
#CUDALIBDIR := $(shell cd $(CMSSW_BASE) && scram tool tag cuda LIBDIR)
endif


ifeq ($(COMPILER), gnu)
	CC = g++
	CXXFLAGS += -std=c++17 -O3 -fopenmp -march=native \
	  -mprefer-vector-width=512 -fopt-info-vec -g \
	  -I$(CUDA_PATH)/include -I$(CUBROOT) \
        -DUSE_GPU -DCACHE_ALLOC -DNUMA_FT -DCALIB_1D #-DOUTPUT #-DACTIVE_STRIPS #-DCALIB_1D #-DOUTPUT -DCPU_DEBUG
	LDFLAGS += -std=c++17 -O3 -fopenmp -march=native \
	  -mprefer-vector-width=512 -fopt-info-vec -g
endif

ifeq ($(COMPILER), intel)
	CC = icpc
	CXXFLAGS += -std=c++17 -O3 -qopenmp -xHost \
	  -qopt-zmm-usage=high -qopt-report=5 \
	  -I$(CUDA_PATH)/include -I$(CUBROOT) -g \
	  -DNUMA_FT -DCALIB_1D #-DOUTPUT #-DOUTPUT #-DCALIB_1D #-DACTIVE_STRIPS -DCALIB_1D #-DOUTPUT -DCPU_DEBUG -DCPU_TIMER
	LDFLAGS += -std=c++17 -O3 -qopenmp -xHost \
	  -qopt-zmm-usage=high -qopt-report=5 -g
endif

NVCC = nvcc
CUDAFLAGS += -std=c++14 -O3 -g --default-stream per-thread -arch=$(GPUARCH) \
 -I$(CUBROOT) --ptxas-options=-v -lineinfo --maxrregcount 32\
 -DCACHE_ALLOC -DCALIB_1D #-DCOPY_ADC #-DGPU_DEBUG #-DGPU_TIMER #-DCOPY_ADC #-DGPU_TIMER #-DGPU_DEBUG #-DCALIB_1D #-DGPU_TIMER #-DCALIB_1D -DCOPY_ADC #-DGPU_TIMER #-DUSE_TEXTURE -DGPU_DEBUG -DCUB_STDERR
 # Note: -arch=sm_60 == -gencode=arch=compute_60,code=\"sm_60,compute_60\"
CUDALDFLAGS += -lcudart -L$(CUDA_PATH)/lib64

ifeq ($(COMPILER), intel)
	CUDAFLAGS += -ccbin=icpc #specify intel for nvcc host compiler
endif

strip-cluster : strip-cluster.o \
	  cluster.o clusterGPU.o SiStripConditions.o FEDChannel.o FEDRawData.o SiStripFEDBuffer.o allocate_host.o allocate_device.o
	$(CC) $(LDFLAGS) $(CUDALDFLAGS) -o strip-cluster strip-cluster.o \
          cluster.o clusterGPU.o SiStripConditions.o FEDChannel.o FEDRawData.o SiStripFEDBuffer.o allocate_host.o allocate_device.o
strip-cluster.o: strip-cluster.cc cluster.h SiStripConditions.h
	$(CC) $(CXXFLAGS) -o strip-cluster.o -c strip-cluster.cc
cluster.o: cluster.cc cluster.h 
	$(CC) $(CXXFLAGS) -o cluster.o -c cluster.cc
clusterGPU.o: clusterGPU.cu 
	$(NVCC) $(CUDAFLAGS) -o clusterGPU.o -c clusterGPU.cu
SiStripConditions.o: SiStripConditions.cc SiStripConditions.h
	$(CC) $(CXXFLAGS) -o SiStripConditions.o -c SiStripConditions.cc
FEDChannel.o: FEDChannel.cc FEDChannel.h host_unique_ptr.h device_unique_ptr.h
	$(CC) $(CXXFLAGS) -o FEDChannel.o -c FEDChannel.cc
FEDRawData.o: FEDRawData.cc FEDRawData.h 
	$(CC) $(CXXFLAGS) -o FEDRawData.o -c FEDRawData.cc
SiStripFEDBuffer.o: SiStripFEDBuffer.cc FEDChannel.h
	$(CC) $(CXXFLAGS) -o SiStripFEDBuffer.o -c SiStripFEDBuffer.cc
allocate_host.o: allocate_host.cc
	$(CC) $(CXXFLAGS) -o allocate_host.o -c allocate_host.cc
allocate_device.o: allocate_device.cc
	$(CC) $(CXXFLAGS) -o allocate_device.o -c allocate_device.cc

clean:
	rm -rf strip-cluster *.o *.optrpt
