#ifndef _CLUSTER_GPU_KERNEL_
#define _CLUSTER_GPU_KERNEL_

#include <cstdint>
#include <iostream>
#include <mm_malloc.h>
#include "cuda_rt_call.h"
#include "cluster.h"

//#ifdef RMM
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
//#endif

#ifdef __cplusplus
   extern "C" {
#endif

typedef	struct {
  float memTransHDTime;
  float memTransDHTime;
  float memAllocTime;
  float memFreeTime;
  float unpackRawDataTime;
  float setSeedStripsTime;
  float setNCSeedStripsTime;
  float setStripIndexTime;
  float findBoundaryTime;
  float checkClusterTime;
  cudaEvent_t start, stop;
} gpu_timing_t;

void allocateSSTDataGPU(int max_strips, sst_data_t *sst_data_d, sst_data_t **pt_sst_data_d, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream,  rmm::mr::device_memory_resource *mr);
void allocateCalibDataGPU(int max_strips, calib_data_t *calib_data_d, calib_data_t **pt_calib_data_t, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream, rmm::mr::device_memory_resource *mr);
void allocateClustDataGPU(int max_strips, clust_data_t *clust_data_d, clust_data_t **pt_clust_data_t, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream, rmm::mr::device_memory_resource *mr);

void freeSSTDataGPU(int max_strips, sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream,  rmm::mr::device_memory_resource *mr);
void freeCalibDataGPU(int max_strips, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_t, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream, rmm::mr::device_memory_resource *mr);
void freeClustDataGPU(int max_strips, clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream, rmm::mr::device_memory_resource *mr);

void cpyGPUToCPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, clust_data_t *clust_data, clust_data_t *clust_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);
void cpySSTDataToGPU(sst_data_t *sst_data, sst_data_t *sst_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);
void cpyCalibDataToGPU(int max_strips, calib_data_t *calib_data, calib_data_t *calib_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);

void unpackRawDataGPU(const SiStripConditions *conditions, const SiStripConditionsGPU *conditionsGPU, const std::vector<FEDRawData>& fedRawDatav, const std::vector<FEDBuffer>& fedBufferv, const std::vector<fedId_t>& fedIndex, sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, const FEDReadoutMode& mode, gpu_timing_t *gpu_timing, cudaStream_t stream);

void findClusterGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, const SiStripConditionsGPU *conditions, clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);

void setSeedStripsNCIndexGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, const SiStripConditionsGPU *conditions, gpu_timing_t *gpu_timing, cudaStream_t stream);

#ifdef __cplusplus
   }
#endif

#endif