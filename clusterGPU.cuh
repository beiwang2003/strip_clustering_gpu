#ifndef _CLUSTER_GPU_KERNEL_
#define _CLUSTER_GPU_KERNEL_

#include <cstdint>
#include <iostream>
#include <mm_malloc.h>
#include "cluster.h"
#include "cuda_rt_call.h"

#ifdef __cplusplus
   extern "C" {
#endif

typedef	struct {
  float memTransHDTime;
  float memTransDHTime;
  float memAllocTime;
  float memFreeTime;
  float setSeedStripsTime;
  float setNCSeedStripsTime;
  float setStripIndexTime;
  float findBoundaryTime;
  float checkClusterTime;
  cudaEvent_t start, stop;
} gpu_timing_t;

void allocateSSTDataGPU(int max_strips, sst_data_t *sst_data_d, sst_data_t **pt_sst_data_d, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream);
void allocateCalibDataGPU(int max_strips, calib_data_t *calib_data_d, calib_data_t **pt_calib_data_t, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream);
void allocateClustDataGPU(int max_strips, clust_data_t *clust_data_d, clust_data_t **pt_clust_data_t, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream);

void freeSSTDataGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream);
void freeCalibDataGPU(calib_data_t *calib_data_d, calib_data_t *pt_calib_data_t, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream);
void freeClustDataGPU(clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream);

void cpyGPUToCPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, clust_data_t *clust_data, clust_data_t *clust_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);
void cpySSTDataToGPU(sst_data_t *sst_data, sst_data_t *sst_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);
void cpyCalibDataToGPU(int max_strips, calib_data_t *calib_data, calib_data_t *calib_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);

void findClusterGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);

void setSeedStripsNCIndexGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);

#ifdef __cplusplus
   }
#endif

#endif