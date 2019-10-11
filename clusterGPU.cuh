#ifndef _CLUSTER_GPU_KERNEL_
#define _CLUSTER_GPU_KERNEL_

#include <cstdint>
#include <iostream>
#include <mm_malloc.h>
#include "cluster.h"

#ifdef __cplusplus
   extern "C" {
#endif

typedef	struct {
  float memTransferTime;
  float setSeedStripsTime;
  float setNCSeedStripsTime;
  float setStripIndexTime;
  float findBoundaryTime;
  float checkClusterTime;
  cudaEvent_t start, stop;
} gpu_timing_t;

void allocateSSTDataGPU(int nStrips, sst_data_t *sst_data_d, sst_data_t **pt_sst_data_d);
void allocateCalibDataGPU(int nStrips, calib_data_t *calib_data_d, calib_data_t **pt_calib_data_t);
void allocateClustDataGPU(int nSeedStripsNC, clust_data_t *clust_data_d, clust_data_t **pt_clust_data_t);

void freeSSTDataGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d);
void freeCalibDataGPU(calib_data_t *calib_data_d, calib_data_t *pt_calib_data_t);
void freeClustDataGPU(clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d);

void cpyGPUToCPU(int event, int nStreams, int max_strips, int nStrips, sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, clust_data_t *clust_data, clust_data_t *clust_data_d, cudaStream_t stream);
void cpySSTDataToGPU(int nStrips, sst_data_t *sst_data, sst_data_t *sst_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);
void cpyCalibDataToGPU(int nStrips,  calib_data_t *calib_data, calib_data_t *calib_data_d, gpu_timing_t *gpu_timing);

void findClusterGPU(int event, int nStreams, int max_strips, int nStrips, sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);

void setSeedStripsNCIndexGPU(int nStrips, sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream);

#ifdef __cplusplus
   }
#endif

#endif