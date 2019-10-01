#ifndef _CLUSTER_GPU_KERNEL_
#define _CLUSTER_GPU_KERNEL_

#include <cstdint>
#include <iostream>
#include <mm_malloc.h>
#include "cluster.h"

#ifdef __cplusplus
   extern "C" {
#endif

#define USE_TEXTURE 1

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

void freeGPUMem(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_t, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d);

void cpyCPUToGPU(int nStrips, sst_data_t *sst_data, sst_data_t *sst_data_d, calib_data_t *calib_data, calib_data_t *calib_data_d, gpu_timing_t *gpu_timing);

void cpyGPUToCPU(int nSeedStripsNC, clust_data_t *clust_data, clust_data_t *clust_data_d);

void findClusterGPU(int SeedStripsNC, int nStrips, sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, gpu_timing_t *gpu_timing);

int setSeedStripsNCIndexGPU(int nStrips, sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, gpu_timing_t *gpu_timing);

#ifdef __cplusplus
   }
#endif

#endif