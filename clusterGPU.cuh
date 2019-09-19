#ifndef _CLUSTER_GPU_KERNEL_
#define _CLUSTER_GPU_KERNEL_

#include <cstdint>
#include <iostream>
#include <mm_malloc.h>

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

void allocateMemAllStripsGPU(int max_strips, uint16_t **stripId_d_pt, uint16_t **adc_d_pt, float **noise_d_pt, float **gain_d_pt, int **seedStripsNCIndex_d_pt);

void allocateMemNCSeedStripsGPU(int nSeedStripsNC, int **clusterLastIndexLeft_d_pt, int **clusterLastIndexRight_d_pt, float **clusterNoiseSquared_d_pt, uint8_t **clusterADCs_d_pt, bool **trueCluster_d_pt);

void freeGPUMem(uint16_t *stripId_d, float *noise_d, int *seedStripNCIndex_d, int *clusterLastIndexLeft_d, float *clusterNoiseSquared_d, uint8_t *clusterADCs_d, bool *trueCluster_d);

void cpyCPUToGPU(int nStrips, uint16_t *stripId_d, uint16_t *stripId, uint16_t *adc_d,  uint16_t *adc, float *noise_d, float *noise, float *gain_d, float *gain, gpu_timing_t *gpu_timing);

void cpyGPUToCPU(int nSeedStripsNC, int *clusterLastIndexLeft_d, int *clusterLastIndexLeft, int *clusterLastIndexRight_d, int *clusterLastIndexRight, uint8_t *clusterADCs_d, uint8_t *clusterADCs, bool *trueCluster_d, bool *trueCluster);

void  findClusterGPU(int nSeedStripsNC, int nStrips, float *clusterNoiseSquared_d, int *clusterLastIndexLeft_d, int *clusterLastIndexRight_d, int *seedStripsNCIndex_d, uint16_t *stripId_d, uint16_t *adc_d,  float *noise_d, float *gain_d, bool *trueCluster_d, uint8_t *clusterADCs_d, gpu_timing_t *gpu_timing);

int setSeedStripsNCIndexGPU(int nStrips, uint16_t *stripId_d, uint16_t *adc_d, float *noise_d, int *seedStripsNCIndex_d, gpu_timing_t *gpu_timing);

#ifdef __cplusplus
   }
#endif
#endif