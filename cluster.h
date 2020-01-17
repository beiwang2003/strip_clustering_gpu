#ifndef _CLUSTER_
#define _CLUSTER_
#include <cstdlib>
#include <cstdint>
//#ifdef USE_GPU
#include <cuda_runtime_api.h>
//#endif
#if _OPENMP
#include <omp.h>
#endif
#include "SiStripConditions.h"

#define IDEAL_ALIGNMENT 64
#define CACHELINE_BYTES 64
#define MAX_STRIPS 600000
#define MAX_SEEDSTRIPS 150000

//using detId_t = uint32_t;

typedef struct {
  detId_t *detId;
  uint16_t *stripId;
  uint8_t *adc;
  fedId_t *fedId;
  fedCh_t *fedCh;
  int *seedStripsNCIndex, *seedStripsMask, *seedStripsNCMask, *prefixSeedStripsNCMask;
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;
  int nSeedStripsNC;
  int nStrips;
} sst_data_t;

typedef struct {
  float *noise, *gain;
  bool *bad;
} calib_data_t;

typedef struct {
  int *clusterLastIndexLeft, *clusterLastIndexRight;
  uint8_t *clusterADCs;
  bool *trueCluster;
  float *barycenter;
} clust_data_t;

typedef struct {
  float setSeedStripsTime;
  float setNCSeedStripsTime;
  float setStripIndexTime;
  float findBoundaryTime;
  float checkClusterTime;
} cpu_timing_t;

void print_binding_info();

void allocateSSTData(int max_strips, sst_data_t *sst_data, cudaStream_t stream);
void allocateCalibData(int max_strips, calib_data_t *calib_data);
void allocateClustData(int max_seedstrips, clust_data_t *clust_data, cudaStream_t stream);

void freeSSTData(sst_data_t *sst_data);
void freeCalibData(calib_data_t *calib_data_t);
void freeClustData(clust_data_t *clust_data_t);

void setSeedStripsNCIndex(sst_data_t *sst_data, calib_data_t *calib_data, const SiStripConditions *conditions, cpu_timing_t *cpu_timing);

void findCluster(sst_data_t *sst_data, calib_data_t *calib_data, const SiStripConditions *conditions, clust_data_t *clust_data, cpu_timing_t *cpu_timing);

#endif
