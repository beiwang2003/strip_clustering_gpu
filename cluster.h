#ifndef _CLUSTER_
#define _CLUSTER_
#include <cstdlib>
#include <cstdint>
#if _OPENMP
#include <omp.h>
#endif

#define IDEAL_ALIGNMENT 64
#define CACHELINE_BYTES 64

using detId_t = uint32_t;

typedef struct {
  detId_t *detId;
  uint16_t *stripId, *adc;
  int *seedStripsNCIndex, *seedStripsMask, *seedStripsNCMask, *prefixSeedStripsNCMask;
} sst_data_t;

typedef struct {
  float *noise, *gain;
  bool *bad;
} calib_data_t;

typedef struct {
  int *clusterLastIndexLeft, *clusterLastIndexRight;
  uint8_t *clusterADCs;
  bool *trueCluster;
} clust_data_t;

typedef struct {
  float setSeedStripsTime;
  float setNCSeedStripsTime;
  float setStripIndexTime;
  float findBoundaryTime;
  float checkClusterTime;
} cpu_timing_t;

void allocateSSTData(int max_strips, sst_data_t *sst_data);
void allocateCalibData(int max_strips, calib_data_t *calib_data);
void allocateClustData(int nSeedStripsNC, clust_data_t *clust_data);

void freeMem(sst_data_t *sst_data, calib_data_t *calib_data, clust_data_t *clust_data);

int setSeedStripsNCIndex(int nStrips, sst_data_t *sst_data, calib_data_t *calib_data, cpu_timing_t *cpu_timing);

void findCluster(int nSeedStripsNC, int nStrips, sst_data_t *sst_data, calib_data_t *calib_data, clust_data_t *clust_data, cpu_timing_t *cpu_timing);

#endif
