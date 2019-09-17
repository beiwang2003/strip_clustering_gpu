#ifndef _CLUSTER_
#define _CLUSTER_
#include <cstdlib>
#include <cstdint>
#if _OPENMP
#include <omp.h>
#endif

#define IDEAL_ALIGNMENT 64
using detId_t = uint32_t;

void allocateMemAllStrips(int max_strips, detId_t **detId_pt, uint16_t **stripId_pt, uint16_t **adc_pt, float **noise_pt, float **gain_pt, bool **bad_pt, int **seedStripsNCIndex_pt);

void allocateMemNCSeedStrips(int nSeedStripsNC, int **clusterLastIndexLeft_pt, int **clusterLastIndexRight_pt, float **clusterNoiseSquared_pt, uint8_t **clusterADCs_pt, bool **trueCluster_pt);

void freeMem(detId_t *detId, uint16_t *stripId, uint16_t *adc, float* noise, float *gain, bool *bad, int *seedStripsNCIndex, int *clusterLastIndexLeft, int *clusterLastIndexRight, float *clusterNoiseSquared, uint8_t *clusterADCs, bool*trueCluster);

int setSeedStripsNCIndex(int nStrips, uint16_t *stripId, uint16_t *adc, float *noise, int *seedStripsNCIndex);

void findCluster(int nSeedStripsNC, int nStrips, float* clusterNoiseSquared, int *clusterLastIndexLeft, int *clusterLastIndexRight, int *seedStripsNCIndex, uint16_t *stripId, uint16_t *adc, float *noise, float *gain, bool *trueCluster, uint8_t *clusterADCs);

#endif
