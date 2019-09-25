#include "cluster.h"
#include <mm_malloc.h>
#include <iostream>
#include <cmath>

void allocateMemAllStrips(int max_strips, detId_t **detId_pt, uint16_t **stripId_pt, uint16_t **adc_pt, float **noise_pt, float **gain_pt, bool **bad_pt, int **seedStripsNCIndex_pt)
{
  *detId_pt = (detId_t *)_mm_malloc(max_strips*sizeof(detId_t), IDEAL_ALIGNMENT);
  *stripId_pt = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  *adc_pt = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  *noise_pt = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  *gain_pt = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  *bad_pt = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);
  *seedStripsNCIndex_pt = (int *)_mm_malloc(max_strips*sizeof(int), IDEAL_ALIGNMENT);
}

void allocateMemNCSeedStrips(int nSeedStripsNC, int **clusterLastIndexLeft_pt, int **clusterLastIndexRight_pt, float **clusterNoiseSquared_pt, uint8_t **clusterADCs_pt, bool **trueCluster_pt)
{
  *clusterLastIndexLeft_pt = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  *clusterLastIndexRight_pt = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  *clusterNoiseSquared_pt = (float *)_mm_malloc(nSeedStripsNC*sizeof(float), IDEAL_ALIGNMENT);
  *clusterADCs_pt = (uint8_t *)_mm_malloc(nSeedStripsNC*256*sizeof(uint8_t), IDEAL_ALIGNMENT);
  *trueCluster_pt = (bool *)_mm_malloc(nSeedStripsNC*sizeof(bool), IDEAL_ALIGNMENT);
}

void freeMem(detId_t *detId, uint16_t *stripId, uint16_t *adc, float* noise, float *gain, bool *bad, int *seedStripsNCIndex, int *clusterLastIndexLeft, int *clusterLastIndexRight, float *clusterNoiseSquared, uint8_t *clusterADCs, bool *trueCluster)
{
  free(detId);
  free(stripId);
  free(adc);
  free(noise);
  free(gain);
  free(bad);
  free(seedStripsNCIndex);

  free(clusterLastIndexLeft);
  free(clusterLastIndexRight);
  free(clusterNoiseSquared);
  free(clusterADCs);
  free(trueCluster);
}

int setSeedStripsNCIndex(int nStrips, uint16_t *stripId, uint16_t *adc, float *noise, int *seedStripsNCIndex, cpu_timing_t *cpu_timing) {
  int nSeedStripsNC, j;
  int nStripsP2 = std::pow(2, std::floor(std::log2(nStrips))+1);
  int depth = std::log2(nStripsP2);

  int *seedStripMask = (int *)_mm_malloc(nStrips*sizeof(int), IDEAL_ALIGNMENT);
  int *seedStripNCMask  = (int *)_mm_malloc(nStrips*sizeof(int), IDEAL_ALIGNMENT);
  int *prefixSeedStripNCMask = (int *)_mm_malloc(nStripsP2*sizeof(int), IDEAL_ALIGNMENT);

  float SeedThreshold = 3.0;
  double t0, t1;

#pragma omp parallel
  {
#pragma omp single
    t0 = omp_get_wtime();
    // mark seed strips
#pragma omp for simd aligned(noise,seedStripMask,seedStripNCMask,prefixSeedStripNCMask: CACHELINE_BYTES)
    for (int i=0; i<nStrips; i++) {
      float noise_i = noise[i];
      uint8_t adc_i = static_cast<uint8_t>(adc[i]);
      seedStripMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? 1:0;
      seedStripNCMask[i] = seedStripMask[i];
      prefixSeedStripNCMask[i] = 0;
    }

#pragma omp for
    for (int i=nStrips; i<nStripsP2; i++) {
      prefixSeedStripNCMask[i] = 0;
    }

#pragma omp single
    {
      t1 = omp_get_wtime();
      cpu_timing->setSeedStripsTime = t1 - t0;
    }

    // mark only non-consecutive seed strips (mask out consecutive seed strips)
#pragma omp for simd aligned(seedStripNCMask,prefixSeedStripNCMask: CACHELINE_BYTES)
    for (int i=0; i<nStrips; i++) {
      int mask = seedStripNCMask[i];
      if (i>0&&seedStripMask[i]&&seedStripMask[i-1]&&(stripId[i]-stripId[i-1])==1) mask = 0;
      prefixSeedStripNCMask[i] = mask;
      seedStripNCMask[i] = mask;
    }

#pragma omp single
    {
      t0 = omp_get_wtime();
      cpu_timing->setNCSeedStripsTime = t0 - t1;
    }

    // set index for non-consecutive seed strips
    // parallel prefix sum implementation
    // The up-sweep (reduce) phase of a work-efficient sum scan algorithm
    for (int d=0; d<depth; d++) {
      int stride = std::pow(2, d);
      int stride2 = 2*stride;
#pragma omp for
      for (int k=0; k<nStripsP2; k+=stride2) {
	prefixSeedStripNCMask[k+stride2-1] += prefixSeedStripNCMask[k+stride-1];
      }
    }

    // The down-sweep phase of a work-efficient sum scan algorithm
#pragma omp single
    {
      nSeedStripsNC = prefixSeedStripNCMask[nStripsP2-1];
      prefixSeedStripNCMask[nStripsP2-1] = 0;
    }
    for (int d=depth-1; d>=0; d--) {
      int stride = std::pow(2, d);
      int stride2 = 2*stride;
#pragma omp for
      for (int k=0; k<nStripsP2; k+=stride2){
	int temp = prefixSeedStripNCMask[k+stride-1];
	prefixSeedStripNCMask[k+stride-1] = prefixSeedStripNCMask[k+stride2-1];
	prefixSeedStripNCMask[k+stride2-1] += temp;
      }
    }

#pragma omp for
    for (int i=0; i<nStrips; i++) {
      if (seedStripNCMask[i]) {
	int index = prefixSeedStripNCMask[i];
	seedStripsNCIndex[index] = i;
      }
    }

#pragma omp single
    cpu_timing->setStripIndexTime = omp_get_wtime() - t0;
  }

#ifdef CPU_DEBUG
  for (int i=0; i<nStrips; i++) {
    if (seedStripNCMask[i])
      std::cout<<" i "<<i<<" mask "<<seedStripNCMask[i]<<" prefix "<<prefixSeedStripNCMask[i]<<" index "<<seedStripsNCIndex[i]<<std::endl;
  }
  std::cout<<"nStrips="<<nStrips<<"nSeedStripsNC="<<nSeedStripsNC<<std::endl;
#endif

  free(seedStripMask);
  free(seedStripNCMask);
  free(prefixSeedStripNCMask);

  return nSeedStripsNC;

}

static void findLeftRightBoundary(int nSeedStripsNC, int nStrips, float* clusterNoiseSquared, int *clusterLastIndexLeft, int *clusterLastIndexRight, int *seedStripsNCIndex, uint16_t *stripId, uint16_t *adc, float *noise) {
  uint8_t MaxSequentialHoles = 0;
  float  ChannelThreshold = 2.0;
  // (currently, we assume no bad strip. fix later)
#pragma omp parallel for
  for (int i=0; i<nSeedStripsNC; i++) {
    clusterNoiseSquared[i] = 0.0;
    int index=seedStripsNCIndex[i];
    int indexLeft = index;
    int indexRight = index;
    float noise_i = noise[index];
    clusterNoiseSquared[i] += noise_i*noise_i;

    // find left boundary
    int testIndex=index-1;
    while(index>0&&((stripId[indexLeft]-stripId[testIndex]-1)>=0)&&((stripId[indexLeft]-stripId[testIndex]-1)<=MaxSequentialHoles)){
      float testNoise = noise[testIndex];
      uint8_t testADC = static_cast<uint8_t>(adc[testIndex]);
      if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	--indexLeft;
        clusterNoiseSquared[i] += testNoise*testNoise;
      }
      --testIndex;
    }

    // find right boundary
    testIndex=index+1;
    while(testIndex<nStrips&&((stripId[testIndex]-stripId[indexRight]-1)>=0)&&((stripId[testIndex]-stripId[indexRight]-1)<=MaxSequentialHoles)) {
      float testNoise = noise[testIndex];
      uint8_t testADC = static_cast<uint8_t>(adc[testIndex]);
      if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	++indexRight;
        clusterNoiseSquared[i] += testNoise*testNoise;
      }
      ++testIndex;
    }

    clusterLastIndexLeft[i] = indexLeft;
    clusterLastIndexRight[i] = indexRight;
  }
}

static void checkClusterCondition(int nSeedStripsNC, float* clusterNoiseSquared, int *clusterLastIndexLeft, int *clusterLastIndexRight, uint16_t *adc, float * gain, bool *trueCluster, uint8_t *clusterADCs)
{
  float minGoodCharge = 1620.0, ClusterThresholdSquared = 25.0;
#pragma omp parallel for
  for (int i=0; i<nSeedStripsNC; i++){
    trueCluster[i] = false;
    int left=clusterLastIndexLeft[i];
    int right=clusterLastIndexRight[i];
    int size=right-left+1;
    int adcsum = 0;
    for (int j=0; j<size; j++) {
      adcsum += (int)adc[left+j];
    }
    bool noiseSquaredPass = clusterNoiseSquared[i]*ClusterThresholdSquared <= ((float)(adcsum)*float(adcsum));
    bool chargePerCMPass = (float)(adcsum)/0.047f > minGoodCharge;
    if (noiseSquaredPass&&chargePerCMPass) {
      for (int j=0; j<size; j++){
        uint8_t adc_j = adc[left+j];
        float gain_j = gain[left+j];
        auto charge = int( float(adc_j)/gain_j + 0.5f );
        if (adc_j < 254) adc_j = ( charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
        clusterADCs[j*nSeedStripsNC+i] = adc_j;
      }
      trueCluster[i] = true;
    }
  }
}

void findCluster(int nSeedStripsNC, int nStrips, float* clusterNoiseSquared, int *clusterLastIndexLeft, int *clusterLastIndexRight, int *seedStripsNCIndex, uint16_t *stripId, uint16_t *adc, float *noise, float *gain, bool *trueCluster, uint8_t *clusterADCs, cpu_timing_t* cpu_timing)
{
  double t0 = omp_get_wtime();
  findLeftRightBoundary(nSeedStripsNC, nStrips, clusterNoiseSquared, clusterLastIndexLeft, clusterLastIndexRight, seedStripsNCIndex, stripId, adc, noise);
  double t1 = omp_get_wtime();
  cpu_timing->findBoundaryTime = t1 - t0;

  checkClusterCondition(nSeedStripsNC, clusterNoiseSquared, clusterLastIndexLeft, clusterLastIndexRight, adc, gain, trueCluster, clusterADCs);
  cpu_timing->checkClusterTime = omp_get_wtime() - t1;
}
