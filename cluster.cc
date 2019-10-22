#include "cluster.h"
#include <mm_malloc.h>
#include <iostream>
#include <cmath>
#include <cuda_runtime_api.h>

void allocateSSTData(int max_strips, sst_data_t *sst_data){
#ifdef USE_GPU
  cudaHostAlloc((void **)&(sst_data->detId), max_strips*sizeof(detId_t), cudaHostAllocDefault);
  cudaHostAlloc((void **)&(sst_data->stripId), max_strips*sizeof(uint16_t), cudaHostAllocDefault);
  cudaHostAlloc((void **)&(sst_data->adc), max_strips*sizeof(uint16_t), cudaHostAllocDefault);
  cudaHostAlloc((void **)&(sst_data->seedStripsNCIndex), max_strips*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&(sst_data->seedStripsMask), max_strips*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&(sst_data->seedStripsNCMask), max_strips*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&(sst_data->prefixSeedStripsNCMask), max_strips*sizeof(int), cudaHostAllocDefault);
#else
  sst_data->detId = (detId_t *)_mm_malloc(max_strips*sizeof(detId_t), IDEAL_ALIGNMENT);
  sst_data->stripId = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  sst_data->adc = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);

  sst_data->seedStripsNCIndex = (int *)_mm_malloc(max_strips*sizeof(int), IDEAL_ALIGNMENT);
  sst_data->seedStripsMask = (int *)_mm_malloc(max_strips*sizeof(int), IDEAL_ALIGNMENT);
  sst_data->seedStripsNCMask = (int *)_mm_malloc(max_strips*sizeof(int), IDEAL_ALIGNMENT);
  sst_data->prefixSeedStripsNCMask = (int *)_mm_malloc(max_strips*sizeof(int), IDEAL_ALIGNMENT);
#endif
}

void allocateCalibData(int max_strips, calib_data_t *calib_data){
  calib_data->noise = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  calib_data->gain = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  calib_data->bad = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);
}

void allocateClustData(int max_strips, clust_data_t *clust_data){
#ifdef USE_GPU
  cudaHostAlloc((void **)&(clust_data->clusterLastIndexLeft), max_strips*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&(clust_data->clusterLastIndexRight), max_strips*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&(clust_data->clusterADCs), max_strips*256*sizeof(uint8_t), cudaHostAllocDefault);
  cudaHostAlloc((void **)&(clust_data->trueCluster), max_strips*sizeof(bool), cudaHostAllocDefault);
#else
  clust_data->clusterLastIndexLeft = (int *)_mm_malloc(max_strips*sizeof(int), IDEAL_ALIGNMENT);
  clust_data->clusterLastIndexRight = (int *)_mm_malloc(max_strips*sizeof(int), IDEAL_ALIGNMENT);
  clust_data->clusterADCs = (uint8_t *)_mm_malloc(max_strips*256*sizeof(uint8_t), IDEAL_ALIGNMENT);
  clust_data->trueCluster = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);
#endif
}

void freeSSTData(sst_data_t *sst_data) {
#ifdef USE_GPU
  cudaFreeHost(sst_data->detId);
  cudaFreeHost(sst_data->stripId);
  cudaFreeHost(sst_data->adc);
  cudaFreeHost(sst_data->seedStripsNCIndex);
  cudaFreeHost(sst_data->seedStripsMask);
  cudaFreeHost(sst_data->seedStripsNCMask);
  cudaFreeHost(sst_data->prefixSeedStripsNCMask);
#else
  free(sst_data->detId);
  free(sst_data->stripId);
  free(sst_data->adc);
  free(sst_data->seedStripsNCIndex);
  free(sst_data->seedStripsMask);
  free(sst_data->seedStripsNCMask);
  free(sst_data->prefixSeedStripsNCMask);
#endif
}

void freeCalibData(calib_data_t *calib_data) {
  free(calib_data->noise);
  free(calib_data->gain);
  free(calib_data->bad);
}

void freeClustData(clust_data_t *clust_data) {
#ifdef USE_GPU
  cudaFreeHost(clust_data->clusterLastIndexLeft);
  cudaFreeHost(clust_data->clusterLastIndexRight);
  cudaFreeHost(clust_data->clusterADCs);
  cudaFreeHost(clust_data->trueCluster);
#else
  free(clust_data->clusterLastIndexLeft);
  free(clust_data->clusterLastIndexRight);
  free(clust_data->clusterADCs);
  free(clust_data->trueCluster);
#endif
}

void setSeedStripsNCIndex(sst_data_t *sst_data, calib_data_t *calib_data, cpu_timing_t *cpu_timing) {
  const detId_t *__restrict__ detId = sst_data->detId;
  const uint16_t *__restrict__ stripId = sst_data->stripId;
  const uint16_t *__restrict__ adc = sst_data->adc;
  const float *__restrict__ noise = calib_data->noise;
  const int nStrips = sst_data->nStrips;
  int *__restrict__ seedStripsNCIndex = sst_data->seedStripsNCIndex;
  int *__restrict__ seedStripsMask = sst_data->seedStripsMask;
  int *__restrict__ seedStripsNCMask = sst_data->seedStripsNCMask;
  int *__restrict__ prefixSeedStripsNCMask = sst_data->prefixSeedStripsNCMask;

  int j;
  int nStripsP2 = std::pow(2, std::floor(std::log2(nStrips))+1);  int depth = std::log2(nStripsP2);

  float SeedThreshold = 3.0;
  double t0, t1;

#pragma omp parallel
  {
#pragma omp single
    t0 = omp_get_wtime();
    // mark seed strips
#pragma omp for simd aligned(noise,seedStripsMask,seedStripsNCMask,prefixSeedStripsNCMask: CACHELINE_BYTES)
    for (int i=0; i<nStrips; i++) {
      float noise_i = noise[i];
      uint8_t adc_i = static_cast<uint8_t>(adc[i]);
      seedStripsMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? 1:0;
      seedStripsNCMask[i] = seedStripsMask[i];
      prefixSeedStripsNCMask[i] = 0;
    }

#pragma omp for
    for (int i=nStrips; i<nStripsP2; i++) {
      prefixSeedStripsNCMask[i] = 0;
    }

#pragma omp single
    {
      t1 = omp_get_wtime();
      cpu_timing->setSeedStripsTime = t1 - t0;
    }

    // mark only non-consecutive seed strips (mask out consecutive seed strips)
#pragma omp for simd aligned(seedStripsNCMask,prefixSeedStripsNCMask: CACHELINE_BYTES)
    for (int i=0; i<nStrips; i++) {
      int mask = seedStripsNCMask[i];
      if (i>0&&seedStripsMask[i]&&seedStripsMask[i-1]&&(stripId[i]-stripId[i-1])==1&&(detId[i]==detId[i-1])) mask = 0;
      prefixSeedStripsNCMask[i] = mask;
      seedStripsNCMask[i] = mask;
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
	prefixSeedStripsNCMask[k+stride2-1] += prefixSeedStripsNCMask[k+stride-1];
      }
    }

    // The down-sweep phase of a work-efficient sum scan algorithm
#pragma omp single
    {
      sst_data->nSeedStripsNC = prefixSeedStripsNCMask[nStripsP2-1];
      prefixSeedStripsNCMask[nStripsP2-1] = 0;
    }
    for (int d=depth-1; d>=0; d--) {
      int stride = std::pow(2, d);
      int stride2 = 2*stride;
#pragma omp for
      for (int k=0; k<nStripsP2; k+=stride2){
	int temp = prefixSeedStripsNCMask[k+stride-1];
	prefixSeedStripsNCMask[k+stride-1] = prefixSeedStripsNCMask[k+stride2-1];
	prefixSeedStripsNCMask[k+stride2-1] += temp;
      }
    }

#pragma omp for
    for (int i=0; i<nStrips; i++) {
      if (seedStripsNCMask[i]) {
	int index = prefixSeedStripsNCMask[i];
	seedStripsNCIndex[index] = i;
      }
    }

#pragma omp single
    cpu_timing->setStripIndexTime = omp_get_wtime() - t0;
  }

#ifdef CPU_DEBUG
  for (int i=0; i<nStrips; i++) {
    if (seedStripNCMask[i])
      std::cout<<" i "<<i<<" mask "<<seedStripsNCMask[i]<<" prefix "<<prefixSeedStripsNCMask[i]<<" index "<<seedStripsNCIndex[i]<<std::endl;
  }
  std::cout<<"nStrips="<<nStrips<<"nSeedStripsNC="<<sst_data->nSeedStripsNC<<std::endl;
#endif

}

static void findLeftRightBoundary(int offset, sst_data_t *sst_data, calib_data_t *calib_data, clust_data_t *clust_data) {
  const int *__restrict__ seedStripsNCIndex = sst_data->seedStripsNCIndex;
  const detId_t *__restrict__ detId = sst_data->detId;
  const uint16_t *__restrict__ stripId = sst_data->stripId;
  const uint16_t *__restrict__ adc = sst_data->adc;
  const int nSeedStripsNC = sst_data->nSeedStripsNC;
  const float *__restrict__ noise = calib_data->noise;
  const int nStrips=sst_data->nStrips;
  int *__restrict__ clusterLastIndexLeft = clust_data->clusterLastIndexLeft+offset;
  int *__restrict__ clusterLastIndexRight = clust_data->clusterLastIndexRight+offset;
  bool *__restrict__ trueCluster = clust_data->trueCluster+offset;

  uint8_t MaxSequentialHoles = 0;
  float  ChannelThreshold = 2.0;
  const float ClusterThresholdSquared = 25.0;

  // (currently, we assume no bad strip. fix later)
#pragma omp parallel for
  for (int i=0; i<nSeedStripsNC; i++) {
    int index=seedStripsNCIndex[i];
    int indexLeft = index;
    int indexRight = index;
    float noise_i = noise[index];
    float noiseSquared_i = noise_i*noise_i;
    float adcSum_i = static_cast<float>(adc[index]);

    // find left boundary
    int testIndexLeft=index-1;
    if (testIndexLeft>=0) {
      int rangeLeft = stripId[indexLeft]-stripId[testIndexLeft]-1;
      bool sameDetLeft = detId[index] == detId[testIndexLeft];
      while(sameDetLeft&&testIndexLeft>=0&&rangeLeft>=0&&rangeLeft<=MaxSequentialHoles) {
	float testNoise = noise[testIndexLeft];
	uint8_t testADC = static_cast<uint8_t>(adc[testIndexLeft]);

	if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	  --indexLeft;
	  noiseSquared_i += testNoise*testNoise;
	  adcSum_i += static_cast<float>(testADC);
	}
	--testIndexLeft;
	if (testIndexLeft>=0) {
	  rangeLeft =stripId[indexLeft]-stripId[testIndexLeft]-1;
	  sameDetLeft = detId[index] == detId[testIndexLeft];
	}
      }
    }

    // find right boundary
    int testIndexRight=index+1;
    if (testIndexRight<nStrips) {
      int rangeRight = stripId[testIndexRight]-stripId[indexRight]-1;
      bool sameDetRight = detId[index] == detId[testIndexRight];
      while(sameDetRight&&testIndexRight<nStrips&&rangeRight>=0&&rangeRight<=MaxSequentialHoles) {
	float testNoise = noise[testIndexRight];
	uint8_t testADC = static_cast<uint8_t>(adc[testIndexRight]);
	if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	  ++indexRight;
	  noiseSquared_i += testNoise*testNoise;
	  adcSum_i += static_cast<float>(testADC);
	}
	++testIndexRight;
	if (testIndexRight<nStrips) {
	  rangeRight = stripId[testIndexRight]-stripId[indexRight]-1;
	  sameDetRight = detId[index] == detId[testIndexRight];
	}
      }
    }

    bool noiseSquaredPass = noiseSquared_i*ClusterThresholdSquared <= adcSum_i*adcSum_i;
    trueCluster[i] = noiseSquaredPass;
    clusterLastIndexLeft[i] = indexLeft;
    clusterLastIndexRight[i] = indexRight;
  }
}

static void checkClusterCondition(int offset, sst_data_t *sst_data, calib_data_t *calib_data, clust_data_t *clust_data) {
  const int *__restrict__ clusterLastIndexLeft = clust_data->clusterLastIndexLeft+offset;
  const int *__restrict__ clusterLastIndexRight = clust_data->clusterLastIndexRight+offset;
  const uint16_t *__restrict__ adc = sst_data->adc;
  const int nSeedStripsNC = sst_data->nSeedStripsNC;
  const float *__restrict__ gain = calib_data->gain;
  bool *__restrict__ trueCluster = clust_data->trueCluster+offset;
  uint8_t *__restrict__ clusterADCs = clust_data->clusterADCs+offset*256;
  const float minGoodCharge = 1620.0;

#pragma omp parallel for
  for (int i=0; i<nSeedStripsNC; i++){
    if (trueCluster[i]) {

      int left=clusterLastIndexLeft[i];
      int right=clusterLastIndexRight[i];
      int size=right-left+1;
      float adcSum=0.0f;

      if (i>0&&clusterLastIndexLeft[i-1]==left) {
        trueCluster[i] = 0;  // ignore duplicates
      } else {
        for (int j=0; j<size; j++) {
          uint8_t adc_j = adc[left+j];
          float gain_j = gain[left+j];
          auto charge = int( float(adc_j)/gain_j + 0.5f );
          if (adc_j < 254) adc_j = ( charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
          clusterADCs[j*nSeedStripsNC+i] = adc_j;
          adcSum += static_cast<float>(adc_j);
        }
      }
      trueCluster[i] = adcSum/0.047f > minGoodCharge;
    }
  }

}

void findCluster(int event, int nStreams, int max_strips, sst_data_t *sst_data, calib_data_t *calib_data, clust_data_t *clust_data, cpu_timing_t *cpu_timing){

  int offset = event *(max_strips/nStreams);
  double t0 = omp_get_wtime();
  findLeftRightBoundary(offset, sst_data, calib_data, clust_data);
  double t1 = omp_get_wtime();
  cpu_timing->findBoundaryTime = t1 - t0;

  checkClusterCondition(offset, sst_data, calib_data, clust_data);
  cpu_timing->checkClusterTime = omp_get_wtime() - t1;
}
