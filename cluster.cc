#include "cluster.h"
#include <mm_malloc.h>

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

int setSeedStripsNCIndex(int nStrips, uint16_t *stripId, uint16_t *adc, float *noise, int *seedStripsNCIndex) {
  int nSeedStripsNC, j;

  bool *seedStripMask = (bool *)_mm_malloc(nStrips*sizeof(bool), IDEAL_ALIGNMENT);
  bool *seedStripNCMask  = (bool *)_mm_malloc(nStrips*sizeof(bool), IDEAL_ALIGNMENT);

  float SeedThreshold = 3.0;

  for (int i=0; i<nStrips; i++) {
    seedStripMask[i] = false;
    seedStripNCMask[i] = false;
  }

  // find the seed strips
  //  nSeedStrips=0;
#pragma omp parallel for
  for (int i=0; i<nStrips; i++) {
    float noise_i = noise[i];
    uint8_t adc_i = static_cast<uint8_t>(adc[i]);
    seedStripMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? true:false;
    //nSeedStrips += static_cast<int>(seedStripMask[i]);
    seedStripNCMask[i] = seedStripMask[i];
  }

  /*
  // find NC seed strips
  nSeedStripsNC=0;
  seedStripNCMask[0] = seedStripMask[0];
  if (seedStripNCMask[0]) nSeedStripsNC++;
#pragma omp parallel for reduction(+:nSeedStripsNC)
  for (int i=1; i<nStrips; i++) {
    if (seedStripMask[i] == true) {
      if (stripId[i]-stripId[i-1]!=1||((stripId[i]-stripId[i-1]==1)&&!seedStripMask[i-1])) {
        seedStripNCMask[i] = true;
        nSeedStripsNC += static_cast<int>(seedStripNCMask[i]);
      }
    }
  }
  */
  nSeedStripsNC = static_cast<int>(seedStripNCMask[0]);
#pragma omp parallel for reduction(+:nSeedStripsNC)
  for (int i=1; i<nStrips; i++) {
    if (seedStripMask[i]&&seedStripMask[i-1]&&(stripId[i]-stripId[i-1])==1) seedStripNCMask[i] = false;
    nSeedStripsNC += static_cast<int>(seedStripNCMask[i]);
  }

  j=0;
  for (int i=0; i<nStrips; i++) {
    if (seedStripNCMask[i] == true) {
      seedStripsNCIndex[j] = i;
      j++;
    }
  }

  free(seedStripMask);
  free(seedStripNCMask);

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
    clusterLastIndexLeft[i] = index;
    clusterLastIndexRight[i] = index;
    //uint8_t adc_i = adc[index];
    float noise_i = noise[index];
    clusterNoiseSquared[i] += noise_i*noise_i;
    // find left boundary
    int testIndex=index-1;
    while(index>0&&((stripId[clusterLastIndexLeft[i]]-stripId[testIndex]-1)>=0)&&((stripId[clusterLastIndexLeft[i]]-stripId[testIndex]-1)<=MaxSequentialHoles)){
      float testNoise = noise[testIndex];
      uint8_t testADC = static_cast<uint8_t>(adc[testIndex]);
      if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
        --clusterLastIndexLeft[i];
        clusterNoiseSquared[i] += testNoise*testNoise;
      }
      --testIndex;
    }

    // find right boundary
    testIndex=index+1;
    while(testIndex<nStrips&&((stripId[testIndex]-stripId[clusterLastIndexRight[i]]-1)>=0)&&((stripId[testIndex]-stripId[clusterLastIndexRight[i]]-1)<=MaxSequentialHoles)) {
      float testNoise = noise[testIndex];
      uint8_t testADC = static_cast<uint8_t>(adc[testIndex]);
      if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
        ++clusterLastIndexRight[i];
        clusterNoiseSquared[i] += testNoise*testNoise;
      }
      ++testIndex;
    }
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

void findCluster(int nSeedStripsNC, int nStrips, float* clusterNoiseSquared, int *clusterLastIndexLeft, int *clusterLastIndexRight, int *seedStripsNCIndex, uint16_t *stripId, uint16_t *adc, float *noise, float *gain, bool *trueCluster, uint8_t *clusterADCs)
{

  findLeftRightBoundary(nSeedStripsNC, nStrips, clusterNoiseSquared, clusterLastIndexLeft, clusterLastIndexRight, seedStripsNCIndex, stripId, adc, noise);

  checkClusterCondition(nSeedStripsNC, clusterNoiseSquared, clusterLastIndexLeft, clusterLastIndexRight, adc, gain, trueCluster, clusterADCs);

}
