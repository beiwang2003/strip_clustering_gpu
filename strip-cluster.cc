#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <mm_malloc.h>
#if _OPENMP
#include <omp.h>
#endif
#define IDEAL_ALIGNMENT 64
using detId_t = uint32_t;
//using fedId_t = uint16_t;
//using fedCh_t = uint8_t;

int main()
{
  int max_strips = 1400000;
  detId_t *detId = (detId_t *)_mm_malloc(max_strips*sizeof(detId_t), IDEAL_ALIGNMENT);
  //  fedId_t *fedId = (fedId_t *)_mm_malloc(max_strips*sizeof(fedId_t), IDEAL_ALIGNMENT);
  //fedCh_t *fedCh = (fedCh_t *)_mm_malloc(max_strips*sizeof(fedCh_t), IDEAL_ALIGNMENT);
  uint16_t *stripId = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  uint16_t *adc = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  float *noise = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  float *gain = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  bool *bad = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);
  bool *seedStripMask = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);
  bool *seedStripNCMask = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);

  // read in the data
  std::ifstream digidata_in("digidata.bin", std::ofstream::in | std::ios::binary);
  int i=0;
  while (digidata_in.read((char*)&detId[i], sizeof(detId_t)).gcount() == sizeof(detId_t)) {
    //digidata_in.read((char*)&fedId[i], sizeof(fedId_t));
    //digidata_in.read((char*)&fedCh[i], sizeof(fedCh_t));
    digidata_in.read((char*)&stripId[i], sizeof(uint16_t));
    digidata_in.read((char*)&adc[i], sizeof(uint16_t));
    digidata_in.read((char*)&noise[i], sizeof(float));
    digidata_in.read((char*)&gain[i], sizeof(float));
    digidata_in.read((char*)&bad[i], sizeof(bool));
    if (bad[i])
      std::cout<<"index "<<i<<" detid "<<detId[i]<<" stripId "<<stripId[i]<<
	" adc "<<adc[i]<<" noise "<<noise[i]<<" gain "<<gain[i]<<" bad "<<bad[i]<<std::endl;
    i++;
  }
  int nStrips=i;

#if _OPENMP
  double start = omp_get_wtime();
#endif
  float ChannelThreshold = 2.0, SeedThreshold = 3.0, ClusterThresholdSquared = 25.0;
  uint8_t MaxSequentialHoles = 0, MaxSequentialBad = 1, MaxAdjacentBad = 0;
  bool RemoveApvShots = true;
  float minGoodCharge = 1620.0;

  for (int i=0; i<nStrips; i++) {
    seedStripMask[i] = false;
    seedStripNCMask[i] = false;
  }
  // find the seed strips
  int nSeedStrips=0;
#pragma omp parallel for reduction(+:nSeedStrips)
  for (int i=0; i<nStrips; i++) {
    float noise_i = noise[i];
    uint8_t adc_i = static_cast<uint8_t>(adc[i]);
    seedStripMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? true:false;
    nSeedStrips += static_cast<int>(seedStripMask[i]);
  }

  int nSeedStripsNC=0;
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

  //  std::cout<<"nStrips "<<nStrips<<"nSeedStrips "<<nSeedStrips<<"nSeedStripsNC "<<nSeedStripsNC<<std::endl;

  int *seedStripsNCIndex = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  int *clusterLastIndexLeft = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  int *clusterLastIndexRight = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  float *clusterNoiseSquared = (float *)_mm_malloc(nSeedStripsNC*sizeof(float), IDEAL_ALIGNMENT);
  uint8_t *clusterADCs = (uint8_t *)_mm_malloc(nSeedStripsNC*256*sizeof(uint8_t), IDEAL_ALIGNMENT);
  bool *trueCluster= (bool *)_mm_malloc(nSeedStripsNC*sizeof(bool), IDEAL_ALIGNMENT);

  int j=0;
  for (int i=0; i<nStrips; i++) {
    if (seedStripNCMask[i] == true) {
      seedStripsNCIndex[j] = i;
      j++;
    }
  }

  if (j!=nSeedStripsNC) {
    std::cout<<"j "<<j<<"nSeedStripsNC "<<nSeedStripsNC<<std::endl;
    exit (1);
  }
  /*
#pragma acc data copyin(clusterNoiseSquared[0:nSeedStripsNC],	\
			seedStripsNCIndex[0:nSeedStripsNC],		\
			clusterLastIndexLeft[0:nSeedStripsNC],		\
			clusterLastIndexRight[0:nSeedStripsNC],		\
			noise[0:nStrips],adc[0:nStrips],stripId[0:nStrips], \
			gain[0:nStrips], bad[0:nStrips], seedStripsMask[0:nStrips], \
			seedStripsNCMask[0:nStrips],			\
			trueCluster[0:nSeedStripsNC],clusterADCs[0:256*nSeedStripsNC])
  */

  // find the left and right bounday of the candidate cluster
  // (currently, we assume no bad strip. fix later)
#pragma omp parallel for simd
#pragma acc parallel loop independent
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
      uint8_t testADC = adc[testIndex];
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
      uint8_t testADC = adc[testIndex];
      if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
        ++clusterLastIndexRight[i];
	clusterNoiseSquared[i] += testNoise*testNoise;
      }
      ++testIndex;
    }
  }

  // check if the candidate cluster is a true cluster
  // if so, do some adjustment for the adc values
#pragma omp parallel for simd
#pragma acc parallel loop independent
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
#if _OPENMP
  double end = omp_get_wtime();

  std::cout<<"clustering time "<<end-start<<std::endl;
#endif

#pragma acc data copyout(trueCluster[0:nSeedStripsNC],	\
			 clusterLastIndexLeft[0:nSeedStripsNC],	\
			 clusterLastIndexRight[0:nSeedStripsNC],	\
			 clusterADCs[0:256*nSeedStripsNC])
  // print out the result
  for (int i=0; i<nSeedStripsNC; i++) {
    if (trueCluster[i]){
      int index = clusterLastIndexLeft[i];
      //std::cout<<"cluster "<<i<<" det Id "<<detId[index]<<" strip "<<stripId[clusterLastIndexLeft[i]]<<" seed strip "<<stripId[seedStripsNCIndex[i]]<<" ADC ";
      std::cout<<" det id "<<detId[index]<<" strip "<<stripId[clusterLastIndexLeft[i]]<< ": ";
      int left=clusterLastIndexLeft[i];
      int right=clusterLastIndexRight[i];
      int size=right-left+1;
      for (int j=0; j<size; j++){
	std::cout<<(int)clusterADCs[j*nSeedStripsNC+i]<<" ";
      }
      std::cout<<std::endl;
    }
  }


  free(detId);
  //free(fedId);
  //free(fedCh);
  free(stripId);
  free(adc);
  free(noise);
  free(gain);
  free(bad);
  free(seedStripMask);
  free(seedStripNCMask);
  free(seedStripsNCIndex);
  free(clusterNoiseSquared);
  free(clusterLastIndexLeft);
  free(clusterLastIndexRight);
  free(clusterADCs);
  free(trueCluster);

  return 0;

}
