#include <fstream>
#include <iostream>
#if _OPENMP
#include <omp.h>
#endif
#include "cluster.h"
#ifdef USE_GPU
#include <cuda_runtime.h>
#include "clusterGPU.cuh"
#endif

int main()
{
  int max_strips = 600000;
  int nStrips, nSeedStripsNC;
  detId_t *detId;

  uint16_t *stripId, *adc;
  float *noise, *gain;
  bool *bad;
  int *seedStripsNCIndex;

  int *clusterLastIndexLeft, *clusterLastIndexRight;
  float *clusterNoiseSquared;
  uint8_t *clusterADCs;
  bool *trueCluster;

  allocateMemAllStrips(max_strips, &detId, &stripId, &adc, &noise, &gain, &bad, &seedStripsNCIndex);

#ifdef USE_GPU
  uint16_t *stripId_d, *adc_d;
  float *noise_d, *gain_d;
  int *seedStripsNCIndex_d;

  int *clusterLastIndexLeft_d, *clusterLastIndexRight_d;
  float *clusterNoiseSquared_d;
  uint8_t *clusterADCs_d;
  bool *trueCluster_d;
  gpu_timing_t *gpu_timing = (gpu_timing_t *)malloc(sizeof(gpu_timing_t));

  allocateMemAllStripsGPU(max_strips, &stripId_d, &adc_d, &noise_d, &gain_d, &seedStripsNCIndex_d);
#endif

  // read in the data
  std::ifstream digidata_in("digidata.bin", std::ofstream::in | std::ios::binary);
  int i=0;
  while (digidata_in.read((char*)&detId[i], sizeof(detId_t)).gcount() == sizeof(detId_t)) {
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
  nStrips=i;

#ifdef USE_GPU
  cpyCPUToGPU(nStrips, stripId_d, stripId, adc_d, adc, noise_d, noise, gain_d, gain, gpu_timing);
#endif

#if _OPENMP
  double t1 = omp_get_wtime();
#endif

#ifdef USE_GPU
  nSeedStripsNC = setSeedStripsNCIndexGPU(nStrips, stripId_d, adc_d, noise_d, seedStripsNCIndex_d,gpu_timing);
  std::cout<<"GPU nStrips="<<nStrips<<"nSeedStripsNC="<<nSeedStripsNC<<std::endl;
#else
  nSeedStripsNC = setSeedStripsNCIndex(nStrips, stripId, adc, noise, seedStripsNCIndex);
  std::cout<<"GPU nStrips="<<nStrips<<"nSeedStripsNC="<<nSeedStripsNC<<std::endl;
#endif

#if _OPENMP
  double t2 = omp_get_wtime();
#endif

#ifdef USE_GPU
  allocateMemNCSeedStripsGPU(nSeedStripsNC, &clusterLastIndexLeft_d, &clusterLastIndexRight_d, &clusterNoiseSquared_d, &clusterADCs_d, &trueCluster_d);
#endif
  allocateMemNCSeedStrips(nSeedStripsNC, &clusterLastIndexLeft, &clusterLastIndexRight, &clusterNoiseSquared, &clusterADCs, &trueCluster);

#if _OPENMP
  double t3 = omp_get_wtime();
#endif

#ifdef USE_GPU
  findClusterGPU(nSeedStripsNC, nStrips, clusterNoiseSquared_d, clusterLastIndexLeft_d, clusterLastIndexRight_d, seedStripsNCIndex_d, stripId_d, adc_d, noise_d, gain_d, trueCluster_d, clusterADCs_d, gpu_timing);
#else
  findCluster(nSeedStripsNC, nStrips, clusterNoiseSquared, clusterLastIndexLeft, clusterLastIndexRight, seedStripsNCIndex, stripId, adc, noise, gain, trueCluster, clusterADCs);

#endif

#if _OPENMP
  double t4 = omp_get_wtime();
  std::cout<<"Time: setStripIndex "<<t2-t1<<" clustering "<<t4-t3<< " total "<<t4-t1<<std::endl;
#endif

#ifdef USE_GPU
  std::cout<<" GPU Memory Transfer Time "<<gpu_timing->memTransferTime<<std::endl;
  std::cout<<" setSeedStrips Kernel Time "<<gpu_timing->setSeedStripsTime<<std::endl;
  std::cout<<" setNCSeedStrips Kernel Time "<<gpu_timing->setNCSeedStripsTime<<std::endl;
  std::cout<<" setStripIndex Kernel Time "<<gpu_timing->setStripIndexTime<<std::endl;
  std::cout<<" findBoundary Kernel Time "<<gpu_timing->findBoundaryTime<<std::endl;
  std::cout<<" checkCluster Kernel Time "<<gpu_timing->checkClusterTime<<std::endl;
#endif

#ifdef USE_GPU
  cpyGPUToCPU(nSeedStripsNC, clusterLastIndexLeft_d, clusterLastIndexLeft, clusterLastIndexRight_d, clusterLastIndexRight, clusterADCs_d, clusterADCs, trueCluster_d, trueCluster);
#endif

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

#ifdef USE_GPU
  free(gpu_timing);
  freeGPUMem(stripId_d, noise_d, seedStripsNCIndex_d, clusterLastIndexLeft_d, clusterNoiseSquared_d, clusterADCs_d, trueCluster_d);
#endif

  freeMem(detId, stripId, adc, noise, gain, bad, seedStripsNCIndex, clusterLastIndexLeft, clusterLastIndexRight, clusterNoiseSquared, clusterADCs, trueCluster);

  return 0;

}
