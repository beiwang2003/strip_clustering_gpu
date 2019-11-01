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
  const int max_strips = 600000;
  const int nStreams = 4;
  sst_data_t *sst_data = (sst_data_t *)malloc(sizeof(sst_data_t));
  calib_data_t *calib_data = (calib_data_t *)malloc(sizeof(calib_data_t));
  clust_data_t *clust_data = (clust_data_t *)malloc(sizeof(clust_data_t));
  cpu_timing_t *cpu_timing = (cpu_timing_t *)malloc(sizeof(cpu_timing_t));

  allocateSSTData(max_strips, sst_data);
  allocateCalibData(max_strips, calib_data);
  allocateClustData(max_strips, clust_data);

  // read in the data
  std::ifstream digidata_in("digidata.bin", std::ofstream::in | std::ios::binary);
  int i=0;
  while (digidata_in.read((char*)&(sst_data->detId[i]), sizeof(detId_t)).gcount() == sizeof(detId_t)) {
    digidata_in.read((char*)&(sst_data->stripId[i]), sizeof(uint16_t));
    digidata_in.read((char*)&(sst_data->adc[i]), sizeof(uint16_t));
    digidata_in.read((char*)&(calib_data->noise[i]), sizeof(float));
    digidata_in.read((char*)&(calib_data->gain[i]), sizeof(float));
    digidata_in.read((char*)&(calib_data->bad[i]), sizeof(bool));
    if (calib_data->bad[i])
      std::cout<<"index "<<i<<" detid "<<sst_data->detId[i]<<" stripId "<<sst_data->stripId[i]<<
	" adc "<<sst_data->adc[i]<<" noise "<<calib_data->noise[i]<<" gain "<<calib_data->gain[i]<<" bad "<<calib_data->bad[i]<<std::endl;
    i++;
  }
  sst_data->nStrips=i;

#ifdef USE_GPU
  sst_data_t *sst_data_d[nStreams], *pt_sst_data_d[nStreams];
  calib_data_t *calib_data_d, *pt_calib_data_d;
  clust_data_t *clust_data_d, *pt_clust_data_d;
  for (int i=0; i<nStreams; i++) {
    sst_data_d[i] = (sst_data_t *)malloc(sizeof(sst_data_t));
    sst_data_d[i]->nStrips = sst_data->nStrips;
  }
  calib_data_d = (calib_data_t *)malloc(sizeof(calib_data_t));
  clust_data_d = (clust_data_t *)malloc(sizeof(clust_data_t));

  gpu_timing_t *gpu_timing[nStreams];
  for (int i=0; i<nStreams; i++) {
    gpu_timing[i] = (gpu_timing_t *)malloc(sizeof(gpu_timing_t));
    gpu_timing[i]->memTransDHTime = 0.0;
    gpu_timing[i]->memTransHDTime = 0.0;
    gpu_timing[i]->memAllocTime = 0.0;
  }
  cudaStream_t stream[nStreams];
#endif

  double t0 = omp_get_wtime();

#if USE_GPU
  allocateClustDataGPU(max_strips, clust_data_d, &pt_clust_data_d, gpu_timing[0]);
  allocateCalibDataGPU(max_strips, calib_data_d, &pt_calib_data_d, gpu_timing[0]);
  cpyCalibDataToGPU(max_strips, calib_data, calib_data_d, gpu_timing[0]);
#pragma omp parallel for num_threads(nStreams)
  for (int i=0; i<nStreams; i++) {
    cudaStreamCreate(&stream[i]);

    allocateSSTDataGPU(max_strips, sst_data_d[i], &pt_sst_data_d[i], gpu_timing[i], stream[i]);

    cpySSTDataToGPU(sst_data, sst_data_d[i], gpu_timing[i], stream[i]);

    setSeedStripsNCIndexGPU(sst_data_d[i], pt_sst_data_d[i], calib_data_d, pt_calib_data_d, gpu_timing[i], stream[i]);

    findClusterGPU(i, nStreams, max_strips, sst_data_d[i], pt_sst_data_d[i], calib_data_d, pt_calib_data_d, clust_data_d, pt_clust_data_d, gpu_timing[i], stream[i]);

    cpyGPUToCPU(i, nStreams, max_strips, sst_data_d[i], pt_sst_data_d[i], clust_data, clust_data_d, gpu_timing[i], stream[i]);
  }
  cudaDeviceSynchronize();
#else
  for (int i=0; i<nStreams; i++) {
    setSeedStripsNCIndex(sst_data, calib_data, cpu_timing);

    findCluster(i, nStreams, max_strips, sst_data, calib_data, clust_data, cpu_timing);
  }
#endif

  double t1 = omp_get_wtime();

#ifdef OUTPUT
  // print out the result
  for (i=0; i<nStreams; i++) {
#ifdef USE_GPU
    //cpyGPUToCPU(i, nStreams, max_strips, sst_data_d[i], pt_sst_data_d[i], clust_data, clust_data_d, stream[i]);
    sst_data->nSeedStripsNC = sst_data_d[i]->nSeedStripsNC;
#endif
    std::cout<<" Event "<<i<<" nSeedStripsNC "<<sst_data->nSeedStripsNC<<std::endl;
    int offset=i*(max_strips/nStreams);
    for (int j=0; j<sst_data->nSeedStripsNC; j++) {
      if (clust_data->trueCluster[j+offset]){
	int index = clust_data->clusterLastIndexLeft[j+offset];
	std::cout<<" det id "<<sst_data->detId[index]<<" strip "<<sst_data->stripId[index]<< ": ";
	int right=clust_data->clusterLastIndexRight[j+offset];
	int size=right-index+1;
	for (int k=0; k<size; k++){
	  std::cout<<(int)clust_data->clusterADCs[offset*256+k*sst_data->nSeedStripsNC+j]<<" ";
	}
	std::cout<<std::endl;
      }
    }
  }
#endif

#ifdef USE_GPU
  std::cout<<" GPU Memory Transfer Host to Device Time "<<gpu_timing[0]->memTransHDTime<<std::endl;
  std::cout<<" GPU Memory Transfer Device to Host Time "<<gpu_timing[0]->memTransDHTime<<std::endl;
  std::cout<<" GPU Memory Allocation Time "<<gpu_timing[0]->memAllocTime<<std::endl;
  std::cout<<" GPU Kernel Time "<<std::endl;
  std::cout<<" --setSeedStrips kernel Time "<<gpu_timing[0]->setSeedStripsTime<<std::endl;
  std::cout<<" --setNCSeedStrips kernel Time "<<gpu_timing[0]->setNCSeedStripsTime<<std::endl;
  std::cout<<" --setStripIndex kernel Time "<<gpu_timing[0]->setStripIndexTime<<std::endl;
  std::cout<<" --findBoundary GPU Kernel Time "<<gpu_timing[0]->findBoundaryTime<<std::endl;
  std::cout<<" --checkCluster GPU Kernel Time "<<gpu_timing[0]->checkClusterTime<<std::endl;
  std::cout<<" Total Time (including data allocation, transfer and kernel cost) "<<t1-t0<<std::endl;
#else
  std::cout<<" setSeedStrips function Time "<<cpu_timing->setSeedStripsTime<<std::endl;
  std::cout<<" setNCSeedStrips function Time "<<cpu_timing->setNCSeedStripsTime<<std::endl;
  std::cout<<" setStripIndex function Time "<<cpu_timing->setStripIndexTime<<std::endl;
  std::cout<<" findBoundary function Time "<<cpu_timing->findBoundaryTime<<std::endl;
  std::cout<<" checkCluster function Time "<<cpu_timing->checkClusterTime<<std::endl;
  std::cout<<" Total Time "<<t1-t0<<std::endl;
#endif

#ifdef USE_GPU
  for (int i=0; i<nStreams; i++) {
    cudaStreamDestroy(stream[i]);
    freeSSTDataGPU(sst_data_d[i], pt_sst_data_d[i]);
    free(sst_data_d[i]);
    free(gpu_timing[i]);
  }
  freeClustDataGPU(clust_data_d, pt_clust_data_d);
  free(clust_data_d);
  freeCalibDataGPU(calib_data_d, pt_calib_data_d);
  free(calib_data_d);
#endif

  freeSSTData(sst_data);
  freeCalibData(calib_data);
  freeClustData(clust_data);
  free(sst_data);
  free(calib_data);
  free(clust_data);
  free(cpu_timing);
  return 0;

}
