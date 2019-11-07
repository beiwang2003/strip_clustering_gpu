#include <fstream>
#include <iostream>
#include <cstring>
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
  const int max_seedstrips = 150000;
  const int nStreams = 10;
  cudaStream_t stream[nStreams];
  sst_data_t *sst_data[nStreams];
  clust_data_t *clust_data[nStreams];
  cpu_timing_t *cpu_timing[nStreams];
  for (int i=0; i<nStreams; i++) {
    cudaStreamCreate(&stream[i]);
    sst_data[i] = (sst_data_t *)malloc(sizeof(sst_data_t));
    clust_data[i] = (clust_data_t *)malloc(sizeof(clust_data_t));
    cpu_timing[i] = (cpu_timing_t *)malloc(sizeof(cpu_timing_t));
  }
  calib_data_t *calib_data = (calib_data_t *)malloc(sizeof(calib_data_t));

  // memory allocation
  for (int i=0; i<nStreams; i++) {
    allocateSSTData(max_strips, sst_data[i], stream[i]);
    allocateClustData(max_seedstrips, clust_data[i], stream[i]);
  }
  allocateCalibData(max_strips, calib_data);

  // read in the data
  std::ifstream digidata_in("digidata.bin", std::ofstream::in | std::ios::binary);
  int i=0;
  while (digidata_in.read((char*)&(sst_data[0]->detId[i]), sizeof(detId_t)).gcount() == sizeof(detId_t)) {
    digidata_in.read((char*)&(sst_data[0]->stripId[i]), sizeof(uint16_t));
    digidata_in.read((char*)&(sst_data[0]->adc[i]), sizeof(uint16_t));
    digidata_in.read((char*)&(calib_data->noise[i]), sizeof(float));
    digidata_in.read((char*)&(calib_data->gain[i]), sizeof(float));
    digidata_in.read((char*)&(calib_data->bad[i]), sizeof(bool));
    if (calib_data->bad[i])
      std::cout<<"index "<<i<<" detid "<<sst_data[0]->detId[i]<<" stripId "<<sst_data[0]->stripId[i]<<
	" adc "<<sst_data[0]->adc[i]<<" noise "<<calib_data->noise[i]<<" gain "<<calib_data->gain[i]<<" bad "<<calib_data->bad[i]<<std::endl;
    i++;
  }
  sst_data[0]->nStrips=i;

  // copy data to other streams
  for (int i=1; i<nStreams; i++) {
    std::memcpy(sst_data[i]->detId, sst_data[0]->detId, sizeof(detId_t)*sst_data[0]->nStrips);
    std::memcpy(sst_data[i]->stripId, sst_data[0]->stripId, sizeof(uint16_t)*sst_data[0]->nStrips);
    std::memcpy(sst_data[i]->adc, sst_data[0]->adc, sizeof(uint16_t)*sst_data[0]->nStrips);
    sst_data[i]->nStrips = sst_data[0]->nStrips;
  }

#ifdef USE_GPU
  sst_data_t *sst_data_d[nStreams], *pt_sst_data_d[nStreams];
  calib_data_t *calib_data_d, *pt_calib_data_d;
  clust_data_t *clust_data_d[nStreams], *pt_clust_data_d[nStreams];
  for (int i=0; i<nStreams; i++) {
    sst_data_d[i] = (sst_data_t *)malloc(sizeof(sst_data_t));
    sst_data_d[i]->nStrips = sst_data[i]->nStrips;
    clust_data_d[i] = (clust_data_t *)malloc(sizeof(clust_data_t));
  }
  calib_data_d = (calib_data_t *)malloc(sizeof(calib_data_t));

  gpu_timing_t *gpu_timing[nStreams];
  for (int i=0; i<nStreams; i++) {
    gpu_timing[i] = (gpu_timing_t *)malloc(sizeof(gpu_timing_t));
    gpu_timing[i]->memTransDHTime = 0.0;
    gpu_timing[i]->memTransHDTime = 0.0;
    gpu_timing[i]->memAllocTime = 0.0;
  }
  int gpu_device = 0;
  cudaSetDevice(gpu_device);
  cudaGetDevice(&gpu_device);
#endif

  double t0 = omp_get_wtime();

#if USE_GPU
  allocateCalibDataGPU(max_strips, calib_data_d, &pt_calib_data_d, gpu_timing[0], gpu_device, stream[0]);
  cpyCalibDataToGPU(max_strips, calib_data, calib_data_d, gpu_timing[0]);
#pragma omp parallel for num_threads(nStreams)
  for (int i=0; i<nStreams; i++) {

    allocateSSTDataGPU(max_strips, sst_data_d[i], &pt_sst_data_d[i], gpu_timing[i], gpu_device, stream[i]);

    cpySSTDataToGPU(sst_data[i], sst_data_d[i], gpu_timing[i], stream[i]);

    setSeedStripsNCIndexGPU(sst_data_d[i], pt_sst_data_d[i], calib_data_d, pt_calib_data_d, gpu_timing[i], stream[i]);
    allocateClustDataGPU(max_seedstrips, clust_data_d[i], &pt_clust_data_d[i], gpu_timing[i], gpu_device, stream[i]);

    findClusterGPU(sst_data_d[i], pt_sst_data_d[i], calib_data_d, pt_calib_data_d, clust_data_d[i], pt_clust_data_d[i], gpu_timing[i], stream[i]);

    cpyGPUToCPU(sst_data_d[i], pt_sst_data_d[i], clust_data[i], clust_data_d[i], gpu_timing[i], stream[i]);
  }
  cudaDeviceSynchronize();
#else
  omp_set_nested(true);
#pragma omp parallel for num_threads(5)
  for (int i=0; i<nStreams; i++) {

    setSeedStripsNCIndex(sst_data[i], calib_data, cpu_timing[i]);

    findCluster(sst_data[i], calib_data, clust_data[i], cpu_timing[i]);
  }
#endif

  double t1 = omp_get_wtime();

#ifdef OUTPUT
  // print out the result
  for (i=0; i<nStreams; i++) {
#ifdef USE_GPU
    sst_data[i]->nSeedStripsNC = sst_data_d[i]->nSeedStripsNC;
#endif
    std::cout<<" Event "<<i<<" nSeedStripsNC "<<sst_data[i]->nSeedStripsNC<<std::endl;
    for (int j=0; j<sst_data[i]->nSeedStripsNC; j++) {
      if (clust_data[i]->trueCluster[j]){
	int index = clust_data[i]->clusterLastIndexLeft[j];
	std::cout<<" det id "<<sst_data[i]->detId[index]<<" strip "<<sst_data[i]->stripId[index]<< ": ";
	int right=clust_data[i]->clusterLastIndexRight[j];
	int size=right-index+1;
	for (int k=0; k<size; k++){
	  std::cout<<(int)clust_data[i]->clusterADCs[k*sst_data[i]->nSeedStripsNC+j]<<" ";
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
  std::cout<<" setSeedStrips function Time "<<cpu_timing[0]->setSeedStripsTime<<std::endl;
  std::cout<<" setNCSeedStrips function Time "<<cpu_timing[0]->setNCSeedStripsTime<<std::endl;
  std::cout<<" setStripIndex function Time "<<cpu_timing[0]->setStripIndexTime<<std::endl;
  std::cout<<" findBoundary function Time "<<cpu_timing[0]->findBoundaryTime<<std::endl;
  std::cout<<" checkCluster function Time "<<cpu_timing[0]->checkClusterTime<<std::endl;
  std::cout<<" Total Time "<<t1-t0<<std::endl;
  std::cout<<"nested? "<<omp_get_nested()<<std::endl;
#endif

#ifdef USE_GPU
  for (int i=0; i<nStreams; i++) {
    freeSSTDataGPU(sst_data_d[i], pt_sst_data_d[i], gpu_device);
    freeClustDataGPU(clust_data_d[i], pt_clust_data_d[i], gpu_device);
    free(sst_data_d[i]);
    free(clust_data_d[i]);
    free(gpu_timing[i]);
  }
  freeCalibDataGPU(calib_data_d, pt_calib_data_d, gpu_device);
  free(calib_data_d);
#endif

  for (int i=0; i<nStreams; i++) {
    freeSSTData(sst_data[i]);
    free(sst_data[i]);
    freeClustData(clust_data[i]);
    free(clust_data[i]);
    free(cpu_timing[i]);
    cudaStreamDestroy(stream[i]);
  }
  freeCalibData(calib_data);
  free(calib_data);

  return 0;

}
