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

  sst_data_t * sst_data = (sst_data_t *)malloc(sizeof(sst_data_t));
  calib_data_t *calib_data = (calib_data_t *)malloc(sizeof(calib_data_t));
  clust_data_t *clust_data = (clust_data_t *)malloc(sizeof(clust_data_t));
  cpu_timing_t *cpu_timing = (cpu_timing_t *)malloc(sizeof(cpu_timing_t));

  allocateSSTData(max_strips, sst_data);
  allocateCalibData(max_strips, calib_data);

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
  nStrips=i;

#ifdef USE_GPU
  sst_data_t *sst_data_d, *pt_sst_data_d;
  calib_data_t *calib_data_d, *pt_calib_data_d;
  clust_data_t *clust_data_d, *pt_clust_data_d;
  sst_data_d = (sst_data_t *)malloc(sizeof(sst_data_t));
  calib_data_d = (calib_data_t *)malloc(sizeof(calib_data_t));
  clust_data_d = (clust_data_t *)malloc(sizeof(clust_data_t));
  gpu_timing_t *gpu_timing = (gpu_timing_t *)malloc(sizeof(gpu_timing_t));

  allocateSSTDataGPU(nStrips, sst_data_d, &pt_sst_data_d);
  allocateCalibDataGPU(nStrips, calib_data_d, &pt_calib_data_d);
  cpyCPUToGPU(nStrips, sst_data, sst_data_d, calib_data, calib_data_d, gpu_timing);
#endif

  double t0 = omp_get_wtime();
#ifdef USE_GPU
  nSeedStripsNC = setSeedStripsNCIndexGPU(nStrips, sst_data_d, pt_sst_data_d, calib_data_d, pt_calib_data_d, gpu_timing);
  std::cout<<"GPU nStrips="<<nStrips<<"nSeedStripsNC="<<nSeedStripsNC<<std::endl;
#else
  nSeedStripsNC = setSeedStripsNCIndex(nStrips, pt_sst_data, pt_calib_data, cpu_timing);
  std::cout<<"CPU nStrips="<<nStrips<<"nSeedStripsNC="<<nSeedStripsNC<<std::endl;
#endif
  double t1 = omp_get_wtime();

#ifdef USE_GPU
  allocateClustDataGPU(nSeedStripsNC, clust_data_d, &pt_clust_data_d);
#endif
  allocateClustData(nSeedStripsNC, clust_data);

  double t2 = omp_get_wtime();
#ifdef USE_GPU
  findClusterGPU(nSeedStripsNC, nStrips, sst_data_d, pt_sst_data_d, calib_data_d, pt_calib_data_d, clust_data_d, pt_clust_data_d, gpu_timing);
#else
  findCluster(nSeedStripsNC, nStrips, pt_sst_data, pt_calib_data, pt_clust_data, cpu_timing);
#endif
  double t3 = omp_get_wtime();

#ifdef USE_GPU
  std::cout<<" GPU Memory Transfer Time "<<gpu_timing->memTransferTime<<std::endl;
  std::cout<<" setStripsNCIndexGPU function Time "<<t1-t0<<std::endl;
  std::cout<<" --setSeedStrips kernel Time "<<gpu_timing->setSeedStripsTime<<std::endl;
  std::cout<<" --setNCSeedStrips kernel Time "<<gpu_timing->setNCSeedStripsTime<<std::endl;
  std::cout<<" --setStripIndex kernel Time "<<gpu_timing->setStripIndexTime<<std::endl;
  std::cout<<" findClusterGPU function Time "<<t3-t2<<std::endl;
  std::cout<<" --findBoundary GPU Kernel Time "<<gpu_timing->findBoundaryTime<<std::endl;
  std::cout<<" --checkCluster GPU Kernel Time "<<gpu_timing->checkClusterTime<<std::endl;
  std::cout<<" total Time (including HtoD data transfer) "<<t1-t0+t3-t2+gpu_timing->memTransferTime<<std::endl;
#else
  std::cout<<" setStripsNCIndex function Time "<<t1-t0<<std::endl;
  std::cout<<" --setSeedStrips Time "<<cpu_timing->setSeedStripsTime<<std::endl;
  std::cout<<" --setNCSeedStrips Time "<<cpu_timing->setNCSeedStripsTime<<std::endl;
  std::cout<<" --setStripIndex Time "<<cpu_timing->setStripIndexTime<<std::endl;
  std::cout<<" findClusterGPU function Time "<<t3-t2<<std::endl;
  std::cout<<" --findBoundary Time "<<cpu_timing->findBoundaryTime<<std::endl;
  std::cout<<" --checkCluster Time "<<cpu_timing->checkClusterTime<<std::endl;
  std::cout<<" total Time "<<t1-t0+t3-t2<<std::endl;
#endif

#ifdef OUTPUT
#ifdef USE_GPU
  cpyGPUToCPU(nSeedStripsNC, clust_data, clust_data_d);
#endif
  // print out the result
  for (int i=0; i<nSeedStripsNC; i++) {
    if (clust_data->trueCluster[i]){
      int index = clust_data->clusterLastIndexLeft[i];
      std::cout<<" det id "<<sst_data->detId[index]<<" strip "<<sst_data->stripId[index]<< ": ";
      int right=clust_data->clusterLastIndexRight[i];
      int size=right-index+1;
      for (int j=0; j<size; j++){
	std::cout<<(int)clust_data->clusterADCs[j*nSeedStripsNC+i]<<" ";
      }
      std::cout<<std::endl;
    }
  }
#endif

#ifdef USE_GPU
  free(sst_data_d);
  free(calib_data_d);
  free(clust_data_d);
  cudaFree(gpu_timing);
  freeGPUMem(sst_data_d, pt_sst_data_d, calib_data_d, pt_calib_data_d, clust_data_d, pt_clust_data_d);
#endif

  free(sst_data);
  free(calib_data);
  free(clust_data);
  free(cpu_timing);
  freeMem(sst_data, calib_data, clust_data);

  return 0;

}
