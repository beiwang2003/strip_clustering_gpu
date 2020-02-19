#include <fstream>
#include <iostream>
#include <cstring>
#include <memory>
#include <functional>
#if _OPENMP
#include <omp.h>
#endif
#include <cuda_runtime.h>
#include "clusterGPU.cuh"
#include "cluster.h"

int main()
{
  const int max_strips = MAX_STRIPS;
  const int max_seedstrips = MAX_SEEDSTRIPS;
  const int nStreams = omp_get_max_threads();
  //  const int nIter = 840/nStreams;
  const int nIter = 1;
  cudaStream_t stream[nStreams];
  sst_data_t *sst_data[nStreams];
  clust_data_t *clust_data[nStreams];
  cpu_timing_t *cpu_timing[nStreams];
  for (int i=0; i<nStreams; i++) {
    CUDA_RT_CALL(cudaStreamCreate(&stream[i]));
    sst_data[i] = (sst_data_t *)malloc(sizeof(sst_data_t));
    clust_data[i] = (clust_data_t *)malloc(sizeof(clust_data_t));
    cpu_timing[i] = (cpu_timing_t *)malloc(sizeof(cpu_timing_t));
  }
  calib_data_t *calib_data = (calib_data_t *)malloc(sizeof(calib_data_t));

  // memory allocation
#ifdef NUMA_FT
#pragma omp parallel for num_threads(nStreams)
#endif
  for (int i=0; i<nStreams; i++) {
    //    print_binding_info();
    allocateSSTData(max_strips, sst_data[i], stream[i]);
    allocateClustData(max_seedstrips, clust_data[i], stream[i]);
  }

#ifdef NUMA_FT
#pragma omp parallel num_threads(nStreams)
  {
#pragma omp single
    {
#endif
      allocateCalibData(max_strips, calib_data);
#ifdef NUMA_FT
    }
  }
#endif

  // read in calibration data (only once)
  std::string condfilename("stripcond.bin");
  auto conditions = std::make_unique<SiStripConditions>(condfilename);

  //#ifdef ACTIVE_STRIPS
  //std::string digifilename("digidata.bin");
  //#else
  //std::string digifilename("digidata_all.bin");
  //#endif
  //readin_raw_digidata(digifilename, conditions.get(), sst_data[0], calib_data);

  /*
  std::string datafilename("stripdata.bin");
  readin_raw_data(datafilename, conditions.get(), sst_data[0], calib_data, stream[0]);

  // copy data to other streams
  for (int i=1; i<nStreams; i++) {
    std::memcpy(sst_data[i]->detId, sst_data[0]->detId, sizeof(detId_t)*sst_data[0]->nStrips);
    std::memcpy(sst_data[i]->stripId, sst_data[0]->stripId, sizeof(uint16_t)*sst_data[0]->nStrips);
    std::memcpy(sst_data[i]->fedId, sst_data[0]->fedId, sizeof(fedId_t)*sst_data[0]->nStrips);
    std::memcpy(sst_data[i]->fedCh, sst_data[0]->fedCh, sizeof(fedCh_t)*sst_data[0]->nStrips);
    std::memcpy(sst_data[i]->adc, sst_data[0]->adc, sizeof(uint8_t)*sst_data[0]->nStrips);
    sst_data[i]->nStrips = sst_data[0]->nStrips;
  }
  */

  // when moving ChannelLocs into a vector and passing to readin_raw_data, it cause SIGFAULT error
  std::vector<ChannelLocs> chanlocsv;
  for (auto i=0; i<nStreams; i++) {
    chanlocsv.emplace_back(conditions->detToFeds().size(), stream[i]);
    //    std::string datafilename("stripdata"+std::to_string(i)+".bin");
    std::string datafilename("stripdata.bin");
    readin_raw_data(datafilename, conditions.get(), chanlocsv[i], sst_data[i], calib_data, stream[i]);
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
  std::unique_ptr<SiStripConditionsGPU, std::function<void(SiStripConditionsGPU*)>> condGPU(conditions->toGPU(), [](SiStripConditionsGPU* p) { cudaFree(p); });

  gpu_timing_t *gpu_timing[nStreams];
  for (int i=0; i<nStreams; i++) {
    gpu_timing[i] = (gpu_timing_t *)malloc(sizeof(gpu_timing_t));
    gpu_timing[i]->memTransDHTime = 0.0;
    gpu_timing[i]->memTransHDTime = 0.0;
    gpu_timing[i]->memAllocTime = 0.0;
    gpu_timing[i]->memFreeTime = 0.0;
  }
  int gpu_device = 0;
  CUDA_RT_CALL(cudaSetDevice(gpu_device));
  CUDA_RT_CALL(cudaGetDevice(&gpu_device));

  //allocateCalibDataGPU(max_strips, calib_data_d, &pt_calib_data_d, gpu_timing[0], gpu_device, stream[0]);
  //cpyCalibDataToGPU(max_strips, calib_data, calib_data_d, gpu_timing[0], stream[0]);
#endif

  double t0 = omp_get_wtime();

#ifdef USE_GPU
#ifdef CALIB_1D
  allocateCalibDataGPU(max_strips, calib_data_d, &pt_calib_data_d, gpu_timing[0], gpu_device, stream[0]);
  cpyCalibDataToGPU(max_strips, calib_data, calib_data_d, gpu_timing[0], stream[0]);
#endif

  for (int iter=0; iter<nIter; iter++) {
#pragma omp parallel for num_threads(nStreams)
    for (int i=0; i<nStreams; i++) {

      allocateSSTDataGPU(max_strips, sst_data_d[i], &pt_sst_data_d[i], gpu_timing[i], gpu_device, stream[i]);

      cpySSTDataToGPU(sst_data[i], sst_data_d[i], gpu_timing[i], stream[i]);

      setSeedStripsNCIndexGPU(sst_data_d[i], pt_sst_data_d[i], calib_data_d, pt_calib_data_d, condGPU.get(), gpu_timing[i], stream[i]);

      allocateClustDataGPU(max_seedstrips, clust_data_d[i], &pt_clust_data_d[i], gpu_timing[i], gpu_device, stream[i]);

      findClusterGPU(sst_data_d[i], pt_sst_data_d[i], calib_data_d, pt_calib_data_d, condGPU.get(), clust_data_d[i], pt_clust_data_d[i], gpu_timing[i], stream[i]);

      cpyGPUToCPU(sst_data_d[i], pt_sst_data_d[i], clust_data[i], clust_data_d[i], gpu_timing[i], stream[i]);

      freeClustDataGPU(clust_data_d[i], pt_clust_data_d[i], gpu_timing[i], gpu_device, stream[i]);

      freeSSTDataGPU(sst_data_d[i], pt_sst_data_d[i], gpu_timing[i], gpu_device, stream[i]);
    }
  }

#ifdef CALIB_1D
  freeCalibDataGPU(calib_data_d, pt_calib_data_d, gpu_timing[0], gpu_device, stream[0]);
#endif

#else
    for (int iter=0; iter<nIter; iter++) {
#pragma omp parallel for num_threads(nStreams)
    for (int i=0; i<nStreams; i++) {

      //      unpack(chanlocsv[i], conditions.get(), sst_data[i], calib_data);

      setSeedStripsNCIndex(sst_data[i], calib_data, conditions.get(), cpu_timing[i]);

      findCluster(sst_data[i], calib_data, conditions.get(), clust_data[i], cpu_timing[i]);
    }
  }
#endif

  double t1 = omp_get_wtime();

#ifdef OUTPUT
#ifdef USE_GPU
  CUDA_RT_CALL(cudaDeviceSynchronize());
#endif
  // print out the result
  for (int i=0; i<nStreams; i++) {
#ifdef USE_GPU
    sst_data[i]->nSeedStripsNC = sst_data_d[i]->nSeedStripsNC;
#endif
    std::cout<<" Event "<<i<<" nSeedStripsNC "<<sst_data[i]->nSeedStripsNC<<std::endl;
    for (int j=0; j<sst_data[i]->nSeedStripsNC; j++) {
      if (clust_data[i]->trueCluster[j]){
	int index = clust_data[i]->clusterLastIndexLeft[j];
	std::cout<<" det id "<<sst_data[i]->detId[index]<<" strip "<<sst_data[i]->stripId[index]
		 <<" bary center "<<clust_data[i]->barycenter[j]<<": ";
	int right=clust_data[i]->clusterLastIndexRight[j];
	int size=right-index+1;
	for (int k=0; k<size; k++){
	  std::cout<<(unsigned int)clust_data[i]->clusterADCs[k*sst_data[i]->nSeedStripsNC+j]<<" ";
	}
	std::cout<<std::endl;
      }
    }
  }
#endif

#ifdef USE_GPU
  std::cout<<" GPU Memory Transfer Host to Device Time: "<<gpu_timing[0]->memTransHDTime<<std::endl;
  std::cout<<" GPU Memory Transfer Device to Host Time: "<<gpu_timing[0]->memTransDHTime<<std::endl;
  std::cout<<" GPU Memory Allocation Time: "<<gpu_timing[0]->memAllocTime<<std::endl;
  std::cout<<" GPU Memory Free Time: "<<gpu_timing[0]->memFreeTime<<std::endl;
  std::cout<<" GPU Kernel Time "<<std::endl;
  std::cout<<" --setSeedStrips kernel Time: "<<gpu_timing[0]->setSeedStripsTime<<std::endl;
  std::cout<<" --setNCSeedStrips kernel Time: "<<gpu_timing[0]->setNCSeedStripsTime<<std::endl;
  std::cout<<" --setStripIndex kernel Time: "<<gpu_timing[0]->setStripIndexTime<<std::endl;
  std::cout<<" --findBoundary GPU Kernel Time: "<<gpu_timing[0]->findBoundaryTime<<std::endl;
  std::cout<<" --checkCluster GPU Kernel Time: "<<gpu_timing[0]->checkClusterTime<<std::endl;
  std::cout<<" Total Time (including data allocation, transfer and kernel cost): "<<t1-t0<<std::endl;
#else
  std::cout<<" setSeedStrips function Time: "<<cpu_timing[0]->setSeedStripsTime<<std::endl;
  std::cout<<" setNCSeedStrips function Time: "<<cpu_timing[0]->setNCSeedStripsTime<<std::endl;
  std::cout<<" setStripIndex function Time: "<<cpu_timing[0]->setStripIndexTime<<std::endl;
  std::cout<<" findBoundary function Time: "<<cpu_timing[0]->findBoundaryTime<<std::endl;
  std::cout<<" checkCluster function Time: "<<cpu_timing[0]->checkClusterTime<<std::endl;
  std::cout<<" Total Time: "<<t1-t0<<std::endl;
  //  std::cout<<"nested? "<<omp_get_nested()<<std::endl;
#endif

#ifdef USE_GPU
  for (int i=0; i<nStreams; i++) {
    free(sst_data_d[i]);
    free(clust_data_d[i]);
    free(gpu_timing[i]);
  }
  //freeCalibDataGPU(calib_data_d, pt_calib_data_d, gpu_timing[0], gpu_device, stream[0]);
  free(calib_data_d);
#endif

  for (int i=0; i<nStreams; i++) {
    freeSSTData(sst_data[i]);
    free(sst_data[i]);
    freeClustData(clust_data[i]);
    free(clust_data[i]);
    free(cpu_timing[i]);
    CUDA_RT_CALL(cudaStreamDestroy(stream[i]));
  }
  freeCalibData(calib_data);
  free(calib_data);

  return 0;

}
