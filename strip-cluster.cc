#include <fstream>
#include <iostream>
#include <cstring>
#include <memory>
#include <functional>
#if _OPENMP
#include <omp.h>
#endif
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "clusterGPU.cuh"
#include "cluster.h"

int main()
{
  const int max_strips = MAX_STRIPS;
  const int max_seedstrips = MAX_SEEDSTRIPS;
  const int nStreams = omp_get_max_threads();
  const int nIter = 840/nStreams;
  //  const int nIter = 840;
  const int totalEvents = nIter*nStreams;
  cudaStream_t stream[nStreams];
  sst_data_t *sst_data[nStreams];
  clust_data_t *clust_data[nStreams];
  calib_data_t *calib_data[nStreams];
  cpu_timing_t *cpu_timing[nStreams];
  for (int i=0; i<nStreams; i++) {
    CUDA_RT_CALL(cudaStreamCreate(&stream[i]));
    sst_data[i] = (sst_data_t *)malloc(sizeof(sst_data_t));
    clust_data[i] = (clust_data_t *)malloc(sizeof(clust_data_t));
    calib_data[i] = (calib_data_t *)malloc(sizeof(calib_data_t));
    cpu_timing[i] = (cpu_timing_t *)malloc(sizeof(cpu_timing_t));
  }

  // memory allocation
#ifdef NUMA_FT
#pragma omp parallel for num_threads(nStreams)
#endif
  for (int i=0; i<nStreams; i++) {
    //    print_binding_info();
    allocateSSTData(max_strips, sst_data[i], stream[i]);
    allocateClustData(max_seedstrips, clust_data[i], stream[i]);
    allocateCalibData(max_strips, calib_data[i], stream[i]);
  }

  // read in calibration data (only once)
  std::string condfilename("stripcond.bin");
  auto conditions = std::make_unique<SiStripConditions>(condfilename);

  // option 1: readin strip info
  //#ifdef ACTIVE_STRIPS
  //std::string digifilename("digidata.bin");
  //#else
  //std::string digifilename("digidata_all.bin");
  //#endif
  //readin_raw_digidata(digifilename, conditions.get(), sst_data[0], calib_data[0]);

  // option 2: read in raw data and convert to strip info
  /*
  std::string datafilename("stripdata.bin");
  readin_raw_data(datafilename, conditions.get(), sst_data[0], calib_data[0], stream[0]);
  // copy data to other streams
  for (int i=1; i<nStreams; i++) {
    std::memcpy(sst_data[i]->detId, sst_data[0]->detId, sizeof(detId_t)*sst_data[0]->nStrips);
    std::memcpy(sst_data[i]->stripId, sst_data[0]->stripId, sizeof(uint16_t)*sst_data[0]->nStrips);
    std::memcpy(sst_data[i]->fedId, sst_data[0]->fedId, sizeof(fedId_t)*sst_data[0]->nStrips);
    std::memcpy(sst_data[i]->fedCh, sst_data[0]->fedCh, sizeof(fedCh_t)*sst_data[0]->nStrips);
    std::memcpy(sst_data[i]->adc, sst_data[0]->adc, sizeof(uint8_t)*sst_data[0]->nStrips);
    sst_data[i]->nStrips = sst_data[0]->nStrips;
    std::memcpy(calib_data[i]->noise, calib_data[0]->noise, sizeof(float)*sst_data[0]->nStrips);
    std::memcpy(calib_data[i]->gain, calib_data[0]->gain, sizeof(float)*sst_data[0]->nStrips);
    std::memcpy(calib_data[i]->bad, calib_data[0]->bad, sizeof(bool)*sst_data[0]->nStrips);
  }
  */

  // option 3: read in raw data only (conversion will be done in the loop)
  std::vector<std::vector<FEDRawData>> fedRawDataAll(nStreams);
  std::vector<std::vector<FEDBuffer>> fedBufferAll(nStreams);
  std::vector<std::vector<fedId_t>> fedIndexAll(nStreams);
  std::vector<FEDReadoutMode> modeAll(nStreams);
  //std::vector<ChannelLocs> chanlocsAll;
  //std::vector<cudautils::host::unique_ptr<uint8_t[]>> fedRawDataHostAll(nStreams);
  for (auto i=0; i<nStreams; i++) {
    std::string datafilename("stripdata.bin");
    readinRawData(datafilename, conditions.get(), fedRawDataAll[i], fedBufferAll[i], fedIndexAll[i], modeAll[i], sst_data[i]);
    //chanlocsAll.emplace_back(conditions->detToFeds().size(), stream[i]);
    //fedRawDataHostAll[i] = cudautils::make_host_unique<uint8_t[]>(sst_data[i]->totalRawSize, stream[i]);
  }

  double t0 = omp_get_wtime();

#ifdef USE_GPU
  sst_data_t *sst_data_d[nStreams], *pt_sst_data_d[nStreams];
  calib_data_t *calib_data_d[nStreams], *pt_calib_data_d[nStreams];
  clust_data_t *clust_data_d[nStreams], *pt_clust_data_d[nStreams];
  //std::vector<ChannelLocsGPU> chanlocsAllGPU;
  //std::vector<cudautils::device::unique_ptr<uint8_t[]>> fedRawDataGPUAll(nStreams);
  for (int i=0; i<nStreams; i++) {
    sst_data_d[i] = (sst_data_t *)malloc(sizeof(sst_data_t));
    sst_data_d[i]->nStrips = sst_data[i]->nStrips;
    sst_data_d[i]->totalRawSize = sst_data[i]->totalRawSize;
    clust_data_d[i] = (clust_data_t *)malloc(sizeof(clust_data_t));
    calib_data_d[i] = (calib_data_t *)malloc(sizeof(calib_data_t));
    //chanlocsAllGPU.emplace_back(chanlocsAll[i].size(), stream[i]);
    //fedRawDataGPUAll[i] = cudautils::make_device_unique<uint8_t[]>(sst_data_d[i]->totalRawSize, stream[i]);
  }

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

  std::unique_ptr<SiStripConditionsGPU, std::function<void(SiStripConditionsGPU*)>> condGPU(conditions->toGPU(), [](SiStripConditionsGPU* p) { cudaFree(p); });

  cudaProfilerStart();

  for (int iter=0; iter<nIter; iter++) {
#pragma omp parallel for num_threads(nStreams)
    for (int i=0; i<nStreams; i++) {
#ifdef CALIB_1D
      allocateCalibDataGPU(max_strips, calib_data_d[i], &pt_calib_data_d[i], gpu_timing[i], gpu_device, stream[i]);
#endif

      allocateSSTDataGPU(max_strips, sst_data_d[i], &pt_sst_data_d[i], gpu_timing[i], gpu_device, stream[i]);

      //unpackRawData(conditions.get(), fedRawDataAll[i], fedBufferAll[i], fedIndexAll[i], sst_data[i], calib_data[i], modeAll[i], cpu_timing[i], stream[i]);
      //cpySSTDataToGPU(sst_data[i], sst_data_d[i], gpu_timing[i], stream[i]);
      //cpyCalibDataToGPU(calib_data[i], calib_data_d[i], gpu_timing[i], stream[i]);
      unpackRawDataGPU(conditions.get(), condGPU.get(), fedRawDataAll[i], fedBufferAll[i], fedIndexAll[i], sst_data_d[i], pt_sst_data_d[i], calib_data_d[i], pt_calib_data_d[i], modeAll[i], gpu_timing[i], stream[i]);

      setSeedStripsNCIndexGPU(sst_data_d[i], pt_sst_data_d[i], calib_data_d[i], pt_calib_data_d[i], condGPU.get(), gpu_timing[i], stream[i]);

      allocateClustDataGPU(max_seedstrips, clust_data_d[i], &pt_clust_data_d[i], gpu_timing[i], gpu_device, stream[i]);

      findClusterGPU(sst_data_d[i], pt_sst_data_d[i], calib_data_d[i], pt_calib_data_d[i], condGPU.get(), clust_data_d[i], pt_clust_data_d[i], gpu_timing[i], stream[i]);

      cpyGPUToCPU(sst_data_d[i], pt_sst_data_d[i], clust_data[i], clust_data_d[i], gpu_timing[i], stream[i]);

      freeClustDataGPU(clust_data_d[i], pt_clust_data_d[i], gpu_timing[i], gpu_device, stream[i]);

      freeSSTDataGPU(sst_data_d[i], pt_sst_data_d[i], gpu_timing[i], gpu_device, stream[i]);

#ifdef CALIB_1D
      freeCalibDataGPU(calib_data_d[i], pt_calib_data_d[i], gpu_timing[i], gpu_device, stream[i]);
#endif
    }
  }

#ifdef USE_GPU
  CUDA_RT_CALL(cudaDeviceSynchronize());
#endif

  cudaProfilerStop();

#else
  for (int iter=0; iter<nIter; iter++) {
#pragma omp parallel for num_threads(nStreams)
    for (int i=0; i<nStreams; i++) {

      unpackRawData(conditions.get(), fedRawDataAll[i], fedBufferAll[i], fedIndexAll[i], sst_data[i], calib_data[i], modeAll[i], cpu_timing[i], stream[i]);

      setSeedStripsNCIndex(sst_data[i], calib_data[i], conditions.get(), cpu_timing[i]);

      findCluster(sst_data[i], calib_data[i], conditions.get(), clust_data[i], cpu_timing[i]);
    }
  }
#endif

  double t1 = omp_get_wtime();

  // print out the result
#ifdef OUTPUT
  for (int i=0; i<nStreams; i++) {
#ifdef USE_GPU
    sst_data[i]->nSeedStripsNC = sst_data_d[i]->nSeedStripsNC;
#endif
    std::cout<<" Event "<<i<<" nSeedStripsNC "<<sst_data[i]->nSeedStripsNC<<std::endl;
    for (int j=0; j<sst_data[i]->nSeedStripsNC; j++) {
      if (clust_data[i]->trueCluster[j]){
	int index = clust_data[i]->clusterLastIndexLeft[j];
	//	std::cout<<" det id "<<sst_data[i]->detId[index]<<" strip "<<sst_data[i]->stripId[index]
	std::cout<<" bary center "<<clust_data[i]->barycenter[j]<<": ";
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
  std::cout<<" --unpackRawData kernel Time: "<<gpu_timing[0]->unpackRawDataTime<<std::endl;
  std::cout<<" --setSeedStrips kernel Time: "<<gpu_timing[0]->setSeedStripsTime<<std::endl;
  std::cout<<" --setNCSeedStrips kernel Time: "<<gpu_timing[0]->setNCSeedStripsTime<<std::endl;
  std::cout<<" --setStripIndex kernel Time: "<<gpu_timing[0]->setStripIndexTime<<std::endl;
  std::cout<<" --findBoundary GPU Kernel Time: "<<gpu_timing[0]->findBoundaryTime<<std::endl;
  std::cout<<" --checkCluster GPU Kernel Time: "<<gpu_timing[0]->checkClusterTime<<std::endl;
  std::cout<<" Total Time (including data allocation, transfer and kernel cost): "<<t1-t0<<" Throughput: "<<totalEvents/(t1-t0)<<std::endl;
#else
  std::cout<<" setSeedStrips function Time: "<<cpu_timing[0]->setSeedStripsTime<<std::endl;
  std::cout<<" setNCSeedStrips function Time: "<<cpu_timing[0]->setNCSeedStripsTime<<std::endl;
  std::cout<<" setStripIndex function Time: "<<cpu_timing[0]->setStripIndexTime<<std::endl;
  std::cout<<" findBoundary function Time: "<<cpu_timing[0]->findBoundaryTime<<std::endl;
  std::cout<<" checkCluster function Time: "<<cpu_timing[0]->checkClusterTime<<std::endl;
  std::cout<<" Total Time: "<<t1-t0<<" Throughput: "<<totalEvents/(t1-t0)<<std::endl;
  std::cout<<"nested? "<<omp_get_nested()<<std::endl;
#endif

#ifdef USE_GPU
  //chanlocsAllGPU.clear();
  //fedRawDataGPUAll.clear();
  for (int i=0; i<nStreams; i++) {
    free(sst_data_d[i]);
    free(clust_data_d[i]);
    free(calib_data_d[i]);
    free(gpu_timing[i]);
  }
#endif

  //fedRawDataHostAll.clear();
  //chanlocsAll.clear();
  for (int i=0; i<nStreams; i++) {
    freeSSTData(sst_data[i]);
    free(sst_data[i]);
    freeClustData(clust_data[i]);
    free(clust_data[i]);
    freeCalibData(calib_data[i]);
    free(calib_data[i]);
    free(cpu_timing[i]);
    CUDA_RT_CALL(cudaStreamDestroy(stream[i]));
  }

  return 0;

}
