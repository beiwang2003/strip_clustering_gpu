#ifndef _CLUSTER_
#define _CLUSTER_
#include "FEDRawData.h"
#include "SiStripFEDBuffer.h"

#include <fstream>
#include <cstdlib>
#include <cstdint>
//#ifdef USE_GPU
#include <cuda_runtime_api.h>
//#endif
#if _OPENMP
#include <omp.h>
#endif
#include "SiStripConditions.h"
#include "FEDChannel.h"

#define IDEAL_ALIGNMENT 64
#define CACHELINE_BYTES 64
#define MAX_STRIPS 810000 // 600000 for active strips
#define MAX_SEEDSTRIPS 200000

//using detId_t = uint32_t;
typedef enum {SST, CALIB} SSTorCALIB;

typedef struct {
  detId_t *detId;
  uint16_t *stripId;
  uint8_t *adc;
  fedId_t *fedId;
  fedCh_t *fedCh;
  int *seedStripsNCIndex, *seedStripsMask, *seedStripsNCMask, *prefixSeedStripsNCMask;
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;
  int nSeedStripsNC;
  int nStrips;
  size_t totalRawSize;
} sst_data_t;

typedef struct {
  float *noise, *gain;
  bool *bad;
} calib_data_t;

typedef struct {
  int *clusterLastIndexLeft, *clusterLastIndexRight;
  uint8_t *clusterADCs;
  bool *trueCluster;
  float *barycenter;
} clust_data_t;

typedef struct {
  float unpackRawDataTime;
  float setSeedStripsTime;
  float setNCSeedStripsTime;
  float setStripIndexTime;
  float findBoundaryTime;
  float checkClusterTime;
} cpu_timing_t;

void print_binding_info();

void readin_raw_digidata(const std::string& digifilename, const SiStripConditions *conditions, sst_data_t *sst_data, calib_data_t *calib_data);
void readin_raw_data(const std::string& datafilename, const SiStripConditions* conditions, sst_data_t *sst_data, calib_data_t *calib_data, cudaStream_t stream);

void readinRawData(const std::string& datafilename, const SiStripConditions *conditions, std::vector<FEDRawData>& fedRawDatav, std::vector<FEDBuffer>& fedBufferv, std::vector<fedId_t>& fedIndex, FEDReadoutMode& mode, sst_data_t* sst_data);
void unpackRawData(const SiStripConditions *conditions, const std::vector<FEDRawData>& fedRawDatav, const std::vector<FEDBuffer>& fedBufferv, const std::vector<fedId_t>& fedIndex, sst_data_t *sst_data, calib_data_t *calib_data, const FEDReadoutMode& mode, cpu_timing_t *cpu_timing, cudaStream_t stream, SSTorCALIB unpack_option);

void unpackSST(const ChannelLocs& chanlocs, const SiStripConditions* conditions, sst_data_t *sst_data);

void unpackCalib(const ChannelLocs& chanlocs, const SiStripConditions* conditions, calib_data_t *calib_data);

void allocateSSTData(int max_strips, sst_data_t *sst_data, cudaStream_t stream);
void allocateCalibData(int max_strips, calib_data_t *calib_data);
void allocateClustData(int max_seedstrips, clust_data_t *clust_data, cudaStream_t stream);

void freeSSTData(sst_data_t *sst_data);
void freeCalibData(calib_data_t *calib_data_t);
void freeClustData(clust_data_t *clust_data_t);

void setSeedStripsNCIndex(sst_data_t *sst_data, calib_data_t *calib_data, const SiStripConditions *conditions, cpu_timing_t *cpu_timing);

void findCluster(sst_data_t *sst_data, calib_data_t *calib_data, const SiStripConditions *conditions, clust_data_t *clust_data, cpu_timing_t *cpu_timing);

#endif
