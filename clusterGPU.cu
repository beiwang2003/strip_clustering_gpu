#include "clusterGPU.cuh"
#include <stdio.h>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#ifdef CACHE_ALLOC
#include "allocate_device.h"
#endif
#include "copyAsync.h"

constexpr auto kStripsPerChannel = SiStripConditionsBase::kStripsPerChannel;

#if USE_TEXTURE
texture<float, 1, cudaReadModeElementType> noiseTexRef;
texture<float, 1, cudaReadModeElementType> gainTexRef;
texture<uint16_t, 1, cudaReadModeElementType> stripIdTexRef;
texture<uint8_t, 1, cudaReadModeElementType> adcTexRef;

static __inline__ __device__ float fetch_noise(int i)
{
  return tex1Dfetch(noiseTexRef, i);
}
static __inline__ __device__ float fetch_gain(int i)
{
  return tex1Dfetch(gainTexRef, i);
}
static __inline__ __device__ uint16_t fetch_stripId(int i)
{
  return tex1Dfetch(stripIdTexRef, i);
}

static __inline__ __device__ uint8_t fetch_adc(int i)
{
  return tex1Dfetch(adcTexRef, i);
}
#define NOISE(i) (fetch_noise(i))
#define GAIN(i) (fetch_gain(i))
#define STRIPID(i) (fetch_stripId(i))
#define ADC(i) (fetch_adc(i))
#else
#define NOISE(i) (noise[i])
#define GAIN(i) (gain[i])
#define STRIPID(i) (stripId[i])
#define ADC(i) (adc[i])
#endif

static void gpu_timer_start(gpu_timing_t *gpu_timing, cudaStream_t stream) {
  CUDA_RT_CALL(cudaEventCreate(&gpu_timing->start));
  CUDA_RT_CALL(cudaEventCreate(&gpu_timing->stop));
  CUDA_RT_CALL(cudaEventRecord(gpu_timing->start, stream));
}

static float gpu_timer_measure(gpu_timing_t *gpu_timing, cudaStream_t stream) {
  float elapsedTime;
  CUDA_RT_CALL(cudaEventRecord(gpu_timing->stop, stream));
  CUDA_RT_CALL(cudaEventSynchronize(gpu_timing->stop));
  CUDA_RT_CALL(cudaEventElapsedTime(&elapsedTime, gpu_timing->start, gpu_timing->stop));
  CUDA_RT_CALL(cudaEventRecord(gpu_timing->start, stream));

  return elapsedTime/1000;
}

static float gpu_timer_measure_end(gpu_timing_t *gpu_timing, cudaStream_t stream) {
  float elapsedTime;
  CUDA_RT_CALL(cudaEventRecord(gpu_timing->stop,stream));
  CUDA_RT_CALL(cudaEventSynchronize(gpu_timing->stop));
  CUDA_RT_CALL(cudaEventElapsedTime(&elapsedTime, gpu_timing->start,gpu_timing->stop));

  CUDA_RT_CALL(cudaEventDestroy(gpu_timing->start));
  CUDA_RT_CALL(cudaEventDestroy(gpu_timing->stop));
  return elapsedTime/1000;
}

__global__
static void unpackChannelsSST(const ChanLocStruct* chanlocs, const SiStripConditionsGPU* conditions, sst_data_t *sst_data_d){
  uint8_t *__restrict__ adc = sst_data_d->adc;
  detId_t *__restrict__ detId = sst_data_d->detId;
  uint16_t *__restrict__ stripId = sst_data_d->stripId;
  fedId_t *__restrict__ fedId = sst_data_d->fedId;
  fedCh_t *__restrict__ fedCh = sst_data_d->fedCh;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const auto chan = nthreads*bid + tid;

  if (chan < chanlocs->size()) {
    const auto fedid = chanlocs->fedID(chan);
    const auto fedch = chanlocs->fedCh(chan);
    const auto detid = conditions->detID(fedid, fedch);
    const auto ipoff = kStripsPerChannel*conditions->iPair(fedid, fedch);

    const auto data = chanlocs->input(chan);
    const auto len = chanlocs->length(chan);

    if (data != nullptr && len > 0) {
      auto aoff = chanlocs->offset(chan);
      auto choff = chanlocs->inoff(chan);
      const auto end = aoff + len;

      while (aoff < end) {
        stripId[aoff] = invStrip;
        detId[aoff] = invDet;
        adc[aoff] = data[(choff++)^7];
        auto stripIndex = adc[aoff++] + ipoff;

        stripId[aoff] = invStrip;
        detId[aoff] = detid;
        adc[aoff] = data[(choff++)^7];
        const auto groupLength = adc[aoff++];

        for (auto i = 0; i < groupLength; ++i) {
	  fedId[aoff] = fedid;
	  fedCh[aoff] = fedch;
          detId[aoff] = detid;
          stripId[aoff] = stripIndex++;
          adc[aoff++] = data[(choff++)^7];
        }
      }
    }
  }
}

__global__
static void unpackChannelsCalib(const ChanLocStruct* chanlocs, const SiStripConditionsGPU* conditions, calib_data_t *calib_data_d) {
  float *__restrict__ noise = calib_data_d->noise;
  float *__restrict__ gain = calib_data_d->gain;
  bool *__restrict__ bad = calib_data_d->bad;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const auto chan = nthreads*bid + tid;

  if (chan < chanlocs->size()) {
    const auto fedid = chanlocs->fedID(chan);
    const auto fedch = chanlocs->fedCh(chan);
    const auto detid = conditions->detID(fedid, fedch);
    const auto ipoff = kStripsPerChannel*conditions->iPair(fedid, fedch);

    const auto data = chanlocs->input(chan);
    const auto len = chanlocs->length(chan);

    if (data != nullptr && len > 0) {
      auto aoff = chanlocs->offset(chan);
      auto choff = chanlocs->inoff(chan);
      const auto end = aoff + len;

      while (aoff < end) {
        auto stripIndex = data[(choff++)^7] + ipoff;
	const auto groupLength = data[(choff++)^7];
	aoff=aoff+2;
        for (auto i = 0; i < groupLength; ++i) {
          noise[aoff] = conditions->noise(fedid, fedch, stripIndex);
          gain[aoff]  = conditions->gain(fedid, fedch, stripIndex);
          bad[aoff++]   = conditions->bad(fedid, fedch, stripIndex);
        }
      }
    }
  }
}

__global__
static void setSeedStripsGPU(sst_data_t *sst_data_d, calib_data_t *calib_data_d, const SiStripConditionsGPU * conditions) {
  const int nStrips = sst_data_d->nStrips;
#ifndef USE_TEXTURE
  const uint8_t *__restrict__ adc = sst_data_d->adc;
  const float *__restrict__ noise = calib_data_d->noise;
  const uint16_t *__restrict__ stripId = sst_data_d->stripId;
  const fedId_t *__restrict__ fedId = sst_data_d->fedId;
  const fedCh_t *__restrict__ fedCh = sst_data_d->fedCh;
#endif
  int *__restrict__ seedStripsMask = sst_data_d->seedStripsMask;
  int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const float SeedThreshold = 3.0;

  int i = nthreads * bid + tid;
  stripId_t strip;
  float noise_i;
  fedId_t fed;
  fedCh_t channel;
  uint8_t adc_i;

  if (i<nStrips) {
    seedStripsMask[i] = 0;
    seedStripsNCMask[i] = 0;
    strip = stripId[i];
    if (strip != invStrip) {
#ifdef CALIB_1D
      noise_i = NOISE(i);
#else
      fed = fedId[i];
      channel = fedCh[i];
      //    stripId_t strip = stripId[i];
      noise_i = conditions->noise(fed, channel, strip);
#endif
      adc_i = ADC(i);
      seedStripsMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? 1:0;
      seedStripsNCMask[i] = seedStripsMask[i];
    }
   }
}

__global__
  static void setNCSeedStripsGPU(sst_data_t *sst_data_d) {
  const int nStrips = sst_data_d->nStrips;
  const detId_t *__restrict__ detId = sst_data_d->detId;
#ifndef USE_TEXTURE
  const uint16_t *__restrict__ stripId = sst_data_d->stripId;
#endif
  const int *__restrict__ seedStripsMask = sst_data_d->seedStripsMask;
  int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  int i = nthreads * bid + tid;

  if (i>0&&i<nStrips) {
     if (seedStripsMask[i]&&seedStripsMask[i-1]&&(STRIPID(i)-STRIPID(i-1))==1&&(detId[i]==detId[i-1])) seedStripsNCMask[i] = 0;
  }
}

__global__
static void setStripIndexGPU(sst_data_t *sst_data_d) {
  const int nStrips = sst_data_d->nStrips;
  const int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;
  const int *__restrict__ prefixSeedStripsNCMask = sst_data_d->prefixSeedStripsNCMask;
  int *__restrict__ seedStripsNCIndex = sst_data_d->seedStripsNCIndex;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  int i = nthreads * bid + tid;
  int index;

  if (i<nStrips) {
    if (seedStripsNCMask[i] == 1) {
      index = prefixSeedStripsNCMask[i];
      seedStripsNCIndex[index] = i;
    }
  }

}

__global__
static void findLeftRightBoundaryGPU(sst_data_t *sst_data_d, calib_data_t *calib_data_d, const SiStripConditionsGPU *conditions, clust_data_t *clust_data_d) {
  const int nStrips = sst_data_d->nStrips;
  const int *__restrict__ seedStripsNCIndex = sst_data_d->seedStripsNCIndex;
  const int nSeedStripsNC = sst_data_d->nSeedStripsNC;
#ifndef USE_TEXTURE
  //#ifdef CALIB_1D
  const float *__restrict__ noise = calib_data_d->noise;
  //#else
  const uint8_t *__restrict__ adc = sst_data_d->adc;
  const uint16_t *__restrict__ stripId = sst_data_d->stripId;
  const detId_t *__restrict__ detId = sst_data_d->detId;
  const fedId_t *__restrict__ fedId = sst_data_d->fedId;
  const fedCh_t *__restrict__ fedCh = sst_data_d->fedCh;
  //#endif
#endif
  int *__restrict__ clusterLastIndexLeft = clust_data_d->clusterLastIndexLeft;
  int *__restrict__ clusterLastIndexRight = clust_data_d->clusterLastIndexRight;
  bool *__restrict__ trueCluster = clust_data_d->trueCluster;

   const uint8_t MaxSequentialHoles = 0;
   const float  ChannelThreshold = 2.0;
   const float ClusterThresholdSquared = 25.0;

   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   const int nthreads = blockDim.x;

   int index, testIndexLeft, testIndexRight, indexLeft, indexRight, rangeLeft, rangeRight;
   uint8_t testADC;
   float noise_i, testNoise, noiseSquared_i, adcSum_i;
   bool noiseSquaredPass, sameDetLeft, sameDetRight;
   fedId_t fed, testFed;
   fedCh_t channel, testChannel;
   stripId_t strip, testStrip;
   int i = nthreads * bid + tid;

   if (i<nSeedStripsNC) {
     index=seedStripsNCIndex[i];
     indexLeft = index;
     indexRight = index;
#ifdef CALIB_1D
     noise_i = NOISE(index);
#else
     fed = fedId[index];
     channel = fedCh[index];
     strip = stripId[index];
     noise_i = conditions->noise(fed, channel, strip);
#endif
     noiseSquared_i = noise_i*noise_i;
     adcSum_i = static_cast<float>(ADC(index));

     // find left boundary
     testIndexLeft=index-1;
     if (testIndexLeft>=0) {
       rangeLeft = STRIPID(indexLeft)-STRIPID(testIndexLeft)-1;
       sameDetLeft = detId[index] == detId[testIndexLeft];
       while(sameDetLeft&&testIndexLeft>=0&&rangeLeft>=0&&rangeLeft<=MaxSequentialHoles) {
#ifdef CALIB_1D
	 testNoise = NOISE(testIndexLeft);
#else
	 testFed = fedId[testIndexLeft];
	 testChannel = fedCh[testIndexLeft];
	 testStrip = stripId[testIndexLeft];
	 testNoise = conditions->noise(testFed, testChannel, testStrip);
#endif
	 testADC = ADC(testIndexLeft);

	 if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	   --indexLeft;
	   noiseSquared_i += testNoise*testNoise;
	   adcSum_i += static_cast<float>(testADC);
	 }
	 --testIndexLeft;
	 if (testIndexLeft>=0) {
	   rangeLeft = STRIPID(indexLeft)-STRIPID(testIndexLeft)-1;
	   sameDetLeft = detId[index] == detId[testIndexLeft];
	 }
       }
     }

     // find right boundary
     testIndexRight=index+1;
     if (testIndexRight<nStrips) {
       rangeRight = STRIPID(testIndexRight)-STRIPID(indexRight)-1;
       sameDetRight = detId[index] == detId[testIndexRight];
       while(sameDetRight&&testIndexRight<nStrips&&rangeRight>=0&&rangeRight<=MaxSequentialHoles) {
#ifdef CALIB_1D
	 testNoise = NOISE(testIndexRight);
#else
         testFed = fedId[testIndexRight];
         testChannel = fedCh[testIndexRight];
         testStrip = stripId[testIndexRight];
         testNoise = conditions->noise(testFed, testChannel, testStrip);
#endif
	 testADC = ADC(testIndexRight);
	 if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	   ++indexRight;
	   noiseSquared_i += testNoise*testNoise;
	   adcSum_i += static_cast<float>(testADC);
	 }
	 ++testIndexRight;
	 if (testIndexRight<nStrips) {
	   rangeRight = STRIPID(testIndexRight)-STRIPID(indexRight)-1;
	   sameDetRight = detId[index] == detId[testIndexRight];
	 }
       }
     }
     noiseSquaredPass = noiseSquared_i*ClusterThresholdSquared <= adcSum_i*adcSum_i;
     trueCluster[i] = noiseSquaredPass;
     clusterLastIndexLeft[i] = indexLeft;
     clusterLastIndexRight[i] = indexRight;

   }
}

__global__
static void checkClusterConditionGPU(sst_data_t *sst_data_d, calib_data_t *calib_data_d, const SiStripConditionsGPU *conditions, clust_data_t *clust_data_d) {
#ifndef USE_TEXTURE
  //#ifdef CALIB_1D
   const float *__restrict__ gain = calib_data_d->gain;
   //#else
   const uint16_t *__restrict__ stripId = sst_data_d->stripId;
   const uint8_t *__restrict__ adc = sst_data_d->adc;
   const fedId_t *__restrict__ fedId = sst_data_d->fedId;
   const fedCh_t *__restrict__ fedCh = sst_data_d->fedCh;
   //#endif
#endif
   const int nSeedStripsNC = sst_data_d->nSeedStripsNC;
   const int *__restrict__ clusterLastIndexLeft = clust_data_d->clusterLastIndexLeft;
   const int *__restrict__ clusterLastIndexRight = clust_data_d->clusterLastIndexRight;
   uint8_t *__restrict__ clusterADCs = clust_data_d->clusterADCs;
   bool *__restrict__ trueCluster = clust_data_d->trueCluster;
   float *__restrict__ barycenter = clust_data_d->barycenter;
   const float minGoodCharge = 1620.0;
   const uint16_t stripIndexMask = 0x7FFF;

   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   const int nthreads = blockDim.x;

   int i = nthreads * bid + tid;

   int left, right, size, j;
   int charge;
   uint8_t adc_j;
   float gain_j;
   float adcSum=0.0f;
   int sumx=0;
   int suma=0;
   fedId_t fed;
   fedCh_t channel;
   stripId_t strip;

   if (i<nSeedStripsNC) {
     if (trueCluster[i]) {
       left=clusterLastIndexLeft[i];
       right=clusterLastIndexRight[i];
       size=right-left+1;

       if (i>0&&clusterLastIndexLeft[i-1]==left) {
         trueCluster[i] = 0;  // ignore duplicates
       } else {
         for (j=0; j<size; j++){
	   adc_j = ADC(left+j);
#ifdef CALIB_1D
	   gain_j = GAIN(left+j);
#else
	   fed = fedId[left+j];
	   channel = fedCh[left+j];
	   strip = stripId[left+j];
	   gain_j = conditions->gain(fed, channel, strip);
#endif
	   charge = static_cast<int>( static_cast<float>(adc_j)/gain_j + 0.5f );
	   if (adc_j < 254) adc_j = ( charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
	   clusterADCs[j*nSeedStripsNC+i] = adc_j;
	   adcSum += static_cast<float>(adc_j);
	   sumx += j*adc_j;
	   suma += adc_j;
         }
	 barycenter[i] = static_cast<float>(stripId[left] & stripIndexMask) + static_cast<float>(sumx)/static_cast<float>(suma) + 0.5f;
       }
       trueCluster[i] = (adcSum/0.047f) > minGoodCharge;
     }
   }
}

extern "C"
void allocateSSTDataGPU(int max_strips, sst_data_t *sst_data_d, sst_data_t **pt_sst_data_d, gpu_timing_t* gpu_timing,  int dev, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif

#ifdef CACHE_ALLOC
  *pt_sst_data_d = (sst_data_t *)cudautils::allocate_device(dev, sizeof(sst_data_t), stream);
  sst_data_d->detId = (detId_t*)cudautils::allocate_device(dev, max_strips*sizeof(detId_t), stream);
  sst_data_d->stripId = (uint16_t *)cudautils::allocate_device(dev, max_strips*sizeof(uint16_t), stream);
  sst_data_d->adc = (uint8_t *)cudautils::allocate_device(dev, max_strips*sizeof(uint8_t), stream);
  sst_data_d->fedId = (fedId_t *)cudautils::allocate_device(dev, max_strips*sizeof(fedId_t), stream);
  sst_data_d->fedCh = (fedCh_t *)cudautils::allocate_device(dev, max_strips*sizeof(fedCh_t), stream);
  sst_data_d->seedStripsMask = (int *)cudautils::allocate_device(dev, 2*max_strips*sizeof(int), stream);
  sst_data_d->prefixSeedStripsNCMask = (int *)cudautils::allocate_device(dev, 2*max_strips*sizeof(int), stream);
#else
  CUDA_RT_CALL(cudaMalloc((void **)pt_sst_data_d, sizeof(sst_data_t)));
  CUDA_RT_CALL(cudaMalloc((void **)&(sst_data_d->detId), max_strips*sizeof(detId_t)));
  CUDA_RT_CALL(cudaMalloc((void **)&(sst_data_d->stripId), 2*max_strips*sizeof(uint16_t)));
  CUDA_RT_CALL(cudaMalloc((void **)&(sst_data_d->adc), max_strips*sizeof(uint8_t)));
  CUDA_RT_CALL(cudaMalloc((void **)&(sst_data_d->fedId), max_strips*sizeof(fedId_t)));
  CUDA_RT_CALL(cudaMalloc((void **)&(sst_data_d->fedCh), max_strips*sizeof(fedCh_t)));
  CUDA_RT_CALL(cudaMalloc((void **)&(sst_data_d->seedStripsMask), 2*max_strips*sizeof(int)));
  CUDA_RT_CALL(cudaMalloc((void **)&(sst_data_d->prefixSeedStripsNCMask), 2*max_strips*sizeof(int)));
#endif

  sst_data_d->seedStripsNCMask = sst_data_d->seedStripsMask + max_strips;
  sst_data_d->seedStripsNCIndex = sst_data_d->prefixSeedStripsNCMask + max_strips;
  sst_data_d->d_temp_storage=NULL;
  sst_data_d->temp_storage_bytes=0;
  cub::DeviceScan::ExclusiveSum(sst_data_d->d_temp_storage, sst_data_d->temp_storage_bytes, sst_data_d->seedStripsNCMask, sst_data_d->prefixSeedStripsNCMask, max_strips);
#ifdef GPU_DEBUG
  std::cout<<"temp_storage_bytes="<<sst_data_d->temp_storage_bytes<<std::endl;
#endif

#ifdef CACHE_ALLOC
  sst_data_d->d_temp_storage = cudautils::allocate_device(dev, sst_data_d->temp_storage_bytes, stream);
#else
  CUDA_RT_CALL(cudaMalloc((void **)&(sst_data_d->d_temp_storage), sst_data_d->temp_storage_bytes));
#endif // end CACHE_ALLOC

  CUDA_RT_CALL(cudaMemcpyAsync((void *)*pt_sst_data_d, sst_data_d, sizeof(sst_data_t), cudaMemcpyHostToDevice, stream));

#ifdef GPU_TIMER
  gpu_timing->memAllocTime = 0;
  gpu_timing->memAllocTime += gpu_timer_measure_end(gpu_timing, stream);
#endif
}

extern "C"
void allocateCalibDataGPU(int max_strips, calib_data_t *calib_data_d, calib_data_t **pt_calib_data_d, gpu_timing_t* gpu_timing, int dev, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif

#ifdef CACHE_ALLOC
  *pt_calib_data_d = (calib_data_t *)cudautils::allocate_device(dev, sizeof(calib_data_t), stream);
  calib_data_d->noise = (float *)cudautils::allocate_device(dev, 2*max_strips*sizeof(float), stream);
  calib_data_d->bad = (bool *)cudautils::allocate_device(dev, max_strips*sizeof(bool), stream);
#else
  CUDA_RT_CALL(cudaMalloc((void **)pt_calib_data_d, sizeof(calib_data_t)));
  CUDA_RT_CALL(cudaMalloc((void **)&(calib_data_d->noise), 2*max_strips*sizeof(float)));
  CUDA_RT_CALL(cudaMalloc((void **)&(calib_data_d->bad), max_strips*sizeof(bool)));
#endif
  calib_data_d->gain = calib_data_d->noise + max_strips;
  CUDA_RT_CALL(cudaMemcpyAsync((void *)*pt_calib_data_d, calib_data_d, sizeof(calib_data_t), cudaMemcpyHostToDevice, stream));
#ifdef GPU_TIMER
  gpu_timing->memAllocTime += gpu_timer_measure_end(gpu_timing, stream);
#endif
}

extern "C"
  void allocateClustDataGPU(int max_strips, clust_data_t *clust_data_d, clust_data_t **pt_clust_data_d, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif

#ifdef CACHE_ALLOC
  *pt_clust_data_d = (clust_data_t *)cudautils::allocate_device(dev, sizeof(clust_data_t), stream);
  clust_data_d->clusterLastIndexLeft = (int *)cudautils::allocate_device(dev, 2*max_strips*sizeof(int), stream);
  clust_data_d->clusterADCs = (uint8_t *)cudautils::allocate_device(dev, max_strips*256*sizeof(uint8_t), stream);
  clust_data_d->trueCluster = (bool *)cudautils::allocate_device(dev, max_strips*sizeof(bool), stream);
  clust_data_d->barycenter = (float *)cudautils::allocate_device(dev, max_strips*sizeof(float), stream);
#else
  CUDA_RT_CALL(cudaMalloc((void **)pt_clust_data_d, sizeof(clust_data_t)));
  CUDA_RT_CALL(cudaMalloc((void **)&(clust_data_d->clusterLastIndexLeft), 2*max_strips*sizeof(int)));
  CUDA_RT_CALL(cudaMalloc((void **)&(clust_data_d->clusterADCs), max_strips*256*sizeof(uint8_t)));
  CUDA_RT_CALL(cudaMalloc((void **)&(clust_data_d->trueCluster), max_strips*sizeof(bool)));
  CUDA_RT_CALL(cudaMalloc((void **)&(clust_data_d->barycenter), max_strips*sizeof(float)));
#endif
  clust_data_d->clusterLastIndexRight = clust_data_d->clusterLastIndexLeft + max_strips;
  CUDA_RT_CALL(cudaMemcpyAsync((void *)*pt_clust_data_d, clust_data_d, sizeof(clust_data_t), cudaMemcpyHostToDevice, stream));

#ifdef GPU_TIMER
  gpu_timing->memAllocTime += gpu_timer_measure_end(gpu_timing, stream);
#endif
}

extern "C"
void freeSSTDataGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif

#ifdef CACHE_ALLOC
  cudautils::free_device(dev, pt_sst_data_d);
  cudautils::free_device(dev, sst_data_d->detId);
  cudautils::free_device(dev, sst_data_d->stripId);
  cudautils::free_device(dev, sst_data_d->adc);
  cudautils::free_device(dev, sst_data_d->fedId);
  cudautils::free_device(dev, sst_data_d->fedCh);
  cudautils::free_device(dev, sst_data_d->seedStripsMask);
  cudautils::free_device(dev, sst_data_d->prefixSeedStripsNCMask);
#else
  CUDA_RT_CALL(cudaFree(pt_sst_data_d));
  CUDA_RT_CALL(cudaFree(sst_data_d->detId));
  CUDA_RT_CALL(cudaFree(sst_data_d->stripId));
  CUDA_RT_CALL(cudaFree(sst_data_d->adc);
  CUDA_RT_CALL(cudaFree(sst_data_d->fedId));
  CUDA_RT_CALL(cudaFree(sst_data_d->fedCh));
  CUDA_RT_CALL(cudaFree(sst_data_d->seedStripsMask));
  CUDA_RT_CALL(cudaFree(sst_data_d->prefixSeedStripsNCMask));
#endif
#if USE_TEXTURE
  cudaUnbindTexture(stripIdTexRef);
  cudaUnbindTexture(adcTexRef);
#endif
#ifdef GPU_TIMER
  gpu_timing->memFreeTime += gpu_timer_measure_end(gpu_timing, stream);
#endif
}

extern "C"
void freeCalibDataGPU(calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif

#ifdef CACHE_ALLOC
  cudautils::free_device(dev, pt_calib_data_d);
  cudautils::free_device(dev, calib_data_d->noise);
  cudautils::free_device(dev, calib_data_d->bad);
#else
  CUDA_RT_CALL(cudaFree(pt_calib_data_d));
  CUDA_RT_CALL(cudaFree(calib_data_d->noise));
  CUDA_RT_CALL(cudaFree(calib_data_d->bad));
#endif
#if USE_TEXTURE
  cudaUnbindTexture(noiseTexRef);
  cudaUnbindTexture(gainTexRef);
#endif
#ifdef GPU_TIMER
  gpu_timing->memFreeTime += gpu_timer_measure_end(gpu_timing, stream);
#endif
}

extern "C"
void freeClustDataGPU(clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, gpu_timing_t *gpu_timing, int dev, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif
#ifdef CACHE_ALLOC
  cudautils::free_device(dev, pt_clust_data_d);
  cudautils::free_device(dev, clust_data_d->clusterLastIndexLeft);
  cudautils::free_device(dev, clust_data_d->clusterADCs);
  cudautils::free_device(dev, clust_data_d->trueCluster);
  cudautils::free_device(dev, clust_data_d->barycenter);
#else
  CUDA_RT_CALL(cudaFree(pt_clust_data_d));
  CUDA_RT_CALL(cudaFree(clust_data_d->clusterLastIndexLeft));
  CUDA_RT_CALL(cudaFree(clust_data_d->clusterADCs));
  CUDA_RT_CALL(cudaFree(clust_data_d->trueCluster));
  CUDA_RT_CALL(cudaFree(clust_data_d->barycenter));
#endif
#ifdef GPU_TIMER
  gpu_timing->memFreeTime = 0;
  gpu_timing->memFreeTime += gpu_timer_measure_end(gpu_timing, stream);
#endif
}

extern "C"
  void unpackRawDataGPU(const SiStripConditions *conditions, const SiStripConditionsGPU *conditionsGPU, const std::vector<FEDRawData>& fedRawDatav, const std::vector<FEDBuffer>& fedBufferv, const std::vector<fedId_t>& fedIndex, sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, const FEDReadoutMode& mode, gpu_timing_t *gpu_timing, cudaStream_t stream, SSTorCALIB unpack_option) {

  ChannelLocs chanlocs(conditions->detToFeds().size(), stream);
  ChannelLocsGPU chanlocsGPU(chanlocs.size(), stream);
  std::vector<uint8_t*> inputGPU(chanlocs.size());
  std::vector<size_t> fedRawDataOffsets;
  fedRawDataOffsets.reserve(SiStripConditions::kFedCount);

  auto fedRawDataHost = cudautils::make_host_unique<uint8_t[]>(sst_data_d->totalRawSize, stream);
  auto fedRawDataGPU = cudautils::make_device_unique<uint8_t[]>(sst_data_d->totalRawSize, stream);

  size_t off = 0;
  for (const auto &d : fedRawDatav) {
    memcpy(fedRawDataHost.get() + off, d.get(), d.size());
    fedRawDataOffsets.push_back(off);
    off += d.size();
  }

  //std::cout<<"fedRawData copy size "<<sst_data_d->totalRawSize<<"at stream "<<stream<<std::endl;
  assert(sst_data_d->totalRawSize==off);

#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif
  cudautils::copyAsync(fedRawDataGPU, fedRawDataHost, sst_data_d->totalRawSize, stream);
#ifdef GPU_TIMER
  gpu_timing->memTransHDTime += gpu_timer_measure(gpu_timing, stream);
#endif

  const auto& detmap = conditions->detToFeds();
  size_t offset = 0;

  // iterate over the detector in DetID/APVPair order
  // mapping out where the data are
  const uint16_t headerlen = mode == READOUT_MODE_ZERO_SUPPRESSED ? 7 : 2;

  for(size_t i = 0; i < detmap.size(); ++i) {
    const auto& detp = detmap[i];

    auto fedId = detp.fedID();
    auto fedi = fedIndex[fedId-SiStripConditions::kFedFirst];
    if (fedi != invFed) {
      const auto& buffer = fedBufferv[fedi];
      const auto& channel = buffer.channel(detp.fedCh());

      if (channel.length() >= headerlen) {
        chanlocs.setChannelLoc(i, channel.data(), channel.offset()+headerlen, offset, channel.length()-headerlen, detp.fedID(), detp.fedCh());
	inputGPU[i] = fedRawDataGPU.get() + fedRawDataOffsets[fedi] + (channel.data() - fedRawDatav[fedi].get());
        offset += channel.length()-headerlen;
      } else {
        chanlocs.setChannelLoc(i, channel.data(), channel.offset(), offset, channel.length(), detp.fedID(),detp.fedCh());
	inputGPU[i] = fedRawDataGPU.get() + fedRawDataOffsets[fedi] + (channel.data() - fedRawDatav[fedi].get());
        offset += channel.length();
        assert(channel.length() == 0);
      }
    } else {
      chanlocs.setChannelLoc(i, nullptr, 0, 0, 0, invFed, 0);
      std::cout << "Missing fed " << fedi << " for detID " << detp.fedID() << std::endl;
      inputGPU[i] = nullptr;
      exit (1);
    }
  }

  sst_data_d->nStrips = offset;
  //std::cout<<"nStrips "<<offset<<std::endl;

  if (unpack_option == SST)
    CUDA_RT_CALL(cudaMemcpyAsync((void *)&(pt_sst_data_d->nStrips), &(sst_data_d->nStrips), sizeof(int), cudaMemcpyHostToDevice, stream));

  chanlocsGPU.reset(chanlocs, inputGPU, stream);

#ifdef GPU_TIMER
  gpu_timing->memTransHDTime += gpu_timer_measure(gpu_timing, stream);
#endif

  constexpr int nthreads = 64;
  const auto channels = chanlocs.size();
  const auto nblocks = (channels + nthreads - 1)/nthreads;

  if (unpack_option == SST)
  //std::cout<<"total channels size "<<channels<<std::endl;
    unpackChannelsSST<<<nblocks, nthreads, 0, stream>>>(chanlocsGPU.chanLocStruct(), conditionsGPU, pt_sst_data_d);
  else if (unpack_option == CALIB)
    unpackChannelsCalib<<<nblocks, nthreads, 0, stream>>>(chanlocsGPU.chanLocStruct(), conditionsGPU, pt_calib_data_d);
  else {
    std::cout<<"other unpack option is not available"<<std::endl;
    exit (2);
  }
  //cudaStreamSynchronize(stream);

#ifdef GPU_TIMER
  gpu_timing->unpackRawDataTime = gpu_timer_measure_end(gpu_timing, stream);
#endif

#ifdef GPU_DEBUG
  cudaStreamSynchronize(stream);
  int nStrips = sst_data_d->nStrips;
  uint16_t *cpu_strip = (uint16_t *)malloc(nStrips*sizeof(uint16_t));
  uint8_t *cpu_adc = (uint8_t *)malloc(nStrips*sizeof(uint8_t));
  fedId_t *cpu_fedid = (fedId_t *)malloc(nStrips*sizeof(fedId_t));

  cudaMemcpy((void *)cpu_strip, sst_data_d->stripId, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_adc, sst_data_d->adc, nStrips*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_fedid, sst_data_d->fedId, nStrips*sizeof(fedId_t), cudaMemcpyDeviceToHost);

  for (int i=0; i<nStrips; i++) {
    std::cout<<" i "<<i<<" cpu fedId "<<cpu_fedid[i]<<" cpu_strip "<<(unsigned int)cpu_strip[i]<<" cpu_adc "<<(unsigned int)cpu_adc[i]<<std::endl;
  }

  free(cpu_strip);
  free(cpu_adc);
  free(cpu_fedid);
#endif

}


extern "C"
void findClusterGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, const SiStripConditionsGPU *conditions, clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif
  int nthreads = 128;
  //int nSeedStripsNC = sst_data_d->nSeedStripsNC;
  int nSeedStripsNC = MAX_SEEDSTRIPS;
  int nblocks = (nSeedStripsNC+nthreads-1)/nthreads;

#ifdef GPU_DEBUG
  int nStrips = sst_data_d->nStrips;
  int *cpu_index = (int *)malloc(nStrips*sizeof(int));
  uint16_t *cpu_strip = (uint16_t *)malloc(nStrips*sizeof(uint16_t));
  uint8_t *cpu_adc = (uint8_t *)malloc(nStrips*sizeof(uint8_t));
  float *cpu_noise = (float *)malloc(nStrips*sizeof(float));

  cudaMemcpy((void *)cpu_strip, sst_data_d->stripId, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_adc, sst_data_d->adc, nStrips*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_noise, calib_data_d->noise, nStrips*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_index, sst_data_d->seedStripsNCIndex, nStrips*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<nStrips; i++) {
      std::cout<<" cpu_strip "<<cpu_strip[i]<<" cpu_adc "<<(unsigned int)cpu_adc[i]<<" cpu index "<<cpu_index[i]<<std::endl;
  }

  free(cpu_index);
  free(cpu_strip);
  free(cpu_adc);
  free(cpu_noise);
#endif

  findLeftRightBoundaryGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d, pt_calib_data_d, conditions, pt_clust_data_d);
  CUDA_RT_CALL(cudaGetLastError());

#ifdef GPU_TIMER
  gpu_timing->findBoundaryTime = gpu_timer_measure(gpu_timing, stream);
#endif

  checkClusterConditionGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d, pt_calib_data_d, conditions, pt_clust_data_d);
  CUDA_RT_CALL(cudaGetLastError());

#ifdef GPU_TIMER
  gpu_timing->checkClusterTime = gpu_timer_measure_end(gpu_timing, stream);
#endif

#ifdef GPU_DEBUG
  int *clusterLastIndexLeft = (int *)malloc(nSeedStripsNC*sizeof(int));
  int *clusterLastIndexRight = (int *)malloc(nSeedStripsNC*sizeof(int));
  bool *trueCluster = (bool *)malloc(nSeedStripsNC*sizeof(bool));
  uint8_t *ADCs = (uint8_t*)malloc(nSeedStripsNC*256*sizeof(uint8_t));
  //nSeedStripsNC=sst_data_d->nSeedStripsNC;
  std::cout<<"findClusterGPU"<<"nSeedStripsNC="<<nSeedStripsNC<<std::endl;
  cudaMemcpyAsync((void *)clusterLastIndexLeft, clust_data_d->clusterLastIndexLeft, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync((void *)clusterLastIndexRight, clust_data_d->clusterLastIndexRight, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync((void *)trueCluster, clust_data_d->trueCluster, nSeedStripsNC*sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync((void *)ADCs, clust_data_d->clusterADCs, nSeedStripsNC*256*sizeof(uint8_t), cudaMemcpyDeviceToHost);

  cudaStreamSynchronize(stream);
  nSeedStripsNC=sst_data_d->nSeedStripsNC;

  for (int i=0; i<nSeedStripsNC; i++) {
    if (trueCluster[i]){
      int left=clusterLastIndexLeft[i];
      int right=clusterLastIndexRight[i];
      std::cout<<"i="<<i<<" left "<<left<<" right "<<right<<" : ";
      int size=right-left+1;
      for (int j=0; j<size; j++){
	std::cout<<(unsigned int)ADCs[j*nSeedStripsNC+i]<<" ";
      }
      std::cout<<std::endl;
    }
  }

  free(clusterLastIndexLeft);
  free(clusterLastIndexRight);
  free(trueCluster);
  free(ADCs);
#endif

}

extern "C"
void setSeedStripsNCIndexGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, const SiStripConditionsGPU *conditions, gpu_timing_t *gpu_timing, cudaStream_t stream) {
#ifdef GPU_DEBUG
  cudaStreamSynchronize(stream);
  int nStrips = sst_data_d->nStrips;
  uint16_t *cpu_strip = (uint16_t *)malloc(nStrips*sizeof(uint16_t));
  uint8_t *cpu_adc = (uint8_t *)malloc(nStrips*sizeof(uint8_t));
  float *cpu_noise = (float *)malloc(nStrips*sizeof(float));

  cudaMemcpy((void *)cpu_strip, sst_data_d->stripId, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_adc, sst_data_d->adc, nStrips*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_noise, calib_data_d->noise, nStrips*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i=0; i<nStrips; i++) {
    if (i%1000==0)
    std::cout<<" cpu_strip "<<cpu_strip[i]<<" cpu_adc "<<(unsigned int)cpu_adc[i]<<std::endl;
  }

  free(cpu_strip);
  free(cpu_adc);
  free(cpu_noise);
#endif

  int nthreads = 256;
  int nblocks = (sst_data_d->nStrips+nthreads-1)/nthreads;

#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif
  CUDA_RT_CALL(cudaMemcpyAsync((void *)&(pt_sst_data_d->nStrips), &(sst_data_d->nStrips), sizeof(int), cudaMemcpyHostToDevice, stream));
#ifdef GPU_TIMER
  gpu_timing->memTransHDTime += gpu_timer_measure(gpu_timing, stream);
#endif

  //  cudaStreamSynchronize(stream);
  //mark seed strips
  setSeedStripsGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d, pt_calib_data_d, conditions);
  CUDA_RT_CALL(cudaGetLastError());
#ifdef GPU_TIMER
  gpu_timing->setSeedStripsTime = gpu_timer_measure(gpu_timing, stream);
#endif

  //mark only non-consecutive seed strips (mask out consecutive seed strips)
  setNCSeedStripsGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d);
  CUDA_RT_CALL(cudaGetLastError());
#ifdef GPU_TIMER
  gpu_timing->setNCSeedStripsTime = gpu_timer_measure(gpu_timing, stream);
#endif
  //std::cout<<"nStrip in setStripIndexGPU is="<<sst_data_d->nStrips<<std::endl;

  cub::DeviceScan::ExclusiveSum(sst_data_d->d_temp_storage, sst_data_d->temp_storage_bytes, sst_data_d->seedStripsNCMask, sst_data_d->prefixSeedStripsNCMask, sst_data_d->nStrips, stream);

#ifdef GPU_TIMER
  gpu_timer_measure(gpu_timing, stream);
#endif

  CUDA_RT_CALL(cudaMemcpyAsync((void *)&(pt_sst_data_d->nSeedStripsNC), sst_data_d->prefixSeedStripsNCMask+sst_data_d->nStrips-1, sizeof(int), cudaMemcpyDeviceToDevice, stream));
#ifdef GPU_TIMER
  gpu_timing->memTransDHTime += gpu_timer_measure(gpu_timing, stream);
#endif

  CUDA_RT_CALL(cudaMemcpyAsync((void *)&(sst_data_d->nSeedStripsNC), &(pt_sst_data_d->nSeedStripsNC), sizeof(int), cudaMemcpyDeviceToHost, stream));

#ifdef GPU_TIMER
  gpu_timing->memTransHDTime += gpu_timer_measure(gpu_timing, stream);
#endif

#ifdef GPU_DEBUG
  cudaStreamSynchronize(stream);
  if (sst_data_d->nSeedStripsNC==0) {
    std::cout<<"nSeedStripsNC in setStripIndexGPU is="<<sst_data_d->nSeedStripsNC<<std::endl;
    exit(2);
  }
#endif

  setStripIndexGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d);
  CUDA_RT_CALL(cudaGetLastError());

#ifdef GPU_TIMER
  gpu_timing->setStripIndexTime = gpu_timer_measure_end(gpu_timing, stream);
#endif

#ifdef GPU_DEBUG
  int *cpu_mask = (int *)malloc(nStrips*sizeof(int));
  int *cpu_prefix= (int *)malloc(nStrips*sizeof(int));
  int *cpu_index = (int *)malloc(nStrips*sizeof(int));

  cudaMemcpy((void *)cpu_mask, sst_data_d->seedStripsNCMask, nStrips*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_prefix, sst_data_d->prefixSeedStripsNCMask, nStrips*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_index, sst_data_d->seedStripsNCIndex, nStrips*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<nStrips; i++) {
    std::cout<<" i "<<i<<" mask "<<cpu_mask[i]<<" prefix "<<cpu_prefix[i]<<" index "<<cpu_index[i]<<std::endl;
  }

  free(cpu_mask);
  free(cpu_prefix);
  free(cpu_index);

  cudaMemcpy((void *)&(sst_data_d->nSeedStripsNC), &(pt_sst_data_d->nSeedStripsNC), sizeof(int), cudaMemcpyDeviceToHost);
  std::cout<<"nStrips="<<nStrips<<"nSeedStripsNC="<<sst_data_d->nSeedStripsNC<<"temp_storage_bytes="<<sst_data_d->temp_storage_bytes<<std::endl;
#endif
}


extern "C"
void cpyGPUToCPU(sst_data_t * sst_data_d, sst_data_t *pt_sst_data_d, clust_data_t *clust_data, clust_data_t *clust_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream) {
  //  cudaDeviceSynchronize();
  //cudaMemcpyAsync((void *)&(sst_data_d->nSeedStripsNC), &(pt_sst_data_d->nSeedStripsNC), sizeof(int), cudaMemcpyDeviceToHost, stream);
  //cudaMemcpy((void *)&(sst_data_d->nSeedStripsNC), &(pt_sst_data_d->nSeedStripsNC), sizeof(int), cudaMemcpyDeviceToHost);
  //cudaStreamSynchronize(stream);

  int nSeedStripsNC = 150000;
  //std::cout<<"cpyGPUtoCPU Event="<<event<<"offset="<<offset<<"nSeedStripsNC="<<nSeedStripsNC<<std::endl;
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif
  CUDA_RT_CALL(cudaMemcpyAsync((void *)(clust_data->clusterLastIndexLeft), clust_data_d->clusterLastIndexLeft, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_RT_CALL(cudaMemcpyAsync((void *)(clust_data->clusterLastIndexRight), clust_data_d->clusterLastIndexRight, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost, stream));
#ifdef COPY_ADC
  CUDA_RT_CALL(cudaMemcpyAsync((void *)(clust_data->clusterADCs), clust_data_d->clusterADCs, nSeedStripsNC*256*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
#endif
  CUDA_RT_CALL(cudaMemcpyAsync((void *)(clust_data->trueCluster), clust_data_d->trueCluster, nSeedStripsNC*sizeof(bool), cudaMemcpyDeviceToHost, stream));
  CUDA_RT_CALL(cudaMemcpyAsync((void *)(clust_data->barycenter), clust_data_d->barycenter, nSeedStripsNC*sizeof(float), cudaMemcpyDeviceToHost, stream));
  //CUDA_RT_CALL(cudaMemcpyAsync((void *)&(sst_data_d->nSeedStripsNC), &(pt_sst_data_d->nSeedStripsNC), sizeof(int), cudaMemcpyDeviceToHost, stream));
  //CUDA_RT_CALL(cudaStreamSynchronize(stream));
  //CUDA_RT_CALL(cudaMemcpy((void *)&(sst_data_d->nSeedStripsNC), &(pt_sst_data_d->nSeedStripsNC), sizeof(int), cudaMemcpyDeviceToHost));
#ifdef GPU_TIMER
  gpu_timing->memTransDHTime = gpu_timer_measure_end(gpu_timing, stream);
#endif
}

extern "C"
void cpyCalibDataToGPU(int max_strips, calib_data_t *calib_data, calib_data_t *calib_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif
  CUDA_RT_CALL(cudaMemcpy((void *)calib_data_d->noise, calib_data->noise, max_strips*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy((void *)calib_data_d->gain, calib_data->gain, max_strips*sizeof(float), cudaMemcpyHostToDevice));
#if USE_TEXTURE
  cudaBindTexture(0, noiseTexRef, (void *)calib_data_d->noise, max_strips*sizeof(float));
  cudaBindTexture(0, gainTexRef, (void *)calib_data_d->gain, max_strips*sizeof(float));
#endif
#ifdef GPU_TIMER
  gpu_timing->memTransHDTime = gpu_timer_measure_end(gpu_timing, stream);
#endif
}

extern "C"
void cpySSTDataToGPU(sst_data_t *sst_data, sst_data_t *sst_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif
  sst_data_d->nStrips = sst_data->nStrips;
  int nStrips = sst_data_d->nStrips;
  std::cout<<"cpySSTDataToGPU nStrips="<<nStrips<<std::endl;

  CUDA_RT_CALL(cudaMemcpyAsync((void *)sst_data_d->stripId, sst_data->stripId, nStrips*sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
  CUDA_RT_CALL(cudaMemcpyAsync((void *)sst_data_d->detId, sst_data->detId, nStrips*sizeof(detId_t), cudaMemcpyHostToDevice, stream));
  CUDA_RT_CALL(cudaMemcpyAsync((void *)sst_data_d->fedId, sst_data->fedId, nStrips*sizeof(fedId_t), cudaMemcpyHostToDevice, stream));
  CUDA_RT_CALL(cudaMemcpyAsync((void *)sst_data_d->fedCh, sst_data->fedCh, nStrips*sizeof(fedCh_t), cudaMemcpyHostToDevice, stream));
  CUDA_RT_CALL(cudaMemcpyAsync((void *)sst_data_d->adc, sst_data->adc, nStrips*sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
#if USE_TEXTURE
  cudaBindTexture(0, stripIdTexRef, (void *)sst_data_d->stripId, nStrips*sizeof(uint16_t));
  cudaBindTexture(0, adcTexRef, (void *)sst_data_d->adc, nStrips*sizeof(uint8_t));
#endif
#ifdef GPU_TIMER
  gpu_timing->memTransHDTime = gpu_timer_measure_end(gpu_timing, stream);
#endif
}
