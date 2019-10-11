#include "clusterGPU.cuh"
#include <cub/cub.cuh>
#include <stdio.h>

#if USE_TEXTURE
texture<float, 1, cudaReadModeElementType> noiseTexRef;
texture<float, 1, cudaReadModeElementType> gainTexRef;
texture<uint16_t, 1, cudaReadModeElementType> stripIdTexRef;
texture<uint16_t, 1, cudaReadModeElementType> adcTexRef;

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

static __inline__ __device__ uint16_t fetch_adc(int i)
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
  cudaEventCreate(&gpu_timing->start);
  cudaEventCreate(&gpu_timing->stop);
  cudaEventRecord(gpu_timing->start, stream);
}

static float gpu_timer_measure(gpu_timing_t *gpu_timing, cudaStream_t stream) {
  float elapsedTime;
  cudaEventRecord(gpu_timing->stop,stream);
  cudaEventSynchronize(gpu_timing->stop);
  cudaEventElapsedTime(&elapsedTime, gpu_timing->start, gpu_timing->stop);
  cudaEventRecord(gpu_timing->start, stream);

  return elapsedTime/1000;
}

static float gpu_timer_measure_end(gpu_timing_t *gpu_timing, cudaStream_t stream) {
  float elapsedTime;
  cudaEventRecord(gpu_timing->stop,stream);
  cudaEventSynchronize(gpu_timing->stop);
  cudaEventElapsedTime(&elapsedTime, gpu_timing->start,gpu_timing->stop);

  cudaEventDestroy(gpu_timing->start);
  cudaEventDestroy(gpu_timing->stop);
  return elapsedTime/1000;
}

__global__
static void setSeedStripsGPU(int nStrips, sst_data_t *sst_data_d, calib_data_t *calib_data_d) {
#ifndef USE_TEXTURE
  const uint16_t *__restrict__ adc = sst_data_d->adc;
  const float *__restrict__ noise = calib_data_d->noise;
#endif
  int *__restrict__ seedStripsMask = sst_data_d->seedStripsMask;
  int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const float SeedThreshold = 3.0;

  int i = nthreads * bid + tid;

  if (i<nStrips) {
    seedStripsMask[i] = 0;
    seedStripsNCMask[i] = 0;
    float noise_i = NOISE(i);
    uint8_t adc_i = static_cast<uint8_t>(ADC(i));
    seedStripsMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? 1:0;
    seedStripsNCMask[i] = seedStripsMask[i];
  }
}

__global__
static void setNCSeedStripsGPU(int nStrips, sst_data_t *sst_data_d) {
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
    if (seedStripsMask[i]&&seedStripsMask[i-1]&&(STRIPID(i)-STRIPID(i-1))==1) seedStripsNCMask[i] = 0;
  }
}

__global__
static void setStripIndexGPU(int nStrips, sst_data_t *sst_data_d) {
  const int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;
  const int *__restrict__ prefixSeedStripsNCMask = sst_data_d->prefixSeedStripsNCMask;
  int *__restrict__ seedStripsNCIndex = sst_data_d->seedStripsNCIndex;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  int i = nthreads * bid + tid;

  if (i<nStrips) {
    if (seedStripsNCMask[i] == 1) {
      int index = prefixSeedStripsNCMask[i];
      seedStripsNCIndex[index] = i;
    }
  }

}

__global__
static void findLeftRightBoundaryGPU(int offset, int nStrips, sst_data_t *sst_data_d, calib_data_t *calib_data_d, clust_data_t *clust_data_d) {
  const int *__restrict__ seedStripsNCIndex = sst_data_d->seedStripsNCIndex;
  const int nSeedStripsNC = sst_data_d->nSeedStripsNC;
#ifndef USE_TEXTURE
  const uint16_t *__restrict__ stripId = sst_data_d->stripId;
  const detId_t *__restrict__ detId = sst_data_d->detId;
  const uint16_t *__restrict__ adc = sst_data_d->adc;
  const float *__restrict__ noise = calib_data_d->noise;
#endif
  int *__restrict__ clusterLastIndexLeft = clust_data_d->clusterLastIndexLeft+offset;
  int *__restrict__ clusterLastIndexRight = clust_data_d->clusterLastIndexRight+offset;
  bool *__restrict__ trueCluster = clust_data_d->trueCluster+offset;

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
   int i = nthreads * bid + tid;

   if (i<nSeedStripsNC) {
     index=seedStripsNCIndex[i];
     indexLeft = index;
     indexRight = index;
     noise_i = NOISE(index);
     noiseSquared_i = noise_i*noise_i;
     adcSum_i = static_cast<float>(ADC(index));

     // find left boundary
     testIndexLeft=index-1;
     if (testIndexLeft>=0) {
       rangeLeft = STRIPID(indexLeft)-STRIPID(testIndexLeft)-1;
       sameDetLeft = detId[index] == detId[testIndexLeft];
       while(sameDetLeft&&testIndexLeft>=0&&rangeLeft>=0&&rangeLeft<=MaxSequentialHoles) {

	 testNoise = NOISE(testIndexLeft);
	 testADC = static_cast<uint8_t>(ADC(testIndexLeft));

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
	 testNoise = NOISE(testIndexRight);
	 testADC = static_cast<uint8_t>(ADC(testIndexRight));
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
static void checkClusterConditionGPU(int offset, sst_data_t *sst_data_d, calib_data_t *calib_data_d, clust_data_t *clust_data_d) {
#ifndef USE_TEXTURE
   const uint16_t *__restrict__ stripId = sst_data_d->stripId;
   const uint16_t *__restrict__ adc = sst_data_d->adc;
   const float *__restrict__ noise = calib_data_d->noise;
   const float *__restrict__ gain = calib_data_d->gain;
#endif
   const int nSeedStripsNC = sst_data_d->nSeedStripsNC;
   int offset256 = offset*256;
   const int *__restrict__ clusterLastIndexLeft = clust_data_d->clusterLastIndexLeft+offset;
   const int *__restrict__ clusterLastIndexRight = clust_data_d->clusterLastIndexRight+offset;
   uint8_t *__restrict__ clusterADCs = clust_data_d->clusterADCs+offset256;
   bool *__restrict__ trueCluster = clust_data_d->trueCluster+offset;
   const float minGoodCharge = 1620.0;

   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   const int nthreads = blockDim.x;

   int i = nthreads * bid + tid;

   int left, right, size, j;
   int charge;
   uint16_t adc_j;
   float gain_j;
   float adcSum=0.0f;

   if (i<nSeedStripsNC) {
     if (trueCluster[i]) {
       left=clusterLastIndexLeft[i];
       right=clusterLastIndexRight[i];
       size=right-left+1;

       for (j=0; j<size; j++){
	 adc_j = ADC(left+j);
	 gain_j = GAIN(left+j);
	 charge = static_cast<int>( static_cast<float>(adc_j)/gain_j + 0.5f );
	 if (adc_j < 254) adc_j = ( charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
	 clusterADCs[j*nSeedStripsNC+i] = adc_j;
	 adcSum += static_cast<float>(adc_j);
       }

       trueCluster[i] = (adcSum/0.047f) > minGoodCharge;
     }
   }
}

extern "C"
void allocateSSTDataGPU(int nStrips, sst_data_t *sst_data_d, sst_data_t **pt_sst_data_d) {
  cudaMalloc((void **)pt_sst_data_d, sizeof(sst_data_t));
  cudaMalloc((void **)&(sst_data_d->stripId), nStrips*sizeof(uint16_t));
  cudaMalloc((void **)&(sst_data_d->detId), nStrips*sizeof(detId_t));
  cudaMalloc((void **)&(sst_data_d->adc), nStrips*sizeof(uint16_t));
  cudaMalloc((void **)&(sst_data_d->seedStripsMask), nStrips*sizeof(int));
  cudaMalloc((void **)&(sst_data_d->seedStripsNCMask), nStrips*sizeof(int));
  cudaMalloc((void **)&(sst_data_d->prefixSeedStripsNCMask), nStrips*sizeof(int));
  cudaMalloc((void **)&(sst_data_d->seedStripsNCIndex), nStrips*sizeof(int));
  sst_data_d->d_temp_storage=NULL;
  sst_data_d->temp_storage_bytes=0;
  cub::DeviceScan::ExclusiveSum(sst_data_d->d_temp_storage, sst_data_d->temp_storage_bytes, sst_data_d->seedStripsNCMask, sst_data_d->prefixSeedStripsNCMask, nStrips);
#ifdef GPU_DEBUG
  std::cout<<"temp_storage_bytes="<<sst_data_d->temp_storage_bytes<<std::endl;
#endif
  cudaMalloc((void **)&(sst_data_d->d_temp_storage), sst_data_d->temp_storage_bytes);
  cudaMemcpy((void *)*pt_sst_data_d, sst_data_d, sizeof(sst_data_t), cudaMemcpyHostToDevice);
}

extern "C"
void allocateCalibDataGPU(int nStrips, calib_data_t *calib_data_d, calib_data_t **pt_calib_data_d) {
  cudaMalloc((void **)pt_calib_data_d, sizeof(calib_data_t));
  cudaMalloc((void **)&(calib_data_d->noise), nStrips*sizeof(float));
  cudaMalloc((void **)&(calib_data_d->gain), nStrips*sizeof(float));
  cudaMalloc((void **)&(calib_data_d->bad), nStrips*sizeof(bool));
  cudaMemcpy((void *)*pt_calib_data_d, calib_data_d, sizeof(calib_data_t), cudaMemcpyHostToDevice);
}

extern "C"
void allocateClustDataGPU(int nSeedStripsNC, clust_data_t *clust_data_d, clust_data_t **pt_clust_data_d) {
  cudaMalloc((void **)pt_clust_data_d, sizeof(clust_data_t));
  cudaMalloc((void **)&(clust_data_d->clusterLastIndexLeft), nSeedStripsNC*sizeof(int));
  cudaMalloc((void **)&(clust_data_d->clusterLastIndexRight), nSeedStripsNC*sizeof(int));
  cudaMalloc((void **)&(clust_data_d->clusterADCs), nSeedStripsNC*256*sizeof(uint8_t));
  cudaMalloc((void **)&(clust_data_d->trueCluster), nSeedStripsNC*sizeof(bool));
  cudaMemcpy((void *)*pt_clust_data_d, clust_data_d, sizeof(clust_data_t), cudaMemcpyHostToDevice);
}

extern "C"
void freeSSTDataGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d) {
  cudaFree(pt_sst_data_d);
  cudaFree(sst_data_d->stripId);
  cudaFree(sst_data_d->detId);
  cudaFree(sst_data_d->adc);
  cudaFree(sst_data_d->seedStripsMask);
  cudaFree(sst_data_d->seedStripsNCMask);
  cudaFree(sst_data_d->prefixSeedStripsNCMask);
  cudaFree(sst_data_d->seedStripsNCIndex);
#if USE_TEXTURE
  cudaUnbindTexture(stripIdTexRef);
  cudaUnbindTexture(adcTexRef);
#endif
}

extern "C"
void freeCalibDataGPU(calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d) {
  cudaFree(pt_calib_data_d);
  cudaFree(calib_data_d->noise);
  cudaFree(calib_data_d->gain);
  cudaFree(calib_data_d->bad);
#if USE_TEXTURE
  cudaUnbindTexture(noiseTexRef);
  cudaUnbindTexture(gainTexRef);
#endif
}

extern "C"
void freeClustDataGPU(clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d) {
  cudaFree(pt_clust_data_d);
  cudaFree(clust_data_d->clusterLastIndexLeft);
  cudaFree(clust_data_d->clusterLastIndexRight);
  cudaFree(clust_data_d->clusterADCs);
  cudaFree(clust_data_d->trueCluster);
}

extern "C"
void findClusterGPU(int event, int nStreams, int max_strips, int nStrips, sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif
  int nthreads = 128;
  //int nSeedStripsNC = sst_data_d->nSeedStripsNC;
  int nSeedStripsNC = 150000;
  int nblocks = (nSeedStripsNC+nthreads-1)/nthreads;
  int offset = event*(max_strips/nStreams);

#ifdef GPU_DEBUG
  int *cpu_index = (int *)malloc(nStrips*sizeof(int));
  uint16_t *cpu_strip = (uint16_t *)malloc(nStrips*sizeof(uint16_t));
  uint16_t *cpu_adc = (uint16_t *)malloc(nStrips*sizeof(uint16_t));
  float *cpu_noise = (float *)malloc(nStrips*sizeof(float));

  cudaMemcpy((void *)cpu_strip, sst_data_d->stripId, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_adc, sst_data_d->adc, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_noise, calib_data_d->noise, nStrips*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_index, sst_data_d->seedStripsNCIndex, nStrips*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<nStrips; i++) {
    std::cout<<" cpu_strip "<<cpu_strip[i]<<" cpu_adc "<<cpu_adc[i]<<" cpu_noise "<<cpu_noise[i]<<" cpu index "<<cpu_index[i]<<std::endl;
  }

  free(cpu_index);
  free(cpu_strip);
  free(cpu_adc);
  free(cpu_noise);
#endif

  findLeftRightBoundaryGPU<<<nblocks, nthreads, 0, stream>>>(offset, nStrips, pt_sst_data_d, pt_calib_data_d, pt_clust_data_d);
#ifdef GPU_TIMER
  gpu_timing->findBoundaryTime = gpu_timer_measure(gpu_timing, stream);
#endif

  checkClusterConditionGPU<<<nblocks, nthreads, 0, stream>>>(offset, pt_sst_data_d, pt_calib_data_d, pt_clust_data_d);

#ifdef GPU_TIMER
  gpu_timing->checkClusterTime = gpu_timer_measure_end(gpu_timing, stream);
#endif

#ifdef GPU_DEBUG
  int *clusterLastIndexLeft = (int *)malloc(nSeedStripsNC*sizeof(int));
  int *clusterLastIndexRight = (int *)malloc(nSeedStripsNC*sizeof(int));
  bool *trueCluster = (bool *)malloc(nSeedStripsNC*sizeof(bool));
  uint8_t *ADCs = (uint8_t*)malloc(nSeedStripsNC*256*sizeof(uint8_t));
  //  cudaStreamSynchronize(stream);
  //nSeedStripsNC=sst_data_d->nSeedStripsNC;
  std::cout<<"findClusterGPU Event="<<event<<"offset="<<offset<<"nSeedStripsNC="<<nSeedStripsNC<<std::endl;
  cudaMemcpyAsync((void *)clusterLastIndexLeft, clust_data_d->clusterLastIndexLeft+offset, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync((void *)clusterLastIndexRight, clust_data_d->clusterLastIndexRight+offset, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync((void *)trueCluster, clust_data_d->trueCluster+offset, nSeedStripsNC*sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync((void *)ADCs, clust_data_d->clusterADCs+offset*256, nSeedStripsNC*256*sizeof(uint8_t), cudaMemcpyDeviceToHost);

  cudaStreamSynchronize(stream);
  nSeedStripsNC=sst_data_d->nSeedStripsNC;

  for (int i=0; i<nSeedStripsNC; i++) {
    if (trueCluster[i]){
      int left=clusterLastIndexLeft[i];
      int right=clusterLastIndexRight[i];
      std::cout<<"i="<<i<<" left "<<left<<" right "<<right<<" : ";
      int size=right-left+1;
      for (int j=0; j<size; j++){
	std::cout<<(int)ADCs[j*nSeedStripsNC+i]<<" ";
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
void setSeedStripsNCIndexGPU(int nStrips, sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, calib_data_t *calib_data_d, calib_data_t *pt_calib_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream) {
#ifdef GPU_DEBUG
  uint16_t *cpu_strip = (uint16_t *)malloc(nStrips*sizeof(uint16_t));
  uint16_t *cpu_adc = (uint16_t *)malloc(nStrips*sizeof(uint16_t));
  float *cpu_noise = (float *)malloc(nStrips*sizeof(float));

  cudaMemcpy((void *)cpu_strip, sst_data_d->stripId, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_adc, sst_data_d->adc, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_noise, calib_data_d->noise, nStrips*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i=0; i<nStrips; i++) {
    std::cout<<" cpu_strip "<<cpu_strip[i]<<" cpu_adc "<<cpu_adc[i]<<" cpu_noise "<<cpu_noise[i]<<std::endl;
  }

  free(cpu_strip);
  free(cpu_adc);
  free(cpu_noise);
#endif
  int nthreads = 256;
  int nblocks = (nStrips+nthreads-1)/nthreads;

#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif
  //mark seed strips
  setSeedStripsGPU<<<nblocks, nthreads, 0, stream>>>(nStrips, pt_sst_data_d, pt_calib_data_d);
#ifdef GPU_TIMER
  gpu_timing->setSeedStripsTime = gpu_timer_measure(gpu_timing, stream);
#endif
  //mark only non-consecutive seed strips (mask out consecutive seed strips)
  setNCSeedStripsGPU<<<nblocks, nthreads, 0, stream>>>(nStrips, pt_sst_data_d);
#ifdef GPU_TIMER
  gpu_timing->setNCSeedStripsTime = gpu_timer_measure(gpu_timing, stream);
#endif

  cub::DeviceScan::ExclusiveSum(sst_data_d->d_temp_storage, sst_data_d->temp_storage_bytes, sst_data_d->seedStripsNCMask, sst_data_d->prefixSeedStripsNCMask, nStrips, stream);

  cudaMemcpyAsync((void *)&(pt_sst_data_d->nSeedStripsNC), sst_data_d->prefixSeedStripsNCMask+nStrips-1, sizeof(int), cudaMemcpyDeviceToDevice, stream);
  //  cudaMemcpyAsync((void *)&(sst_data_d->nSeedStripsNC), &(pt_sst_data_d->nSeedStripsNC), sizeof(int), cudaMemcpyDeviceToHost, stream);

  setStripIndexGPU<<<nblocks, nthreads, 0, stream>>>(nStrips, pt_sst_data_d);

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
void cpyGPUToCPU(int event, int nStreams, int max_strips, int nStrips, sst_data_t * sst_data_d, sst_data_t *pt_sst_data_d, clust_data_t *clust_data, clust_data_t *clust_data_d, cudaStream_t stream) {
  int offset = event*(max_strips/nStreams);
  //  cudaDeviceSynchronize();
  cudaMemcpyAsync((void *)&(sst_data_d->nSeedStripsNC), &(pt_sst_data_d->nSeedStripsNC), sizeof(int), cudaMemcpyDeviceToHost, stream);
  int nSeedStripsNC = sst_data_d->nSeedStripsNC;
  std::cout<<"cpyGPUtoCPU Event="<<event<<"offset="<<offset<<"nSeedStripsNC="<<nSeedStripsNC<<std::endl;
  cudaMemcpyAsync((void *)(clust_data->clusterLastIndexLeft+offset), clust_data_d->clusterLastIndexLeft+offset, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync((void *)(clust_data->clusterLastIndexRight+offset), clust_data_d->clusterLastIndexRight+offset, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync((void *)(clust_data->clusterADCs+offset*256), clust_data_d->clusterADCs+offset*256, nSeedStripsNC*256*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync((void *)(clust_data->trueCluster+offset), clust_data_d->trueCluster+offset, nSeedStripsNC*sizeof(bool), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
}

extern "C"
void cpyCalibDataToGPU(int nStrips,  calib_data_t *calib_data, calib_data_t *calib_data_d, gpu_timing_t *gpu_timing) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, 0);
#endif
  cudaMemcpy((void *)calib_data_d->noise, calib_data->noise, nStrips*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void *)calib_data_d->gain, calib_data->gain, nStrips*sizeof(float), cudaMemcpyHostToDevice);
#if USE_TEXTURE
  cudaBindTexture(0, noiseTexRef, (void *)calib_data_d->noise, nStrips*sizeof(float));
  cudaBindTexture(0, gainTexRef, (void *)calib_data_d->gain, nStrips*sizeof(float));
#endif
#ifdef GPU_TIMER
  gpu_timing->memTransferTime = gpu_timer_measure_end(gpu_timing, 0);
#endif
}

extern "C"
void cpySSTDataToGPU(int nStrips, sst_data_t *sst_data, sst_data_t *sst_data_d, gpu_timing_t *gpu_timing, cudaStream_t stream) {
#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, stream);
#endif
  cudaMemcpyAsync((void *)sst_data_d->stripId, sst_data->stripId, nStrips*sizeof(uint16_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync((void *)sst_data_d->detId, sst_data->detId, nStrips*sizeof(detId_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync((void *)sst_data_d->adc, sst_data->adc, nStrips*sizeof(uint16_t), cudaMemcpyHostToDevice, stream);
#if USE_TEXTURE
  cudaBindTexture(0, stripIdTexRef, (void *)sst_data_d->stripId, nStrips*sizeof(uint16_t));
  cudaBindTexture(0, adcTexRef, (void *)sst_data_d->adc, nStrips*sizeof(uint16_t));
#endif
#ifdef GPU_TIMER
  gpu_timing->memTransferTime = gpu_timer_measure_end(gpu_timing, stream);
#endif
}
