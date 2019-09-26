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
#define NOISE(i) (noise_d[i])
#define GAIN(i) (gain_d[i])
#define STRIPID(i) (stripId_d[i])
#define ADC(i) (adc_d[i])
#endif

static void gpu_timer_start(gpu_timing_t *gpu_timing)
{
  cudaEventCreate(&gpu_timing->start);
  cudaEventCreate(&gpu_timing->stop);
  cudaEventRecord(gpu_timing->start,0);
}

static float gpu_timer_measure(gpu_timing_t *gpu_timing)
{
  float elapsedTime;
  cudaEventRecord(gpu_timing->stop,0);
  cudaEventSynchronize(gpu_timing->stop);
  cudaEventElapsedTime(&elapsedTime, gpu_timing->start, gpu_timing->stop);
  cudaEventRecord(gpu_timing->start,0);

  return elapsedTime/1000;
}

static float gpu_timer_measure_end(gpu_timing_t *gpu_timing)
{
  float elapsedTime;
  cudaEventRecord(gpu_timing->stop,0);
  cudaEventSynchronize(gpu_timing->stop);
  cudaEventElapsedTime(&elapsedTime, gpu_timing->start,gpu_timing->stop);

  cudaEventDestroy(gpu_timing->start);
  cudaEventDestroy(gpu_timing->stop);
  return elapsedTime/1000;
}


__global__
static void setSeedStripsGPU(int nStrips, float *noise_d, uint16_t *adc_d, int *seedStripMask_d, int *seedStripNCMask_d)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  float SeedThreshold = 3.0;

  int i = nthreads * bid + tid;

  if (i<nStrips) {
    seedStripMask_d[i] = 0;
    seedStripNCMask_d[i] = 0;
    float noise_i = NOISE(i);
    uint8_t adc_i = static_cast<uint8_t>(ADC(i));
    seedStripMask_d[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? 1:0;
    seedStripNCMask_d[i] = seedStripMask_d[i];
  }
}

__global__
static void setNCSeedStripsGPU(int nStrips, uint16_t *stripId_d, int *seedStripMask_d, int *seedStripNCMask_d)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  int i = nthreads * bid + tid;

  if (i>0&&i<nStrips) {
    if (seedStripMask_d[i]&&seedStripMask_d[i-1]&&(STRIPID(i)-STRIPID(i-1))==1) seedStripNCMask_d[i] = 0;
  }
}

__global__
static void setStripIndexGPU(int nStrips, int *seedStripNCMask_d, int *prefixSeedStripNCMask_d, int *seedStripsNCIndex_d)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  int i = nthreads * bid + tid;

  if (i<nStrips) {
    if (seedStripNCMask_d[i] == 1) {
      int index = prefixSeedStripNCMask_d[i];
      seedStripsNCIndex_d[index] = i;
    }
  }

}

__global__
static void findLeftRightBoundaryGPU(int nSeedStripsNC, int nStrips, int *clusterLastIndexLeft_d, int *clusterLastIndexRight_d, int *seedStripsNCIndex_d, uint16_t *stripId_d, uint16_t *adc_d, float *noise_d, bool *trueCluster_d)
{
   const uint8_t MaxSequentialHoles = 0;
   const float  ChannelThreshold = 2.0;
   const float minGoodCharge = 1620.0;
   const float ClusterThresholdSquared = 25.0;

   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   const int nthreads = blockDim.x;

   int index, testIndexLeft, testIndexRight, indexLeft, indexRight, rangeLeft, rangeRight;
   uint8_t testADC;
   float noise_i, testNoise, noiseSquared_i, adcSum_i;
   bool noiseSquaredPass, chargePerCMPass;
   int i = nthreads * bid + tid;

   if (i<nSeedStripsNC) {

     index=seedStripsNCIndex_d[i];
     indexLeft = index;
     indexRight = index;
     noise_i = NOISE(index);
     noiseSquared_i = noise_i*noise_i;
     adcSum_i = static_cast<float>(ADC(index));

     // find left boundary
     testIndexLeft=index-1;
     rangeLeft = STRIPID(indexLeft)-STRIPID(testIndexLeft)-1;

     while(testIndexLeft>=0&&rangeLeft>=0&&rangeLeft<=MaxSequentialHoles) {
       testNoise = NOISE(testIndexLeft);
       testADC = static_cast<uint8_t>(ADC(testIndexLeft));

       if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	 --indexLeft;
	 noiseSquared_i += testNoise*testNoise;
	 adcSum_i += static_cast<float>(testADC);
       }
       --testIndexLeft;
       rangeLeft = STRIPID(indexLeft)-STRIPID(testIndexLeft)-1;
     }

     // find right boundary
     testIndexRight=index+1;
     rangeRight = STRIPID(testIndexRight)-STRIPID(indexRight)-1;

     while(testIndexRight<nStrips&&rangeRight>=0&&rangeRight<=MaxSequentialHoles) {
       testNoise = NOISE(testIndexRight);
       testADC = static_cast<uint8_t>(ADC(testIndexRight));
       if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	 ++indexRight;
	 noiseSquared_i += testNoise*testNoise;
	 adcSum_i += static_cast<float>(testADC);
       }
       ++testIndexRight;
       rangeRight = STRIPID(testIndexRight)-STRIPID(indexRight)-1;
     }

     noiseSquaredPass = noiseSquared_i*ClusterThresholdSquared <= adcSum_i*adcSum_i;
     chargePerCMPass = adcSum_i/0.047f > minGoodCharge;

     trueCluster_d[i] = noiseSquaredPass&chargePerCMPass;
     clusterLastIndexLeft_d[i] = indexLeft;
     clusterLastIndexRight_d[i] = indexRight;
   }
}

__global__
static void checkClusterConditionGPU(int nSeedStripsNC, int *clusterLastIndexLeft_d, int *clusterLastIndexRight_d, uint16_t *adc_d, float * gain_d, bool *trueCluster_d, uint8_t *clusterADCs_d)
{
   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   const int nthreads = blockDim.x;

   int i = nthreads * bid + tid;

   int left, right, size, j;
   int charge;
   uint16_t adc_j;
   float gain_j;

   if (i<nSeedStripsNC) {
     if (trueCluster_d[i]) {
       left=clusterLastIndexLeft_d[i];
       right=clusterLastIndexRight_d[i];
       size=right-left+1;

       for (j=0; j<size; j++){
	 adc_j = ADC(left+j);
	 gain_j = GAIN(left+j);
	 charge = static_cast<int>( static_cast<float>(adc_j)/gain_j + 0.5f );
	 if (adc_j < 254) adc_j = ( charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
	 clusterADCs_d[j*nSeedStripsNC+i] = adc_j;
       }
     }
   }
}

extern "C"
void allocateMemAllStripsGPU(int max_strips, uint16_t **stripId_d_pt, uint16_t **adc_d_pt, float **noise_d_pt, float **gain_d_pt, int **seedStripsNCIndex_d_pt, int **seedStripsMask_d_pt, int **seedStripsNCMask_d_pt, int **prefixSeedStripsNCMask_d_pt)
{
  cudaMalloc(stripId_d_pt, max_strips*sizeof(uint16_t));
  cudaMalloc(adc_d_pt, max_strips*sizeof(uint16_t));
  cudaMalloc(noise_d_pt, max_strips*sizeof(float));
  cudaMalloc(gain_d_pt, max_strips*sizeof(float));
  cudaMalloc(seedStripsNCIndex_d_pt, max_strips*sizeof(int));

  cudaMalloc(seedStripsMask_d_pt, max_strips*sizeof(int));
  cudaMalloc(seedStripsNCMask_d_pt, max_strips*sizeof(int));
  cudaMalloc(prefixSeedStripsNCMask_d_pt, max_strips*sizeof(int));
}

extern "C"
void allocateMemNCSeedStripsGPU(int nSeedStripsNC, int **clusterLastIndexLeft_d_pt, int **clusterLastIndexRight_d_pt, uint8_t **clusterADCs_d_pt, bool **trueCluster_d_pt)
{
  cudaMalloc(clusterLastIndexLeft_d_pt, 2*nSeedStripsNC*sizeof(int));
  *clusterLastIndexRight_d_pt = *clusterLastIndexLeft_d_pt + nSeedStripsNC;
  cudaMalloc(clusterADCs_d_pt, nSeedStripsNC*256*sizeof(uint8_t));
  cudaMalloc(trueCluster_d_pt, nSeedStripsNC*sizeof(bool));
}

extern "C"
void freeGPUMem(uint16_t *stripId_d, uint16_t *adc_d, float *noise_d, float *gain_d, int *seedStripNCIndex_d, int *seedStripsMask_d, int *seedStripsNCMask_d, int *prefixSeedStripsNCMask_d, int *clusterLastIndexLeft_d, uint8_t *clusterADCs_d, bool *trueCluster_d)
{
   cudaFree(stripId_d);
   cudaFree(adc_d);
   cudaFree(noise_d);
   cudaFree(gain_d);
#if USE_TEXTURE
   cudaUnbindTexture(stripIdTexRef);
   cudaUnbindTexture(adcTexRef);
   cudaUnbindTexture(noiseTexRef);
   cudaUnbindTexture(gainTexRef);
#endif
   cudaFree(seedStripsMask_d);
   cudaFree(seedStripsNCMask_d);
   cudaFree(prefixSeedStripsNCMask_d);
   cudaFree(seedStripNCIndex_d);
   cudaFree(clusterLastIndexLeft_d);
   cudaFree(clusterADCs_d);
   cudaFree(trueCluster_d);
}

extern "C"
void  findClusterGPU(int nSeedStripsNC, int nStrips, int *clusterLastIndexLeft_d,  int *clusterLastIndexRight_d, int *seedStripsNCIndex_d, uint16_t *stripId_d, uint16_t *adc_d, float *noise_d, float *gain_d, bool *trueCluster_d, uint8_t *clusterADCs_d, gpu_timing_t *gpu_timing)
{
  gpu_timer_start(gpu_timing);
  int nthreads = 128;
  int nblocks = (nSeedStripsNC+nthreads-1)/nthreads;

#ifdef GPU_DEBUG
  int *cpu_index = (int *)malloc(nSeedStripsNC*sizeof(int));
  uint16_t *cpu_strip = (uint16_t *)malloc(nStrips*sizeof(uint16_t));
  uint16_t *cpu_adc = (uint16_t *)malloc(nStrips*sizeof(uint16_t));
  float *cpu_noise = (float *)malloc(nStrips*sizeof(float));

  cudaMemcpy((void *)cpu_strip, stripId_d, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_adc, adc_d, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_noise, noise_d, nStrips*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_index, seedStripsNCIndex_d, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<nStrips; i++) {
    std::cout<<" cpu_strip "<<cpu_strip[i]<<" cpu_adc "<<cpu_adc[i]<<" cpu_noise "<<cpu_noise[i]<<" cpu index "<<cpu_index[i]<<std::endl;
  }

  free(cpu_index);
  free(cpu_strip);
  free(cpu_adc);
  free(cpu_noise);
#endif

  findLeftRightBoundaryGPU<<<nblocks, nthreads>>>(nSeedStripsNC, nStrips, clusterLastIndexLeft_d, clusterLastIndexRight_d, seedStripsNCIndex_d, stripId_d, adc_d, noise_d, trueCluster_d);

  gpu_timing->findBoundaryTime = gpu_timer_measure(gpu_timing);

  checkClusterConditionGPU<<<nblocks, nthreads>>>(nSeedStripsNC, clusterLastIndexLeft_d, clusterLastIndexRight_d, adc_d, gain_d, trueCluster_d, clusterADCs_d);

  gpu_timing->checkClusterTime = gpu_timer_measure_end(gpu_timing);

#ifdef GPU_DEBUG
  int *clusterLastIndexLeft = (int *)malloc(nSeedStripsNC*sizeof(int));
  int *clusterLastIndexRight = (int *)malloc(nSeedStripsNC*sizeof(int));
  bool *trueCluster = (bool *)malloc(nSeedStripsNC*sizeof(bool));
  uint8_t *ADCs = (uint8_t*)malloc(nSeedStripsNC*256*sizeof(uint8_t));

  cudaMemcpy((void *)clusterLastIndexLeft, clusterLastIndexLeft_d, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)clusterLastIndexRight, clusterLastIndexRight_d, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)trueCluster, trueCluster_d, nSeedStripsNC*sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)ADCs, clusterADCs_d, nSeedStripsNC*256*sizeof(uint8_t), cudaMemcpyDeviceToHost);

  for (int i=0; i<nSeedStripsNC; i++) {
    if (trueCluster[i]){
      int left=clusterLastIndexLeft[i];
      int right=clusterLastIndexRight[i];
      std::cout<<" left "<<left<<" right "<<right<<" : ";
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
int setSeedStripsNCIndexGPU(int nStrips, uint16_t *stripId_d, uint16_t *adc_d, float *noise_d, int *seedStripsNCIndex_d, int *seedStripsMask_d, int *seedStripsNCMask_d, int *prefixSeedStripsNCMask_d, gpu_timing_t *gpu_timing){
  int nSeedStripsNC;

  gpu_timer_start(gpu_timing);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

#ifdef GPU_DEBUG
  int *cpu_mask = (int *)malloc(nStrips*sizeof(int));
  int *cpu_prefix= (int *)malloc(nStrips*sizeof(int));
  int *cpu_index = (int *)malloc(nStrips*sizeof(int));
#endif

  int nthreads = 256;
  int nblocks = (nStrips+nthreads-1)/nthreads;

  // mark seed strips
  setSeedStripsGPU<<<nblocks, nthreads>>>(nStrips, noise_d, adc_d, seedStripsMask_d, seedStripsNCMask_d);
  gpu_timing->setSeedStripsTime = gpu_timer_measure(gpu_timing);

  // mark only non-consecutive seed strips (mask out consecutive seed strips)
  setNCSeedStripsGPU<<<nblocks, nthreads>>>(nStrips, stripId_d, seedStripsMask_d, seedStripsNCMask_d);
  gpu_timing->setNCSeedStripsTime = gpu_timer_measure(gpu_timing);

  // set index for non-consecutive seed strips
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, seedStripsNCMask_d, prefixSeedStripsNCMask_d, nStrips);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, seedStripsNCMask_d, prefixSeedStripsNCMask_d, nStrips);

  cudaMemcpy((void *)&nSeedStripsNC, prefixSeedStripsNCMask_d+nStrips-1, sizeof(int), cudaMemcpyDeviceToHost);

  setStripIndexGPU<<<nblocks, nthreads>>>(nStrips, seedStripsNCMask_d, prefixSeedStripsNCMask_d, seedStripsNCIndex_d);

  cudaFree(d_temp_storage);

#ifdef GPU_DEBUG
  cudaMemcpy((void *)cpu_mask, seedStripsNCMask_d, nStrips*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_prefix, prefixSeedStripsNCMask_d, nStrips*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)cpu_index, seedStripsNCIndex_d, nStrips*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<nStrips; i++) {
    std::cout<<" i "<<i<<" mask "<<cpu_mask[i]<<" prefix "<<cpu_prefix[i]<<" index "<<cpu_index[i]<<std::endl;
  }

  free(cpu_mask);
  free(cpu_prefix);
  free(cpu_index);

  std::cout<<"nStrips="<<nStrips<<"nSeedStripsNC="<<nSeedStripsNC<<"temp_storage_bytes="<<temp_storage_bytes<<std::endl;
#endif

  gpu_timing->setStripIndexTime = gpu_timer_measure_end(gpu_timing);

  return nSeedStripsNC;
}

extern "C"
void cpyGPUToCPU(int nSeedStripsNC, int *clusterLastIndexLeft_d, int *clusterLastIndexLeft, int *clusterLastIndexRight_d, int *clusterLastIndexRight, uint8_t *clusterADCs_d, uint8_t *clusterADCs, bool *trueCluster_d, bool *trueCluster) {
  cudaMemcpy((void *)clusterLastIndexLeft, clusterLastIndexLeft_d, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)clusterLastIndexRight, clusterLastIndexRight_d, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)clusterADCs, clusterADCs_d, nSeedStripsNC*256*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)trueCluster, trueCluster_d, nSeedStripsNC*sizeof(bool), cudaMemcpyDeviceToHost);
}

extern "C"
void cpyCPUToGPU(int nStrips, uint16_t *stripId_d, uint16_t *stripId, uint16_t *adc_d,  uint16_t *adc, float *noise_d, float *noise, float *gain_d, float *gain, gpu_timing_t *gpu_timing) {
  gpu_timer_start(gpu_timing);
  cudaMemcpy((void *)stripId_d, stripId, nStrips*sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy((void *)adc_d, adc, nStrips*sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy((void *)noise_d, noise, nStrips*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void *)gain_d, gain, nStrips*sizeof(float), cudaMemcpyHostToDevice);
#if USE_TEXTURE
  cudaBindTexture(0, stripIdTexRef, (void *)stripId_d, nStrips*sizeof(uint16_t));
  cudaBindTexture(0, adcTexRef, (void *)adc_d, nStrips*sizeof(uint16_t));
  cudaBindTexture(0, noiseTexRef, (void *)noise_d, nStrips*sizeof(float));
  cudaBindTexture(0, gainTexRef, (void *)gain_d, nStrips*sizeof(float));
#endif
  gpu_timing->memTransferTime = gpu_timer_measure_end(gpu_timing);
}
