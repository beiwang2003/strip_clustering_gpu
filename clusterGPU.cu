#include "clusterGPU.cuh"
#include <cub/cub.cuh>
#include <stdio.h>

#ifdef USE_TEXTURE
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
static void findLeftRightBoundaryGPU(int nSeedStripsNC, int nStrips, float* clusterNoiseSquared_d, int *clusterLastIndexLeft_d, int *clusterLastIndexRight_d, int *seedStripsNCIndex_d, uint16_t *stripId_d, uint16_t *adc_d, float *noise_d)
{
   const uint8_t MaxSequentialHoles = 0;
   const float  ChannelThreshold = 2.0;
   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   const int nthreads = blockDim.x;

   int index, testIndexLeft, testIndexRight, indexLeft, indexRight, rangeLeft, rangeRight;
   uint8_t testADC;
   float noise_i, testNoise;

   int i = nthreads * bid + tid;

   if (i<nSeedStripsNC) {

     clusterNoiseSquared_d[i] = 0.0;
     index=seedStripsNCIndex_d[i];
     indexLeft = index;
     indexRight = index;
     noise_i = NOISE(index);
     //noise_i = noise_d[index];
     clusterNoiseSquared_d[i] += noise_i*noise_i;

     // find left boundary
     testIndexLeft=index-1;
     rangeLeft = STRIPID(indexLeft)-STRIPID(testIndexLeft)-1;
     //rangeLeft = stripId_d[indexLeft]-stripId_d[testIndexLeft]-1;
     while(testIndexLeft>=0&&rangeLeft>=0&&rangeLeft<=MaxSequentialHoles) {
       //testNoise = NOISE(testIndexLeft);
       //testADC = static_cast<uint8_t>(ADC(testIndexLeft));
       testNoise = noise_d[testIndexLeft];
       testADC = static_cast<uint8_t>(adc_d[testIndexLeft]);
       if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	 --indexLeft;
	 clusterNoiseSquared_d[i] += testNoise*testNoise;
       }
       --testIndexLeft;
       rangeLeft = STRIPID(indexLeft)-STRIPID(testIndexLeft)-1;
       //rangeLeft = stripId_d[indexLeft]-stripId_d[testIndexLeft]-1;
     }

     // find right boundary
     testIndexRight=index+1;
     rangeRight = STRIPID(testIndexRight)-STRIPID(indexRight)-1;
     //rangeRight = stripId_d[testIndexRight]-stripId_d[indexRight]-1;
     while(testIndexRight<nStrips&&rangeRight>=0&&rangeRight<=MaxSequentialHoles) {
       //testNoise = NOISE(testIndexRight);
       //testADC = static_cast<uint8_t>(ADC(testIndexRight));
       testNoise = noise_d[testIndexRight];
       testADC = static_cast<uint8_t>(adc_d[testIndexRight]);
       if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	 ++indexRight;
	 clusterNoiseSquared_d[i] += testNoise*testNoise;
       }
       ++testIndexRight;
       rangeRight = STRIPID(testIndexRight)-STRIPID(indexRight)-1;
       //rangeRight= stripId_d[testIndexRight]-stripId_d[indexRight]-1;
     }

     clusterLastIndexLeft_d[i] = indexLeft;
     clusterLastIndexRight_d[i] = indexRight;
   }
}

__global__
static void checkClusterConditionGPU(int nSeedStripsNC, float* clusterNoiseSquared_d, int *clusterLastIndexLeft_d, int *clusterLastIndexRight_d, uint16_t *adc_d, float * gain_d, bool *trueCluster_d, uint8_t *clusterADCs_d)
{
   const float minGoodCharge = 1620.0;
   const float ClusterThresholdSquared = 25.0;

   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   const int nthreads = blockDim.x;

   int i = nthreads * bid + tid;

   int left, right, size, j;
   int adcsum = 0, charge;
   bool noiseSquaredPass, chargePerCMPass;
   uint8_t adc_j;
   float gain_j;

   if (i<nSeedStripsNC) {

    trueCluster_d[i] = false;
    left=clusterLastIndexLeft_d[i];
    right=clusterLastIndexRight_d[i];
    size=right-left+1;
    for (j=0; j<size; j++) {
      adcsum += (int)ADC(left+j);
    }
    noiseSquaredPass = clusterNoiseSquared_d[i]*ClusterThresholdSquared <= ((float)(adcsum)*float(adcsum));
    chargePerCMPass = (float)(adcsum)/0.047f > minGoodCharge;
    if (noiseSquaredPass&&chargePerCMPass) {
      for (j=0; j<size; j++){
        adc_j = ADC(left+j);
        gain_j = GAIN(left+j);
        charge = int( float(adc_j)/gain_j + 0.5f );
        if (adc_j < 254) adc_j = ( charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
        clusterADCs_d[j*nSeedStripsNC+i] = adc_j;
      }
      trueCluster_d[i] = true;
    }
  }
}

extern "C"
void allocateMemAllStripsGPU(int max_strips, uint16_t **stripId_d_pt, uint16_t **adc_d_pt, float **noise_d_pt, float **gain_d_pt, int **seedStripsNCIndex_d_pt)
{
  cudaMalloc(stripId_d_pt, max_strips*sizeof(uint16_t));
  cudaMalloc(adc_d_pt, max_strips*sizeof(uint16_t));
  cudaMalloc(noise_d_pt, max_strips*sizeof(float));
  cudaMalloc(gain_d_pt, max_strips*sizeof(float));
  cudaMalloc(seedStripsNCIndex_d_pt, max_strips*sizeof(int));
}

extern "C"
void allocateMemNCSeedStripsGPU(int nSeedStripsNC, int **clusterLastIndexLeft_d_pt, int **clusterLastIndexRight_d_pt, float **clusterNoiseSquared_d_pt, uint8_t **clusterADCs_d_pt, bool **trueCluster_d_pt)
{
  cudaMalloc(clusterLastIndexLeft_d_pt, 2*nSeedStripsNC*sizeof(int));
  *clusterLastIndexRight_d_pt = *clusterLastIndexLeft_d_pt + nSeedStripsNC;
  cudaMalloc(clusterNoiseSquared_d_pt, nSeedStripsNC*sizeof(float));
  cudaMalloc(clusterADCs_d_pt, nSeedStripsNC*256*sizeof(uint8_t));
  cudaMalloc(trueCluster_d_pt, nSeedStripsNC*sizeof(bool));
}

extern "C"
void freeGPUMem(uint16_t *stripId_d, uint16_t *adc_d, float *noise_d, float *gain_d, int *seedStripNCIndex_d, int *clusterLastIndexLeft_d, float *clusterNoiseSquared_d, uint8_t *clusterADCs_d, bool *trueCluster_d)
{
   cudaFree(stripId_d);
   cudaFree(adc_d);
   cudaFree(noise_d);
   cudaFree(gain_d);
#ifdef USE_TEXTURE
   cudaUnbindTexture(stripIdTexRef);
   cudaUnbindTexture(adcTexRef);
   cudaUnbindTexture(noiseTexRef);
   cudaUnbindTexture(gainTexRef);
#endif
   cudaFree(seedStripNCIndex_d);
   cudaFree(clusterLastIndexLeft_d);
   cudaFree(clusterNoiseSquared_d);
   cudaFree(clusterADCs_d);
   cudaFree(trueCluster_d);
}

extern "C"
void  findClusterGPU(int nSeedStripsNC, int nStrips, float *clusterNoiseSquared_d, int *clusterLastIndexLeft_d,  int *clusterLastIndexRight_d, int *seedStripsNCIndex_d, uint16_t *stripId_d, uint16_t *adc_d, float *noise_d, float *gain_d, bool *trueCluster_d, uint8_t *clusterADCs_d, gpu_timing_t *gpu_timing)
{
  gpu_timer_start(gpu_timing);
  int nthreads = 256;
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

  int *clusterL_d, *clusterR_d;
  float *clusterNoiseSq_d;
  bool *trueC_d;
  uint8_t *cADCs_d;
  cudaMalloc(&clusterL_d, nSeedStripsNC*sizeof(int));
  cudaMalloc(&clusterR_d, nSeedStripsNC*sizeof(int));
  cudaMalloc(&clusterNoiseSq_d, nSeedStripsNC*sizeof(float));
  cudaMalloc(&trueC_d, nSeedStripsNC*sizeof(bool));
  cudaMalloc(&cADCs_d, nSeedStripsNC*256*sizeof(uint8_t));
#endif

  findLeftRightBoundaryGPU<<<nblocks, nthreads>>>(nSeedStripsNC, nStrips, clusterNoiseSquared_d, clusterLastIndexLeft_d, clusterLastIndexRight_d, seedStripsNCIndex_d, stripId_d, adc_d, noise_d);
  //findLeftRightBoundaryGPU<<<nblocks, nthreads>>>(nSeedStripsNC, nStrips, clusterNoiseSq_d, clusterL_d, clusterR_d, seedStripsNCIndex_d, stripId_d, adc_d, noise_d);

  gpu_timing->findBoundaryTime = gpu_timer_measure(gpu_timing);

  //cudaDeviceSynchronize();

  checkClusterConditionGPU<<<nblocks, nthreads>>>(nSeedStripsNC, clusterNoiseSquared_d, clusterLastIndexLeft_d, clusterLastIndexRight_d, adc_d, gain_d, trueCluster_d, clusterADCs_d);
  //checkClusterConditionGPU<<<nblocks, nthreads>>>(nSeedStripsNC, clusterNoiseSq_d, clusterL_d, clusterR_d, adc_d, gain_d, trueC_d, cADCs_d);

  //cudaDeviceSynchronize();
  gpu_timing->checkClusterTime = gpu_timer_measure_end(gpu_timing);

#ifdef GPU_DEBUG
  int *clusterLastIndexLeft = (int *)malloc(nSeedStripsNC*sizeof(int));
  int *clusterLastIndexRight = (int *)malloc(nSeedStripsNC*sizeof(int));
  bool *trueCluster = (bool *)malloc(nSeedStripsNC*sizeof(bool));
  uint8_t *ADCs = (uint8_t*)malloc(nSeedStripsNC*256*sizeof(uint8_t));


  cudaMemcpy((void *)clusterLastIndexLeft, clusterL_d, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)clusterLastIndexRight, clusterR_d, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)trueCluster, trueC_d, nSeedStripsNC*sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)ADCs, cADCs_d, nSeedStripsNC*256*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)clusterLastIndexLeft, clusterLastIndexLeft_d, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)clusterLastIndexRight, clusterLastIndexRight_d, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)trueCluster, trueCluster_d, nSeedStripsNC*sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)ADCs, clusterADCs_d, nSeedStripsNC*256*sizeof(uint8_t), cudaMemcpyDeviceToHost);

  /*
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
  */

  free(cpu_index);
  free(cpu_strip);
  free(cpu_adc);
  free(cpu_noise);

  free(clusterLastIndexLeft);
  free(clusterLastIndexRight);
  free(trueCluster);
  free(ADCs);

  cudaFree(clusterL_d);
  cudaFree(clusterR_d);
  cudaFree(clusterNoiseSq_d);
  cudaFree(trueC_d);
  cudaFree(cADCs_d);
#endif

}

extern "C"
int setSeedStripsNCIndexGPU(int nStrips, uint16_t *stripId_d, uint16_t *adc_d, float *noise_d, int *seedStripsNCIndex_d, gpu_timing_t *gpu_timing){
  int nSeedStripsNC;

  gpu_timer_start(gpu_timing);

  int *seedStripsMask_d, *seedStripsNCMask_d, *prefixSeedStripsNCMask_d;
  cudaMalloc((void **)&seedStripsMask_d, 3*nStrips*sizeof(int));
  seedStripsNCMask_d = seedStripsMask_d + nStrips;
  prefixSeedStripsNCMask_d = seedStripsMask_d + 2*nStrips;

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

  cudaFree(seedStripsMask_d);

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
#ifdef USE_TEXTURE
  cudaBindTexture(0, stripIdTexRef, (void *)stripId_d, nStrips*sizeof(uint16_t));
  cudaBindTexture(0, adcTexRef, (void *)adc_d, nStrips*sizeof(uint16_t));
  cudaBindTexture(0, noiseTexRef, (void *)noise_d, nStrips*sizeof(float));
  cudaBindTexture(0, gainTexRef, (void *)gain_d, nStrips*sizeof(float));
#endif
  gpu_timing->memTransferTime = gpu_timer_measure_end(gpu_timing);
}
