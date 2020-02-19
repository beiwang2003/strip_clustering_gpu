#include <cassert>

#ifdef USE_GPU
#include <cuda_runtime.h>
#include "cudaCheck.h"
#include "copyAsync.h"
#endif

#include "FEDChannel.h"

ChannelLocs::ChannelLocs(size_t size, cudaStream_t stream)
  : ChannelLocsBase(size)
{
  if (size > 0) {
    input_ = cudautils::make_host_unique<const uint8_t*[]>(size, stream);
    inoff_ = cudautils::make_host_unique<size_t[]>(size, stream);
    offset_ = cudautils::make_host_unique<size_t[]>(size, stream);
    length_ = cudautils::make_host_unique<uint16_t[]>(size, stream);
    fedID_ = cudautils::make_host_unique<fedId_t[]>(size, stream);
    fedCh_ = cudautils::make_host_unique<fedCh_t[]>(size, stream);
  }
}

ChannelLocs::~ChannelLocs() {}

void ChanLocStruct::Fill(const ChannelLocsGPU& c)
{
  input_ = c.input();
  inoff_ = c.inoff();
  offset_ = c.offset();
  length_ = c.length();
  fedID_ = c.fedID();
  fedCh_ = c.fedCh();
  size_ = c.size();
}

ChannelLocsGPU::ChannelLocsGPU(size_t size, cudaStream_t stream)
  : ChannelLocsBase(size)
{
#ifdef USE_GPU
  if (size > 0) {
    input_ = cudautils::make_device_unique<const uint8_t*[]>(size, stream);
    inoff_ = cudautils::make_device_unique<size_t[]>(size, stream);
    offset_ = cudautils::make_device_unique<size_t[]>(size, stream);
    length_ = cudautils::make_device_unique<uint16_t[]>(size, stream);
    fedID_ = cudautils::make_device_unique<fedId_t[]>(size, stream);
    fedCh_ = cudautils::make_device_unique<fedCh_t[]>(size, stream);

    ChanLocStruct chanstruct;
    chanstruct.Fill(*this);
    chanstruct_ = cudautils::make_device_unique<ChanLocStruct>(stream);
    cudaCheck(cudaMemcpyAsync(chanstruct_.get(), &chanstruct, sizeof(ChanLocStruct), cudaMemcpyDefault, stream));
  }
#endif
}

void ChannelLocsGPU::reset(const ChannelLocs& c, const std::vector<uint8_t*>& inputGPU, cudaStream_t stream)
{
#ifdef USE_GPU
  assert(c.size() == size_);
  cudaCheck(cudaMemcpyAsync(input_.get(), inputGPU.data(), sizeof(uint8_t*)*size_, cudaMemcpyDefault, stream));
  cudautils::copyAsync(inoff_, c.inoff_, size_, stream);
  cudautils::copyAsync(offset_, c.offset_, size_, stream);
  cudautils::copyAsync(length_, c.length_, size_, stream);
  cudautils::copyAsync(fedID_, c.fedID_, size_, stream);
  cudautils::copyAsync(fedCh_, c.fedCh_, size_, stream);
#endif
}

ChannelLocsGPU::~ChannelLocsGPU()
{
}
