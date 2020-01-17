#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>

#ifdef USE_GPU
#include <cuda_runtime.h>
#include "cuda_rt_call.h"
#endif

#include "SiStripConditions.h"

struct FileChannel {
  fedId_t fedId_;
  detId_t  detId_;
  fedCh_t fedCh_;
  APVPair_t ipair_;
  float noise_[ChannelConditions::kStripsPerChannel];
  float gain_[ChannelConditions::kStripsPerChannel];
  bool bad_[ChannelConditions::kStripsPerChannel];
};

SiStripConditions::SiStripConditions(const std::string& file)
{
  std::ifstream datafile(file, std::ios::in | std::ios::binary);

  FileChannel fc;

  while (datafile.read((char*) &fc, sizeof(fc)).gcount() == sizeof(fc)) {
    auto fedi = fc.fedId_ - kFedFirst;
    auto fch = fc.fedCh_;
    detID_[fedi][fch] = fc.detId_;
    iPair_[fedi][fch] = fc.ipair_;
    detToFeds_.emplace_back(fc.detId_, fc.ipair_, fc.fedId_, fc.fedCh_);

    auto choff = fch*kStripsPerChannel;
    for (auto i = 0; i < ChannelConditions::kStripsPerChannel; ++i, ++choff) {
      noise_[fedi][choff] = fc.noise_[i];
      gain_[fedi][choff] = fc.gain_[i];
      bad_[fedi][choff] = fc.bad_[i];
    }
  }
  std::sort(detToFeds_.begin(), detToFeds_.end(),
    [](const DetToFed& a, const DetToFed& b){ 
      return a.detID() < b.detID() || (a.detID() == b.detID() && a.pair() < b.pair());
    });
#ifdef DBGPRINT
  for (const auto& d : detToFeds_) {
    std::cout << d.detID() << ":" << d.pair() << " --> " << d.fedID() << ":" << (uint16_t) d.fedCh() << std::endl;
  }
#endif
}

const ChannelConditions SiStripConditionsBase::operator()(fedId_t fed, fedCh_t channel) const
{
  fed -= kFedFirst;
  const auto detID = detID_[fed][channel];
  const auto pair = iPair_[fed][channel];
  const auto choff = channel*kStripsPerChannel;
  return ChannelConditions(detID, pair, &noise_[fed][choff], &gain_[fed][choff], &bad_[fed][choff]);
}

SiStripConditionsGPU* SiStripConditionsBase::toGPU() const
{
  SiStripConditionsGPU* s = nullptr;
#ifdef USE_GPU
  CUDA_RT_CALL(cudaMalloc((void**) &s, sizeof(SiStripConditionsGPU)));
  CUDA_RT_CALL(cudaMemcpyAsync(s, this, sizeof(SiStripConditionsGPU), cudaMemcpyDefault));
#endif
  return s;
}
