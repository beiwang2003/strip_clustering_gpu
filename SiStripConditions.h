#pragma once

#include <limits>
#include <cstdint>
#include <string>
#include <vector>

#include "cudaCompat.h"

using detId_t = uint32_t;
using fedId_t = uint16_t;
using fedCh_t = uint8_t;
using APVPair_t = uint16_t;
using stripId_t = uint16_t;

static constexpr detId_t invDet = std::numeric_limits<detId_t>::max();
static constexpr fedId_t invFed = std::numeric_limits<fedId_t>::max();
static constexpr stripId_t invStrip = std::numeric_limits<stripId_t>::max();

class DetToFed {
public:
  DetToFed(detId_t detid, APVPair_t ipair, fedId_t fedid, fedCh_t fedch)
    : detid_(detid), ipair_(ipair), fedid_(fedid), fedch_(fedch) {}
  detId_t detID() const { return detid_; }
  APVPair_t pair() const { return ipair_; }
  fedId_t fedID() const { return fedid_; }
  fedCh_t fedCh() const { return fedch_; }
private:
  detId_t detid_;
  APVPair_t ipair_;
  fedId_t fedid_;
  fedCh_t fedch_;
};
using DetToFeds = std::vector<DetToFed>;

class ChannelConditions {
public:
  static constexpr int kStripsPerChannel = 256;
  
  ChannelConditions(detId_t det, uint16_t pair, const float* noise, const float* gain, const bool* bad)
    : detId_(det), ipair_(pair), noise_(noise), gain_(gain), bad_(bad) {}

  detId_t detID() const { return detId_; }
  APVPair_t iPair() const { return ipair_; }
  float noise(int strip) const { return noise_[strip-ipair_*kStripsPerChannel]; }
  float gain(int strip) const { return gain_[strip-ipair_*kStripsPerChannel]; }
  bool bad(int strip) const { return bad_[strip-ipair_*kStripsPerChannel]; }

private:
  detId_t detId_ = invDet;
  APVPair_t ipair_;
  const float* noise_;
  const float* gain_;
  const bool* bad_;
};

class SiStripConditionsGPU;

class SiStripConditionsBase {
public:
  static constexpr int kStripsPerChannel = ChannelConditions::kStripsPerChannel;
  static constexpr int kFedFirst = 50;
  static constexpr int kFedLast = 489;
  static constexpr int kFedCount = kFedLast - kFedFirst + 1;
  static constexpr int kChannelCount = 96;

  SiStripConditionsBase() {}

  const ChannelConditions operator()(fedId_t fed, fedCh_t channel) const;
  SiStripConditionsGPU* toGPU() const;

  __host__ __device__ detId_t detID(fedId_t fed, fedCh_t channel) const { return detID_[fed-kFedFirst][channel]; }
  __host__ __device__ APVPair_t iPair(fedId_t fed, fedCh_t channel) const { return iPair_[fed-kFedFirst][channel]; }

  __host__ __device__ float noise(fedId_t fed, fedCh_t channel, stripId_t strip) const
    { return noise_[fed-kFedFirst][channel*kStripsPerChannel + (strip % kStripsPerChannel)]; }

  __host__ __device__ float gain(fedId_t fed, fedCh_t channel, stripId_t strip) const
    { return gain_[fed-kFedFirst][channel*kStripsPerChannel + (strip % kStripsPerChannel)]; }

  __host__ __device__ bool bad(fedId_t fed, fedCh_t channel, stripId_t strip) const
    { return bad_[fed-kFedFirst][channel*kStripsPerChannel + (strip % kStripsPerChannel)]; }

protected:
  float noise_[kFedCount][kChannelCount*kStripsPerChannel];
  float gain_[kFedCount][kChannelCount*kStripsPerChannel];
  bool bad_[kFedCount][kChannelCount*kStripsPerChannel];
  detId_t detID_[kFedCount][kChannelCount];
  APVPair_t iPair_[kFedCount][kChannelCount];
};

class SiStripConditions : public SiStripConditionsBase {
public:
  SiStripConditions(const std::string& file);
  const DetToFeds& detToFeds() const { return detToFeds_; }
private:
  DetToFeds detToFeds_;
};

class SiStripConditionsGPU : public SiStripConditionsBase {
};
