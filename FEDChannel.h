#pragma once

#include <memory>

#include "host_unique_ptr.h"
#include "device_unique_ptr.h"

#include "SiStripConditions.h"

class ChannelLocsGPU;

template<template <typename> class T>
class ChannelLocsBase {
public:
  ChannelLocsBase(size_t size) : size_(size) {}
  ~ChannelLocsBase() {}

  ChannelLocsBase(ChannelLocsBase&& arg)
    : input_(std::move(arg.input_)),
      inoff_(std::move(arg.inoff_)),
      offset_(std::move(arg.offset_)),
      length_(std::move(arg.length_)),
      fedID_(std::move(arg.fedID_)),
      fedCh_(std::move(arg.fedCh_)),
      size_(arg.size_) {}

  void setChannelLoc(uint32_t index, const uint8_t* input, size_t inoff, size_t offset, uint16_t length, fedId_t fedID, fedCh_t fedCh)
  {
    input_[index] = input;
    inoff_[index] = inoff;
    offset_[index] = offset;
    length_[index] = length;
    fedID_[index] = fedID;
    fedCh_[index] = fedCh;
  }

  __host__ __device__ size_t size() const { return size_; }

  __host__ __device__ const uint8_t* input(uint32_t index) const { return input_[index]; }
  __host__ __device__ size_t inoff(uint32_t index) const { return inoff_[index]; }
  __host__ __device__ size_t offset(uint32_t index) const { return offset_[index]; }
  __host__ __device__ uint16_t length(uint32_t index) const { return length_[index]; }
  __host__ __device__ fedId_t fedID(uint32_t index) const { return fedID_[index]; }
  __host__ __device__ fedCh_t fedCh(uint32_t index) const { return fedCh_[index]; }

  const uint8_t** input() const { return input_.get(); }
  size_t* inoff() const { return inoff_.get(); }
  size_t* offset() const { return offset_.get(); }
  uint16_t* length() const { return length_.get(); }
  fedId_t* fedID() const { return fedID_.get(); }
  fedCh_t* fedCh() const { return fedCh_.get(); }

protected:
  T<const uint8_t*[]> input_; // input raw data for channel
  T<size_t[]> inoff_;         // offset in input raw data
  T<size_t[]> offset_;        // global offset in alldata
  T<uint16_t[]> length_;      // length of channel data
  T<fedId_t[]> fedID_;
  T<fedCh_t[]> fedCh_;
  size_t size_ = 0;
};

class ChannelLocs : public ChannelLocsBase<cudautils::host::unique_ptr> {
  friend class ChannelLocsGPU;
public:
  ChannelLocs(size_t size, cudaStream_t stream);
  ChannelLocs(ChannelLocs&& arg) : ChannelLocsBase(std::move(arg)) {}
  ~ChannelLocs();
};

struct ChanLocStruct {
  void Fill(const ChannelLocsGPU& c);

  __host__ __device__ size_t size() const { return size_; }

  __host__ __device__ const uint8_t* input(uint32_t index) const { return input_[index]; }
  __host__ __device__ size_t inoff(uint32_t index) const { return inoff_[index]; }
  __host__ __device__ size_t offset(uint32_t index) const { return offset_[index]; }
  __host__ __device__ uint16_t length(uint32_t index) const { return length_[index]; }
  __host__ __device__ fedId_t fedID(uint32_t index) const { return fedID_[index]; }
  __host__ __device__ fedCh_t fedCh(uint32_t index) const { return fedCh_[index]; }


  const uint8_t** input_; // input raw data for channel
  size_t* inoff_;         // offset in input raw data
  size_t* offset_;        // global offset in alldata
  uint16_t* length_;      // length of channel data
  fedId_t* fedID_;
  fedCh_t* fedCh_;
  size_t size_;
};

class ChannelLocsGPU : public ChannelLocsBase<cudautils::device::unique_ptr> {
public:
  //using Base = ChannelLocsBase<cudautils::device::unique_ptr>;
  ChannelLocsGPU(size_t size, cudaStream_t stream);
  ChannelLocsGPU(ChannelLocsGPU&& arg)
    : ChannelLocsBase(std::move(arg)), chanstruct_(std::move(arg.chanstruct_)) {}
  ~ChannelLocsGPU();
  void reset(const ChannelLocs&, const std::vector<uint8_t*>& inputGPU, cudaStream_t stream);
  const ChanLocStruct* chanLocStruct() const { return chanstruct_.get(); }
private:
  cudautils::device::unique_ptr<ChanLocStruct> chanstruct_;
};

//holds information about position of a channel in the buffer for use by unpacker
class FEDChannel {
public:
  FEDChannel(const uint8_t* const data, const size_t offset, const uint16_t length);
  //gets length from first 2 bytes (assuming normal FED channel)
  FEDChannel(const uint8_t* const data, const size_t offset);
  uint16_t length() const { return length_; }
  const uint8_t* data() const { return data_; }
  size_t offset() const { return offset_; }
  uint16_t cmMedian(const uint8_t apvIndex) const;
  //third byte of channel data for normal FED channels
  uint8_t packetCode() const;

private:
  friend class FEDBuffer;
  const uint8_t* data_;
  size_t offset_;
  uint16_t length_;
};

inline FEDChannel::FEDChannel(const uint8_t* const data, const size_t offset) : data_(data), offset_(offset) {
  length_ = (data_[(offset_) ^ 7] + (data_[(offset_ + 1) ^ 7] << 8));
}

inline FEDChannel::FEDChannel(const uint8_t*const data, const size_t offset, const uint16_t length)
  : data_(data),
    offset_(offset),
    length_(length)
{
}
