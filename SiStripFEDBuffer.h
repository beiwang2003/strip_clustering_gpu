#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <cstring>

#include "FEDChannel.h"

static const uint8_t INVALID=0xFF;

static const uint16_t FEDCH_PER_FEUNIT = 12;
static const uint16_t FEUNITS_PER_FED = 8;
static const uint16_t FEDCH_PER_FED = FEDCH_PER_FEUNIT * FEUNITS_PER_FED;  // 96

//these are the values which appear in the buffer.
static const uint8_t BUFFER_FORMAT_CODE_OLD = 0xED;
static const uint8_t BUFFER_FORMAT_CODE_NEW = 0xC5;

enum FEDReadoutMode { READOUT_MODE_INVALID=INVALID,
                      READOUT_MODE_SCOPE=0x1,
                      READOUT_MODE_VIRGIN_RAW=0x2,
                      READOUT_MODE_ZERO_SUPPRESSED_LITE10=0x3,
                      READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE=0x4,
                      READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT=0x5,
                      READOUT_MODE_PROC_RAW=0x6,
                      READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT_CMOVERRIDE=0x7,
                      READOUT_MODE_ZERO_SUPPRESSED_LITE8_CMOVERRIDE=0x8,
                      READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT=0x9,
                      READOUT_MODE_ZERO_SUPPRESSED=0xA,
                      READOUT_MODE_ZERO_SUPPRESSED_FAKE=0xB,
                      READOUT_MODE_ZERO_SUPPRESSED_LITE8=0xC,
                      READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE=0xD,
                      READOUT_MODE_SPY=0xE,
                      READOUT_MODE_PREMIX_RAW=0xF
                    };

class TrackerSpecialHeader
{
public:
  TrackerSpecialHeader() {}
  //construct with a pointer to the data. The data will be coppied and swapped if necessary. 
  explicit TrackerSpecialHeader(const uint8_t* headerPointer);
  FEDReadoutMode readoutMode() const { return FEDReadoutMode(specialHeader_[BUFFERTYPE] & 0x0F); }
  bool feEnabled(const uint8_t internalFEUnitNum) const { return (0x1<<internalFEUnitNum) & feEnableRegister(); }
  uint8_t feEnableRegister() const { return specialHeader_[FEENABLE]; }

private:
  enum byteIndicies { FEDSTATUS=0, FEOVERFLOW=2, FEENABLE=3, ADDRESSERROR=4, APVEADDRESS=5, BUFFERTYPE=6, BUFFERFORMAT=7 };
  //copy of header, 32 bit word swapped if needed
  uint8_t specialHeader_[8];
  //was the header word swapped wrt order in buffer?
  bool wordSwapped_;
};

class FEDFullDebugHeader
{
public:
  size_t lengthInBytes() const { return FULL_DEBUG_HEADER_SIZE_IN_BYTES; }
  explicit FEDFullDebugHeader(const uint8_t* headerBuffer) { memcpy(header_,headerBuffer,FULL_DEBUG_HEADER_SIZE_IN_BYTES); }
  bool fePresent(const uint8_t internalFEUnitNum) const { return (feUnitLength(internalFEUnitNum) != 0); }
  uint16_t feUnitLength(const uint8_t internalFEUnitNum) const { return (feWord(internalFEUnitNum)[15]<<8) | (feWord(internalFEUnitNum)[14]); }
  const uint8_t* feWord(const uint8_t internalFEUnitNum) const { return header_+internalFEUnitNum*2*8; }
private:
  static const size_t FULL_DEBUG_HEADER_SIZE_IN_64BIT_WORDS = FEUNITS_PER_FED*2;
  static const size_t FULL_DEBUG_HEADER_SIZE_IN_BYTES = FULL_DEBUG_HEADER_SIZE_IN_64BIT_WORDS*8;
  uint8_t header_[FULL_DEBUG_HEADER_SIZE_IN_BYTES];
};

class FEDBuffer
{
public:
  //construct from buffer
  FEDBuffer(const uint8_t* fedBuffer, const uint16_t fedBufferSize, const bool allowBadBuffer = false);
  ~FEDBuffer() {}

  // move constructor
  FEDBuffer(FEDBuffer&& arg);

  const uint8_t* data() const { return orderedBuffer_; }
  const uint8_t* getPointerToDataAfterTrackerSpecialHeader() const { return orderedBuffer_ + 16; }
  const uint8_t* getPointerToByteAfterEndOfPayload() const { return orderedBuffer_+bufferSize_-8; }

  const FEDChannel& channel(const uint8_t internalFEDChannelNum) const { return channels_[internalFEDChannelNum]; }
  const std::vector<FEDChannel>& channels() const { return channels_; }
  uint8_t validChannels() const { return validChannels_; }
  size_t bufferSize() const { return bufferSize_; }
  FEDReadoutMode readoutMode() const { return specialHeader_.readoutMode(); }

  bool fePresent(uint8_t internalFEUnitNum) const { return fePresent_[internalFEUnitNum]; }
  bool feEnabled(const uint8_t internalFEUnitNum) const { return specialHeader_.feEnabled(internalFEUnitNum); }

private:
  void findChannels();
  std::unique_ptr<FEDFullDebugHeader> feHeader_;
  const uint8_t* payloadPointer_;
  uint16_t payloadLength_;
  std::vector<FEDChannel> channels_;
  const uint8_t* orderedBuffer_;
  const size_t bufferSize_;
  uint8_t validChannels_;
  TrackerSpecialHeader specialHeader_;
  std::array<bool, FEUNITS_PER_FED> fePresent_;
};
