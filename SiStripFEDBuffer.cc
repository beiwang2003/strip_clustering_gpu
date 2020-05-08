#include <iostream>

#include "SiStripFEDBuffer.h"

TrackerSpecialHeader::TrackerSpecialHeader(const uint8_t* headerPointer)
{
  //the buffer format byte is one of the valid values if we assume the buffer is not swapped
  const bool validFormatByteWhenNotWordSwapped = ( (headerPointer[BUFFERFORMAT] == BUFFER_FORMAT_CODE_NEW) ||
					     (headerPointer[BUFFERFORMAT] == BUFFER_FORMAT_CODE_OLD) );
  //the buffer format byte is the old value if we assume the buffer is swapped
  const bool validFormatByteWhenWordSwapped = (headerPointer[BUFFERFORMAT^4] == BUFFER_FORMAT_CODE_OLD);
  //if the buffer format byte is valid if the buffer is not swapped or it is never valid
  if (validFormatByteWhenNotWordSwapped || (!validFormatByteWhenNotWordSwapped && !validFormatByteWhenWordSwapped) ) {
    memcpy(specialHeader_,headerPointer,8);
    wordSwapped_ = false;
  } else {
    memcpy(specialHeader_,headerPointer+4,4);
    memcpy(specialHeader_+4,headerPointer,4);
    wordSwapped_ = true;
  }
}

FEDBuffer::FEDBuffer(const uint8_t* fedBuffer, const uint16_t fedBufferSize, const bool allowBadBuffer)
  : orderedBuffer_(fedBuffer),bufferSize_(fedBufferSize)
{
  channels_.reserve(FEDCH_PER_FED);

  //build the correct type of FE header object
  feHeader_ = std::make_unique<FEDFullDebugHeader>(getPointerToDataAfterTrackerSpecialHeader());
  payloadPointer_ = getPointerToDataAfterTrackerSpecialHeader()+feHeader_->lengthInBytes();
  payloadLength_ = getPointerToByteAfterEndOfPayload()-payloadPointer_;

  specialHeader_ = TrackerSpecialHeader(orderedBuffer_+8);

  const FEDFullDebugHeader* fdHeader = dynamic_cast<FEDFullDebugHeader*>(feHeader_.get());
  if (fdHeader) {
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (fdHeader->fePresent(iFE)) fePresent_[iFE] = true;
      else fePresent_[iFE] = false;
    }
  }

  //try to find channels
  validChannels_ = 0;
  findChannels();
}

FEDBuffer::FEDBuffer(FEDBuffer&& arg)
  : feHeader_(std::move(arg.feHeader_)),
    payloadPointer_(arg.payloadPointer_),
    payloadLength_(arg.payloadLength_),
    channels_(std::move(arg.channels_)),
    orderedBuffer_(arg.orderedBuffer_),
    bufferSize_(arg.bufferSize_),
    validChannels_(arg.validChannels_),
    specialHeader_(arg.specialHeader_),
    fePresent_(arg.fePresent_) {}

void FEDBuffer::findChannels()
{
  uint16_t offsetBeginningOfChannel = 0;
  for (uint16_t i = 0; i < FEDCH_PER_FED; i++) {
    //if FE unit is not enabled then skip rest of FE unit adding NULL pointers
    if ( !(fePresent(i/FEDCH_PER_FEUNIT) && feEnabled(i/FEDCH_PER_FEUNIT)) ) {
      channels_.insert(channels_.end(),uint16_t(FEDCH_PER_FEUNIT),FEDChannel(payloadPointer_,0,0));
      i += FEDCH_PER_FEUNIT-1;
      validChannels_ += FEDCH_PER_FEUNIT;
      continue;
    }

    channels_.push_back(FEDChannel(payloadPointer_,offsetBeginningOfChannel));
    //get length and check that whole channel fits into buffer
    uint16_t channelLength = channels_.back().length();

    validChannels_++;
    const uint16_t offsetEndOfChannel = offsetBeginningOfChannel+channelLength;
    //add padding if necessary and calculate offset for begining of next channel
    if (!( (i+1) % FEDCH_PER_FEUNIT )) {
      uint8_t numPaddingBytes = 8 - (offsetEndOfChannel % 8);
      if (numPaddingBytes == 8) numPaddingBytes = 0;
      offsetBeginningOfChannel = offsetEndOfChannel + numPaddingBytes;
    } else {
      offsetBeginningOfChannel = offsetEndOfChannel;
    }
  }
}
