/** \file
   implementation of class FedRawData

   \author Stefano ARGIRO
   \date 28 Jun 2005
*/

#include <cassert>

#include "FEDRawData.h"

using namespace std;

FEDRawData::FEDRawData(size_t size)
  : size_(size) {
  assert(size_ % 8 == 0);
  data_ = std::make_unique<unsigned char[]>(size_);
}

FEDRawData::~FEDRawData() {}
