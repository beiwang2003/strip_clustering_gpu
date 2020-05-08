#ifndef FEDRawData_FEDRawData_h
#define FEDRawData_FEDRawData_h

/** \class FEDRawData
 *
 *  Class representing the raw data for one FED.
 *  The raw data is owned as a binary buffer. It is required that the 
 *  lenght of the data is a multiple of the S-Link64 word lenght (8 byte).
 *  The FED data should include the standard FED header and trailer.
 *
 *  \author G. Bruno - CERN, EP Division
 *  \author S. Argiro - CERN and INFN - 
 *                      Refactoring and Modifications to fit into CMSSW
 */

#include <cstddef>
#include <memory>

class FEDRawData {
public:
  typedef std::unique_ptr<unsigned char[]> Data;

  /// Ctor specifying the size to be preallocated, in bytes.
  /// It is required that the size is a multiple of the size of a FED
  /// word (8 bytes)
  FEDRawData(size_t newsize);

  /// Move constructor
  FEDRawData(FEDRawData &&arg)
    : size_(arg.size_), data_(std::move(arg.data_)) {}
  
  /// Dtor
  ~FEDRawData();

  /// Return a const pointer to the beginning of the data buffer
  const unsigned char *get() const { return data_.get(); }

  /// Return a pointer to the beginning of the data buffer
  unsigned char *get() { return data_.get(); }

  /// return ref to underlying unique_ptr
  const Data& data() const { return data_; }

  /// Length of the data buffer in bytes
  size_t size() const { return size_; }

private:
  size_t size_ = 0;
  Data data_;
};

#endif
