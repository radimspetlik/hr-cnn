/**
 * @date Wed May 11 12:39:37 MDT 2016
 * @author Manuel Gunther <siebenkopf@googlemail.com>
 *
 * @brief The file provides an easy C++ interface to read and write BMP images using our own codec
 *
 * Copyright (c) 2016, Regents of the University of Colorado on behalf of the University of Colorado Colorado Springs.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef BOB_IO_IMAGE_BMP_H
#define BOB_IO_IMAGE_BMP_H

#include <stdexcept>
#include <string>

#include <boost/shared_ptr.hpp>
#include <blitz/array.h>

#include <bob.io.base/File.h>


/**
 * @brief Array submodule API of the I/O module
 */
namespace bob { namespace io { namespace image {

  class BMPFile: public bob::io::base::File {

    public: //api

      BMPFile(const char* path, char mode);

      virtual ~BMPFile() { }

      virtual const char* filename() const {
        return m_filename.c_str();
      }

      virtual const bob::io::base::array::typeinfo& type_all() const {
        return m_type;
      }

      virtual const bob::io::base::array::typeinfo& type() const {
        return m_type;
      }

      virtual size_t size() const {
        return m_length;
      }

      virtual const char* name() const {
        return s_codecname.c_str();
      }

      virtual void read_all(bob::io::base::array::interface& buffer) {
        read(buffer, 0); ///we only have 1 image in an image file anyways
      }

      virtual void read(bob::io::base::array::interface& buffer, size_t index);

      virtual size_t append (const bob::io::base::array::interface& buffer);

      virtual void write (const bob::io::base::array::interface& buffer);

      using bob::io::base::File::write;
      using bob::io::base::File::read;

    private: //representation
      std::string m_filename;
      bool m_newfile;
      bob::io::base::array::typeinfo m_type;
      size_t m_length;

      static std::string s_codecname;

  };

  inline blitz::Array<uint8_t,3> read_bmp(const std::string& filename){
    BMPFile bmp(filename.c_str(), 'r');
    return bmp.read<uint8_t,3>(0);
  }

  inline void write_bmp(const blitz::Array<uint8_t,3>& image, const std::string& filename){
    BMPFile bmp(filename.c_str(), 'w');
    bmp.write(image);
  }

}}}

#endif /* BOB_IO_IMAGE_BMP_H */
