/**
 * @date Wed May 25 13:05:58 MDT 2016
 * @author Manuel Gunther <siebenkopf@googlemail.com>
 *
 * @brief The file provides a generic interface to read any kind of images that we support
 *
 * Copyright (c) 2016, Regents of the University of Colorado on behalf of the University of Colorado Colorado Springs.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef BOB_IO_IMAGE_IMAGE_H
#define BOB_IO_IMAGE_IMAGE_H

#include <bob.io.image/bmp.h>
#include <bob.io.image/png.h>
#include <bob.io.image/gif.h>
#include <bob.io.image/jpeg.h>
#include <bob.io.image/netpbm.h>
#include <bob.io.image/tiff.h>
#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <algorithm>

namespace bob { namespace io { namespace image {

const std::string& get_correct_image_extension(const std::string& image_name);

bool is_color_image(const std::string& filename, std::string extension="");

inline blitz::Array<uint8_t,3> read_color_image(const std::string& filename, std::string extension=""){
  if (extension.empty())
    extension = boost::filesystem::path(filename).extension().string();
  boost::algorithm::to_lower(extension);
  if (extension == ".bmp") return read_bmp(filename);
#ifdef HAVE_GIFLIB
  if (extension == ".gif") return read_gif(filename);
#endif
#ifdef HAVE_LIBPNG
  if (extension == ".png") return read_png<uint8_t,3>(filename);
#endif
#ifdef HAVE_LIBJPEG
  if (extension == ".jpg" || extension == ".jpeg") return read_jpeg<3>(filename);
#endif
#ifdef HAVE_LIBTIFF
  if (extension == ".tif" || extension == ".tiff") return read_tiff<uint8_t,3>(filename);
#endif
  if (extension == ".ppm") return read_ppm<uint8_t>(filename);

  throw std::runtime_error("The filename extension '" + extension + "' is not known or not supported for color images");
}

inline blitz::Array<uint8_t,2> read_gray_image(const std::string& filename, std::string extension=""){
  if (extension.empty())
    extension = boost::filesystem::path(filename).extension().string();
  boost::algorithm::to_lower(extension);
#ifdef HAVE_LIBPNG
  if (extension == ".png") return read_png<uint8_t,2>(filename);
#endif
#ifdef HAVE_LIBJPEG
  if (extension == ".jpg" || extension == ".jpeg") return read_jpeg<2>(filename); // this will only work for T=uint8_t
#endif
#ifdef HAVE_LIBTIFF
  if (extension == ".tif" || extension == ".tiff") return read_tiff<uint8_t,2>(filename);
#endif
  if (extension == ".pgm") return read_pgm<uint8_t>(filename);
  if (extension == ".pbm") return read_pbm<uint8_t>(filename);

  throw std::runtime_error("The filename extension '" + extension + "' is not known or not supported for gray images");
}


inline void write_color_image(const blitz::Array<uint8_t,3>& image, const std::string& filename, std::string extension=""){
  if (extension.empty())
    extension = boost::filesystem::path(filename).extension().string();
  boost::algorithm::to_lower(extension);
  if (extension == ".bmp") return write_bmp(image, filename); // this will only work for T=uint8_t
#ifdef HAVE_GIFLIB
  if (extension == ".gif") return write_gif(image, filename); // this will only work for T=uint8_t
#endif
#ifdef HAVE_LIBPNG
  if (extension == ".png") return write_png(image, filename);
#endif
#ifdef HAVE_LIBJPEG
  if (extension == ".jpg" || extension == ".jpeg") return write_jpeg(image, filename); // this will only work for T=uint8_t
#endif
#ifdef HAVE_LIBTIFF
  if (extension == ".tif" || extension == ".tiff") return write_tiff(image, filename);
#endif
  if (extension == ".ppm") return write_ppm(image, filename);

  throw std::runtime_error("The filename extension '" + extension + "' is not known or not supported for color images");
}

inline void write_gray_image(const blitz::Array<uint8_t,2>& image, const std::string& filename, std::string extension=""){
  if (extension.empty())
    extension = boost::filesystem::path(filename).extension().string();
  boost::algorithm::to_lower(extension);
#ifdef HAVE_LIBPNG
  if (extension == ".png") return write_png(image, filename);
#endif
#ifdef HAVE_LIBJPEG
  if (extension == ".jpg" || extension == ".jpeg") return write_jpeg(image, filename); // this will only work for T=uint8_t
#endif
#ifdef HAVE_LIBTIFF
  if (extension == ".tif" || extension == ".tiff") return write_tiff(image, filename);
#endif
  if (extension == ".pgm") return write_pgm(image, filename);
  if (extension == ".pbm") return write_pbm(image, filename);

  throw std::runtime_error("The filename extension '" + extension + "' is not known or not supported for gray images");
}

} } } // namespace

#endif // BOB_IO_IMAGE_IMAGE_H
