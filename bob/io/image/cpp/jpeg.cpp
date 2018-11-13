/**
 * @file io/cxx/ImageJpegFile.cc
 * @date Wed Oct 10 16:38:00 2012 +0200
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 * @author Manuel Gunther <siebenkopf@googlemail.com>
 *
 * @brief Implements an image format reader/writer using libjpeg.
 * This codec is only able to work with 3D input/output.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * Copyright (c) 2016, Regents of the University of Colorado on behalf of the University of Colorado Colorado Springs.
 */

#ifdef HAVE_LIBJPEG

#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <string>

#include <bob.core/logging.h>
#include <bob.io.image/jpeg.h>

#include <jpeglib.h>

// Default JPEG quality
static int s_jpeg_quality = 92;

static boost::shared_ptr<std::FILE> make_cfile(const char *filename, const char *flags)
{
  std::FILE* fp = std::fopen(filename, flags);
  if(fp == 0) {
    boost::format m("the file `%s' could not be opened - verify permissions and availability");
    m % filename;
    throw std::runtime_error(m.str());
  }
  return boost::shared_ptr<std::FILE>(fp, std::fclose);
}


/**
 * ERROR HANDLING
 */
static void my_error_exit (j_common_ptr cinfo){
  // get error message
  char message[JMSG_LENGTH_MAX];
  cinfo->err->format_message(cinfo, message);
  // format error
  boost::format m("In image '%s' fatal JPEG error (%d) has occurred -> %s");
  m % reinterpret_cast<char*>(cinfo->client_data) % cinfo->err->msg_code % message;

  // Clean-up JPEG structures IS NOT required,
  // just raise the exception
  throw std::runtime_error(m.str());
}

static void my_output_message(j_common_ptr cinfo){
  // get warning message
  char message[JMSG_LENGTH_MAX];
  cinfo->err->format_message(cinfo, message);

  // log message as debug
  bob::core::debug << "In image '" << reinterpret_cast<char*>(cinfo->client_data) << "' JPEG warning has occured -> " << message << std::endl;
}


/**
 * LOADING
 */
static void im_peek(const std::string& path, bob::io::base::array::typeinfo& info) {
  // 1. JPEG structures
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jerr.error_exit = my_error_exit;
  jerr.output_message = my_output_message;
  // set image name as client data; used for warning and error messages
  cinfo.client_data = const_cast<char*>(path.c_str());
  jpeg_create_decompress(&cinfo);

  // 2. JPEG file opening
  boost::shared_ptr<std::FILE> in_file = make_cfile(path.c_str(), "rb");
  jpeg_stdio_src(&cinfo, in_file.get());

  // 3. Read header
  jpeg_read_header(&cinfo, TRUE);

  // 4. Set parameters for decompression if any

  // 5. Start decompression and get information
  jpeg_start_decompress(&cinfo);

  // Set depth and number of dimensions
  info.dtype = bob::io::base::array::t_uint8;
  info.nd = (cinfo.output_components == 1? 2 : 3);
  if(info.nd == 2)
  {
    info.shape[0] = cinfo.output_height;
    info.shape[1] = cinfo.output_width;
  }
  else
  {
    info.shape[0] = 3;
    info.shape[1] = cinfo.output_height;
    info.shape[2] = cinfo.output_width;
  }
  info.update_strides();

  // 6. clean up
  jpeg_destroy_decompress(&cinfo);
}

template <typename T> static
void im_load_gray(struct jpeg_decompress_struct *cinfo, bob::io::base::array::interface& b) {
  const bob::io::base::array::typeinfo& info = b.type();

  T *element = static_cast<T*>(b.ptr());
  const int row_stride = info.shape[1];
  JSAMPROW buffer_pptr[1];
  while (cinfo->output_scanline < cinfo->image_height) {
    buffer_pptr[0] = element;
    jpeg_read_scanlines(cinfo, buffer_pptr, 1);
    element += row_stride;
  }
}

template <typename T> static
void imbuffer_to_rgb(size_t size, const T* im, T* r, T* g, T* b) {
  for (size_t k=0; k<size; ++k) {
    r[k] = im[3*k];
    g[k] = im[3*k +1];
    b[k] = im[3*k +2];
  }
}

template <typename T> static
void cmyk_imbuffer_to_rgb(size_t size, const T* im, T* r, T* g, T* b, bool adobe_marker) {
  T C,M,Y,K;
  for (size_t k=0; k<size; ++k) {
    if (adobe_marker){
      C = *im++;
      M = *im++;
      Y = *im++;
      K = *im++;
    } else {
      C = 255-*im++;
      M = 255-*im++;
      Y = 255-*im++;
      K = 255-*im++;
    }
    *r++ = C * K / 255;
    *g++ = M * K / 255;
    *b++ = Y * K / 255;
  }
}

template <typename T> static
void im_load_color(struct jpeg_decompress_struct *cinfo, bob::io::base::array::interface& b) {
  const bob::io::base::array::typeinfo& info = b.type();

  long unsigned int frame_size = info.shape[1] * info.shape[2];
  T *element_r = static_cast<T*>(b.ptr());
  T *element_g = element_r+frame_size;
  T *element_b = element_g+frame_size;

  const int row_stride = cinfo->output_width * cinfo->output_components;
  JSAMPROW buffer_pptr[1];
  boost::shared_array<JSAMPLE> buffer(new JSAMPLE[row_stride]);
  buffer_pptr[0] = buffer.get();
  while (cinfo->output_scanline < cinfo->output_height) {
    jpeg_read_scanlines(cinfo, buffer_pptr, 1);
    if (cinfo->output_components == 3)
      imbuffer_to_rgb<T>(info.shape[2], reinterpret_cast<T*>(buffer_pptr[0]), element_r, element_g, element_b);
    else
      cmyk_imbuffer_to_rgb<T>(info.shape[2], reinterpret_cast<T*>(buffer_pptr[0]), element_r, element_g, element_b, cinfo->saw_Adobe_marker);

    element_r += cinfo->output_width;
    element_g += cinfo->output_width;
    element_b += cinfo->output_width;
  }
}

static void im_load(const std::string& filename, bob::io::base::array::interface& b) {
  // 1. JPEG structures
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jerr.error_exit = my_error_exit;
  jerr.output_message = my_output_message;
  // set image name as client data; used for warning and error messages
  cinfo.client_data = const_cast<char*>(filename.c_str());
  jpeg_create_decompress(&cinfo);

  // 2. JPEG file opening
  boost::shared_ptr<std::FILE> in_file = make_cfile(filename.c_str(), "rb");
  jpeg_stdio_src(&cinfo, in_file.get());

  // 3. Read header
  jpeg_read_header(&cinfo, TRUE);

  // 4. Set parameters for decompression
  if (cinfo.output_components == 4){
    // assure to get CMYK output
    cinfo.out_color_space = JCS_CMYK;
  }

  // 5. Start decompression and get information
  jpeg_start_decompress(&cinfo);

  // 6. Read content
  const bob::io::base::array::typeinfo& info = b.type();
  if(info.dtype == bob::io::base::array::t_uint8) {
    if(info.nd == 2) im_load_gray<uint8_t>(&cinfo, b);
    else if( info.nd == 3) im_load_color<uint8_t>(&cinfo, b);
    else {
      boost::format m("the image in file `%s' has a number of dimensions this jpeg codec has no support for: %s");
      m % filename % info.str();
      throw std::runtime_error(m.str());
    }
  }
  else {
    boost::format m("the image in file `%s' has a data type this jpeg codec has no support for: %s");
    m % filename % info.str();
    throw std::runtime_error(m.str());
  }

  // 7. Finish decompression
  jpeg_finish_decompress(&cinfo);

  // 8. Release JPEG decompression object
  jpeg_destroy_decompress(&cinfo);
}

/**
 * SAVING
 */
template <typename T>
static void im_save_gray(const bob::io::base::array::interface& b, struct jpeg_compress_struct *cinfo) {
  const bob::io::base::array::typeinfo& info = b.type();

  const T* element = static_cast<const T*>(b.ptr());

  // pointer to a single row  (JSAMPLE is a typedef to unsigned char or char)
  JSAMPROW row_pointer[1];
  int row_stride = info.shape[1]; // JSAMPLEs per row in image_buffer
  while(cinfo->next_scanline < cinfo->image_height) {
    row_pointer[0] = const_cast<T*>(element);
    jpeg_write_scanlines(cinfo, row_pointer, 1);
    element += row_stride;
  }
}

template <typename T> static
void rgb_to_imbuffer(size_t size, const T* r, const T* g, const T* b, T* im) {
  for (size_t k=0; k<size; ++k) {
    im[3*k]   = r[k];
    im[3*k+1] = g[k];
    im[3*k+2] = b[k];
  }
}

template <typename T>
static void im_save_color(const bob::io::base::array::interface& b, struct jpeg_compress_struct *cinfo) {
  const bob::io::base::array::typeinfo& info = b.type();

  long unsigned int frame_size = info.shape[1] * info.shape[2];

  const T *element_r = static_cast<const T*>(b.ptr());
  const T *element_g = element_r + frame_size;
  const T *element_b = element_g + frame_size;

  // pointer to a single row  (JSAMPLE is a typedef to unsigned char or char)
  boost::shared_array<JSAMPLE> row(new JSAMPLE[3*info.shape[2]]);
  JSAMPROW array_ptr[1];
  array_ptr[0] = row.get();
  int row_color_stride = info.shape[2]; // JSAMPLEs per row in image_buffer
  while(cinfo->next_scanline < cinfo->image_height) {
    rgb_to_imbuffer(row_color_stride, element_r, element_g, element_b, reinterpret_cast<T*>(array_ptr[0]));
    jpeg_write_scanlines(cinfo, array_ptr, 1);
    element_r += row_color_stride;
    element_g += row_color_stride;
    element_b += row_color_stride;
  }
}

static void im_save (const std::string& filename, const bob::io::base::array::interface& array) {
  const bob::io::base::array::typeinfo& info = array.type();

  // 1. JPEG structures
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jerr.error_exit = my_error_exit;
  jerr.output_message = my_output_message;
  // set image name as client data; used for warning and error messages
  cinfo.client_data = const_cast<char*>(filename.c_str());
  jpeg_create_compress(&cinfo);

  // 2. JPEG opening
  boost::shared_ptr<std::FILE> out_file = make_cfile(filename.c_str(), "wb");
  jpeg_stdio_dest(&cinfo, out_file.get());

  // 3. Set compression parameters
  cinfo.image_height = (info.nd == 2 ? info.shape[0] : info.shape[1]);
  cinfo.image_width = (info.nd == 2 ? info.shape[1] : info.shape[2]);
  cinfo.input_components = (info.nd == 2 ? 1 : 3);
  cinfo.in_color_space = (info.nd == 2 ? JCS_GRAYSCALE : JCS_RGB); // colorspace of input image
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, s_jpeg_quality, TRUE);

  // 4.
  jpeg_start_compress(&cinfo, TRUE);

  // Writes content
  if(info.dtype == bob::io::base::array::t_uint8) {

    if(info.nd == 2) im_save_gray<uint8_t>(array, &cinfo);
    else if(info.nd == 3) {
      if(info.shape[0] != 3) throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      im_save_color<uint8_t>(array, &cinfo);
    }
    else {
      boost::format m("the image array to be written at file `%s' has a number of dimensions this jpeg codec has no support for: %s");
      m % filename % info.str();
      throw std::runtime_error(m.str());
    }
  }
  else {
    boost::format m("the image array to be written at file `%s' has a data type this jpeg codec has no support for: %s");
    m % filename % info.str();
    throw std::runtime_error(m.str());
  }

  // 6.
  jpeg_finish_compress(&cinfo);

  // 7.
  jpeg_destroy_compress(&cinfo);
}


/**
 * JPEG class
*/

bob::io::image::JPEGFile::JPEGFile(const char* path, char mode)
: m_filename(path),
  m_newfile(true)
{
  //checks if file exists
  if (mode == 'r' && !boost::filesystem::exists(path)) {
    boost::format m("file '%s' is not readable");
    m % path;
    throw std::runtime_error(m.str());
  }

  if (mode == 'r' || (mode == 'a' && boost::filesystem::exists(path))) {
    im_peek(path, m_type);
    m_length = 1;
    m_newfile = false;
  } else {
    m_length = 0;
    m_newfile = true;
  }
}


void bob::io::image::JPEGFile::read(bob::io::base::array::interface& buffer, size_t index) {
  if (m_newfile)
    throw std::runtime_error("uninitialized image file cannot be read");

  if (!buffer.type().is_compatible(m_type)) buffer.set(m_type);

  if (index != 0)
    throw std::runtime_error("cannot read image with index > 0 -- there is only one image in an image file");

  if(!buffer.type().is_compatible(m_type)) buffer.set(m_type);

  // load jpeg
  im_load(m_filename, buffer);
}

size_t bob::io::image::JPEGFile::append(const bob::io::base::array::interface& buffer) {
  if (m_newfile) {
    im_save(m_filename, buffer);
    m_type = buffer.type();
    m_newfile = false;
    m_length = 1;
    return 0;
  }

  throw std::runtime_error("image files only accept a single array");
}

void bob::io::image::JPEGFile::write(const bob::io::base::array::interface& buffer) {
  //overwriting position 0 should always work
  if (m_newfile) {
    append(buffer);
    return;
  }

  throw std::runtime_error("image files only accept a single array");
}



std::string bob::io::image::JPEGFile::s_codecname = "bob.image_jpeg";

boost::shared_ptr<bob::io::base::File> make_jpeg_file (const char* path, char mode) {
  return boost::make_shared<bob::io::image::JPEGFile>(path, mode);
}

#endif // HAVE_LIBJPEG
