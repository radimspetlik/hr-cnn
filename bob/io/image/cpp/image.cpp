/**
 * @date Thu Sep  8 12:05:08 MDT 2016
 * @author Manuel Gunther <siebenkopf@googlemail.com>
 *
 * @brief Implements an generic image functionalities.
 *
 * Copyright (c) 2016, Regents of the University of Colorado on behalf of the University of Colorado Colorado Springs.
 */

#include <stdint.h>
#include <boost/assign/list_of.hpp>
#include <bob.io.image/image.h>

namespace bob { namespace io { namespace image {

static std::map<std::string, std::vector<std::vector<uint8_t>>> _initialize_magic_numbers(){
  // these numbers are based on: http://stackoverflow.com/questions/26350342/determining-the-extension-type-of-an-image-file-using-binary/26350431#26350431
  std::map<std::string, std::vector<std::vector<uint8_t>>> magic_numbers;
  // BMP
  magic_numbers[".bmp"].push_back(boost::assign::list_of(0x42)(0x4D));
  // NetPBM, see https://en.wikipedia.org/wiki/Netpbm_format
  magic_numbers[".pbm"].push_back(boost::assign::list_of(0x50)(0x31));// P1
  magic_numbers[".pbm"].push_back(boost::assign::list_of(0x50)(0x34));// P4
  magic_numbers[".pgm"].push_back(boost::assign::list_of(0x50)(0x32));// P2
  magic_numbers[".pgm"].push_back(boost::assign::list_of(0x50)(0x35));// P5
  magic_numbers[".ppm"].push_back(boost::assign::list_of(0x50)(0x33));// P3
  magic_numbers[".ppm"].push_back(boost::assign::list_of(0x50)(0x36));// P6
  // TODO: what about P7?

#ifdef HAVE_GIFLIB
  // GIF
  magic_numbers[".gif"].push_back(boost::assign::list_of(0x47)(0x49)(0x46)(0x38)(0x37)(0x61));
  magic_numbers[".gif"].push_back(boost::assign::list_of(0x47)(0x49)(0x46)(0x38)(0x39)(0x61));
#endif // HAVE_GIFLIB
#ifdef HAVE_LIBPNG
  // PNG
  magic_numbers[".png"].push_back(boost::assign::list_of(0x89)(0x50)(0x4E)(0x47)(0x0D)(0x0A)(0x1A)(0x0A));
#endif // HAVE_LIB_PNG
#ifdef HAVE_LIBJPEG
  // JPEG
  magic_numbers[".jpg"].push_back(boost::assign::list_of(0xFF)(0xD8)(0xFF));
  magic_numbers[".jpg"].push_back(boost::assign::list_of(0x00)(0x00)(0x00)(0x0C)(0x6A)(0x50)(0x20)(0x20));
#endif // HAVE_LIBJPEG
#ifdef HAVE_LIBTIFF
  // TIFF
  magic_numbers[".tiff"].push_back(boost::assign::list_of(0x0C)(0xED));
  magic_numbers[".tiff"].push_back(boost::assign::list_of(0x49)(0x20)(0x49));
  magic_numbers[".tiff"].push_back(boost::assign::list_of(0x49)(0x49)(0x2A)(0x00));
  magic_numbers[".tiff"].push_back(boost::assign::list_of(0x4D)(0x4D)(0x00)(0x2A));
  magic_numbers[".tiff"].push_back(boost::assign::list_of(0x4D)(0x4D)(0x00)(0x2B));
#endif // HAVE_LIBTIFF
  return magic_numbers;
}

// global static thread-safe map of known magic numbers
static std::map<std::string, std::vector<std::vector<uint8_t>>> known_magic_numbers = _initialize_magic_numbers();


const std::string& get_correct_image_extension(const std::string& image_name){
  // read first 8 bytes from file
  uint8_t image_bytes[8];
  std::ifstream f(image_name.c_str());
  if (!f) throw std::runtime_error("The given image '" + image_name + "' could not be opened for reading");
  f.read(reinterpret_cast<char*>(image_bytes), 8);

  // iterate over all extensions
  for (auto eit = known_magic_numbers.begin(); eit != known_magic_numbers.end(); ++eit){
    // iterate over all magic bytes
    for (auto mit = eit->second.begin(); mit != eit->second.end(); ++mit){
      // check magic number
      if (std::equal(mit->begin(), mit->end(), image_bytes))
        return eit->first;
    }
  }

  throw std::runtime_error("The given image '" + image_name + "' does not contain an image of a known type");
}

bool is_color_image(const std::string& filename, std::string extension){
  if (extension.empty())
    extension = boost::filesystem::path(filename).extension().string();
  boost::algorithm::to_lower(extension);
  if (extension == ".bmp") return true;
#ifdef HAVE_GIFLIB
  if (extension == ".gif") return true;
#endif
#ifdef HAVE_LIBPNG
  if (extension == ".png") return is_color_png(filename);
#endif
#ifdef HAVE_LIBJPEG
  if (extension == ".jpg" || extension == ".jpeg") return is_color_jpeg(filename);
#endif
#ifdef HAVE_LIBTIFF
  if (extension == ".tif" || extension == ".tiff") return is_color_tiff(filename);
#endif
  if (extension == ".pgm" || extension == ".pbm") return false;
  if (extension == ".ppm") return true;

  throw std::runtime_error("The filename extension '" + extension + "' is not known");
}

} } } // namespaces
