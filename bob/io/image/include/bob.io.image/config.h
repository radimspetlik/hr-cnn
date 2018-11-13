/**
 * @author Manuel Gunther <siebenkopf@googlemail.com>
 * @date Mon May 23 09:58:35 MDT 2016
 *
 * @brief General directives for all modules in bob.io.image
 */

#ifndef BOB_IO_IMAGE_CONFIG_H
#define BOB_IO_IMAGE_CONFIG_H

/* Macros that define versions and important names */
#define BOB_IO_IMAGE_API_VERSION 0x0201


#ifdef BOB_IMPORT_VERSION

  /***************************************
  * Here we define some functions that should be used to build version dictionaries in the version.cpp file
  * There will be a compiler warning, when these functions are not used, so use them!
  ***************************************/

  #include <Python.h>
  #include <boost/preprocessor/stringize.hpp>

#ifdef HAVE_LIBJPEG
  #include <jpeglib.h>
#endif

#ifdef HAVE_LIBPNG
  #define PNG_SKIP_SETJMP_CHECK
  // #define requires because of the problematic pngconf.h.
  // Look at the thread here:
  // https://bugs.launchpad.net/ubuntu/+source/libpng/+bug/218409
  #include <png.h>
#endif

#ifdef HAVE_GIFLIB
  #include <gif_lib.h>
#endif

#ifdef HAVE_LIBTIFF
  #include <tiffio.h>
#endif

  /**
   * LibJPEG version
   */
#ifdef HAVE_LIBJPEG
  static PyObject* libjpeg_version() {
    boost::format f("%d (compiled with %d bits depth)");
    f % LIBJPEG_VERSION % BITS_IN_JSAMPLE;
    return Py_BuildValue("s", f.str().c_str());
  }
#endif

  /**
   * Libpng version
   */
#ifdef HAVE_LIBPNG
  static PyObject* libpng_version() {
    return Py_BuildValue("s", PNG_LIBPNG_VER_STRING);
  }
#endif

  /**
   * Libtiff version
   */
#ifdef HAVE_LIBTIFF
  static PyObject* libtiff_version() {
    static const std::string beg_str("LIBTIFF, Version ");
    static const size_t beg_len = beg_str.size();
    std::string vtiff(TIFFGetVersion());

    // Remove first part if it starts with "LIBTIFF, Version "
    if(vtiff.compare(0, beg_len, beg_str) == 0)
      vtiff = vtiff.substr(beg_len);

    // Remove multiple (copyright) lines if any
    size_t end_line = vtiff.find("\n");
    if(end_line != std::string::npos)
      vtiff = vtiff.substr(0,end_line);

    return Py_BuildValue("s", vtiff.c_str());
  }
#endif

  /**
   * Version of giflib support
   */
#ifdef HAVE_GIFLIB
  static PyObject* giflib_version() {
  #ifdef GIFLIB_VERSION
   return Py_BuildValue("s", GIFLIB_VERSION);
  #else
    boost::format f("%s.%s.%s");
    f % BOOST_PP_STRINGIZE(GIFLIB_MAJOR) % BOOST_PP_STRINGIZE(GIFLIB_MINOR) % BOOST_PP_STRINGIZE(GIFLIB_RELEASE);
    return Py_BuildValue("s", f.str().c_str());
  #endif
  }
#endif

  /**
   * bob.io.image c/c++ api version
   */
  static PyObject* bob_io_image_version() {
    return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_IO_IMAGE_API_VERSION));
  }

#endif // BOB_IMPORT_VERSION


#endif /* BOB_IO_IMAGE_CONFIG_H */
