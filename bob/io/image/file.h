/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 14 May 14:42:34 2014 CEST
 *
 * @brief Implements the matlab (.mat) array codec using matio
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IO_IMAGE_FILE_H
#define BOB_IO_IMAGE_FILE_H

#include <boost/shared_ptr.hpp>
#include <bob.io.base/File.h>

/**
 * This defines the factory method F that can create codecs of this type.
 *
 * Here are the meanings of the mode flag that should be respected by your
 * factory implementation:
 *
 * 'r': opens for reading only - no modifications can occur; it is an
 *      error to open a file that does not exist for read-only operations.
 * 'w': opens for reading and writing, but truncates the file if it
 *      exists; it is not an error to open files that do not exist with
 *      this flag.
 * 'a': opens for reading and writing - any type of modification can
 *      occur. If the file does not exist, this flag is effectively like
 *      'w'.
 *
 * Returns a newly allocated File object that can read and write data to the
 * file using a specific backend.
 */
boost::shared_ptr<bob::io::base::File> make_tiff_file (const char* path, char mode);
#ifdef HAVE_LIBJPEG
boost::shared_ptr<bob::io::base::File> make_jpeg_file (const char* path, char mode);
#endif
boost::shared_ptr<bob::io::base::File> make_gif_file (const char* path, char mode);
boost::shared_ptr<bob::io::base::File> make_netpbm_file (const char* path, char mode);
boost::shared_ptr<bob::io::base::File> make_png_file (const char* path, char mode);
boost::shared_ptr<bob::io::base::File> make_bmp_file (const char* path, char mode);

#endif /* BOB_IO_IMAGE_FILE_H */
