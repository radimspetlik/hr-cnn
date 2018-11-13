/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  3 Oct 08:36:48 2012
 *
 * @brief Implementation of some compile-time I/O utitlites
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.io.base/CodecRegistry.h>
#include <bob.io.base/utils.h>

boost::shared_ptr<bob::io::base::File> bob::io::base::open (const char* filename,
    char mode, const char* pretend_extension) {
  boost::shared_ptr<bob::io::base::CodecRegistry> instance = bob::io::base::CodecRegistry::instance();
  return instance->findByExtension(pretend_extension)(filename, mode);
}

boost::shared_ptr<bob::io::base::File> bob::io::base::open (const char* filename, char mode) {
  boost::shared_ptr<bob::io::base::CodecRegistry> instance = bob::io::base::CodecRegistry::instance();
  return instance->findByFilenameExtension(filename)(filename, mode);
}

bob::io::base::array::typeinfo bob::io::base::peek (const char* filename) {
  return open(filename, 'r')->type();
}

bob::io::base::array::typeinfo bob::io::base::peek_all (const char* filename) {
  return open(filename, 'r')->type_all();
}
