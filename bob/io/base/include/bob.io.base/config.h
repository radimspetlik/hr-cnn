/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri  1 Nov 07:10:59 2013
 *
 * @brief General directives for all modules in bob.io
 */

#ifndef BOB_IO_BASE_CONFIG_H
#define BOB_IO_BASE_CONFIG_H

/* Macros that define versions and important names */
#define BOB_IO_BASE_API_VERSION 0x0201


#ifdef BOB_IMPORT_VERSION

  /***************************************
  * Here we define some functions that should be used to build version dictionaries in the version.cpp file
  * There will be a compiler warning, when these functions are not used, so use them!
  ***************************************/

  #include <Python.h>
  #include <boost/preprocessor/stringize.hpp>
  #include <hdf5.h>

  /**
  * The version of HDF5
  */
  static PyObject* hdf5_version() {
    boost::format f("%s.%s.%s");
    f % BOOST_PP_STRINGIZE(H5_VERS_MAJOR);
    f % BOOST_PP_STRINGIZE(H5_VERS_MINOR);
    f % BOOST_PP_STRINGIZE(H5_VERS_RELEASE);
    return Py_BuildValue("s", f.str().c_str());
  }

  /**
   * bob.io.base c/c++ api version
   */
  static PyObject* bob_io_base_version() {
    return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_IO_BASE_API_VERSION));
  }

#endif // BOB_IMPORT_VERSION


#endif /* BOB_IO_BASE_CONFIG_H */
