/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Manuel Gunther <siebenkopf@googlemail.com>
 * @date Wed 14 May 14:00:33 2014 CEST
 *
 * @brief Binds configuration information available from bob
 */

#define BOB_IMPORT_VERSION
#include <bob.blitz/config.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/config.h>
#include <bob.io.base/config.h>
#include <bob.io.image/config.h>

#include <boost/preprocessor/stringize.hpp>

static PyObject* build_version_dictionary() {

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

#ifdef HAVE_LIBJPEG
  if (!dict_steal(retval, "libjpeg", libjpeg_version())) return 0;
#endif
#ifdef HAVE_LIBPNG
  if (!dict_steal(retval, "libpng", libpng_version())) return 0;
#endif
#ifdef HAVE_LIBTIFF
  if (!dict_steal(retval, "libtiff", libtiff_version())) return 0;
#endif
#ifdef HAVE_GIFLIB
  if (!dict_steal(retval, "giflib", giflib_version())) return 0;
#endif
  if (!dict_steal(retval, "HDF5", hdf5_version())) return 0;
  if (!dict_steal(retval, "Boost", boost_version())) return 0;
  if (!dict_steal(retval, "Compiler", compiler_version())) return 0;
  if (!dict_steal(retval, "Python", python_version())) return 0;
  if (!dict_steal(retval, "NumPy", numpy_version())) return 0;
  if (!dict_steal(retval, "Blitz++", blitz_version())) return 0;
  if (!dict_steal(retval, "bob.blitz", bob_blitz_version())) return 0;
  if (!dict_steal(retval, "bob.core", bob_core_version())) return 0;
  if (!dict_steal(retval, "bob.io.base", bob_io_base_version())) return 0;

  return Py_BuildValue("O", retval);
}

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr,
"Information about software used to compile the C++ Bob API"
);

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
  auto m_ = make_xsafe(m);
  const char* ret = "O";
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!m) return 0;

  /* register version numbers and constants */
  if (PyModule_AddIntConstant(m, "api", BOB_IO_IMAGE_API_VERSION) < 0) return 0;
  if (PyModule_AddStringConstant(m, "module", BOB_EXT_MODULE_VERSION) < 0) return 0;
  if (PyModule_AddObject(m, "externals", build_version_dictionary()) < 0) return 0;

  // call bob_io_image_version once to avoid compiler warning
  auto _ = make_safe(bob_io_image_version());

  return Py_BuildValue(ret, m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
