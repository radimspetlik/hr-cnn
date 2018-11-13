/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 14 May 14:00:33 2014 CEST
 *
 * @brief Binds configuration information available from bob
 */

#define BOB_IMPORT_VERSION
#include <bob.blitz/config.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/config.h>
#include <bob.io.base/config.h>

extern "C" {

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>

}

static int dict_set(PyObject* d, const char* key, const char* value) {
  PyObject* v = Py_BuildValue("s", value);
  return dict_steal(d, key, v);
}

/**
 * FFmpeg version
 */
static PyObject* ffmpeg_version() {
  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

# if defined(FFMPEG_VERSION)
  if (std::strlen(FFMPEG_VERSION) && !dict_set(retval, "ffmpeg", FFMPEG_VERSION)) return 0;
# endif
  if (!dict_set(retval, "avformat", BOOST_PP_STRINGIZE(LIBAVFORMAT_VERSION))) return 0;
  if (!dict_set(retval, "avcodec", BOOST_PP_STRINGIZE(LIBAVCODEC_VERSION))) return 0;
  if (!dict_set(retval, "avutil", BOOST_PP_STRINGIZE(LIBAVUTIL_VERSION))) return 0;
  if (!dict_set(retval, "swscale", BOOST_PP_STRINGIZE(LIBSWSCALE_VERSION))) return 0;

  return Py_BuildValue("O", retval);
}

static PyObject* build_version_dictionary() {

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  if (!dict_steal(retval, "FFmpeg", ffmpeg_version())) return 0;
  if (!dict_steal(retval, "Boost", boost_version())) return 0;
  if (!dict_steal(retval, "Compiler", compiler_version())) return 0;
  if (!dict_steal(retval, "Python", python_version())) return 0;
  if (!dict_steal(retval, "NumPy", numpy_version())) return 0;
  if (!dict_steal(retval, "Blitz++", blitz_version())) return 0;
  if (!dict_steal(retval, "HDF5", hdf5_version())) return 0;
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
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  /* register version numbers and constants */
  if (PyModule_AddStringConstant(module, "module", BOB_EXT_MODULE_VERSION) < 0) return 0;
  if (PyModule_AddObject(module, "externals", build_version_dictionary()) < 0) return 0;

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
