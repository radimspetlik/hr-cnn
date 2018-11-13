/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Manuel Gunther <siebenkopf@googlemail.com>
 * @date Fri 16 May 12:33:38 2014 CEST
 *
 * @brief Pythonic bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * Copyright (c) 2016, Regents of the University of Colorado on behalf of the University of Colorado Colorado Springs.
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif

#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.core/array_convert.h>
#include <bob.io.base/api.h>

#include "file.h"

#include <bob.extension/documentation.h>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <bob.io.image/image.h>


#ifdef HAVE_LIBJPEG
#include <jpeglib.h>
#endif


static auto s_image_extension = bob::extension::FunctionDoc(
  "get_correct_image_extension",
  "Estimates the image type and return a corresponding extension based on file content",
  "This function loads the first bytes of the given image, and matches it with known magic numbers of image files. "
  "If a match is found, it returns the corresponding image extension (including the leading ``'.'`` that can be used, e.g., in :py:func:`bob.io.image.load`."
)
.add_prototype("image_name", "extension")
.add_parameter("image_name", "str", "The name (including path) of the image to check")
.add_return("extension", "str", "The extension of the image based on the file content")
;
static PyObject* image_extension(PyObject*, PyObject *args, PyObject* kwds) {
BOB_TRY
  static char** kwlist = s_image_extension.kwlist();

  const char* image_name;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &image_name)) return 0;

  return Py_BuildValue("s", bob::io::image::get_correct_image_extension(image_name).c_str());

BOB_CATCH_FUNCTION("get_correct_image_extension", 0)
}


static PyMethodDef module_methods[] = {
  {
    s_image_extension.name(),
    (PyCFunction)image_extension,
    METH_VARARGS|METH_KEYWORDS,
    s_image_extension.doc(),
  },
  {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Image I/O support for Bob");

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

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;

  /* activates image plugins */
  if (!PyBobIoCodec_Register(".tif", "TIFF, compresssed (libtiff)",
        &make_tiff_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".tiff", "TIFF, compresssed (libtiff)",
        &make_tiff_file)) {
    PyErr_Print();
  }

#ifdef HAVE_LIBJPEG
  if (BITS_IN_JSAMPLE == 8) {
    if (!PyBobIoCodec_Register(".jpg", "JPEG, compresssed (libjpeg)",
          &make_jpeg_file)) {
      PyErr_Print();
    }
    if (!PyBobIoCodec_Register(".jpeg", "JPEG, compresssed (libjpeg)",
          &make_jpeg_file)) {
      PyErr_Print();
    }
  }
  else {
    PyErr_Format(PyExc_RuntimeError, "libjpeg compiled with `%d' bits depth (instead of 8). JPEG images are hence not supported.", BITS_IN_JSAMPLE);
    PyErr_Print();
  }
#endif

#ifdef HAVE_GIFLIB
  if (!PyBobIoCodec_Register(".gif", "GIF (giflib)", &make_gif_file)) {
    PyErr_Print();
  }
#endif // HAVE_GIFLIB

  if (!PyBobIoCodec_Register(".pbm", "PBM, indexed (libnetpbm)",
        &make_netpbm_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".pgm", "PGM, indexed (libnetpbm)",
        &make_netpbm_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".ppm", "PPM, indexed (libnetpbm)",
        &make_netpbm_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".png", "PNG, compressed (libpng)", &make_png_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".bmp", "BMP, (built-in codec)", &make_bmp_file)) {
    PyErr_Print();
  }

  return Py_BuildValue(ret, m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
