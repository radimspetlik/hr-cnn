/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 16 Oct 17:40:24 2013
 *
 * @brief Pythonic bindings to C++ constructs on bob.io.video
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif

#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>
#include <bob.core/api.h>
#include <bob.io.base/api.h>

#include "main.h"


extern "C" {

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#if LIBAVUTIL_VERSION_INT >= 0x320f01 //50.15.1 @ ffmpeg-0.6
#  include <libavutil/opt.h>
#  include <libavutil/pixdesc.h>
#endif
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#if LIBAVCODEC_VERSION_INT < 0x363b64 //54.59.100 @ ffmpeg-1.0
#  define AVCodecID CodecID
#endif

}


/**
 * Creates an str object, from a C or C++ string. Returns a **new
 * reference**.
 */
static PyObject* make_object(PyObject* o) {
  return Py_BuildValue("O", o);
}

static PyObject* make_object(const char* s) {
  return Py_BuildValue("s", s);
}

static PyObject* make_object(bool v) {
  if (v) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyObject* make_object(unsigned int v) {
  return Py_BuildValue("n", v);
}

static PyObject* make_object(double v) {
  return PyFloat_FromDouble(v);
}


/**
 * Sets a dictionary entry using a string as key and another one as value.
 * Returns 1 in case of success, 0 in case of failure.
 */
template <typename T>
bool dict_set(PyObject* d, const char* key, T value) {
  auto pyvalue = make_xsafe(make_object(value));
  if (!pyvalue) return false;
  int retval = PyDict_SetItemString(d, key, pyvalue.get());
  return !retval;
}

/**
 * Sets a dictionary entry using a string as key and another one as value.
 * Returns 1 in case of success, 0 in case of faiulre.
 */
template <typename T>
bool list_append(PyObject* l, T value) {
  auto pyvalue = make_xsafe(make_object(value));
  if (!pyvalue) return false;
  int retval = PyList_Append(l, pyvalue.get());
  return !retval;
}
/**
 *  * Checks if it is a Python string for Python 2.x or 3.x
 *   */
int check_string(PyObject* o) {
#     if PY_VERSION_HEX >= 0x03000000
        return PyUnicode_Check(o);
#     else
        return PyString_Check(o);
#     endif
}

/**
 * Describes a given codec. We return a **new reference** to a dictionary
 * containing the codec properties.
 */
static PyObject* describe_codec(const AVCodec* codec) {

  /**
   * We wrap the returned object into a smart pointer until we
   * are absolutely sure all went good. At this point, we free
   * the PyObject* from the boost encapsulation and return it.
   */
  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  /* Sets basic properties for the codec */
  if (!dict_set(retval, "name", codec->name)) return 0;
  if (!dict_set(retval, "long_name", codec->long_name)) return 0;
  if (!dict_set(retval, "id", (unsigned int)codec->id)) return 0;

  /**
   * If pixel formats are available, creates and attaches a
   * list with all their names
   */

  PyObject* pixfmt = 0;
  if (codec->pix_fmts) {
    pixfmt = PyList_New(0);
    if (!pixfmt) return 0;
    auto pixfmt_ = make_safe(pixfmt);

    unsigned int i=0;
    while(codec->pix_fmts[i] != -1) {
      if (!list_append(pixfmt, av_get_pix_fmt_name(codec->pix_fmts[i++])))
        return 0;
    }
    pixfmt = PySequence_Tuple(pixfmt);
  }
  else {
    Py_INCREF(Py_None);
    pixfmt = Py_None;
  }

  auto pixfmt_ = make_safe(pixfmt);
  if (!dict_set(retval, "pixfmts", pixfmt)) return 0;

  /* Get specific framerates for the codec, if any */
  const AVRational* rate = codec->supported_framerates;
  PyObject* rates = PyList_New(0);
  if (!rates) return 0;
  auto rates_ = make_safe(rates);

  while (rate && rate->num && rate->den) {
    list_append(rates, ((double)rate->num)/((double)rate->den));
    ++rate;
  }

  rates = PySequence_Tuple(rates);
  rates_ = make_safe(rates);

  if (!dict_set(retval, "specific_framerates_hz", rates)) return 0;

  /* Other codec capabilities */
# ifdef CODEC_CAP_LOSSLESS
  if (!dict_set(retval, "lossless", (bool)(codec->capabilities & CODEC_CAP_LOSSLESS))) return 0;
# endif
# ifdef CODEC_CAP_EXPERIMENTAL
  if (!dict_set(retval, "experimental", (bool)(codec->capabilities & CODEC_CAP_EXPERIMENTAL))) return 0;
# endif
# ifdef CODEC_CAP_DELAY
  if (!dict_set(retval, "delay", (bool)(codec->capabilities & CODEC_CAP_DELAY))) return 0;
# endif
# ifdef CODEC_CAP_HWACCEL
  if (!dict_set(retval, "hardware_accelerated", (bool)(codec->capabilities & CODEC_CAP_HWACCEL))) return 0;
# endif
  if (!dict_set(retval, "encode", (bool)(avcodec_find_encoder(codec->id)))) return 0;
  if (!dict_set(retval, "decode", (bool)(avcodec_find_decoder(codec->id)))) return 0;

  /* If all went OK, detach the returned value from the smart pointer **/
  return Py_BuildValue("O", retval);

}

/**
 * Describes a given codec or raises, in case the codec cannot be accessed
 */
auto s_describe_encoder = bob::extension::FunctionDoc(
  "describe_encoder",
  "Returns a dictionary containing a description of properties in the given encoder."
)
.add_prototype("key", "description")
.add_parameter("key", "`int` or `str`", "A key which can be either the encoder name or its numerical identifier.")
.add_return("description", "dict", "The description of the requested encoder")
;
static PyObject* PyBobIoVideo_DescribeEncoder(PyObject*, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_describe_encoder.kwlist();

  PyObject* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &key)) return 0;

  if (!PyBob_NumberCheck(key) && !check_string(key)) {
    PyErr_SetString(PyExc_TypeError, "input `key' must be a number identifier or a string with the codec name");
    return 0;
  }

  if (PyBob_NumberCheck(key)) {

    /* If you get to this point, the user passed a number - re-parse */
    int id = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &id)) return 0;

    AVCodec* codec = avcodec_find_encoder((AVCodecID)id);
    if (!codec) {
      PyErr_Format(PyExc_RuntimeError, "ffmpeg::avcodec_find_encoder(%d == 0x%x) did not return a valid codec", id, id);
      return 0;
    }

    return describe_codec(codec);
  }

  /* If you get to this point, the user passed a string - re-parse */
  const char* name = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &name)) return 0;

  AVCodec* codec = avcodec_find_encoder_by_name(name);
  if (!codec) {
    PyErr_Format(PyExc_RuntimeError, "ffmpeg::avcodec_find_encoder_by_name(`%s') did not return a valid codec", name);
    return 0;
  }

  return describe_codec(codec);
BOB_CATCH_FUNCTION("describe_encoder", 0)
}


/**
 * Describes a given codec or raises, in case the codec cannot be accessed
 */
auto s_describe_decoder = bob::extension::FunctionDoc(
  "describe_decoder",
  "Returns a dictionary containing a description of properties in the given decoder."
)
.add_prototype("key", "description")
.add_parameter("key", "`int` or `str`", "A key which can be either the decoder name or its numerical identifier.")
.add_return("description", "dict", "The description of the requested encoder")
;
static PyObject* PyBobIoVideo_DescribeDecoder(PyObject*, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_describe_decoder.kwlist();

  PyObject* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &key)) return 0;

  if (!PyBob_NumberCheck(key) && !check_string(key)) {
    PyErr_SetString(PyExc_TypeError, "input `key' must be a number identifier or a string with the codec name");
    return 0;
  }

  if (PyBob_NumberCheck(key)) {

    /* If you get to this point, the user passed a number - re-parse */
    int id = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &id)) return 0;

    AVCodec* codec = avcodec_find_decoder((AVCodecID)id);
    if (!codec) {
      PyErr_Format(PyExc_RuntimeError, "ffmpeg::avcodec_find_decoder(%d == 0x%x) did not return a valid codec", id, id);
      return 0;
    }

    return describe_codec(codec);
  }

  /* If you get to this point, the user passed a string - re-parse */
  const char* name = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &name)) return 0;

  AVCodec* codec = avcodec_find_decoder_by_name(name);
  if (!codec) {
    PyErr_Format(PyExc_RuntimeError, "ffmpeg::avcodec_find_decoder_by_name(`%s') did not return a valid codec", name);
    return 0;
  }

  return describe_codec(codec);
BOB_CATCH_FUNCTION("describe_decoder", 0)
}


static PyObject* get_video_codecs(bool supported) {
BOB_TRY
  std::map<std::string, const AVCodec*> m;
  if (supported)
    bob::io::video::codecs_supported(m);
  else
    bob::io::video::codecs_installed(m);

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (auto k=m.begin(); k!=m.end(); ++k) {
    dict_set(retval, k->first.c_str(), describe_codec(k->second));
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_FUNCTION("get_video_codecs", 0)
}

auto s_supported_codecs = bob::extension::FunctionDoc(
  "supported_video_codecs",
  "Returns a dictionary with currently supported video codec properties",
  "Returns a dictionary containing a detailed description of the built-in codecs for videos that are fully supported."
)
.add_prototype("", "codecs")
.add_return("codecs", "dict", "A dictionary of supported codecs")
;
static PyObject* PyBobIoVideo_SupportedCodecs(PyObject*) {
  return get_video_codecs(true);
}

auto s_available_codecs = bob::extension::FunctionDoc(
  "available_video_codecs",
  "Returns a dictionary with currently supported video codec properties",
  "Returns a dictionary containing a detailed description of the built-in codecs for videos that are available but **not necessarily supported**."
)
.add_prototype("", "codecs")
.add_return("codecs", "dict", "A dictionary of available codecs")
;
static PyObject* PyBobIoVideo_AvailableCodecs(PyObject*) {
  return get_video_codecs(false);
}


static PyObject* get_video_iformats(bool supported) {
BOB_TRY
  std::map<std::string, AVInputFormat*> m;
  if (supported)
    bob::io::video::iformats_supported(m);
  else
    bob::io::video::iformats_installed(m);

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (auto k=m.begin(); k!=m.end(); ++k) {

    PyObject* property = PyDict_New();
    if (!property) return 0;
    auto property_ = make_safe(property);

    if (!dict_set(property, "name", k->second->name)) return 0;

    if (!dict_set(property, "long_name", k->second->long_name)) return 0;

    // get extensions
    std::vector<std::string> exts;
    bob::io::video::tokenize_csv(k->second->extensions, exts);

    PyObject* ext_list = PyList_New(0);
    if (!ext_list) return 0;
    auto ext_list_ = make_safe(ext_list);

    for (auto ext=exts.begin(); ext!=exts.end(); ++ext) {
      if (!list_append(ext_list, ext->c_str())) return 0;
    }

    if (!dict_set(property, "extensions", ext_list)) return 0;

    if (!dict_set(retval, k->first.c_str(), property)) return 0;

  }

  return Py_BuildValue("O", retval);
BOB_CATCH_FUNCTION("get_video_iformats", 0)
}

auto s_supported_iformats = bob::extension::FunctionDoc(
  "supported_videoreader_formats",
  "Returns a dictionary with currently supported video input formats",
  "Returns a dictionary containing a detailed description of the built-in input formats for videos that are fully supported."
)
.add_prototype("", "formats")
.add_return("formats", "dict", "A dictionary of supported input formats")
;
static PyObject* PyBobIoVideo_SupportedInputFormats(PyObject*) {
  return get_video_iformats(true);
}

auto s_available_iformats = bob::extension::FunctionDoc(
  "available_videoreader_formats",
  "Returns a dictionary with currently available video input formats",
  "Returns a dictionary containing a detailed description of the built-in input formats for videos that are available, but **not necessarily supported** by this library."
)
.add_prototype("", "formats")
.add_return("formats", "dict", "A dictionary of available input formats")
;
static PyObject* PyBobIoVideo_AvailableInputFormats(PyObject*) {
  return get_video_iformats(false);
}


static PyObject* get_video_oformats(bool supported) {
BOB_TRY
  std::map<std::string, AVOutputFormat*> m;
  if (supported)
    bob::io::video::oformats_supported(m);
  else
    bob::io::video::oformats_installed(m);

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (auto k=m.begin(); k!=m.end(); ++k) {

    PyObject* property = PyDict_New();
    if (!property) return 0;
    auto property_ = make_safe(property);

    if (!dict_set(property, "name", k->second->name)) return 0;

    if (!dict_set(property, "long_name", k->second->long_name)) return 0;

    if (!dict_set(property, "mime_type", k->second->mime_type)) return 0;

    // get extensions
    std::vector<std::string> exts;
    bob::io::video::tokenize_csv(k->second->extensions, exts);
    Py_ssize_t size = exts.size();
    PyObject* ext_list = PyList_New(size);
    if (!ext_list) return 0;
    auto ext_list_ = make_safe(ext_list);

    for (Py_ssize_t i = 0; i < size; i++)
      PyList_SET_ITEM(ext_list, i, make_object(exts[i].c_str()));

    if (!dict_set(property, "extensions", ext_list)) return 0;

    /**  get recommended codec **/
    PyObject* default_codec = 0;
    if (k->second->video_codec) {
      AVCodec* codec = avcodec_find_encoder(k->second->video_codec);
      if (codec) {
        default_codec = describe_codec(codec);
        if (!default_codec) return 0;
      }
    }
    auto default_codec_ = make_xsafe(default_codec);

    if (!default_codec) {
      default_codec = Py_None;
    }

    if (!dict_set(property, "default_codec", default_codec)) return 0;

    /** get supported codec list **/
    if (supported) {
      std::vector<const AVCodec*> codecs;
      bob::io::video::oformat_supported_codecs(k->second->name, codecs);

      PyObject* supported_codecs = PyDict_New();
      if (!supported_codecs) return 0;
      auto supported_codecs_ = make_safe(supported_codecs);

      for (auto c=codecs.begin(); c!=codecs.end(); ++c) {
        PyObject* codec_descr = describe_codec(*c);
        auto codec_descr_ = make_safe(codec_descr);
        if (!codec_descr) return 0;
        if (!dict_set(supported_codecs, (*c)->name, codec_descr)) return 0;
      }

      if (!dict_set(property, "supported_codecs", supported_codecs)) return 0;
    }

    if (!dict_set(retval, k->first.c_str(), property)) return 0;

  }

  return Py_BuildValue("O", retval);
BOB_CATCH_FUNCTION("get_video_oformats", 0)
}

auto s_supported_oformats = bob::extension::FunctionDoc(
  "supported_videowriter_formats",
  "Returns a dictionary with currently supported video output formats",
  "Returns a dictionary containing a detailed description of the built-in output formats for videos that are fully supported."
)
.add_prototype("", "formats")
.add_return("formats", "dict", "A dictionary of supported output formats")
;
static PyObject* PyBobIoVideo_SupportedOutputFormats(PyObject*) {
  return get_video_oformats(true);
}

auto s_available_oformats = bob::extension::FunctionDoc(
  "available_videowriter_formats",
  "Returns a dictionary with currently available video output formats",
  "Returns a dictionary containing a detailed description of the built-in output formats for videos that are available, but **not necessarily supported** by this library."
)
.add_prototype("", "formats")
.add_return("formats", "dict", "A dictionary of available output formats")
;
static PyObject* PyBobIoVideo_AvailableOutputFormats(PyObject*) {
  return get_video_oformats(false);
}


/**
 * Arranges a listing of input and output file formats
 */
static void list_formats(std::map<std::string, std::string>& formats) {
  std::map<std::string, AVInputFormat*> iformat;
  bob::io::video::iformats_supported(iformat);
  std::map<std::string, AVOutputFormat*> oformat;
  bob::io::video::oformats_supported(oformat);

  for (auto k=iformat.begin(); k!=iformat.end(); ++k) {
    auto o=oformat.find(k->first);
    if (o!=oformat.end()) {
      //format can be used for input and output
      std::vector<std::string> extensions;
      bob::io::video::tokenize_csv(o->second->extensions, extensions);
      for (auto e=extensions.begin(); e!=extensions.end(); ++e) {
        std::string key = ".";
        key += *e;
        std::string value = k->second->long_name;
        value += " (video/ffmpeg)";
        formats[key] = value;
      }
    }
  }
}

static PyMethodDef module_methods[] = {
    {
      s_describe_encoder.name(),
      (PyCFunction)PyBobIoVideo_DescribeEncoder,
      METH_VARARGS|METH_KEYWORDS,
      s_describe_encoder.doc(),
    },
    {
      s_describe_decoder.name(),
      (PyCFunction)PyBobIoVideo_DescribeDecoder,
      METH_VARARGS|METH_KEYWORDS,
      s_describe_decoder.doc(),
    },
    {
      s_supported_codecs.name(),
      (PyCFunction)PyBobIoVideo_SupportedCodecs,
      METH_NOARGS,
      s_supported_codecs.doc(),
    },
    {
      s_available_codecs.name(),
      (PyCFunction)PyBobIoVideo_AvailableCodecs,
      METH_NOARGS,
      s_available_codecs.doc(),
    },
    {
      s_supported_iformats.name(),
      (PyCFunction)PyBobIoVideo_SupportedInputFormats,
      METH_NOARGS,
      s_supported_iformats.doc(),
    },
    {
      s_available_iformats.name(),
      (PyCFunction)PyBobIoVideo_AvailableInputFormats,
      METH_NOARGS,
      s_available_iformats.doc(),
    },
    {
      s_supported_oformats.name(),
      (PyCFunction)PyBobIoVideo_SupportedOutputFormats,
      METH_NOARGS,
      s_supported_oformats.doc(),
    },
    {
      s_available_oformats.name(),
      (PyCFunction)PyBobIoVideo_AvailableOutputFormats,
      METH_NOARGS,
      s_available_oformats.doc(),
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Video I/O support for Bob");

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

  /* Initialize libavcodec, and register all codecs and formats. */
  av_log_set_level(AV_LOG_QUIET);
  avcodec_register_all();
  av_register_all();

# if PY_VERSION_HEX >= 0x03000000
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  if (!init_BobIoVideoReader(module)) return 0;
  if (!init_BobIoVideoWriter(module)) return 0;

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;

  /* activates video plugins */
  std::map<std::string, std::string> formats;
  list_formats(formats);
  for (auto k=formats.begin(); k!=formats.end(); ++k) {
    if (!PyBobIoCodec_Register(k->first.c_str(), k->second.c_str(), &make_file)) {
      PyErr_Print();
      //do not return 0, or we may crash badly
    }
  }

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
