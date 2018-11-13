/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  6 Nov 21:44:34 2013
 *
 * @brief Bindings to bob::io::video::Writer
 */

#include "main.h"

static auto s_writer = bob::extension::ClassDoc(
  "writer",
  "Use this object to write frames to video files"
  "Video writer objects can write data to video files. "
  "The current implementation uses `FFmpeg <http://ffmpeg.org>`_ (or `libav <http://libav.org>`_ if FFmpeg is not available) which is a stable freely available video encoding and decoding library, designed specifically for these tasks. "
  "Videos are objects composed of RGB colored frames. "
  "Each frame inserted should be a 3D :py:class:`numpy.ndarray` composed of unsigned integers of 8 bits. "
  "Each frame should have a shape equivalent to ``(plane, height, width)``."
).add_constructor(
  bob::extension::FunctionDoc(
    "writer",
    "Create a video writer",
    "The video will be created if the combination of format and codec are known to work and have been tested, otherwise an exception is raised. "
    "If you set the ``check`` parameter to ``False``, though, we will ignore this check.",
    true
  )
  .add_prototype("filename, height, width, [framerate], [bitrate], [gop], [codec], [format], [check]", "")
  .add_parameter("filename", "str", "The file path to the file you want to write data to")
  .add_parameter("height", "int", "The height of the video (must be a multiple of 2)")
  .add_parameter("width", "int", "The width of the video (must be a multiple of 2)")
  .add_parameter("framerate", "float", "[Default: 25.] The number of frames per second")
  .add_parameter("bitrate", "float", "[Default: 150000.] The estimated bitrate of the output video")
  .add_parameter("gop", "int", "[Default: 12] Group-of-Pictures (emit one intra frame every ``gop`` frames at most)")
  .add_parameter("codec", "str", "[Default: ``''``] If you must, specify a valid FFmpeg codec name here and that will be used to encode the video stream on the output file")
  .add_parameter("format", "str", "[Default: ``''``] If you must, specify a valid FFmpeg output format name and that will be used to encode the video on the output file. Leave it empty to guess from the filename extension")
  .add_parameter("check", "bool", "[Default: ``True``] ")
);


static void PyBobIoVideoWriter_Delete (PyBobIoVideoWriterObject* o) {

  o->v.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);

}

/* The __init__(self) method */
static int PyBobIoVideoWriter_Init(PyBobIoVideoWriterObject* self,
    PyObject *args, PyObject* kwds) {
BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = s_writer.kwlist();

  char* filename = 0;

  Py_ssize_t height = 0;
  Py_ssize_t width = 0;

  double framerate = 25.;
  double bitrate = 1500000.;
  Py_ssize_t gop = 12;
  char* codec = 0;
  char* format = 0;
  PyObject* pycheck = Py_True;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "snn|ddnssO", kwlist,
        &filename,
        &height, &width, &framerate, &bitrate, &gop, &codec,
        &format, &pycheck)) return -1;

  std::string codec_str = codec?codec:"";
  std::string format_str = format?format:"";
  bool check = PyObject_IsTrue(pycheck);

  self->v = boost::make_shared<bob::io::video::Writer>(filename,
      height, width, framerate, bitrate, gop, codec_str, format_str, check);

  return 0; ///< SUCCESS
BOB_CATCH_MEMBER("constructor", -1)
}

static auto s_filename = bob::extension::VariableDoc(
  "filename",
  "str",
  "The full path to the file that will be decoded by this object"
);
PyObject* PyBobIoVideoWriter_Filename(PyBobIoVideoWriterObject* self) {
  return Py_BuildValue("s", self->v->filename().c_str());
}

static auto s_height = bob::extension::VariableDoc(
  "height",
  "int",
  "The height of each frame in the video (a multiple of 2)"
);
PyObject* PyBobIoVideoWriter_Height(PyBobIoVideoWriterObject* self) {
  return Py_BuildValue("n", self->v->height());
}

static auto s_width = bob::extension::VariableDoc(
  "width",
  "int",
  "The width of each frame in the video (a multiple of 2)"
);
PyObject* PyBobIoVideoWriter_Width(PyBobIoVideoWriterObject* self) {
  return Py_BuildValue("n", self->v->width());
}

static auto s_number_of_frames = bob::extension::VariableDoc(
  "number_of_frames",
  "int",
  "The number of frames in this video file"
);
PyObject* PyBobIoVideoWriter_NumberOfFrames(PyBobIoVideoWriterObject* self) {
  return Py_BuildValue("n", self->v->numberOfFrames());
}

static auto s_duration = bob::extension::VariableDoc(
  "duration",
  "int",
  "Total duration of this video file in microseconds (long)"
);
PyObject* PyBobIoVideoWriter_Duration(PyBobIoVideoWriterObject* self) {
  return Py_BuildValue("n", self->v->duration());
}

static auto s_format_name = bob::extension::VariableDoc(
  "format_name",
  "str",
  "Short name of the format in which this video file will be written in"
);
PyObject* PyBobIoVideoWriter_FormatName(PyBobIoVideoWriterObject* self) {
  if (!self->v->is_opened()) {
    PyErr_Format(PyExc_RuntimeError, "`%s' for `%s' is closed",
        Py_TYPE(self)->tp_name, self->v->filename().c_str());
    return 0;
  }
  return Py_BuildValue("s", self->v->formatName().c_str());
}

static auto s_format_long_name = bob::extension::VariableDoc(
  "format_long_name",
  "str",
  "Full name of the format in which this video file will be written in"
);
PyObject* PyBobIoVideoWriter_FormatLongName(PyBobIoVideoWriterObject* self) {
  if (!self->v->is_opened()) {
    PyErr_Format(PyExc_RuntimeError, "`%s' for `%s' is closed",
        Py_TYPE(self)->tp_name, self->v->filename().c_str());
    return 0;
  }
  return Py_BuildValue("s", self->v->formatLongName().c_str());
}

static auto s_codec_name = bob::extension::VariableDoc(
  "codec_name",
  "str",
  "Short name of the codec in which this video file will be written in"
);
PyObject* PyBobIoVideoWriter_CodecName(PyBobIoVideoWriterObject* self) {
  if (!self->v->is_opened()) {
    PyErr_Format(PyExc_RuntimeError, "`%s' for `%s' is closed",
        Py_TYPE(self)->tp_name, self->v->filename().c_str());
    return 0;
  }
  return Py_BuildValue("s", self->v->codecName().c_str());
}

static auto s_codec_long_name = bob::extension::VariableDoc(
  "codec_long_name",
  "str",
  "Full name of the codec in which this video file will be written in"
);
PyObject* PyBobIoVideoWriter_CodecLongName(PyBobIoVideoWriterObject* self) {
  if (!self->v->is_opened()) {
    PyErr_Format(PyExc_RuntimeError, "`%s' for `%s' is closed",
        Py_TYPE(self)->tp_name, self->v->filename().c_str());
    return 0;
  }
  return Py_BuildValue("s", self->v->codecLongName().c_str());
}

static auto s_frame_rate = bob::extension::VariableDoc(
  "frame_rate",
  "float",
  "Video's announced frame rate (note there are video formats with variable frame rates)"
);
PyObject* PyBobIoVideoWriter_FrameRate(PyBobIoVideoWriterObject* self) {
  return PyFloat_FromDouble(self->v->frameRate());
}

static auto s_bit_rate = bob::extension::VariableDoc(
  "bit_rate",
  "float",
  "The indicative bit rate for this video file, given as a hint to `FFmpeg` (compression levels are subject to the picture textures)"
);
PyObject* PyBobIoVideoWriter_BitRate(PyBobIoVideoWriterObject* self) {
  return PyFloat_FromDouble(self->v->bitRate());
}

static auto s_gop = bob::extension::VariableDoc(
  "gop",
  "int",
  "Group of pictures setting (see the `Wikipedia entry <http://en.wikipedia.org/wiki/Group_of_pictures>`_ for details on this setting)"
);
PyObject* PyBobIoVideoWriter_GOP(PyBobIoVideoWriterObject* self) {
  return Py_BuildValue("n", self->v->gop());
}

static auto s_video_type = bob::extension::VariableDoc(
  "video_type",
  "tuple",
  "Typing information to load all of the file at once",
  ".. todo:: Explain, what exactly is contained in this tuple"
);
PyObject* PyBobIoVideoWriter_VideoType(PyBobIoVideoWriterObject* self) {
  return PyBobIo_TypeInfoAsTuple(self->v->video_type());
}

static auto s_frame_type = bob::extension::VariableDoc(
  "frame_type",
  "tuple",
  "Typing information to load each frame separatedly",
  ".. todo:: Explain, what exactly is contained in this tuple"
);
PyObject* PyBobIoVideoWriter_FrameType(PyBobIoVideoWriterObject* self) {
  return PyBobIo_TypeInfoAsTuple(self->v->frame_type());
}

static auto s_info = bob::extension::VariableDoc(
  "info",
  "str",
  "A string with lots of video information (same as ``str(x)``)"
);
static PyObject* PyBobIoVideoWriter_Print(PyBobIoVideoWriterObject* self) {
  if (!self->v->is_opened()) {
    PyErr_Format(PyExc_RuntimeError, "`%s' for `%s' is closed",
        Py_TYPE(self)->tp_name, self->v->filename().c_str());
    return 0;
  }

  return Py_BuildValue("s", self->v->info().c_str());
}

static auto s_is_opened = bob::extension::VariableDoc(
  "is_opened",
  "bool",
  "A flag, indicating if the video is still opened for writing (or has already been closed by the user using :py:meth:`close`)"
);
static PyObject* PyBobIoVideoWriter_IsOpened(PyBobIoVideoWriterObject* self) {
  if (self->v->is_opened()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyGetSetDef PyBobIoVideoWriter_getseters[] = {
    {
      s_filename.name(),
      (getter)PyBobIoVideoWriter_Filename,
      0,
      s_filename.doc(),
      0,
    },
    {
      s_height.name(),
      (getter)PyBobIoVideoWriter_Height,
      0,
      s_height.doc(),
      0,
    },
    {
      s_width.name(),
      (getter)PyBobIoVideoWriter_Width,
      0,
      s_width.doc(),
      0,
    },
    {
      s_number_of_frames.name(),
      (getter)PyBobIoVideoWriter_NumberOfFrames,
      0,
      s_number_of_frames.doc(),
      0,
    },
    {
      s_duration.name(),
      (getter)PyBobIoVideoWriter_Duration,
      0,
      s_duration.doc(),
      0,
    },
    {
      s_format_name.name(),
      (getter)PyBobIoVideoWriter_FormatName,
      0,
      s_format_name.doc(),
      0,
    },
    {
      s_format_long_name.name(),
      (getter)PyBobIoVideoWriter_FormatLongName,
      0,
      s_format_long_name.doc(),
      0,
    },
    {
      s_codec_name.name(),
      (getter)PyBobIoVideoWriter_CodecName,
      0,
      s_codec_name.doc(),
      0,
    },
    {
      s_codec_long_name.name(),
      (getter)PyBobIoVideoWriter_CodecLongName,
      0,
      s_codec_long_name.doc(),
      0,
    },
    {
      s_frame_rate.name(),
      (getter)PyBobIoVideoWriter_FrameRate,
      0,
      s_frame_rate.doc(),
      0,
    },
    {
      s_bit_rate.name(),
      (getter)PyBobIoVideoWriter_BitRate,
      0,
      s_bit_rate.doc(),
      0,
    },
    {
      s_gop.name(),
      (getter)PyBobIoVideoWriter_GOP,
      0,
      s_gop.doc(),
      0,
    },
    {
      s_video_type.name(),
      (getter)PyBobIoVideoWriter_VideoType,
      0,
      s_video_type.doc(),
      0,
    },
    {
      s_frame_type.name(),
      (getter)PyBobIoVideoWriter_FrameType,
      0,
      s_frame_type.doc(),
      0,
    },
    {
      s_info.name(),
      (getter)PyBobIoVideoWriter_Print,
      0,
      s_info.doc(),
      0,
    },
    {
      s_is_opened.name(),
      (getter)PyBobIoVideoWriter_IsOpened,
      0,
      s_is_opened.doc(),
      0,
    },
    {0}  /* Sentinel */
};

static PyObject* PyBobIoVideoWriter_Repr(PyBobIoVideoWriterObject* self) {
  if (!self->v->is_opened()) {
    PyErr_Format(PyExc_RuntimeError, "`%s' for `%s' is closed",
        Py_TYPE(self)->tp_name, self->v->filename().c_str());
    return 0;
  }

  return
# if PY_VERSION_HEX >= 0x03000000
  PyUnicode_FromFormat
# else
  PyString_FromFormat
# endif
  ("%s(filename='%s', height=%" PY_FORMAT_SIZE_T "d, width=%" PY_FORMAT_SIZE_T "d, framerate=%g, bitrate=%g, gop=%" PY_FORMAT_SIZE_T "d, codec='%s', format='%s')", Py_TYPE(self)->tp_name, self->v->filename().c_str(), self->v->height(), self->v->width(), self->v->frameRate(), self->v->bitRate(), self->v->gop(), self->v->codecName().c_str(), self->v->formatName().c_str());
}


static auto s_append = bob::extension::FunctionDoc(
  "append",
  "Writes a new frame or set of frames to the file.",
  "The frame should be setup as a array with 3 dimensions organized in this way (RGB color-bands, height, width). "
  "Sets of frames should be setup as a 4D array in this way: (frame-number, RGB color-bands, height, width). "
  "Arrays should contain only unsigned integers of 8 bits.\n\n"
  ".. note::\n"
  "  At present time we only support arrays that have C-style storages (if you pass reversed arrays or arrays with Fortran-style storage, the result is undefined).",
  true
)
.add_prototype("frame")
.add_parameter("frame", "3D or 4D :py:class:`numpy.ndarray` of ``uint8``", "The frame or set of frames to write")
;
static PyObject* PyBobIoVideoWriter_Append(PyBobIoVideoWriterObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  if (!self->v->is_opened()) {
    PyErr_Format(PyExc_RuntimeError, "`%s' for `%s' is closed",
        Py_TYPE(self)->tp_name, self->v->filename().c_str());
    return 0;
  }

  char** kwlist = s_append.kwlist();

  PyBlitzArrayObject* frame = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBlitzArray_BehavedConverter, &frame)) return 0;
  auto frame_ = make_safe(frame);

  if (frame->ndim != 3 && frame->ndim != 4) {
    PyErr_Format(PyExc_ValueError, "input array should have 3 or 4 dimensions, but you passed an array with %" PY_FORMAT_SIZE_T "d dimensions", frame->ndim);
    return 0;
  }

  if (frame->type_num != NPY_UINT8) {
    PyErr_Format(PyExc_TypeError, "input array should have dtype `uint8', but you passed an array with dtype == `%s'", PyBlitzArray_TypenumAsString(frame->type_num));
    return 0;
  }

  if (frame->ndim == 3) {
    self->v->append(*PyBlitzArrayCxx_AsBlitz<uint8_t,3>(frame));
  }
  else {
    self->v->append(*PyBlitzArrayCxx_AsBlitz<uint8_t,4>(frame));
  }
  Py_RETURN_NONE;
BOB_CATCH_MEMBER("append", 0)
}


static auto s_close = bob::extension::FunctionDoc(
  "close",
  "Closes the current video stream and forces writing the trailer.",
  "After this point the video is finalized and cannot be written to anymore.",
  true
)
.add_prototype("")
;
static PyObject* PyBobIoVideoWriter_Close(PyBobIoVideoWriterObject* self) {
BOB_TRY
  self->v->close();
  Py_RETURN_NONE;
BOB_CATCH_MEMBER("close", 0)
}

static PyMethodDef PyBobIoVideoWriter_Methods[] = {
    {
      s_append.name(),
      (PyCFunction)PyBobIoVideoWriter_Append,
      METH_VARARGS|METH_KEYWORDS,
      s_append.doc(),
    },
    {
      s_close.name(),
      (PyCFunction)PyBobIoVideoWriter_Close,
      METH_NOARGS,
      s_close.doc(),
    },
    {0}  /* Sentinel */
};

Py_ssize_t PyBobIoVideoWriter_Len(PyBobIoVideoWriterObject* self) {
  return self->v->numberOfFrames();
}

static PyMappingMethods PyBobIoVideoWriter_Mapping = {
    (lenfunc)PyBobIoVideoWriter_Len, //mp_lenght
    0, /* (binaryfunc)PyBobIoVideoWriter_GetItem, //mp_subscript */
    0  /* (objobjargproc)PyBobIoVideoWriter_SetItem //mp_ass_subscript */
};

PyTypeObject PyBobIoVideoWriter_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    0
};

bool init_BobIoVideoWriter(PyObject* module){

  // initialize the writer
  PyBobIoVideoWriter_Type.tp_name = s_writer.name();
  PyBobIoVideoWriter_Type.tp_basicsize = sizeof(PyBobIoVideoWriterObject);
  PyBobIoVideoWriter_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobIoVideoWriter_Type.tp_doc = s_writer.doc();

  // set the functions
  PyBobIoVideoWriter_Type.tp_new = PyType_GenericNew;
  PyBobIoVideoWriter_Type.tp_init = reinterpret_cast<initproc>(PyBobIoVideoWriter_Init);
  PyBobIoVideoWriter_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIoVideoWriter_Delete);
  PyBobIoVideoWriter_Type.tp_methods = PyBobIoVideoWriter_Methods;
  PyBobIoVideoWriter_Type.tp_getset = PyBobIoVideoWriter_getseters;

  PyBobIoVideoWriter_Type.tp_str = reinterpret_cast<reprfunc>(PyBobIoVideoWriter_Print);
  PyBobIoVideoWriter_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobIoVideoWriter_Repr);
  PyBobIoVideoWriter_Type.tp_as_mapping = &PyBobIoVideoWriter_Mapping;

  // check that everything is fine
  if (PyType_Ready(&PyBobIoVideoWriter_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIoVideoWriter_Type);
  return PyModule_AddObject(module, "writer", (PyObject*)&PyBobIoVideoWriter_Type) >= 0;
}
