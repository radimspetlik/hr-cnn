/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  6 Nov 21:44:34 2013
 *
 * @brief Bindings to bob::io::video::Reader
 */

#include "main.h"

static auto s_reader = bob::extension::ClassDoc(
  "reader",
  "Use this object to read frames from video files."
  "Video reader objects can read data from video files. "
  "The current implementation uses `FFmpeg <http://ffmpeg.org>`_ (or `libav <http://libav.org>`_ if FFmpeg is not available) which is a stable freely available video encoding and decoding library, designed specifically for these tasks. "
  "You can read an entire video in memory by using the :py:meth:`bob.io.video.reader.load` method or use iterators to read it frame by frame and avoid overloading your machine\'s memory. "
  "The maximum precision data `FFmpeg`_ will yield is a 24-bit (8-bit per band) representation of each pixel (32-bit depths are also supported by `FFmpeg`_, but not by this extension presently). "
  "So, the output of data is done with ``uint8`` as data type. "
  "Output will be colored using the RGB standard, with each band varying between 0 and 255, with zero meaning pure black and 255, pure white (color).\n\n"
).add_constructor(
  bob::extension::FunctionDoc(
    "reader",
    "Opens a video file for reading",
    "By default, if the format and/or the codec are not supported by this version of Bob, an exception will be raised. "
    "You can (at your own risk) set the ``check`` flag to ``False`` to  avoid this check.",
    true
  )
  .add_prototype("filename, [check]", "")
  .add_parameter("filename", "str", "The file path to the file you want to read data from")
  .add_parameter("check", "bool", "Format and codec will be extracted from the video metadata.")
);


static void PyBobIoVideoReader_Delete (PyBobIoVideoReaderObject* o) {
  o->v.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);
}

/* The __init__(self) method */
static int PyBobIoVideoReader_Init(PyBobIoVideoReaderObject* self,
    PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_reader.kwlist();

  char* filename = 0;

  PyObject* pycheck = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|O", kwlist,
        &filename, &pycheck)) return -1;

  bool check = (pycheck && PyObject_IsTrue(pycheck));

  self->v.reset(new bob::io::video::Reader(filename, check));
  return 0; ///< SUCCESS
BOB_CATCH_MEMBER("constructor", -1)
}

static auto s_filename = bob::extension::VariableDoc(
  "filename",
  "str",
  "The full path to the file that will be decoded by this object"
);
PyObject* PyBobIoVideoReader_Filename(PyBobIoVideoReaderObject* self) {
  return Py_BuildValue("s", self->v->filename().c_str());
}

static auto s_height = bob::extension::VariableDoc(
  "height",
  "int",
  "The height of each frame in the video (a multiple of 2)"
);
PyObject* PyBobIoVideoReader_Height(PyBobIoVideoReaderObject* self) {
  return Py_BuildValue("n", self->v->height());
}

static auto s_width = bob::extension::VariableDoc(
  "width",
  "int",
  "The width of each frame in the video (a multiple of 2)"
);
PyObject* PyBobIoVideoReader_Width(PyBobIoVideoReaderObject* self) {
  return Py_BuildValue("n", self->v->width());
}

static auto s_number_of_frames = bob::extension::VariableDoc(
  "number_of_frames",
  "int",
  "The number of frames in this video file"
);

PyObject* PyBobIoVideoReader_NumberOfFrames(PyBobIoVideoReaderObject* self) {
  return Py_BuildValue("n", self->v->numberOfFrames());
}

static auto s_duration = bob::extension::VariableDoc(
  "duration",
  "int",
  "Total duration of this video file in microseconds (long)"
);
PyObject* PyBobIoVideoReader_Duration(PyBobIoVideoReaderObject* self) {
  return Py_BuildValue("n", self->v->duration());
}

static auto s_format_name = bob::extension::VariableDoc(
  "format_name",
  "str",
  "Short name of the format in which this video file was recorded in"
);
PyObject* PyBobIoVideoReader_FormatName(PyBobIoVideoReaderObject* self) {
  return Py_BuildValue("s", self->v->formatName().c_str());
}

static auto s_format_long_name = bob::extension::VariableDoc(
  "format_long_name",
  "str",
  "Full name of the format in which this video file was recorded in"
);
PyObject* PyBobIoVideoReader_FormatLongName(PyBobIoVideoReaderObject* self) {
  return Py_BuildValue("s", self->v->formatLongName().c_str());
}

static auto s_codec_name = bob::extension::VariableDoc(
  "codec_name",
  "str",
  "Short name of the codec in which this video file was recorded in"
);
PyObject* PyBobIoVideoReader_CodecName(PyBobIoVideoReaderObject* self) {
  return Py_BuildValue("s", self->v->codecName().c_str());
}

static auto s_codec_long_name = bob::extension::VariableDoc(
  "codec_long_name",
  "str",
  "Full name of the codec in which this video file was recorded in"
);
PyObject* PyBobIoVideoReader_CodecLongName(PyBobIoVideoReaderObject* self) {
  return Py_BuildValue("s", self->v->codecLongName().c_str());
}

static auto s_frame_rate = bob::extension::VariableDoc(
  "frame_rate",
  "float",
  "Video's announced frame rate (note there are video formats with variable frame rates)"
);
PyObject* PyBobIoVideoReader_FrameRate(PyBobIoVideoReaderObject* self) {
  return PyFloat_FromDouble(self->v->frameRate());
}

static auto s_video_type = bob::extension::VariableDoc(
  "video_type",
  "tuple",
  "Typing information to load all of the file at once",
  ".. todo:: Explain, what exactly is contained in this tuple"
);
PyObject* PyBobIoVideoReader_VideoType(PyBobIoVideoReaderObject* self) {
  return PyBobIo_TypeInfoAsTuple(self->v->video_type());
}

static auto s_frame_type = bob::extension::VariableDoc(
  "frame_type",
  "tuple",
  "Typing information to load each frame separatedly",
  ".. todo:: Explain, what exactly is contained in this tuple"
);
PyObject* PyBobIoVideoReader_FrameType(PyBobIoVideoReaderObject* self) {
  return PyBobIo_TypeInfoAsTuple(self->v->frame_type());
}

static auto s_info = bob::extension::VariableDoc(
  "info",
  "str",
  "A string with lots of video information (same as ``str(x)``)"
);
static PyObject* PyBobIoVideoReader_Print(PyBobIoVideoReaderObject* self) {
  return Py_BuildValue("s", self->v->info().c_str());
}

static PyGetSetDef PyBobIoVideoReader_getseters[] = {
    {
      s_filename.name(),
      (getter)PyBobIoVideoReader_Filename,
      0,
      s_filename.doc(),
      0,
    },
    {
      s_height.name(),
      (getter)PyBobIoVideoReader_Height,
      0,
      s_height.doc(),
      0,
    },
    {
      s_width.name(),
      (getter)PyBobIoVideoReader_Width,
      0,
      s_width.doc(),
      0,
    },
    {
      s_number_of_frames.name(),
      (getter)PyBobIoVideoReader_NumberOfFrames,
      0,
      s_number_of_frames.doc(),
      0,
    },
    {
      s_duration.name(),
      (getter)PyBobIoVideoReader_Duration,
      0,
      s_duration.doc(),
      0,
    },
    {
      s_format_name.name(),
      (getter)PyBobIoVideoReader_FormatName,
      0,
      s_format_name.doc(),
      0,
    },
    {
      s_format_long_name.name(),
      (getter)PyBobIoVideoReader_FormatLongName,
      0,
      s_format_long_name.doc(),
      0,
    },
    {
      s_codec_name.name(),
      (getter)PyBobIoVideoReader_CodecName,
      0,
      s_codec_name.doc(),
      0,
    },
    {
      s_codec_long_name.name(),
      (getter)PyBobIoVideoReader_CodecLongName,
      0,
      s_codec_long_name.doc(),
      0,
    },
    {
      s_frame_rate.name(),
      (getter)PyBobIoVideoReader_FrameRate,
      0,
      s_frame_rate.doc(),
      0,
    },
    {
      s_video_type.name(),
      (getter)PyBobIoVideoReader_VideoType,
      0,
      s_video_type.doc(),
      0,
    },
    {
      s_frame_type.name(),
      (getter)PyBobIoVideoReader_FrameType,
      0,
      s_frame_type.doc(),
      0,
    },
    {
      s_info.name(),
      (getter)PyBobIoVideoReader_Print,
      0,
      s_info.doc(),
      0,
    },
    {0}  /* Sentinel */
};

static PyObject* PyBobIoVideoReader_Repr(PyBobIoVideoReaderObject* self) {
  return
# if PY_VERSION_HEX >= 0x03000000
  PyUnicode_FromFormat
# else
  PyString_FromFormat
# endif
  ("%s(filename='%s')", Py_TYPE(self)->tp_name, self->v->filename().c_str());
}

/**
 * If a keyboard interruption occurs, then it is translated into a C++
 * exception that makes the loop stops.
 */
static void Check_Interrupt() {
  if(PyErr_CheckSignals() == -1) {
    if (!PyErr_Occurred()) PyErr_SetInterrupt();
    throw std::runtime_error("error is already set");
  }
}

static auto s_load = bob::extension::FunctionDoc(
  "load",
  "Loads all of the video stream in a numpy ndarray organized in this way: (frames, color-bands, height, width). "
  "I'll dynamically allocate the output array and return it to you",
  "  The flag ``raise_on_error``, which is set to ``False`` by default influences the error reporting in case problems are found with the video file. "
  "If you set it to ``True``, we will report problems raising exceptions. "
  "If you set it to ``False`` (the default), we will truncate the file at the frame with problems and will not report anything. "
  "It is your task to verify if the number of frames returned matches the expected number of frames as reported by the :py:attr:`number_of_frames` (or ``len``) of this object.",
  true
)
.add_prototype("raise_on_error", "video")
.add_parameter("raise_on_error", "bool", "[Default: ``False``] Raise an excpetion in case of errors?")
.add_return("video", "3D or 4D :py:class:`numpy.ndarray`", "The video stream organized as: (frames, color-bands, height, width")
;
static PyObject* PyBobIoVideoReader_Load(PyBobIoVideoReaderObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_load.kwlist();

  PyObject* raise = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &raise)) return 0;

  bool raise_on_error = (raise && PyObject_IsTrue(raise));

  const bob::io::base::array::typeinfo& info = self->v->video_type();

  npy_intp shape[NPY_MAXDIMS];
  for (size_t k=0; k<info.nd; ++k) shape[k] = info.shape[k];

  int type_num = PyBobIo_AsTypenum(info.dtype);
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  PyObject* retval = PyArray_SimpleNew(info.nd, shape, type_num);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  Py_ssize_t frames_read = 0;

  bobskin skin((PyArrayObject*)retval, info.dtype);
  frames_read = self->v->load(skin, raise_on_error, &Check_Interrupt);

  if (frames_read != shape[0]) {
    //resize
    shape[0] = frames_read;
    PyArray_Dims newshape;
    newshape.ptr = shape;
    newshape.len = info.nd;
    PyArray_Resize((PyArrayObject*)retval, &newshape, 1, NPY_ANYORDER);
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER("load", 0)
}


static PyMethodDef PyBobIoVideoReader_Methods[] = {
    {
      s_load.name(),
      (PyCFunction)PyBobIoVideoReader_Load,
      METH_VARARGS|METH_KEYWORDS,
      s_load.doc(),
    },
    {0}  /* Sentinel */
};

static PyObject* PyBobIoVideoReader_GetIndex (PyBobIoVideoReaderObject* self, Py_ssize_t i) {
BOB_TRY
  if (i < 0) i += self->v->numberOfFrames(); ///< adjust for negative indexing

  if (i < 0 || (size_t)i >= self->v->numberOfFrames()) {
    PyErr_Format(PyExc_IndexError, "video frame index out of range - `%s' only contains %" PY_FORMAT_SIZE_T "d frame(s)", self->v->filename().c_str(), self->v->numberOfFrames());
    return 0;
  }

  const bob::io::base::array::typeinfo& info = self->v->frame_type();

  npy_intp shape[NPY_MAXDIMS];
  for (size_t k=0; k<info.nd; ++k) shape[k] = info.shape[k];

  int type_num = PyBobIo_AsTypenum(info.dtype);
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  PyObject* retval = PyArray_SimpleNew(info.nd, shape, type_num);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  auto it = self->v->begin();
  it += i;
  bobskin skin((PyArrayObject*)retval, info.dtype);
  it.read(skin);

  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER("get_index", 0)
}

static PyObject* PyBobIoVideoReader_GetSlice (PyBobIoVideoReaderObject* self, PySliceObject* slice) {
BOB_TRY
  Py_ssize_t start, stop, step, slicelength;
#if PY_VERSION_HEX < 0x03000000
  if (PySlice_GetIndicesEx(slice,
#else
  if (PySlice_GetIndicesEx(reinterpret_cast<PyObject*>(slice),
#endif
        self->v->numberOfFrames(), &start, &stop, &step, &slicelength) < 0) return 0;

  //creates the return array
  const bob::io::base::array::typeinfo& info = self->v->frame_type();

  int type_num = PyBobIo_AsTypenum(info.dtype);
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  if (slicelength <= 0) return PyArray_SimpleNew(0, 0, type_num);

  npy_intp shape[NPY_MAXDIMS];
  shape[0] = slicelength;
  for (size_t k=0; k<info.nd; ++k) shape[k+1] = info.shape[k];

  PyObject* retval = PyArray_SimpleNew(info.nd+1, shape, type_num);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  Py_ssize_t counter;
  Py_ssize_t lo, hi, st;
  auto it = self->v->begin();
  if (start <= stop) {
    lo = start, hi = stop, st = step;
    it += lo, counter = 0;
  }
  else {
    lo = stop, hi = start, st = -step;
    it += lo + (hi-lo)%st, counter = slicelength - 1;
  }

  for (auto i=lo; i<hi; i+=st) {

    //get slice to fill
    PyObject* islice = Py_BuildValue("n", counter);
    counter = (st == -step)? counter-1 : counter+1;
    if (!islice) return 0;
    auto islice_ = make_safe(islice);

    PyObject* item = PyObject_GetItem(retval, islice);
    if (!item) return 0;
    auto item_ = make_safe(item);

    bobskin skin((PyArrayObject*)item, info.dtype);
    it.read(skin);
    it += (st-1);
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER("get_slice", 0)
}

static PyObject* PyBobIoVideoReader_GetItem (PyBobIoVideoReaderObject* self, PyObject* item) {
BOB_TRY
   if (PyIndex_Check(item)) {
     Py_ssize_t i = PyNumber_AsSsize_t(item, PyExc_IndexError);
     if (i == -1 && PyErr_Occurred()) return 0;
     return PyBobIoVideoReader_GetIndex(self, i);
   }
   if (PySlice_Check(item)) {
     return PyBobIoVideoReader_GetSlice(self, (PySliceObject*)item);
   }
   else {
     PyErr_Format(PyExc_TypeError, "VideoReader indices must be integers, not `%s'",
         Py_TYPE(item)->tp_name);
     return 0;
   }
BOB_CATCH_MEMBER("get_item", 0)
}

Py_ssize_t PyBobIoVideoReader_Len(PyBobIoVideoReaderObject* self) {
  return self->v->numberOfFrames();
}

static PyMappingMethods PyBobIoVideoReader_Mapping = {
    (lenfunc)PyBobIoVideoReader_Len, //mp_lenght
    (binaryfunc)PyBobIoVideoReader_GetItem, //mp_subscript
    0 /* (objobjargproc)PyBobIoVideoReader_SetItem //mp_ass_subscript */
};

/*****************************************
 * Definition of Iterator to VideoReader *
 *****************************************/

static const char* s_videoreaderiterator(BOB_EXT_MODULE_PREFIX ".reader.iter");


static PyObject* PyBobIoVideoReaderIterator_Iter (PyBobIoVideoReaderIteratorObject* self) {
  return Py_BuildValue("O", self);
}

static PyObject* PyBobIoVideoReaderIterator_Next (PyBobIoVideoReaderIteratorObject* self) {

  if ((*self->iter == self->pyreader->v->end()) ||
      (self->iter->cur() == self->pyreader->v->numberOfFrames())) {
    self->iter->reset();
    self->iter.reset();
    Py_XDECREF((PyObject*)self->pyreader);
    return 0;
  }

  const bob::io::base::array::typeinfo& info = self->pyreader->v->frame_type();

  npy_intp shape[NPY_MAXDIMS];
  for (size_t k=0; k<info.nd; ++k) shape[k] = info.shape[k];

  int type_num = PyBobIo_AsTypenum(info.dtype);
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  PyObject* retval = PyArray_SimpleNew(info.nd, shape, type_num);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  try {
    bobskin skin((PyArrayObject*)retval, info.dtype);
    self->iter->read(skin);
  }
  catch (std::exception& e) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    if (!PyErr_Occurred()) PyErr_Format(PyExc_RuntimeError, "caught unknown exception while reading frame #%" PY_FORMAT_SIZE_T "d from file `%s'", self->iter->cur(), self->pyreader->v->filename().c_str());
    return 0;
  }

  Py_INCREF(retval);
  return retval;

}

#if PY_VERSION_HEX >= 0x03000000
#  define Py_TPFLAGS_HAVE_ITER 0
#endif

static PyObject* PyBobIoVideoReader_Iter (PyBobIoVideoReaderObject* self) {

  /* Allocates the python object itself */
  PyBobIoVideoReaderIteratorObject* retval = (PyBobIoVideoReaderIteratorObject*)PyBobIoVideoReaderIterator_Type.tp_new(&PyBobIoVideoReaderIterator_Type, 0, 0);
  if (!retval) return 0;

  Py_INCREF(self);
  retval->pyreader = self;
  retval->iter.reset(new bob::io::video::Reader::const_iterator(self->v->begin()));
  return Py_BuildValue("N", retval);
}


PyTypeObject PyBobIoVideoReader_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    0
};

PyTypeObject PyBobIoVideoReaderIterator_Type = {
    PyVarObject_HEAD_INIT(0, 0)
  0
};

bool init_BobIoVideoReader(PyObject* module){

  // initialize the reader
  PyBobIoVideoReader_Type.tp_name = s_reader.name();
  PyBobIoVideoReader_Type.tp_basicsize = sizeof(PyBobIoVideoReaderObject);
  PyBobIoVideoReader_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobIoVideoReader_Type.tp_doc = s_reader.doc();

  // set the functions
  PyBobIoVideoReader_Type.tp_new = PyType_GenericNew;
  PyBobIoVideoReader_Type.tp_init = reinterpret_cast<initproc>(PyBobIoVideoReader_Init);
  PyBobIoVideoReader_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIoVideoReader_Delete);
  PyBobIoVideoReader_Type.tp_methods = PyBobIoVideoReader_Methods;
  PyBobIoVideoReader_Type.tp_getset = PyBobIoVideoReader_getseters;
  PyBobIoVideoReader_Type.tp_iter = reinterpret_cast<getiterfunc>(PyBobIoVideoReader_Iter);


  PyBobIoVideoReader_Type.tp_str = reinterpret_cast<reprfunc>(PyBobIoVideoReader_Print);
  PyBobIoVideoReader_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobIoVideoReader_Repr);
  PyBobIoVideoReader_Type.tp_as_mapping = &PyBobIoVideoReader_Mapping;

  // check that everything is fine
  if (PyType_Ready(&PyBobIoVideoReader_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIoVideoReader_Type);
  if (PyModule_AddObject(module, "reader", (PyObject*)&PyBobIoVideoReader_Type) < 0)
    return false;

  // initialize the iterator
  PyBobIoVideoReaderIterator_Type.tp_name = s_videoreaderiterator;
  PyBobIoVideoReaderIterator_Type.tp_basicsize = sizeof(PyBobIoVideoReaderIteratorObject);
  PyBobIoVideoReaderIterator_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER;

  PyBobIoVideoReaderIterator_Type.tp_new = PyType_GenericNew;
  PyBobIoVideoReaderIterator_Type.tp_iter = reinterpret_cast<getiterfunc>(PyBobIoVideoReaderIterator_Iter);
  PyBobIoVideoReaderIterator_Type.tp_iternext = reinterpret_cast<getiterfunc>(PyBobIoVideoReaderIterator_Next);

  // check that everything is fine
  if (PyType_Ready(&PyBobIoVideoReaderIterator_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIoVideoReaderIterator_Type);
  return true;
}
