/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  5 Nov 11:16:09 2013
 *
 * @brief Bindings to bob::io::base::File
 */

#define BOB_IO_BASE_MODULE
#include "bobskin.h"
#include <bob.io.base/api.h>
#include <numpy/arrayobject.h>
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>
#include <stdexcept>

#include <bob.io.base/CodecRegistry.h>
#include <bob.io.base/utils.h>

/* Creates an exception message including the name of the given file, if possible */
inline const std::string exception_message(PyBobIoFileObject* self, const std::string& name){
  std::ostringstream str;
  str << name << " (";
  try{
    str << "'" << self->f->filename() << "'";
  } catch (...){
    str << "<unkown>";
  }
  str << ")";
  return  str.str();
}

static auto s_file = bob::extension::ClassDoc(
  "File",
  "Use this object to read and write data into files"
)
.add_constructor(
  bob::extension::FunctionDoc(
    "File",
    "Opens a file for reading or writing",
    "Normally, we read the file matching the extension to one of the available codecs installed with the present release of Bob. "
    "If you set the ``pretend_extension`` parameter though, we will read the file as it had a given extension. "
    "The value should start with a ``'.'``. "
    "For example ``'.hdf5'``, to make the file be treated like an HDF5 file.",
    true
  )
  .add_prototype("filename, [mode], [pretend_extension]", "")
  .add_parameter("filename", "str", "The file path to the file you want to open")
  .add_parameter("mode", "one of ('r', 'w', 'a')", "[Default: ``'r'``] A single character indicating if you'd like to ``'r'``\\ ead, ``'w'``\\ rite or ``'a'``\\ ppend into the file; if you choose ``'w'`` and the file already exists, it will be truncated")
  .add_parameter("pretend_extension", "str", "[optional] An extension to use; see :py:func:`bob.io.base.extensions` for a list of (currently) supported extensions")
);
/* How to create a new PyBobIoFileObject */
static PyObject* PyBobIoFile_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBobIoFileObject* self = (PyBobIoFileObject*)type->tp_alloc(type, 0);

  self->f.reset();

  return reinterpret_cast<PyObject*>(self);
}

static void PyBobIoFile_Delete (PyBobIoFileObject* o) {

  o->f.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);

}

int PyBobIo_FilenameConverter (PyObject* o, const char** b) {
#if PY_VERSION_HEX >= 0x03000000
  if (PyUnicode_Check(o)) {
    *b = PyUnicode_AsUTF8(o);
  } else {
    PyObject* temp = PyObject_Bytes(o);
    if (!temp) return 0;
    auto temp_ = make_safe(temp);
    *b = PyBytes_AsString(temp);
  }
#else
  if (PyUnicode_Check(o)) {
    PyObject* temp = PyUnicode_AsEncodedString(o, Py_FileSystemDefaultEncoding, "strict");
    if (!temp) return 0;
    auto temp_ = make_safe(temp);
    *b = PyString_AsString(temp);
  } else {
    *b = PyString_AsString(o);
  }
#endif
  return b != 0;
}

/* The __init__(self) method */
static int PyBobIoFile_init(PyBobIoFileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_file.kwlist();

  const char* filename;
  const char* pretend_extension = 0;

#if PY_VERSION_HEX >= 0x03000000
#  define MODE_CHAR "C"
  int mode = 'r';
#else
#  define MODE_CHAR "c"
  char mode = 'r';
#endif

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|" MODE_CHAR "s", kwlist,
        &PyBobIo_FilenameConverter, &filename, &mode, &pretend_extension)) return -1;

#undef MODE_CHAR

  if (mode != 'r' && mode != 'w' && mode != 'a') {
    PyErr_Format(PyExc_ValueError, "file open mode string should have 1 element and be either 'r' (read), 'w' (write) or 'a' (append)");
    return -1;
  }

  if (pretend_extension) {
    self->f = bob::io::base::open(filename, mode, pretend_extension);
  }
  else {
    self->f = bob::io::base::open(filename, mode);
  }

  return 0; ///< SUCCESS
BOB_CATCH_MEMBER("constructor", -1);
}

static PyObject* PyBobIoFile_repr(PyBobIoFileObject* self) {
  return PyString_FromFormat("%s(filename='%s', codec='%s')", Py_TYPE(self)->tp_name, self->f->filename(), self->f->name());
}

static auto s_filename = bob::extension::VariableDoc(
  "filename",
  "str",
  "The path to the file being read/written"
);
static PyObject* PyBobIoFile_Filename(PyBobIoFileObject* self) {
  return Py_BuildValue("s", self->f->filename());
}

static auto s_codec_name = bob::extension::VariableDoc(
  "codec_name",
  "str",
  "Name of the File class implementation",
  "This variable is available for compatibility reasons with the previous versions of this library."
);
static PyObject* PyBobIoFile_CodecName(PyBobIoFileObject* self) {
  return Py_BuildValue("s", self->f->name());
}


static PyGetSetDef PyBobIoFile_getseters[] = {
    {
      s_filename.name(),
      (getter)PyBobIoFile_Filename,
      0,
      s_filename.doc(),
      0,
    },
    {
      s_codec_name.name(),
      (getter)PyBobIoFile_CodecName,
      0,
      s_codec_name.doc(),
      0,
    },
    {0}  /* Sentinel */
};

static Py_ssize_t PyBobIoFile_len (PyBobIoFileObject* self) {
  Py_ssize_t retval = self->f->size();
  return retval;
}

int PyBobIo_AsTypenum (bob::io::base::array::ElementType type) {

  switch(type) {
    case bob::io::base::array::t_bool:
      return NPY_BOOL;
    case bob::io::base::array::t_int8:
      return NPY_INT8;
    case bob::io::base::array::t_int16:
      return NPY_INT16;
    case bob::io::base::array::t_int32:
      return NPY_INT32;
    case bob::io::base::array::t_int64:
      return NPY_INT64;
    case bob::io::base::array::t_uint8:
      return NPY_UINT8;
    case bob::io::base::array::t_uint16:
      return NPY_UINT16;
    case bob::io::base::array::t_uint32:
      return NPY_UINT32;
    case bob::io::base::array::t_uint64:
      return NPY_UINT64;
    case bob::io::base::array::t_float32:
      return NPY_FLOAT32;
    case bob::io::base::array::t_float64:
      return NPY_FLOAT64;
#ifdef NPY_FLOAT128
    case bob::io::base::array::t_float128:
      return NPY_FLOAT128;
#endif
    case bob::io::base::array::t_complex64:
      return NPY_COMPLEX64;
    case bob::io::base::array::t_complex128:
      return NPY_COMPLEX128;
#ifdef NPY_COMPLEX256
    case bob::io::base::array::t_complex256:
      return NPY_COMPLEX256;
#endif
    default:
      PyErr_Format(PyExc_TypeError, "unsupported Bob/C++ element type (%s)", bob::io::base::array::stringize(type));
  }

  return NPY_NOTYPE;

}

static PyObject* PyBobIoFile_getIndex (PyBobIoFileObject* self, Py_ssize_t i) {
  if (i < 0) i += self->f->size(); ///< adjust for negative indexing

  if (i < 0 || (size_t)i >= self->f->size()) {
    PyErr_Format(PyExc_IndexError, "file index out of range - `%s' only contains %" PY_FORMAT_SIZE_T "d object(s)", self->f->filename(), self->f->size());
    return 0;
  }

  const bob::io::base::array::typeinfo& info = self->f->type();

  npy_intp shape[NPY_MAXDIMS];
  for (size_t k=0; k<info.nd; ++k) shape[k] = info.shape[k];

  int type_num = PyBobIo_AsTypenum(info.dtype);
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  PyObject* retval = PyArray_SimpleNew(info.nd, shape, type_num);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  bobskin skin((PyArrayObject*)retval, info.dtype);
  self->f->read(skin, i);
  return Py_BuildValue("O", retval);
}

static PyObject* PyBobIoFile_getSlice (PyBobIoFileObject* self, PySliceObject* slice) {

  Py_ssize_t start, stop, step, slicelength;
#if PY_VERSION_HEX < 0x03000000
  if (PySlice_GetIndicesEx(slice,
#else
  if (PySlice_GetIndicesEx(reinterpret_cast<PyObject*>(slice),
#endif
        self->f->size(), &start, &stop, &step, &slicelength) < 0) return 0;

  //creates the return array
  const bob::io::base::array::typeinfo& info = self->f->type();

  int type_num = PyBobIo_AsTypenum(info.dtype);
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  if (slicelength <= 0) return PyArray_SimpleNew(0, 0, type_num);

  npy_intp shape[NPY_MAXDIMS];
  shape[0] = slicelength;
  for (size_t k=0; k<info.nd; ++k) shape[k+1] = info.shape[k];

  PyObject* retval = PyArray_SimpleNew(info.nd+1, shape, type_num);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  Py_ssize_t counter = 0;
  for (auto i = start; (start<=stop)?i<stop:i>stop; i+=step) {

    //get slice to fill
    PyObject* islice = Py_BuildValue("n", counter++);
    if (!islice) return 0;
    auto islice_ = make_safe(islice);

    PyObject* item = PyObject_GetItem(retval, islice);
    if (!item) return 0;
    auto item_ = make_safe(item);

    bobskin skin((PyArrayObject*)item, info.dtype);
    self->f->read(skin, i);
  }

  return Py_BuildValue("O", retval);
}

static PyObject* PyBobIoFile_getItem (PyBobIoFileObject* self, PyObject* item) {
  if (PyIndex_Check(item)) {
   Py_ssize_t i = PyNumber_AsSsize_t(item, PyExc_IndexError);
   if (i == -1 && PyErr_Occurred()) return 0;
   return PyBobIoFile_getIndex(self, i);
  }
  if (PySlice_Check(item)) {
   return PyBobIoFile_getSlice(self, (PySliceObject*)item);
  }
  else {
   PyErr_Format(PyExc_TypeError, "File indices must be integers, not %s", Py_TYPE(item)->tp_name);
   return 0;
  }
}

static PyMappingMethods PyBobIoFile_Mapping = {
    (lenfunc)PyBobIoFile_len, //mp_length
    (binaryfunc)PyBobIoFile_getItem, //mp_subscript
    0 /* (objobjargproc)PyBobIoFile_SetItem //mp_ass_subscript */
};


static auto s_read = bob::extension::FunctionDoc(
  "read",
  "Reads a specific object in the file, or the whole file",
  "This method reads data from the file. "
  "If you specified an ``index``, it reads just the object indicated by the index, as you would do using the ``[]`` operator. "
  "If the ``index`` is not specified, reads the whole contents of the file into a :py:class:`numpy.ndarray`.",
  true
)
.add_prototype("[index]", "data")
.add_parameter("index", "int", "[optional] The index to the object one wishes to retrieve from the file; negative indexing is supported; if not given, implies retrieval of the whole file contents.")
.add_return("data", ":py:class:`numpy.ndarray`", "The contents of the file, as array")
;
static PyObject* PyBobIoFile_read(PyBobIoFileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_read.kwlist();

  Py_ssize_t i = PY_SSIZE_T_MIN;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &i)) return 0;

  if (i != PY_SSIZE_T_MIN) {

    // reads a specific object inside the file

    if (i < 0) i += self->f->size();

    if (i < 0 || (size_t)i >= self->f->size()) {
      PyErr_Format(PyExc_IndexError, "file index out of range - `%s' only contains %" PY_FORMAT_SIZE_T "d object(s)", self->f->filename(), self->f->size());
      return 0;
    }

    return PyBobIoFile_getIndex(self, i);

  }

  // reads the whole file in a single shot

  const bob::io::base::array::typeinfo& info = self->f->type_all();

  npy_intp shape[NPY_MAXDIMS];
  for (size_t k=0; k<info.nd; ++k) shape[k] = info.shape[k];

  int type_num = PyBobIo_AsTypenum(info.dtype);
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  PyObject* retval = PyArray_SimpleNew(info.nd, shape, type_num);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  bobskin skin((PyArrayObject*)retval, info.dtype);
  self->f->read_all(skin);

  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER(exception_message(self, s_read.name()).c_str(), 0)
}


static auto s_write = bob::extension::FunctionDoc(
  "write",
  "Writes the contents of an object to the file",
  "This method writes data to the file. "
  "It acts like the given array is the only piece of data that will ever be written to such a file. "
  "No more data appending may happen after a call to this method.",
  true
)
.add_prototype("data")
.add_parameter("data", "array_like", "The array to be written into the file; it can be a :py:class:`numpy.ndarray`, a :py:class:`bob.blitz.array` or any other object which can be converted to either of them")
;
static PyObject* PyBobIoFile_write(PyBobIoFileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_write.kwlist();

  PyBlitzArrayObject* bz = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBlitzArray_Converter, &bz)) return 0;

  auto bz_ = make_safe(bz);

  bobskin skin(bz);
  self->f->write(skin);

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_write.name()).c_str(), 0)
}


static auto s_append = bob::extension::FunctionDoc(
  "append",
  "Adds the contents of an object to the file",
  "This method appends data to the file. "
  "If the file does not exist, creates a new file, else, makes sure that the inserted array respects the previously set file structure.",
  true
)
.add_prototype("data", "position")
.add_parameter("data", "array_like", "The array to be written into the file; it can be a :py:class:`numpy.ndarray`, a :py:class:`bob.blitz.array` or any other object which can be converted to either of them")
.add_return("position", "int", "The current position of the newly written data")
;
static PyObject* PyBobIoFile_append(PyBobIoFileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_append.kwlist();

  PyBlitzArrayObject* bz = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBlitzArray_Converter, &bz)) return 0;
  auto bz_ = make_safe(bz);
  Py_ssize_t pos = -1;

  bobskin skin(bz);
  pos = self->f->append(skin);

  return Py_BuildValue("n", pos);
BOB_CATCH_MEMBER(exception_message(self, s_append.name()).c_str(), 0)
}


PyObject* PyBobIo_TypeInfoAsTuple (const bob::io::base::array::typeinfo& ti) {

  int type_num = PyBobIo_AsTypenum(ti.dtype);
  if (type_num == NPY_NOTYPE) return 0;

  PyObject* retval = Py_BuildValue("NNN",
      reinterpret_cast<PyObject*>(PyArray_DescrFromType(type_num)),
      PyTuple_New(ti.nd), //shape
      PyTuple_New(ti.nd)  //strides
      );
  if (!retval) return 0;

  PyObject* shape = PyTuple_GET_ITEM(retval, 1);
  PyObject* stride = PyTuple_GET_ITEM(retval, 2);
  for (Py_ssize_t i=0; (size_t)i<ti.nd; ++i) {
    PyTuple_SET_ITEM(shape, i, Py_BuildValue("n", ti.shape[i]));
    PyTuple_SET_ITEM(stride, i, Py_BuildValue("n", ti.stride[i]));
  }

  return retval;
}

static auto s_describe = bob::extension::FunctionDoc(
  "describe",
  "Returns a description (dtype, shape, stride) of data at the file",
  0,
  true
)
.add_prototype("[all]", "dtype, shape, stride")
.add_parameter("all", "bool", "[Default: ``False``]  If set to ``True``, returns the shape and strides for reading the whole file contents in one shot.")
.add_return("dtype", ":py:class:`numpy.dtype`", "The data type of the object")
.add_return("shape", "tuple", "The shape of the object")
.add_return("stride", "tuple", "The stride of the object")
;
static PyObject* PyBobIoFile_describe(PyBobIoFileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"all", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* all = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &all)) return 0;

  const bob::io::base::array::typeinfo* info = 0;
  if (all && PyObject_IsTrue(all)) info = &self->f->type_all();
  else info = &self->f->type();

  /* Now return type description and tuples with shape and strides */
  return PyBobIo_TypeInfoAsTuple(*info);
BOB_CATCH_MEMBER(exception_message(self, s_describe.name()).c_str(), 0)
}


static PyMethodDef PyBobIoFile_methods[] = {
    {
      s_read.name(),
      (PyCFunction)PyBobIoFile_read,
      METH_VARARGS|METH_KEYWORDS,
      s_read.doc(),
    },
    {
      s_write.name(),
      (PyCFunction)PyBobIoFile_write,
      METH_VARARGS|METH_KEYWORDS,
      s_write.doc(),
    },
    {
      s_append.name(),
      (PyCFunction)PyBobIoFile_append,
      METH_VARARGS|METH_KEYWORDS,
      s_append.doc(),
    },
    {
      s_describe.name(),
      (PyCFunction)PyBobIoFile_describe,
      METH_VARARGS|METH_KEYWORDS,
      s_describe.doc(),
    },
    {0}  /* Sentinel */
};

/**********************************
 * Definition of Iterator to File *
 **********************************/

PyDoc_STRVAR(s_fileiterator_str, BOB_EXT_MODULE_PREFIX ".File.iter");

/* How to create a new PyBobIoFileIteratorObject */
static PyObject* PyBobIoFileIterator_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBobIoFileIteratorObject* self = (PyBobIoFileIteratorObject*)type->tp_alloc(type, 0);

  return reinterpret_cast<PyObject*>(self);
}

static PyObject* PyBobIoFileIterator_iter (PyBobIoFileIteratorObject* self) {
  return reinterpret_cast<PyObject*>(self);
}

static PyObject* PyBobIoFileIterator_next (PyBobIoFileIteratorObject* self) {
  if ((size_t)self->curpos >= self->pyfile->f->size()) {
    Py_XDECREF((PyObject*)self->pyfile);
    self->pyfile = 0;
    return 0;
  }
  return PyBobIoFile_getIndex(self->pyfile, self->curpos++);
}

static PyObject* PyBobIoFile_iter (PyBobIoFileObject* self) {
  PyBobIoFileIteratorObject* retval = (PyBobIoFileIteratorObject*)PyBobIoFileIterator_New(&PyBobIoFileIterator_Type, 0, 0);
  if (!retval) return 0;
  retval->pyfile = self;
  retval->curpos = 0;
  return Py_BuildValue("N", retval);
}

#if PY_VERSION_HEX >= 0x03000000
#  define Py_TPFLAGS_HAVE_ITER 0
#endif

PyTypeObject PyBobIoFileIterator_Type = {
  PyVarObject_HEAD_INIT(0, 0)
  0
};


PyTypeObject PyBobIoFile_Type = {
  PyVarObject_HEAD_INIT(0, 0)
  0
};

bool init_File(PyObject* module){

  // initialize the iterator
  PyBobIoFileIterator_Type.tp_name = s_fileiterator_str;
  PyBobIoFileIterator_Type.tp_basicsize = sizeof(PyBobIoFileIteratorObject);
  PyBobIoFileIterator_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER;
  PyBobIoFileIterator_Type.tp_iter = (getiterfunc)PyBobIoFileIterator_iter;
  PyBobIoFileIterator_Type.tp_iternext = (iternextfunc)PyBobIoFileIterator_next;

  // initialize the File
  PyBobIoFile_Type.tp_name = s_file.name();
  PyBobIoFile_Type.tp_basicsize = sizeof(PyBobIoFileObject);
  PyBobIoFile_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobIoFile_Type.tp_doc = s_file.doc();

  // set the functions
  PyBobIoFile_Type.tp_new = PyBobIoFile_New;
  PyBobIoFile_Type.tp_init = reinterpret_cast<initproc>(PyBobIoFile_init);
  PyBobIoFile_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIoFile_Delete);
  PyBobIoFile_Type.tp_methods = PyBobIoFile_methods;
  PyBobIoFile_Type.tp_getset = PyBobIoFile_getseters;
  PyBobIoFile_Type.tp_iter = (getiterfunc)PyBobIoFile_iter;

  PyBobIoFile_Type.tp_str = reinterpret_cast<reprfunc>(PyBobIoFile_repr);
  PyBobIoFile_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobIoFile_repr);
  PyBobIoFile_Type.tp_as_mapping = &PyBobIoFile_Mapping;


  // check that everything is fine
  if (PyType_Ready(&PyBobIoFile_Type) < 0)
    return false;
  if (PyType_Ready(&PyBobIoFileIterator_Type) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&PyBobIoFile_Type);
  bool success = PyModule_AddObject(module, s_file.name(), (PyObject*)&PyBobIoFile_Type) >= 0;
  if (!success) return false;
  Py_INCREF(&PyBobIoFileIterator_Type);
  success = PyModule_AddObject(module, s_fileiterator_str, (PyObject*)&PyBobIoFileIterator_Type) >= 0;
  return success;
}
