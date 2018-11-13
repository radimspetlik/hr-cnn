/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 12 Nov 18:19:22 2013
 *
 * @brief Bindings to bob::io::base::HDF5File
 */

#define BOB_IO_BASE_MODULE
#include <bob.io.base/api.h>

#include <boost/make_shared.hpp>
#include <numpy/arrayobject.h>
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>
#include <stdexcept>
#include <cstring>

/* Creates an exception message including the name of the given file, if possible */
inline const std::string exception_message(PyBobIoHDF5FileObject* self, const std::string& name){
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

static auto s_hdf5file = bob::extension::ClassDoc(
  "HDF5File",
  "Reads and writes data to HDF5 files.",
  "HDF5 stands for Hierarchical Data Format version 5. "
  "It is a flexible, binary file format that allows one to store and read data efficiently into or from files. "
  "It is a cross-platform, cross-architecture format.\n\n"
  "Objects of this class allows users to read and write data from and to files in HDF5 format. "
  "For an introduction to HDF5, visit the `HDF5 Website <http://www.hdfgroup.org/HDF5>`_."
)
.add_constructor(
  bob::extension::FunctionDoc(
    "HDF5File",
    "Opens an HFF5 file for reading, writing or appending.",
    "For the ``open`` mode, use ``'r'`` for read-only ``'a'`` for read/write/append, ``'w'`` for read/write/truncate or ``'x'`` for (read/write/exclusive). "
    "When another :py:class:`HDF5File` object is given, a shallow copy is created, pointing to the same file."
  )
  .add_prototype("filename, [mode]","")
  .add_prototype("hdf5", "")
  .add_parameter("filename", "str", "The file path to the file you want to open for reading or writing")
  .add_parameter("mode", "one of ('r', 'w', 'a', 'x')", "[Default: ``'r'``]  The opening mode")
  .add_parameter("hdf5", ":py:class:`HDF5File`", "An HDF5 file to copy-construct")
);


int PyBobIoHDF5File_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIoHDF5File_Type));
}

int PyBobIoHDF5File_Converter(PyObject* o, PyBobIoHDF5FileObject** a) {
  if (!PyBobIoHDF5File_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBobIoHDF5FileObject*>(o);
  return 1;
}

/* How to create a new PyBobIoHDF5FileObject */
static PyObject* PyBobIoHDF5File_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBobIoHDF5FileObject* self = (PyBobIoHDF5FileObject*)type->tp_alloc(type, 0);

  self->f.reset();

  return reinterpret_cast<PyObject*>(self);
}

static void PyBobIoHDF5File_Delete (PyBobIoHDF5FileObject* o) {

  o->f.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);
}

static bob::io::base::HDF5File::mode_t mode_from_char (char mode) {

  bob::io::base::HDF5File::mode_t new_mode = bob::io::base::HDF5File::inout;

  switch (mode) {
    case 'r': new_mode = bob::io::base::HDF5File::in; break;
    case 'a': new_mode = bob::io::base::HDF5File::inout; break;
    case 'w': new_mode = bob::io::base::HDF5File::trunc; break;
    case 'x': new_mode = bob::io::base::HDF5File::excl; break;
    default:
      PyErr_SetString(PyExc_RuntimeError, "Supported flags are 'r' (read-only), 'a' (read/write/append), 'w' (read/write/truncate) or 'x' (read/write/exclusive)");
  }

  return new_mode;

}

/* The __init__(self) method */
static int PyBobIoHDF5File_init(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist1 = s_hdf5file.kwlist(0);
  static char** kwlist2 = s_hdf5file.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  if (!nargs){
    // at least one argument is required
    PyErr_Format(PyExc_TypeError, "`%s' constructor requires at least one parameter", Py_TYPE(self)->tp_name);
    return -1;
  } // nargs == 0

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (
    (kwds && PyDict_Contains(kwds, k)) ||
    (args && PyBobIoHDF5File_Check(PyTuple_GET_ITEM(args, 0)))
  ){
    PyBobIoHDF5FileObject* other;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist2, &PyBobIoHDF5File_Converter, &other))
      return -1;
    auto other_ = make_safe(other);
    self->f = other->f;
    return 0;
  }

#if PY_VERSION_HEX >= 0x03000000
#  define MODE_CHAR "C"
  int mode = 'r';
#else
#  define MODE_CHAR "c"
  char mode = 'r';
#endif

  const char* filename;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|" MODE_CHAR, kwlist1,
        &PyBobIo_FilenameConverter, &filename, &mode))
    return -1;

#undef MODE_CHAR

  if (mode != 'r' && mode != 'w' && mode != 'a' && mode != 'x') {
    PyErr_Format(PyExc_ValueError, "file open mode string should have 1 element and be either 'r' (read), 'w' (write), 'a' (append), 'x' (exclusive)");
    return -1;
  }
  bob::io::base::HDF5File::mode_t mode_mode = mode_from_char(mode);
  if (PyErr_Occurred()) return -1;

  self->f.reset(new bob::io::base::HDF5File(filename, mode_mode));
  return 0; ///< SUCCESS
BOB_CATCH_MEMBER("hdf5 constructor", -1)
}


static PyObject* PyBobIoHDF5File_repr(PyBobIoHDF5FileObject* self) {
BOB_TRY
  return PyString_FromFormat("%s(filename='%s')", Py_TYPE(self)->tp_name, self->f->filename().c_str());
BOB_CATCH_MEMBER("__repr__", 0)
}


static auto s_flush = bob::extension::FunctionDoc(
  "flush",
  "Flushes the content of the HDF5 file to disk",
  "When the HDF5File is open for writing, this function synchronizes the contents on the disk with the one from the file. "
  "When the file is open for reading, nothing happens.",
  true
)
  .add_prototype("")
;
static PyObject* PyBobIoHDF5File_flush(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_flush.kwlist();

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist)) return 0;

  self->f->flush();
  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_flush.name()).c_str(), 0)
}


static auto s_close = bob::extension::FunctionDoc(
  "close",
  "Closes this file",
  "This function closes the HDF5File after flushing all its contents to disk. "
  "After the HDF5File is closed, any operation on it will result in an exception.",
  true
)
.add_prototype("")
;
static PyObject* PyBobIoHDF5File_close(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_close.kwlist();

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist)) return 0;

  self->f->close();

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_close.name()).c_str(), 0);
}


static auto s_cd = bob::extension::FunctionDoc(
  "cd",
  "Changes the current prefix path",
  "When this object is created the prefix path is empty, which means all following paths to data objects should be given using the full path. "
  "If you set the ``path`` to a different value, it will be used as a prefix to any subsequent operation until you reset it. "
  "If ``path`` starts with ``'/'``, it is treated as an absolute path. "
  "If the value is relative, it is added to the current path; ``'..'`` and ``'.'`` are supported. "
  "If it is absolute, it causes the prefix to be reset.\n\n"
  "..note:: All operations taking a relative path, following a :py:func:`cd`, will be considered relative to the value defined by the :py:attr:`cwd` property of this object.",
  true
)
.add_prototype("path")
.add_parameter("path", "str", "The path to change directories to")
;

static PyObject* PyBobIoHDF5File_changeDirectory(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_cd.kwlist();

  const char* path = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &path)) return 0;

  self->f->cd(path);

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_cd.name()).c_str(), 0)
}


static auto s_has_group = bob::extension::FunctionDoc(
  "has_group",
  "Checks if a path (group) exists inside a file",
  "This method does not work for datasets, only for directories. "
  "If the given path is relative, it is take w.r.t. to the current working directory.",
  true
)
.add_prototype("path")
.add_parameter("path", "str", "The path to check")
;
static PyObject* PyBobIoHDF5File_hasGroup(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_has_group.kwlist();

  const char* path = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &path)) return 0;

  if (self->f->hasGroup(path)) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
BOB_CATCH_MEMBER(exception_message(self, s_has_group.name()).c_str(), 0)
}


static auto s_create_group = bob::extension::FunctionDoc(
  "create_group",
  "Creates a new path (group) inside the file",
  "A relative path is taken w.r.t. to the current directory. "
  "If the directory already exists (check it with :py:meth:`has_group`), an exception will be raised.",
  true
)
.add_prototype("path")
.add_parameter("path", "str", "The path to create.")
;
static PyObject* PyBobIoHDF5File_createGroup(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_create_group.kwlist();

  const char* path = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &path)) return 0;

  self->f->createGroup(path);

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_create_group.name()).c_str(), 0)
}


static auto s_has_dataset = bob::extension::FunctionDoc(
  "has_dataset",
  "Checks if a dataset exists inside a file",
  "Checks if a dataset exists inside a file, on the specified path. "
  "If the given path is relative, it is take w.r.t. to the current working directory.\n\n"
  ".. note:: The functions :py:meth:`has_dataset` and :py:meth:`has_key` are synonyms. "
  "You can also use the Python's ``in`` operator instead of :py:meth:`has_key`: ``key in hdf5file``.",
  true
)
.add_prototype("key")
.add_parameter("key", "str", "The dataset path to check")
;
auto s_has_key = s_has_dataset.clone("has_key");
static PyObject* PyBobIoHDF5File_hasDataset(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_has_dataset.kwlist();

  const char* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &key)) return 0;

  if (self->f->contains(key)) Py_RETURN_TRUE;

  Py_RETURN_FALSE;
BOB_CATCH_MEMBER(exception_message(self, s_has_dataset.name()).c_str(), 0)
}

static bob::io::base::hdf5type PyBobIo_H5FromTypenum (int type_num) {

  switch(type_num) {
    case NPY_STRING:     return bob::io::base::s;
    case NPY_BOOL:       return bob::io::base::b;
    case NPY_INT8:       return bob::io::base::i8;
    case NPY_INT16:      return bob::io::base::i16;
    case NPY_INT32:      return bob::io::base::i32;
    case NPY_INT64:      return bob::io::base::i64;
    case NPY_UINT8:      return bob::io::base::u8;
    case NPY_UINT16:     return bob::io::base::u16;
    case NPY_UINT32:     return bob::io::base::u32;
    case NPY_UINT64:     return bob::io::base::u64;
    case NPY_FLOAT32:    return bob::io::base::f32;
    case NPY_FLOAT64:    return bob::io::base::f64;
#ifdef NPY_FLOAT128
    case NPY_FLOAT128:   return bob::io::base::f128;
#endif
    case NPY_COMPLEX64:  return bob::io::base::c64;
    case NPY_COMPLEX128: return bob::io::base::c128;
#ifdef NPY_COMPLEX256
    case NPY_COMPLEX256: return bob::io::base::c256;
#endif
#if defined(__LP64__) || defined(__APPLE__)
    case NPY_LONGLONG:
                         switch (NPY_BITSOF_LONGLONG) {
                           case 8: return bob::io::base::i8;
                           case 16: return bob::io::base::i16;
                           case 32: return bob::io::base::i32;
                           case 64: return bob::io::base::i64;
                           default: return bob::io::base::unsupported;
                         }
                         break;
    case NPY_ULONGLONG:
                         switch (NPY_BITSOF_LONGLONG) {
                           case 8: return bob::io::base::u8;
                           case 16: return bob::io::base::u16;
                           case 32: return bob::io::base::u32;
                           case 64: return bob::io::base::u64;
                           default: return bob::io::base::unsupported;
                         }
                         break;
#endif
    default:             return bob::io::base::unsupported;
  }

}

static int PyBobIo_H5AsTypenum (bob::io::base::hdf5type type) {

  switch(type) {
    case bob::io::base::s:    return NPY_STRING;
    case bob::io::base::b:    return NPY_BOOL;
    case bob::io::base::i8:   return NPY_INT8;
    case bob::io::base::i16:  return NPY_INT16;
    case bob::io::base::i32:  return NPY_INT32;
    case bob::io::base::i64:  return NPY_INT64;
    case bob::io::base::u8:   return NPY_UINT8;
    case bob::io::base::u16:  return NPY_UINT16;
    case bob::io::base::u32:  return NPY_UINT32;
    case bob::io::base::u64:  return NPY_UINT64;
    case bob::io::base::f32:  return NPY_FLOAT32;
    case bob::io::base::f64:  return NPY_FLOAT64;
#ifdef NPY_FLOAT128
    case bob::io::base::f128: return NPY_FLOAT128;
#endif
    case bob::io::base::c64:  return NPY_COMPLEX64;
    case bob::io::base::c128: return NPY_COMPLEX128;
#ifdef NPY_COMPLEX256
    case bob::io::base::c256: return NPY_COMPLEX256;
#endif
    default:            return NPY_NOTYPE;
  }

}

static PyObject* PyBobIo_HDF5TypeAsTuple (const bob::io::base::HDF5Type& t) {

  const bob::io::base::HDF5Shape& sh = t.shape();
  size_t ndim = sh.n();
  const hsize_t* shptr = sh.get();

  int type_num = PyBobIo_H5AsTypenum(t.type());
  if (type_num == NPY_NOTYPE) {
    PyErr_Format(PyExc_TypeError, "unsupported HDF5 element type (%d) found during conversion to numpy type number", (int)t.type());
    return 0;
  }

  PyObject* dtype = reinterpret_cast<PyObject*>(PyArray_DescrFromType(type_num));
  if (!dtype) return 0;
  auto dtype_ = make_safe(dtype);

  PyObject* shape = PyTuple_New(ndim);
  if (!shape) return 0;
  auto shape_ = make_safe(shape);

  PyObject* retval = Py_BuildValue("OO", dtype, shape);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (Py_ssize_t i=0; i<(Py_ssize_t)ndim; ++i) {
    PyObject* value = Py_BuildValue("n", shptr[i]);
    if (!value) return 0;
    PyTuple_SET_ITEM(shape, i, value);
  }

  return Py_BuildValue("O", retval);

}

static PyObject* PyBobIo_HDF5DescriptorAsTuple (const bob::io::base::HDF5Descriptor& d) {
  return Py_BuildValue("NnO",
    PyBobIo_HDF5TypeAsTuple(d.type),
    d.size,
    d.expandable? Py_True : Py_False
  ); //steals references, except for True/False
}


static auto s_describe = bob::extension::FunctionDoc(
  "describe",
  "Describes a dataset type/shape, if it exists inside a file",
  "If a given ``key`` to an HDF5 dataset exists inside the file, returns a type description of objects recorded in such a dataset, otherwise, raises an exception. "
  "The returned value type is a list of tuples (HDF5Type, number-of-objects, expandable) describing the capabilities if the file is read using these formats. \n\n",
  true
)
.add_prototype("key", "[(hdf5type, size, expandable)]")
.add_parameter("key", "str", "The dataset path to describe")
.add_return("hdf5type", "tuple", "The HDF5Type of the returned array")
.add_return("size", "int", "The number of objects in the dataset")
.add_return("expandable", "bool", "Defines if this object can be resized.")
;
static PyObject* PyBobIoHDF5File_describe(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_describe.kwlist();

  const char* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &key)) return 0;

  const std::vector<bob::io::base::HDF5Descriptor>& dv = self->f->describe(key);
  PyObject* retval = PyList_New(dv.size());
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (size_t k=0; k<dv.size(); ++k) {
    PyObject* entry = PyBobIo_HDF5DescriptorAsTuple(dv[k]);
    if (!entry) return 0;
    PyList_SET_ITEM(retval, k, entry);
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER(exception_message(self, s_describe.name()).c_str(), 0)
}


static auto s_unlink = bob::extension::FunctionDoc(
  "unlink",
  "Unlinks datasets inside the file making them invisible",
  "If a given path to an HDF5 dataset exists inside the file, unlinks it."
  "Please note this will note remove the data from the file, just make it inaccessible. "
  "If you wish to cleanup, save the reacheable objects from this file to another :py:class:`HDF5File` object using :py:meth:`copy`, for example.",
  true
)
.add_prototype("key")
.add_parameter("key", "str", "The dataset path to unlink")
;
static PyObject* PyBobIoHDF5File_unlink(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_unlink.kwlist();

  const char* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &key)) return 0;

  self->f->unlink(key);

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_unlink.name()).c_str(), 0)
}


static auto s_rename = bob::extension::FunctionDoc(
  "rename",
  "Renames datasets in a file",
  0,
  true
)
.add_prototype("from, to")
.add_parameter("from", "str", "The path to the data to be renamed")
.add_parameter("to", "str", "The new name of the dataset")
;
static PyObject* PyBobIoHDF5File_rename(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_rename.kwlist();

  const char* from = 0;
  const char* to = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss", kwlist, &from, &to)) return 0;

  self->f->rename(from, to);

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_rename.name()).c_str(), 0)
}


static auto s_paths = bob::extension::FunctionDoc(
  "paths",
  "Lists datasets available inside this file",
  "Returns all paths to datasets available inside this file, stored under the current working directory. "
  "If ``relative`` is set to ``True``, the returned paths are relative to the current working directory, otherwise they are absolute.\n\n"
  ".. note:: The functions :py:meth:`keys` and :py:meth:`paths` are synonyms.",
  true
)
.add_prototype("[relative]", "paths")
.add_parameter("relative", "bool", "[Default: ``False``] If set to ``True``, the returned paths are relative to the current working directory, otherwise they are absolute")
.add_return("paths", "[str]", "A list of paths inside this file")
;
auto s_keys = s_paths.clone("keys");
static PyObject* PyBobIoHDF5File_paths(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_paths.kwlist();

  PyObject* pyrel = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &pyrel)) return 0;

  bool relative = false;
  if (pyrel && PyObject_IsTrue(pyrel)) relative = true;

  std::vector<std::string> values;
  self->f->paths(values, relative);
  PyObject* retval = PyList_New(values.size());
  if (!retval) return 0;
  auto retval_ = make_safe(retval);
  for (size_t i=0; i<values.size(); ++i) {
    PyList_SET_ITEM(retval, i, Py_BuildValue("s", values[i].c_str()));
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER(exception_message(self, s_paths.name()).c_str(), 0)
}


static auto s_sub_groups = bob::extension::FunctionDoc(
  "sub_groups",
  "Lists groups (directories) in the current file",
  0,
  true
)
.add_prototype("[relative], [recursive]", "groups")
.add_parameter("relative", "bool", "[Default: ``False``] If set to ``True``, the returned sub-groups are relative to the current working directory, otherwise they are absolute")
.add_parameter("recursive", "bool", "[Default: ``True``] If set to ``False``, the returned sub-groups   are only the ones in the current directory, otherwise recurses down the directory structure")
.add_return("groups", "[str]", "The list of directories (groups) inside this file")
;
static PyObject* PyBobIoHDF5File_subGroups(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_sub_groups.kwlist();

  PyObject* pyrel = 0;
  PyObject* pyrec = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &pyrel, &pyrec)) return 0;

  bool relative = (pyrel && PyObject_IsTrue(pyrel));
  bool recursive = (!pyrec || PyObject_IsTrue(pyrec));

  std::vector<std::string> values;
  self->f->sub_groups(values, relative, recursive);
  PyObject* retval = PyList_New(values.size());
  if (!retval) return 0;
  auto retval_ = make_safe(retval);
  for (size_t i=0; i<values.size(); ++i) {
    PyList_SET_ITEM(retval, i, Py_BuildValue("s", values[i].c_str()));
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER(exception_message(self, s_sub_groups.name()).c_str(), 0)
}


static PyObject* PyBobIoHDF5File_Xread(PyBobIoHDF5FileObject* self, const char* p, int descriptor, int pos) {

  const std::vector<bob::io::base::HDF5Descriptor> D = self->f->describe(p);

  //last descriptor always contains the full readout.
  const bob::io::base::HDF5Type& type = D[descriptor].type;
  const bob::io::base::HDF5Shape& shape = type.shape();

  if (shape.n() == 1 && shape[0] == 1) { //read as scalar
    switch(type.type()) {
      case bob::io::base::s:
        return Py_BuildValue("s", self->f->read<std::string>(p, pos).c_str());
      case bob::io::base::b:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<bool>(p, pos));
      case bob::io::base::i8:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<int8_t>(p, pos));
      case bob::io::base::i16:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<int16_t>(p, pos));
      case bob::io::base::i32:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<int32_t>(p, pos));
      case bob::io::base::i64:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<int64_t>(p, pos));
      case bob::io::base::u8:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<uint8_t>(p, pos));
      case bob::io::base::u16:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<uint16_t>(p, pos));
      case bob::io::base::u32:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<uint32_t>(p, pos));
      case bob::io::base::u64:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<uint64_t>(p, pos));
      case bob::io::base::f32:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<float>(p, pos));
      case bob::io::base::f64:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<double>(p, pos));
      case bob::io::base::f128:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<long double>(p, pos));
      case bob::io::base::c64:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<std::complex<float> >(p, pos));
      case bob::io::base::c128:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<std::complex<double> >(p, pos));
      case bob::io::base::c256:
        return PyBlitzArrayCxx_FromCScalar(self->f->read<std::complex<long double> >(p, pos));
      default:
        PyErr_Format(PyExc_TypeError, "unsupported HDF5 type: %s", type.str().c_str());
        return 0;
    }
  }

  //read as an numpy array
  int type_num = PyBobIo_H5AsTypenum(type.type());
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  npy_intp pyshape[NPY_MAXDIMS];
  for (size_t k=0; k<shape.n(); ++k) pyshape[k] = shape.get()[k];

  PyObject* retval = PyArray_SimpleNew(shape.n(), pyshape, type_num);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  self->f->read_buffer(p, pos, type, PyArray_DATA((PyArrayObject*)retval));

  return Py_BuildValue("O", retval);
}


static auto s_read = bob::extension::FunctionDoc(
  "read",
  "Reads whole datasets from the file",
  "This function reads full data sets from this file. "
  "The data type is dependent on the stored data, but is generally a :py:class:`numpy.ndarray`.\n\n"
  ".. note:: The functions :py:func:`read` and :py:func:`get` are synonyms."
)
.add_prototype("key", "data")
.add_parameter("key", "str", "The path to the dataset to read data from; can be an absolute value (starting with a leading ``'/'``) or relative to the current working directory :py:attr:`cwd`")
.add_return("data", ":py:class:`numpy.ndarray` or other", "The data read from this file at the given key")
;
auto s_get = s_read.clone("get");
static PyObject* PyBobIoHDF5File_read(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY

  /* Parses input arguments in a single shot */
  static char** kwlist = s_read.kwlist();

  const char* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &key)) return 0;

  return PyBobIoHDF5File_Xread(self, key, 1, 0);
BOB_CATCH_MEMBER(exception_message(self, s_read.name()).c_str(), 0)
}


static auto s_lread = bob::extension::FunctionDoc(
  "lread",
  "Reads some contents of the dataset",
  "This method reads contents from a dataset, treating the N-dimensional dataset like a container for multiple objects with N-1 dimensions. "
  "It returns a single :py:class:`numpy.ndarray` in case ``pos`` is set to a value >= 0, or a list of arrays otherwise."
)
.add_prototype("key, [pos]", "data")
.add_parameter("key", "str", "The path to the dataset to read data from, can be an absolute value (starting with a leading ``'/'``) or relative to the current working directory :py:attr:`cwd`")
.add_parameter("pos", "int", "If given and >= 0 returns the data object with the given index, otherwise returns a list by reading all objects in sequence")
.add_return("data", ":py:class:`numpy.ndarray` or [:py:class:`numpy.ndarray`]", "The data read from this file")
;
static PyObject* PyBobIoHDF5File_listRead(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_lread.kwlist();

  const char* key = 0;
  Py_ssize_t pos = -1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|n", kwlist, &key, &pos)) return 0;

  if (pos >= 0) return PyBobIoHDF5File_Xread(self, key, 0, pos);

  //otherwise returns as a list
  const std::vector<bob::io::base::HDF5Descriptor>& D = self->f->describe(key);

  PyObject* retval = PyList_New(D[0].size);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (uint64_t k=0; k<D[0].size; ++k) {
    PyObject* item = PyBobIoHDF5File_Xread(self, key, 0, k);
    if (!item) return 0;
    PyList_SET_ITEM(retval, k, item);
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER(exception_message(self, s_lread.name()).c_str(), 0)
}


/**
 * Sets at 't', the type of the object 'o' according to our support types.
 * Raise in case of problems. Furthermore, returns 'true' if the object is as
 * simple scalar.
 */

static void null_char_array_deleter(char*) {}

#if PY_VERSION_HEX >= 0x03000000
static void char_array_deleter(char* o) { delete[] o; }
#endif

static boost::shared_ptr<char> PyBobIo_GetString(PyObject* o) {

#if PY_VERSION_HEX < 0x03000000

  return boost::shared_ptr<char>(PyString_AsString(o), null_char_array_deleter);

#else

  if (PyBytes_Check(o)) {
    //fast way out
    return boost::shared_ptr<char>(PyBytes_AsString(o), null_char_array_deleter);
  }

  PyObject* bytes = 0;

  if (PyUnicode_Check(o)) {
    //re-encode using utf-8
    bytes = PyUnicode_AsEncodedString(o, "utf-8", "strict");
  }
  else {
    //tries coercion
    bytes = PyObject_Bytes(o);
  }
  auto bytes_ = make_safe(bytes); ///< protects acquired resource

  Py_ssize_t length = PyBytes_GET_SIZE(bytes)+1;
  char* copy = new char[length];
  std::strncpy(copy, PyBytes_AsString(bytes), length);

  return boost::shared_ptr<char>(copy, char_array_deleter);

#endif

}

static int PyBobIoHDF5File_setStringType(bob::io::base::HDF5Type& t, PyObject* o) {
  auto value = PyBobIo_GetString(o);
  if (!value) return -1;
  t = bob::io::base::HDF5Type(value.get());
  return 0;
}

template <typename T> int PyBobIoHDF5File_setType(bob::io::base::HDF5Type& t) {
  T v;
  t = bob::io::base::HDF5Type(v);
  return 0;
}

/**
 * A function to check for python scalars that works with numpy-1.6.x
 */
static bool PyBobIoHDF5File_isPythonScalar(PyObject* obj) {
  return (
    PyBool_Check(obj) ||
#if PY_VERSION_HEX < 0x03000000
    PyString_Check(obj) ||
#else
    PyBytes_Check(obj) ||
#endif
    PyUnicode_Check(obj) ||
#if PY_VERSION_HEX < 0x03000000
    PyInt_Check(obj) ||
#endif
    PyLong_Check(obj) ||
    PyFloat_Check(obj) ||
    PyComplex_Check(obj)
    );
}

/**
 * Returns the type of object `op' is - a scalar (return value = 0), a
 * bob.blitz.array (return value = 1), a numpy.ndarray (return value = 2), an
 * object which is convertible to a numpy.ndarray (return value = 3) or returns
 * -1 if the object cannot be converted. No error is set on the python stack.
 *
 * If the object is convertible into a numpy.ndarray, then it is converted into
 * a numpy ndarray and the resulting object is placed in `converted'. If
 * `*converted' is set to 0 (NULL), then we don't try a conversion, returning
 * -1.
 */
static int PyBobIoHDF5File_getObjectType(PyObject* o, bob::io::base::HDF5Type& t,
    PyObject** converted=0) {

  if (PyArray_IsScalar(o, Generic) || PyBobIoHDF5File_isPythonScalar(o)) {

    if (PyArray_IsScalar(o, String))
      return PyBobIoHDF5File_setStringType(t, o);

    else if (PyBool_Check(o))
      return PyBobIoHDF5File_setType<bool>(t);

#if PY_VERSION_HEX < 0x03000000
    else if (PyString_Check(o))
      return PyBobIoHDF5File_setStringType(t, o);

#else
    else if (PyBytes_Check(o))
      return PyBobIoHDF5File_setStringType(t, o);

#endif
    else if (PyUnicode_Check(o))
      return PyBobIoHDF5File_setStringType(t, o);

#if PY_VERSION_HEX < 0x03000000
    else if (PyInt_Check(o))
      return PyBobIoHDF5File_setType<int32_t>(t);

#endif
    else if (PyLong_Check(o))
      return PyBobIoHDF5File_setType<int64_t>(t);

    else if (PyFloat_Check(o))
      return PyBobIoHDF5File_setType<double>(t);

    else if (PyComplex_Check(o))
      return PyBobIoHDF5File_setType<std::complex<double> >(t);

    else if (PyArray_IsScalar(o, Bool))
      return PyBobIoHDF5File_setType<bool>(t);

    else if (PyArray_IsScalar(o, Int8))
      return PyBobIoHDF5File_setType<int8_t>(t);

    else if (PyArray_IsScalar(o, UInt8))
      return PyBobIoHDF5File_setType<uint8_t>(t);

    else if (PyArray_IsScalar(o, Int16))
      return PyBobIoHDF5File_setType<int16_t>(t);

    else if (PyArray_IsScalar(o, UInt16))
      return PyBobIoHDF5File_setType<uint16_t>(t);

    else if (PyArray_IsScalar(o, Int32))
      return PyBobIoHDF5File_setType<int32_t>(t);

    else if (PyArray_IsScalar(o, UInt32))
      return PyBobIoHDF5File_setType<uint32_t>(t);

    else if (PyArray_IsScalar(o, Int64))
      return PyBobIoHDF5File_setType<int64_t>(t);

    else if (PyArray_IsScalar(o, UInt64))
      return PyBobIoHDF5File_setType<uint64_t>(t);

    else if (PyArray_IsScalar(o, Float))
      return PyBobIoHDF5File_setType<float>(t);

    else if (PyArray_IsScalar(o, Double))
      return PyBobIoHDF5File_setType<double>(t);

    else if (PyArray_IsScalar(o, LongDouble))
      return PyBobIoHDF5File_setType<long double>(t);

    else if (PyArray_IsScalar(o, CFloat))
      return PyBobIoHDF5File_setType<std::complex<float> >(t);

    else if (PyArray_IsScalar(o, CDouble))
      return PyBobIoHDF5File_setType<std::complex<double> >(t);

    else if (PyArray_IsScalar(o, CLongDouble))
      return PyBobIoHDF5File_setType<std::complex<long double> >(t);

    //if you get to this, point, it is an unsupported scalar
    return -1;

  }

  else if (PyBlitzArray_Check(o)) {

    PyBlitzArrayObject* bz = reinterpret_cast<PyBlitzArrayObject*>(o);
    bob::io::base::hdf5type h5type = PyBobIo_H5FromTypenum(bz->type_num);
    if (h5type == bob::io::base::unsupported) return -1;
    bob::io::base::HDF5Shape h5shape(bz->ndim, bz->shape);
    t = bob::io::base::HDF5Type(h5type, h5shape);
    return 1;

  }

  else if (PyArray_CheckExact(o) && PyArray_ISCARRAY_RO((PyArrayObject*)o)) {

    PyArrayObject* np = reinterpret_cast<PyArrayObject*>(o);
    bob::io::base::hdf5type h5type = PyBobIo_H5FromTypenum(PyArray_DESCR(np)->type_num);
    if (h5type == bob::io::base::unsupported) return -1;
    bob::io::base::HDF5Shape h5shape(PyArray_NDIM(np), PyArray_DIMS(np));
    t = bob::io::base::HDF5Type(h5type, h5shape);
    return 2;

  }

  else if (converted) {

    *converted = PyArray_FromAny(o, 0, 1, 0,
#if     NPY_FEATURE_VERSION >= NUMPY17_API /* NumPy C-API version >= 1.7 */
        NPY_ARRAY_CARRAY_RO,
#       else
        NPY_CARRAY_RO,
#       endif
        0);
    if (!*converted) return -1; ///< error condition

    PyArrayObject* np = reinterpret_cast<PyArrayObject*>(*converted);
    bob::io::base::hdf5type h5type = PyBobIo_H5FromTypenum(PyArray_DESCR(np)->type_num);
    if (h5type == bob::io::base::unsupported) {
      Py_CLEAR(*converted);
      return -1;
    }
    bob::io::base::HDF5Shape h5shape(PyArray_NDIM(np), PyArray_DIMS(np));
    t = bob::io::base::HDF5Type(h5type, h5shape);
    return 3;

  }

  //if you get to this, point, it is an unsupported type
  return -1;

}

template <typename T>
static PyObject* PyBobIoHDF5File_replaceScalar(PyBobIoHDF5FileObject* self,
    const char* path, Py_ssize_t pos, PyObject* o) {

  T value = PyBlitzArrayCxx_AsCScalar<T>(o);
  if (PyErr_Occurred()) return 0;
  self->f->replace(path, pos, value);

  Py_RETURN_NONE;

}


static auto s_replace = bob::extension::FunctionDoc(
  "replace",
  "Modifies the value of a scalar/array in a dataset.",
  0,
  true
)
.add_prototype("path, pos, data")
.add_parameter("path", "str", "The path to the dataset to read data from; can be an absolute value (starting with a leading ``'/'``) or relative to the current working directory :py:attr:`cwd`")
.add_parameter("pos", "int", "Position, within the dataset, of the object to be replaced; the object position on the dataset must exist, or an exception is raised")
.add_parameter("data", ":py:class:`numpy.ndarray` or scalar", "Object to replace the value with; this value must be compatible with the typing information on the dataset, or an exception will be raised")
;
static PyObject* PyBobIoHDF5File_replace(PyBobIoHDF5FileObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_replace.kwlist();

  const char* path = 0;
  Py_ssize_t pos = -1;
  PyObject* data = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "snO", kwlist, &path, &pos, &data)) return 0;

  bob::io::base::HDF5Type type;
  PyObject* converted = 0;
  int is_array = PyBobIoHDF5File_getObjectType(data, type, &converted);
  auto converted_ = make_xsafe(converted);

  if (is_array < 0) { ///< error condition, signal
    const char* filename = "<unknown>";
    try{ filename = self->f->filename().c_str(); } catch(...){}
    PyErr_Format(PyExc_TypeError, "error replacing position %" PY_FORMAT_SIZE_T "d of dataset `%s' at HDF5 file `%s': no support for storing objects of type `%s' on HDF5 files", pos, path, filename, Py_TYPE(data)->tp_name);
    return 0;
  }

  if (!is_array) { //write as a scalar

    switch(type.type()) {
      case bob::io::base::s:
        {
          auto value = PyBobIo_GetString(data);
          if (!value) return 0;
          self->f->replace<std::string>(path, pos, value.get());
          Py_RETURN_NONE;
        }
      case bob::io::base::b:
        return PyBobIoHDF5File_replaceScalar<bool>(self, path, pos, data);
      case bob::io::base::i8:
        return PyBobIoHDF5File_replaceScalar<int8_t>(self, path, pos, data);
      case bob::io::base::i16:
        return PyBobIoHDF5File_replaceScalar<int16_t>(self, path, pos, data);
      case bob::io::base::i32:
        return PyBobIoHDF5File_replaceScalar<int32_t>(self, path, pos, data);
      case bob::io::base::i64:
        return PyBobIoHDF5File_replaceScalar<int64_t>(self, path, pos, data);
      case bob::io::base::u8:
        return PyBobIoHDF5File_replaceScalar<uint8_t>(self, path, pos, data);
      case bob::io::base::u16:
        return PyBobIoHDF5File_replaceScalar<uint16_t>(self, path, pos, data);
      case bob::io::base::u32:
        return PyBobIoHDF5File_replaceScalar<uint32_t>(self, path, pos, data);
      case bob::io::base::u64:
        return PyBobIoHDF5File_replaceScalar<uint64_t>(self, path, pos, data);
      case bob::io::base::f32:
        return PyBobIoHDF5File_replaceScalar<float>(self, path, pos, data);
      case bob::io::base::f64:
        return PyBobIoHDF5File_replaceScalar<double>(self, path, pos, data);
      case bob::io::base::f128:
        return PyBobIoHDF5File_replaceScalar<long double>(self, path, pos, data);
      case bob::io::base::c64:
        return PyBobIoHDF5File_replaceScalar<std::complex<float> >(self, path, pos, data);
      case bob::io::base::c128:
        return PyBobIoHDF5File_replaceScalar<std::complex<double> >(self, path, pos, data);
      case bob::io::base::c256:
        return PyBobIoHDF5File_replaceScalar<std::complex<long double> >(self, path, pos, data);
      default:
        break;
    }

  }

  else { //write as array

    switch (is_array) {
      case 1: //bob.blitz.array
        self->f->write_buffer(path, pos, type, ((PyBlitzArrayObject*)data)->data);
        break;

      case 2: //numpy.ndarray
        self->f->write_buffer(path, pos, type, PyArray_DATA((PyArrayObject*)data));
        break;

      case 3: //converted numpy.ndarray
        self->f->write_buffer(path, pos, type, PyArray_DATA((PyArrayObject*)converted));
        break;

      default:
        const char* filename = "<unknown>";
        try{ filename = self->f->filename().c_str(); } catch(...){}
        PyErr_Format(PyExc_NotImplementedError, "error replacing position %" PY_FORMAT_SIZE_T "d of dataset `%s' at HDF5 file `%s': HDF5 replace function is uncovered for array type %d (DEBUG ME)", pos, path, filename, is_array);
        return 0;
    }

  }

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_replace.name()).c_str(), 0)
}


template <typename T>
static int PyBobIoHDF5File_appendScalar(PyBobIoHDF5FileObject* self,
    const char* path, PyObject* o) {

  T value = PyBlitzArrayCxx_AsCScalar<T>(o);
  if (PyErr_Occurred()) return 0;
  self->f->append(path, value);

  return 1;

}

static int PyBobIoHDF5File_innerAppend(PyBobIoHDF5FileObject* self, const char* path, PyObject* data, Py_ssize_t compression) {

  bob::io::base::HDF5Type type;
  PyObject* converted = 0;
  int is_array = PyBobIoHDF5File_getObjectType(data, type, &converted);
  auto converted_ = make_xsafe(converted);

  if (is_array < 0) { ///< error condition, signal
    const char* filename = "<unknown>";
    try{ filename = self->f->filename().c_str(); } catch(...){}
    PyErr_Format(PyExc_TypeError, "error appending to object `%s' of HDF5 file `%s': no support for storing objects of type `%s' on HDF5 files", path, filename, Py_TYPE(data)->tp_name);
    return 0;
  }

  try {

    if (!is_array) { //write as a scalar

      switch(type.type()) {
        case bob::io::base::s:
          {
            auto value = PyBobIo_GetString(data);
            if (!value) return 0;
            self->f->append<std::string>(path, value.get());
            return 1;
          }
        case bob::io::base::b:
          return PyBobIoHDF5File_appendScalar<bool>(self, path, data);
        case bob::io::base::i8:
          return PyBobIoHDF5File_appendScalar<int8_t>(self, path, data);
        case bob::io::base::i16:
          return PyBobIoHDF5File_appendScalar<int16_t>(self, path, data);
        case bob::io::base::i32:
          return PyBobIoHDF5File_appendScalar<int32_t>(self, path, data);
        case bob::io::base::i64:
          return PyBobIoHDF5File_appendScalar<int64_t>(self, path, data);
        case bob::io::base::u8:
          return PyBobIoHDF5File_appendScalar<uint8_t>(self, path, data);
        case bob::io::base::u16:
          return PyBobIoHDF5File_appendScalar<uint16_t>(self, path, data);
        case bob::io::base::u32:
          return PyBobIoHDF5File_appendScalar<uint32_t>(self, path, data);
        case bob::io::base::u64:
          return PyBobIoHDF5File_appendScalar<uint64_t>(self, path, data);
        case bob::io::base::f32:
          return PyBobIoHDF5File_appendScalar<float>(self, path, data);
        case bob::io::base::f64:
          return PyBobIoHDF5File_appendScalar<double>(self, path, data);
        case bob::io::base::f128:
          return PyBobIoHDF5File_appendScalar<long double>(self, path, data);
        case bob::io::base::c64:
          return PyBobIoHDF5File_appendScalar<std::complex<float> >(self, path, data);
        case bob::io::base::c128:
          return PyBobIoHDF5File_appendScalar<std::complex<double> >(self, path, data);
        case bob::io::base::c256:
          return PyBobIoHDF5File_appendScalar<std::complex<long double> >(self, path, data);
        default:
          break;
      }

    }

    else { //write as array

      switch (is_array) {
        case 1: //bob.blitz.array
          if (!self->f->contains(path)) self->f->create(path, type, true, compression);
          self->f->extend_buffer(path, type, ((PyBlitzArrayObject*)data)->data);
          break;

        case 2: //numpy.ndarray
          if (!self->f->contains(path)) self->f->create(path, type, true, compression);
          self->f->extend_buffer(path, type, PyArray_DATA((PyArrayObject*)data));
          break;

        case 3: //converted numpy.ndarray
          if (!self->f->contains(path)) self->f->create(path, type, true, compression);
          self->f->extend_buffer(path, type, PyArray_DATA((PyArrayObject*)converted));
          break;

        default:{
          const char* filename = "<unknown>";
          try{ filename = self->f->filename().c_str(); } catch(...){}
          PyErr_Format(PyExc_NotImplementedError, "error appending to object `%s' at HDF5 file `%s': HDF5 replace function is uncovered for array type %d (DEBUG ME)", path, filename, is_array);
          return 0;
        }
      }

    }

  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    const char* filename = "<unknown>";
    try{ filename = self->f->filename().c_str(); } catch(...){}
    PyErr_Format(PyExc_RuntimeError, "cannot append to object `%s' at HDF5 file `%s': unknown exception caught", path, filename);
    return 0;
  }

  return 1;

}


static auto s_append = bob::extension::FunctionDoc(
  "append",
  "Appends a scalar or an array to a dataset",
  "The object must be compatible with the typing information on the dataset, or an exception will be raised. "
  "You can also, optionally, set this to an iterable of scalars or arrays. "
  "This will cause this method to iterate over the elements and add each individually.\n\n"
  "The ``compression`` parameter is effective when appending arrays. "
  "Set this to a number betwen 0 (default) and 9 (maximum) to compress the contents of this dataset. "
  "This setting is only effective if the dataset does not yet exist, otherwise, the previous setting is respected.",
  true
)
.add_prototype("path, data, [compression]")
.add_parameter("path", "str", "The path to the dataset to append data at; can be an absolute value (starting with a leading ``'/'``) or relative to the current working directory :py:attr:`cwd`")
.add_parameter("data", ":py:class:`numpy.ndarray` or scalar", "Object to append to the dataset")
.add_parameter("compression", "int", "A compression value between 0 and 9")
;
static PyObject* PyBobIoHDF5File_append(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_append.kwlist();

  char* path = 0;
  PyObject* data = 0;
  Py_ssize_t compression = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|n", kwlist, &path, &data, &compression)) return 0;

  if (compression < 0 || compression > 9) {
    PyErr_SetString(PyExc_ValueError, "compression should be set to an integer value between and including 0 and 9");
    return 0;
  }

  // special case: user passes a tuple or list of arrays or scalars to append
  if (PyTuple_Check(data) || PyList_Check(data)) {
    PyObject* iter = PyObject_GetIter(data);
    if (!iter) return 0;
    auto iter_ = make_safe(iter);
    while (PyObject* item = PyIter_Next(iter)) {
      auto item_ = make_safe(item);
      int ok = PyBobIoHDF5File_innerAppend(self, path, item, compression);
      if (!ok) return 0;
    }
    Py_RETURN_NONE;
  }

  int ok = PyBobIoHDF5File_innerAppend(self, path, data, compression);
  if (!ok) return 0;
  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_append.name()).c_str(), 0)
}


template <typename T>
static PyObject* PyBobIoHDF5File_setScalar(PyBobIoHDF5FileObject* self,
    const char* path, PyObject* o) {

  T value = PyBlitzArrayCxx_AsCScalar<T>(o);
  if (PyErr_Occurred()) return 0;
  self->f->set(path, value);

  Py_RETURN_NONE;

}


static auto s_set = bob::extension::FunctionDoc(
  "set",
  "Sets the scalar or array at position 0 to the given value",
  "This method is equivalent to checking if the scalar or array at position 0 exists and then replacing it. "
  "If the path does not exist, we append the new scalar or array.\n\n"
  "The ``data`` must be compatible with the typing information on the dataset, or an exception will be raised. "
  "You can also, optionally, set this to an iterable of scalars or arrays. "
  "This will cause this method to iterate over the elements and add each individually.\n\n"
  "The ``compression`` parameter is effective when writing arrays. "
  "Set this to a number betwen 0 (default) and 9 (maximum) to compress the contents of this dataset. "
  "This setting is only effective if the dataset does not yet exist, otherwise, the previous setting is respected.\n\n"
  ".. note:: The functions :py:meth:`set` and :py:meth:`write` are synonyms.",
  true
)
.add_prototype("path, data, [compression]")
.add_parameter("path", "str", "The path to the dataset to write data to; can be an absolute value (starting with a leading ``'/'``) or relative to the current working directory :py:attr:`cwd`")
.add_parameter("data", ":py:class:`numpy.ndarray` or scalar", "Object to write to the dataset")
.add_parameter("compression", "int", "A compression value between 0 and 9")
;
auto s_write = s_set.clone("write");
static PyObject* PyBobIoHDF5File_set(PyBobIoHDF5FileObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"path", "data", "compression", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* path = 0;
  PyObject* data = 0;
  Py_ssize_t compression = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|n", kwlist, &path, &data, &compression)) return 0;

  if (compression < 0 || compression > 9) {
    PyErr_SetString(PyExc_ValueError, "compression should be set to an integer value between and including 0 and 9");
    return 0;
  }

  bob::io::base::HDF5Type type;
  PyObject* converted = 0;
  int is_array = PyBobIoHDF5File_getObjectType(data, type, &converted);
  auto converted_ = make_xsafe(converted);

  if (is_array < 0) { ///< error condition, signal
    const char* filename = "<unknown>";
    try{ filename = self->f->filename().c_str(); } catch(...){}
    PyErr_Format(PyExc_TypeError, "error setting object `%s' of HDF5 file `%s': no support for storing objects of type `%s' on HDF5 files", path, filename, Py_TYPE(data)->tp_name);
    return 0;
  }

  if (!is_array) { //write as a scalar

    switch(type.type()) {
      case bob::io::base::s:
        {
          auto value = PyBobIo_GetString(data);
          if (!value) return 0;
          self->f->set<std::string>(path, value.get());
          Py_RETURN_NONE;
        }
        break;
      case bob::io::base::b:
        return PyBobIoHDF5File_setScalar<bool>(self, path, data);
      case bob::io::base::i8:
        return PyBobIoHDF5File_setScalar<int8_t>(self, path, data);
      case bob::io::base::i16:
        return PyBobIoHDF5File_setScalar<int16_t>(self, path, data);
      case bob::io::base::i32:
        return PyBobIoHDF5File_setScalar<int32_t>(self, path, data);
      case bob::io::base::i64:
        return PyBobIoHDF5File_setScalar<int64_t>(self, path, data);
      case bob::io::base::u8:
        return PyBobIoHDF5File_setScalar<uint8_t>(self, path, data);
      case bob::io::base::u16:
        return PyBobIoHDF5File_setScalar<uint16_t>(self, path, data);
      case bob::io::base::u32:
        return PyBobIoHDF5File_setScalar<uint32_t>(self, path, data);
      case bob::io::base::u64:
        return PyBobIoHDF5File_setScalar<uint64_t>(self, path, data);
      case bob::io::base::f32:
        return PyBobIoHDF5File_setScalar<float>(self, path, data);
      case bob::io::base::f64:
        return PyBobIoHDF5File_setScalar<double>(self, path, data);
      case bob::io::base::f128:
        return PyBobIoHDF5File_setScalar<long double>(self, path, data);
      case bob::io::base::c64:
        return PyBobIoHDF5File_setScalar<std::complex<float> >(self, path, data);
      case bob::io::base::c128:
        return PyBobIoHDF5File_setScalar<std::complex<double> >(self, path, data);
      case bob::io::base::c256:
        return PyBobIoHDF5File_setScalar<std::complex<long double> >(self, path, data);
      default:
        break;
    }

  }

  else { //write as array

    switch (is_array) {
      case 1: //bob.blitz.array
        if (!self->f->contains(path)) self->f->create(path, type, false, compression);
        self->f->write_buffer(path, 0, type, ((PyBlitzArrayObject*)data)->data);
        break;

      case 2: //numpy.ndarray
        if (!self->f->contains(path)) self->f->create(path, type, false, compression);
        self->f->write_buffer(path, 0, type, PyArray_DATA((PyArrayObject*)data));
        break;

      case 3: //converted numpy.ndarray
        if (!self->f->contains(path)) self->f->create(path, type, false, compression);
        self->f->write_buffer(path, 0, type, PyArray_DATA((PyArrayObject*)converted));
        break;

      default:
        const char* filename = "<unknown>";
        try{ filename = self->f->filename().c_str(); } catch(...){}
        PyErr_Format(PyExc_NotImplementedError, "error setting object `%s' at HDF5 file `%s': HDF5 replace function is uncovered for array type %d (DEBUG ME)", path, filename, is_array);
        return 0;
    }

  }

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_set.name()).c_str(), 0)
}


static auto s_copy = bob::extension::FunctionDoc(
  "copy",
  "Copies all accessible content to another HDF5 file",
  "Unlinked contents of this file will not be copied. "
  "This can be used as a method to trim unwanted content in a file.",
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`HDF5File`", "The HDF5 file (already opened for writing), to copy the contents to")
;
static PyObject* PyBobIoHDF5File_copy(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_copy.kwlist();

  PyBobIoHDF5FileObject* other = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBobIoHDF5File_Converter, &other)) return 0;

  self->f->copy(*other->f);

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_copy.name()).c_str(), 0)
}


template <typename T> static PyObject* PyBobIoHDF5File_readScalarAttribute
(PyBobIoHDF5FileObject* self, const char* path, const char* name,
 const bob::io::base::HDF5Type& type) {
  T value;
  self->f->read_attribute(path, name, type, static_cast<void*>(&value));
  return PyBlitzArrayCxx_FromCScalar(value);
}

template <> PyObject* PyBobIoHDF5File_readScalarAttribute<const char*>
(PyBobIoHDF5FileObject* self, const char* path, const char* name,
 const bob::io::base::HDF5Type& type) {
  std::string retval;
  self->f->getAttribute(path, name, retval);
  return Py_BuildValue("s", retval.c_str());
}

static PyObject* PyBobIoHDF5File_readAttribute(PyBobIoHDF5FileObject* self,
    const char* path, const char* name, const bob::io::base::HDF5Type& type) {

  //no error detection: this should be done before reaching this method

  const bob::io::base::HDF5Shape& shape = type.shape();

  if (type.type() == bob::io::base::s || (shape.n() == 1 && shape[0] == 1)) {
    //read as scalar
    switch(type.type()) {
      case bob::io::base::s:
        return PyBobIoHDF5File_readScalarAttribute<const char*>(self, path, name, type);
      case bob::io::base::b:
        return PyBobIoHDF5File_readScalarAttribute<bool>(self, path, name, type);
      case bob::io::base::i8:
        return PyBobIoHDF5File_readScalarAttribute<int8_t>(self, path, name, type);
      case bob::io::base::i16:
        return PyBobIoHDF5File_readScalarAttribute<int16_t>(self, path, name, type);
      case bob::io::base::i32:
        return PyBobIoHDF5File_readScalarAttribute<int32_t>(self, path, name, type);
      case bob::io::base::i64:
        return PyBobIoHDF5File_readScalarAttribute<int64_t>(self, path, name, type);
      case bob::io::base::u8:
        return PyBobIoHDF5File_readScalarAttribute<uint8_t>(self, path, name, type);
      case bob::io::base::u16:
        return PyBobIoHDF5File_readScalarAttribute<uint16_t>(self, path, name, type);
      case bob::io::base::u32:
        return PyBobIoHDF5File_readScalarAttribute<uint32_t>(self, path, name, type);
      case bob::io::base::u64:
        return PyBobIoHDF5File_readScalarAttribute<uint64_t>(self, path, name, type);
      case bob::io::base::f32:
        return PyBobIoHDF5File_readScalarAttribute<float>(self, path, name, type);
      case bob::io::base::f64:
        return PyBobIoHDF5File_readScalarAttribute<double>(self, path, name, type);
      case bob::io::base::f128:
        return PyBobIoHDF5File_readScalarAttribute<long double>(self, path, name, type);
      case bob::io::base::c64:
        return PyBobIoHDF5File_readScalarAttribute<std::complex<float> >(self, path, name, type);
      case bob::io::base::c128:
        return PyBobIoHDF5File_readScalarAttribute<std::complex<double> >(self, path, name, type);
      case bob::io::base::c256:
        return PyBobIoHDF5File_readScalarAttribute<std::complex<long double> >(self, path, name, type);
      default:
        break;
    }
  }

  //read as an numpy array
  int type_num = PyBobIo_H5AsTypenum(type.type());
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  npy_intp pyshape[NPY_MAXDIMS];
  for (size_t k=0; k<shape.n(); ++k) pyshape[k] = shape.get()[k];

  PyObject* retval = PyArray_SimpleNew(shape.n(), pyshape, type_num);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  self->f->read_attribute(path, name, type, PyArray_DATA((PyArrayObject*)retval));

  return Py_BuildValue("O", retval);
}


static auto s_get_attribute = bob::extension::FunctionDoc(
  "get_attribute",
  "Retrieve a given attribute from the named resource",
  "This method returns a single value corresponding to what is stored inside the attribute container for the given resource. "
  "If you would like to retrieve all attributes at once, use :py:meth:`get_attributes` instead.",
  true
)
.add_prototype("name, [path]", "attribute")
.add_parameter("name", "str", "The name of the attribute to retrieve; if the attribute is not available, a ``RuntimeError`` is raised")
.add_parameter("path", "str", "[Default: ``'.'``] The path leading to the resource (dataset or group|directory) you would like to get an attribute from; if the path does not exist, a ``RuntimeError`` is raised")
.add_return("attribute", ":py:class:`numpy.ndarray` or scalar", "The read attribute")
;
static PyObject* PyBobIoHDF5File_getAttribute(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_get_attribute.kwlist();

  const char* name = 0;
  const char* path = ".";
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|s", kwlist, &name, &path)) return 0;

  bob::io::base::HDF5Type type;

  self->f->getAttributeType(path, name, type);

  if (type.type() == bob::io::base::unsupported) {
    const char* filename = "<unknown>";
    try{ filename = self->f->filename().c_str(); } catch(...){}
    boost::format m("unsupported HDF5 data type detected for attribute `%s' at path `%s' of file `%s' - returning None");
    m % name % path % filename;
    PyErr_Warn(PyExc_UserWarning, m.str().c_str());
    Py_RETURN_NONE;
  }

  return PyBobIoHDF5File_readAttribute(self, path, name, type);
BOB_CATCH_MEMBER(exception_message(self, s_get_attribute.name()).c_str(), 0)
}


static auto s_get_attributes = bob::extension::FunctionDoc(
  "get_attributes",
  "Reads all attributes of the given path",
  "Attributes are returned in a dictionary in which each key corresponds to the attribute name and each value corresponds to the value stored inside the HDF5 file. "
  "To retrieve only a specific attribute, use :py:meth:`get_attribute`.",
  true
)
.add_prototype("[path]", "attributes")
.add_parameter("path", "str", "[Default: ``'.'``] The path leading to the resource (dataset or group|directory) you would like to get all attributes from; if the path does not exist, a ``RuntimeError`` is raised.")
.add_return("attributes", "{str:value}", "The attributes organized in dictionary, where ``value`` might be a :py:class:`numpy.ndarray` or a scalar")
;
static PyObject* PyBobIoHDF5File_getAttributes(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_get_attributes.kwlist();

  const char* path = ".";
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist, &path)) return 0;

  std::map<std::string, bob::io::base::HDF5Type> attributes;
  self->f->listAttributes(path, attributes);
  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (auto k=attributes.begin(); k!=attributes.end(); ++k) {
    PyObject* item = 0;
    if (k->second.type() == bob::io::base::unsupported) {
      const char* filename = "<unknown>";
      try{ filename = self->f->filename().c_str(); } catch(...){}
      boost::format m("unsupported HDF5 data type detected for attribute `%s' at path `%s' of file `%s' - returning None");
      m % k->first % k->second.str() % filename;
      PyErr_Warn(PyExc_UserWarning, m.str().c_str());
      item = Py_BuildValue("");
    }
    else item = PyBobIoHDF5File_readAttribute(self, path, k->first.c_str(), k->second);

    if (!item) return 0;
    auto item_ = make_safe(item);

    if (PyDict_SetItemString(retval, k->first.c_str(), item) != 0) return 0;
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER(exception_message(self, s_get_attributes.name()).c_str(), 0)
}


template <typename T> PyObject* PyBobIoHDF5File_writeScalarAttribute(PyBobIoHDF5FileObject* self, const char* path, const char* name, const bob::io::base::HDF5Type& type, PyObject* o) {
  T value = PyBlitzArrayCxx_AsCScalar<T>(o);
  if (PyErr_Occurred()) return 0;

  self->f->write_attribute(path, name, type, static_cast<void*>(&value));

  Py_RETURN_NONE;
}

template <> PyObject* PyBobIoHDF5File_writeScalarAttribute<const char*>(PyBobIoHDF5FileObject* self, const char* path, const char* name, const bob::io::base::HDF5Type& type, PyObject* o) {
  auto value = PyBobIo_GetString(o);
  if (!value) return 0;
  self->f->write_attribute(path, name, type, static_cast<const void*>(value.get()));
  Py_RETURN_NONE;
}

static PyObject* PyBobIoHDF5File_writeAttribute(PyBobIoHDF5FileObject* self,
    const char* path, const char* name, const bob::io::base::HDF5Type& type,
    PyObject* o, int is_array, PyObject* converted) {

  //no error detection: this should be done before reaching this method

  if (!is_array) { //write as a scalar
    switch(type.type()) {
      case bob::io::base::s:
        return PyBobIoHDF5File_writeScalarAttribute<const char*>(self, path, name, type, o);
      case bob::io::base::b:
        return PyBobIoHDF5File_writeScalarAttribute<bool>(self, path, name, type, o);
      case bob::io::base::i8:
        return PyBobIoHDF5File_writeScalarAttribute<int8_t>(self, path, name, type, o);
      case bob::io::base::i16:
        return PyBobIoHDF5File_writeScalarAttribute<int16_t>(self, path, name, type, o);
      case bob::io::base::i32:
        return PyBobIoHDF5File_writeScalarAttribute<int32_t>(self, path, name, type, o);
      case bob::io::base::i64:
        return PyBobIoHDF5File_writeScalarAttribute<int64_t>(self, path, name, type, o);
      case bob::io::base::u8:
        return PyBobIoHDF5File_writeScalarAttribute<uint8_t>(self, path, name, type, o);
      case bob::io::base::u16:
        return PyBobIoHDF5File_writeScalarAttribute<uint16_t>(self, path, name, type, o);
      case bob::io::base::u32:
        return PyBobIoHDF5File_writeScalarAttribute<uint32_t>(self, path, name, type, o);
      case bob::io::base::u64:
        return PyBobIoHDF5File_writeScalarAttribute<uint64_t>(self, path, name, type, o);
      case bob::io::base::f32:
        return PyBobIoHDF5File_writeScalarAttribute<float>(self, path, name, type, o);
      case bob::io::base::f64:
        return PyBobIoHDF5File_writeScalarAttribute<double>(self, path, name, type, o);
      case bob::io::base::f128:
        return PyBobIoHDF5File_writeScalarAttribute<long double>(self, path, name, type, o);
      case bob::io::base::c64:
        return PyBobIoHDF5File_writeScalarAttribute<std::complex<float> >(self, path, name, type, o);
      case bob::io::base::c128:
        return PyBobIoHDF5File_writeScalarAttribute<std::complex<double> >(self, path, name, type, o);
      case bob::io::base::c256:
        return PyBobIoHDF5File_writeScalarAttribute<std::complex<long double> >(self, path, name, type, o);
      default:
        break;
    }
  }

  else { //write as an numpy array

    switch (is_array) {

      case 1: //bob.blitz.array
        self->f->write_attribute(path, name, type, ((PyBlitzArrayObject*)o)->data);
        break;

      case 2: //numpy.ndarray
        self->f->write_attribute(path, name, type, PyArray_DATA((PyArrayObject*)o));
        break;

      case 3: //converted numpy.ndarray
        self->f->write_attribute(path, name, type, PyArray_DATA((PyArrayObject*)converted));
        break;

      default:{
        const char* filename = "<unknown>";
        try{ filename = self->f->filename().c_str(); } catch(...){}
        PyErr_Format(PyExc_NotImplementedError, "error setting attribute `%s' at resource `%s' of HDF5 file `%s': HDF5 attribute setting function is uncovered for array type %d (DEBUG ME)", name, path, filename, is_array);
        return 0;
      }
    }
  }
  Py_RETURN_NONE;
}

static auto s_set_attribute = bob::extension::FunctionDoc(
  "set_attribute",
  "Sets a given attribute at the named resource",
  "Only simple  scalars (booleans, integers, floats and complex numbers) and arrays of those are supported at the time being. "
  "You can use :py:mod:`numpy` scalars to set values with arbitrary precision (e.g. :py:class:`numpy.uint8`).\n\n"
  ".. warning:: Attributes in HDF5 files are supposed to be small containers or simple scalars that provide extra information about the data stored on the main resource (dataset or group|directory). "
  "Attributes cannot be retrieved in chunks, contrary to data in datasets. "
  "Currently, **no limitations** for the size of values stored on attributes is imposed.",
  true
)
.add_prototype("name, value, [path]")
.add_parameter("name", "str", "The name of the attribute to set")
.add_parameter("value", ":py:class:`numpy.ndarray` or scalar", "A simple scalar to set for the given attribute on the named resources ``path``")
.add_parameter("path", "str", "[Default: ``'.'``] The path leading to the resource (dataset or group|directory) you would like to set an attribute at")
;
static PyObject* PyBobIoHDF5File_setAttribute(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_set_attribute.kwlist();

  const char* name = 0;
  PyObject* value = 0;
  const char* path = ".";
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|s", kwlist, &name, &value, &path)) return 0;

  bob::io::base::HDF5Type type;
  PyObject* converted = 0;
  int is_array = PyBobIoHDF5File_getObjectType(value, type, &converted);
  auto converted_ = make_xsafe(converted);

  if (is_array < 0) { ///< error condition, signal
    const char* filename = "<unknown>";
    try{ filename = self->f->filename().c_str(); } catch(...){}
    PyErr_Format(PyExc_TypeError, "error setting attribute `%s' of resource `%s' at HDF5 file `%s': no support for storing objects of type `%s' on HDF5 files", name, path, filename, Py_TYPE(value)->tp_name);
    return 0;
  }

  return PyBobIoHDF5File_writeAttribute(self, path, name, type, value, is_array, converted);
BOB_CATCH_MEMBER(exception_message(self, s_set_attribute.name()).c_str(), 0)
}


static auto s_set_attributes = bob::extension::FunctionDoc(
  "set_attributes",
  "Sets several attribute at the named resource using a dictionary",
  "Each value in the dictionary should be simple scalars (booleans, integers, floats and complex numbers) or arrays of those are supported at the time being. "
  "You can use :py:mod:`numpy` scalars to set values with arbitrary precision (e.g. :py:class:`numpy.uint8`).\n\n"
  ".. warning:: Attributes in HDF5 files are supposed to be small containers or simple scalars that provide extra information about the data stored on the main resource (dataset or group|directory). "
  "Attributes cannot be retrieved in chunks, contrary to data in datasets. "
  "Currently, **no limitations** for the size of values stored on attributes is imposed.",
  true
)
.add_prototype("attributes, [path]")
.add_parameter("attributes", "{str: value}", "A python dictionary containing pairs of strings and values, which can be a py:class:`numpy.ndarray` or a scalar")
.add_parameter("path", "str", "[Default: ``'.'``] The path leading to the resource (dataset or group|directory) you would like to set attributes at")
;
static PyObject* PyBobIoHDF5File_setAttributes(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_set_attributes.kwlist();

  PyObject* attrs = 0;
  const char* path = ".";
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|s", kwlist, &attrs, &path)) return 0;

  if (!PyDict_Check(attrs)) {
    PyErr_Format(PyExc_TypeError, "parameter `%s' should be a dictionary where keys are strings and values are the attribute values", kwlist[0]);
    return 0;
  }

  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(attrs, &pos, &key, &value)) {
    bob::io::base::HDF5Type type;
    PyObject* converted = 0;

    auto name = PyBobIo_GetString(key);
    if (!name) return 0;

    int is_array = PyBobIoHDF5File_getObjectType(value, type, &converted);
    auto converted_ = make_xsafe(converted);

    if (is_array < 0) { ///< error condition, signal
      const char* filename = "<unknown>";
      try{ filename = self->f->filename().c_str(); } catch(...){}
      PyErr_Format(PyExc_TypeError, "error setting attribute `%s' of resource `%s' at HDF5 file `%s': no support for storing objects of type `%s' on HDF5 files", name.get(), path, filename, Py_TYPE(value)->tp_name);
      return 0;
    }

    PyObject* retval = PyBobIoHDF5File_writeAttribute(self, path, name.get(), type, value, is_array, converted);
    if (!retval) return 0;
    Py_DECREF(retval);

  }

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_set_attributes.name()).c_str(), 0)
}


static auto s_del_attribute = bob::extension::FunctionDoc(
  "del_attribute",
  "Removes a given attribute at the named resource",
  0,
  true
)
.add_prototype("name, [path]")
.add_parameter("name", "str", "The name of the attribute to delete; if the attribute is not available, a ``RuntimeError`` is raised")
.add_parameter("path", "str", "[Default: ``'.'``] The path leading to the resource (dataset or group|directory) you would like to delete an attribute from; if the path does not exist, a ``RuntimeError`` is raised")
;
static PyObject* PyBobIoHDF5File_delAttribute(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_del_attribute.kwlist();

  const char* name = 0;
  const char* path = ".";
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|s", kwlist, &name, &path)) return 0;

  self->f->deleteAttribute(path, name);

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_del_attribute.name()).c_str(), 0)
}


static auto s_del_attributes = bob::extension::FunctionDoc(
  "del_attributes",
  "Removes attributes in a given (existing) path",
  "If the ``attributes`` are not given or set to ``None``, then remove all attributes at the named resource.",
  true
)
.add_prototype("[attributes], [path]")
.add_parameter("attributes", "[str] or None", "[Default: ``None``] An iterable containing the names of the attributes to be removed, or ``None``")
.add_parameter("path", "str", "[Default: ``'.'``] The path leading to the resource (dataset or group|directory) you would like to delete attributes from; if the path does not exist, a ``RuntimeError`` is raised")
;
static PyObject* PyBobIoHDF5File_delAttributes(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_del_attributes.kwlist();

  PyObject* attrs = 0;
  const char* path = ".";
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Os", kwlist, &attrs, &path)) return 0;

  if (attrs && !PyIter_Check(attrs)) {
    PyErr_Format(PyExc_TypeError, "parameter `%s', if set, must be an iterable of strings", kwlist[0]);
    return 0;
  }

  if (attrs) {
    PyObject* iter = PyObject_GetIter(attrs);
    if (!iter) return 0;
    auto iter_ = make_safe(iter);
    while (PyObject* item = PyIter_Next(iter)) {
      auto item_ = make_safe(item);
      auto name = PyBobIo_GetString(item);
      if (!name) return 0;
      self->f->deleteAttribute(path, name.get());
    }
    Py_RETURN_NONE;
  }

  //else, find the attributes and remove all of them
  std::map<std::string, bob::io::base::HDF5Type> attributes;
  self->f->listAttributes(path, attributes);
  for (auto k=attributes.begin(); k!=attributes.end(); ++k) {
    self->f->deleteAttribute(path, k->first);
  }

  Py_RETURN_NONE;
BOB_CATCH_MEMBER(exception_message(self, s_del_attributes.name()).c_str(), 0)
}


static auto s_has_attribute = bob::extension::FunctionDoc(
  "has_attribute",
  "Checks existence of a given attribute at the named resource",
  0,
  true
)
.add_prototype("name, [path]", "existence")
.add_parameter("name", "str", "The name of the attribute to check")
.add_parameter("path", "str", "[Default: ``'.'``] The path leading to the resource (dataset or group|directory) you would like to delete attributes from; if the path does not exist, a ``RuntimeError`` is raised")
.add_return("existence", "bool", "``True``, if the attribute ``name`` exists, otherwise ``False``")
;
static PyObject* PyBobIoHDF5File_hasAttribute(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = s_has_attribute.kwlist();

  const char* name = 0;
  const char* path = ".";
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|s", kwlist, &name, &path)) return 0;

  if (self->f->hasAttribute(path, name))
    Py_RETURN_TRUE;
  Py_RETURN_FALSE;

BOB_CATCH_MEMBER(exception_message(self, s_has_attribute.name()).c_str(), 0)
}


static PyMethodDef PyBobIoHDF5File_methods[] = {
  {
    s_close.name(),
    (PyCFunction)PyBobIoHDF5File_close,
    METH_VARARGS|METH_KEYWORDS,
    s_close.doc()
  },
  {
    s_flush.name(),
    (PyCFunction)PyBobIoHDF5File_flush,
    METH_VARARGS|METH_KEYWORDS,
    s_flush.doc()
  },
  {
    s_cd.name(),
    (PyCFunction)PyBobIoHDF5File_changeDirectory,
    METH_VARARGS|METH_KEYWORDS,
    s_cd.doc(),
  },
  {
    s_has_group.name(),
    (PyCFunction)PyBobIoHDF5File_hasGroup,
    METH_VARARGS|METH_KEYWORDS,
    s_has_group.doc(),
  },
  {
    s_create_group.name(),
    (PyCFunction)PyBobIoHDF5File_createGroup,
    METH_VARARGS|METH_KEYWORDS,
    s_create_group.doc(),
  },
  {
    s_has_dataset.name(),
    (PyCFunction)PyBobIoHDF5File_hasDataset,
    METH_VARARGS|METH_KEYWORDS,
    s_has_dataset.doc(),
  },
  {
    s_has_key.name(),
    (PyCFunction)PyBobIoHDF5File_hasDataset,
    METH_VARARGS|METH_KEYWORDS,
    s_has_key.doc(),
  },
  {
    s_describe.name(),
    (PyCFunction)PyBobIoHDF5File_describe,
    METH_VARARGS|METH_KEYWORDS,
    s_describe.doc(),
  },
  {
    s_unlink.name(),
    (PyCFunction)PyBobIoHDF5File_unlink,
    METH_VARARGS|METH_KEYWORDS,
    s_unlink.doc(),
  },
  {
    s_rename.name(),
    (PyCFunction)PyBobIoHDF5File_rename,
    METH_VARARGS|METH_KEYWORDS,
    s_rename.doc(),
  },
  {
    s_paths.name(),
    (PyCFunction)PyBobIoHDF5File_paths,
    METH_VARARGS|METH_KEYWORDS,
    s_paths.doc(),
  },
  {
    s_keys.name(),
    (PyCFunction)PyBobIoHDF5File_paths,
    METH_VARARGS|METH_KEYWORDS,
    s_keys.doc(),
  },
  {
    s_sub_groups.name(),
    (PyCFunction)PyBobIoHDF5File_subGroups,
    METH_VARARGS|METH_KEYWORDS,
    s_sub_groups.doc(),
  },
  {
    s_read.name(),
    (PyCFunction)PyBobIoHDF5File_read,
    METH_VARARGS|METH_KEYWORDS,
    s_read.doc(),
  },
  {
    s_get.name(),
    (PyCFunction)PyBobIoHDF5File_read,
    METH_VARARGS|METH_KEYWORDS,
    s_get.doc(),
  },
  {
    s_lread.name(),
    (PyCFunction)PyBobIoHDF5File_listRead,
    METH_VARARGS|METH_KEYWORDS,
    s_lread.doc(),
  },
  {
    s_replace.name(),
    (PyCFunction)PyBobIoHDF5File_replace,
    METH_VARARGS|METH_KEYWORDS,
    s_replace.doc(),
  },
  {
    s_append.name(),
    (PyCFunction)PyBobIoHDF5File_append,
    METH_VARARGS|METH_KEYWORDS,
    s_append.doc(),
  },
  {
    s_set.name(),
    (PyCFunction)PyBobIoHDF5File_set,
    METH_VARARGS|METH_KEYWORDS,
    s_set.doc(),
  },
  {
    s_write.name(),
    (PyCFunction)PyBobIoHDF5File_set,
    METH_VARARGS|METH_KEYWORDS,
    s_write.doc(),
  },
  {
    s_copy.name(),
    (PyCFunction)PyBobIoHDF5File_copy,
    METH_VARARGS|METH_KEYWORDS,
    s_copy.doc(),
  },
  {
    s_get_attribute.name(),
    (PyCFunction)PyBobIoHDF5File_getAttribute,
    METH_VARARGS|METH_KEYWORDS,
    s_get_attribute.doc(),
  },
  {
    s_get_attributes.name(),
    (PyCFunction)PyBobIoHDF5File_getAttributes,
    METH_VARARGS|METH_KEYWORDS,
    s_get_attributes.doc(),
  },
  {
    s_set_attribute.name(),
    (PyCFunction)PyBobIoHDF5File_setAttribute,
    METH_VARARGS|METH_KEYWORDS,
    s_set_attribute.doc(),
  },
  {
    s_set_attributes.name(),
    (PyCFunction)PyBobIoHDF5File_setAttributes,
    METH_VARARGS|METH_KEYWORDS,
    s_set_attributes.doc(),
  },
  {
    s_del_attribute.name(),
    (PyCFunction)PyBobIoHDF5File_delAttribute,
    METH_VARARGS|METH_KEYWORDS,
    s_del_attribute.doc(),
  },
  {
    s_del_attributes.name(),
    (PyCFunction)PyBobIoHDF5File_delAttributes,
    METH_VARARGS|METH_KEYWORDS,
    s_del_attributes.doc(),
  },
  {
    s_has_attribute.name(),
    (PyCFunction)PyBobIoHDF5File_hasAttribute,
    METH_VARARGS|METH_KEYWORDS,
    s_has_attribute.doc(),
  },
  {0}  /* Sentinel */
};

static auto s_cwd = bob::extension::VariableDoc(
  "cwd",
  "str",
  "The current working directory set on the file"
);
static PyObject* PyBobIoHDF5File_cwd(PyBobIoHDF5FileObject* self) {
BOB_TRY
  return Py_BuildValue("s", self->f->cwd().c_str());
BOB_CATCH_MEMBER(exception_message(self, s_cwd.name()).c_str(), 0)
}

static auto s_filename = bob::extension::VariableDoc(
  "filename",
  "str",
  "The name (and path) of the underlying file on hard disk"
);
static PyObject* PyBobIoHDF5File_filename(PyBobIoHDF5FileObject* self) {
BOB_TRY
  return Py_BuildValue("s", self->f->filename().c_str());
BOB_CATCH_MEMBER(exception_message(self, s_filename.name()).c_str(), 0)
}


static auto s_writable = bob::extension::VariableDoc(
  "writable",
  "bool",
  "Has this file been opened in writable mode?"
);
static PyObject* PyBobIoHDF5File_writable(PyBobIoHDF5FileObject* self) {
BOB_TRY
  return Py_BuildValue("b", self->f->writable());
BOB_CATCH_MEMBER(exception_message(self, s_writable.name()).c_str(), 0)
}

static PyGetSetDef PyBobIoHDF5File_getseters[] = {
    {
      s_cwd.name(),
      (getter)PyBobIoHDF5File_cwd,
      0,
      s_cwd.doc(),
      0,
    },
    {
      s_filename.name(),
      (getter)PyBobIoHDF5File_filename,
      0,
      s_filename.doc(),
      0,
    },
    {
      s_writable.name(),
      (getter)PyBobIoHDF5File_writable,
      0,
      s_writable.doc(),
      0,
    },
    {0}  /* Sentinel */
};

PyTypeObject PyBobIoHDF5File_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    0
};

bool init_HDF5File(PyObject* module){

  // initialize the HDF5 file
  PyBobIoHDF5File_Type.tp_name = s_hdf5file.name();
  PyBobIoHDF5File_Type.tp_basicsize = sizeof(PyBobIoHDF5FileObject);
  PyBobIoHDF5File_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobIoHDF5File_Type.tp_doc = s_hdf5file.doc();

  // set the functions
  PyBobIoHDF5File_Type.tp_new = PyBobIoHDF5File_New;
  PyBobIoHDF5File_Type.tp_init = reinterpret_cast<initproc>(PyBobIoHDF5File_init);
  PyBobIoHDF5File_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIoHDF5File_Delete);
  PyBobIoHDF5File_Type.tp_methods = PyBobIoHDF5File_methods;
  PyBobIoHDF5File_Type.tp_getset = PyBobIoHDF5File_getseters;

  PyBobIoHDF5File_Type.tp_str = reinterpret_cast<reprfunc>(PyBobIoHDF5File_repr);
  PyBobIoHDF5File_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobIoHDF5File_repr);


  // check that everyting is fine
  if (PyType_Ready(&PyBobIoHDF5File_Type) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&PyBobIoHDF5File_Type);
  return PyModule_AddObject(module, s_hdf5file.name(), (PyObject*)&PyBobIoHDF5File_Type) >= 0;
}
