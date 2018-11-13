/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  6 Nov 07:57:57 2013
 *
 * @brief Implementation of our bobskin class
 */

#include "bobskin.h"
#include <stdexcept>

bobskin::bobskin(PyObject* array, bob::io::base::array::ElementType eltype) {

  if (!PyArray_CheckExact(array)) {
    PyErr_SetString(PyExc_TypeError, "input object to bobskin constructor is not (exactly) a numpy.ndarray");
    throw std::runtime_error("error is already set");
  }

  m_type.set<npy_intp>(eltype, PyArray_NDIM((PyArrayObject*)array),
      PyArray_DIMS((PyArrayObject*)array),
      PyArray_STRIDES((PyArrayObject*)array));

  m_ptr = PyArray_DATA((PyArrayObject*)array);

}

bobskin::bobskin(PyArrayObject* array, bob::io::base::array::ElementType eltype) {

  m_type.set<npy_intp>(eltype, PyArray_NDIM((PyArrayObject*)array),
      PyArray_DIMS((PyArrayObject*)array),
      PyArray_STRIDES((PyArrayObject*)array));

  m_ptr = PyArray_DATA((PyArrayObject*)array);

}

static bob::io::base::array::ElementType signed_integer_type(int bits) {
  switch(bits) {
    case 8:
      return bob::io::base::array::t_int8;
    case 16:
      return bob::io::base::array::t_int16;
    case 32:
      return bob::io::base::array::t_int32;
    case 64:
      return bob::io::base::array::t_int64;
    default:
      PyErr_Format(PyExc_TypeError, "unsupported signed integer element type with %d bits", bits);
  }
  return bob::io::base::array::t_unknown;
}

static bob::io::base::array::ElementType unsigned_integer_type(int bits) {
  switch(bits) {
    case 8:
      return bob::io::base::array::t_uint8;
    case 16:
      return bob::io::base::array::t_uint16;
    case 32:
      return bob::io::base::array::t_uint32;
    case 64:
      return bob::io::base::array::t_uint64;
    default:
      PyErr_Format(PyExc_TypeError, "unsupported unsigned signed integer element type with %d bits", bits);
  }
  return bob::io::base::array::t_unknown;
}

static bob::io::base::array::ElementType num_to_type (int num) {
  switch(num) {
    case NPY_BOOL:
      return bob::io::base::array::t_bool;

    //signed integers
    case NPY_BYTE:
      return signed_integer_type(NPY_BITSOF_CHAR);
    case NPY_SHORT:
      return signed_integer_type(NPY_BITSOF_SHORT);
    case NPY_INT:
      return signed_integer_type(NPY_BITSOF_INT);
    case NPY_LONG:
      return signed_integer_type(NPY_BITSOF_LONG);
    case NPY_LONGLONG:
      return signed_integer_type(NPY_BITSOF_LONGLONG);

    //unsigned integers
    case NPY_UBYTE:
      return unsigned_integer_type(NPY_BITSOF_CHAR);
    case NPY_USHORT:
      return unsigned_integer_type(NPY_BITSOF_SHORT);
    case NPY_UINT:
      return unsigned_integer_type(NPY_BITSOF_INT);
    case NPY_ULONG:
      return unsigned_integer_type(NPY_BITSOF_LONG);
    case NPY_ULONGLONG:
      return unsigned_integer_type(NPY_BITSOF_LONGLONG);

    //floats
    case NPY_FLOAT32:
      return bob::io::base::array::t_float32;
    case NPY_FLOAT64:
      return bob::io::base::array::t_float64;
#ifdef NPY_FLOAT128
    case NPY_FLOAT128:
      return bob::io::base::array::t_float128;
#endif

    //complex
    case NPY_COMPLEX64:
      return bob::io::base::array::t_complex64;
    case NPY_COMPLEX128:
      return bob::io::base::array::t_complex128;
#ifdef NPY_COMPLEX256
    case NPY_COMPLEX256:
      return bob::io::base::array::t_complex256;
#endif

    default:
      PyErr_Format(PyExc_TypeError, "unsupported NumPy element type (%d)", num);
  }

  return bob::io::base::array::t_unknown;
}

bobskin::bobskin(PyBlitzArrayObject* array) {
  bob::io::base::array::ElementType eltype = num_to_type(array->type_num);
  if (eltype == bob::io::base::array::t_unknown) {
    throw std::runtime_error("error is already set");
  }
  m_type.set<Py_ssize_t>(num_to_type(array->type_num), array->ndim,
      array->shape, array->stride);
  m_ptr = array->data;
}

bobskin::~bobskin() { }

void bobskin::set(const interface&) {
  PyErr_SetString(PyExc_NotImplementedError, "setting C++ bobskin with (const interface&) is not implemented - DEBUG ME!");
  throw std::runtime_error("error is already set");
}

void bobskin::set(boost::shared_ptr<interface>) {
  PyErr_SetString(PyExc_NotImplementedError, "setting C++ bobskin with (boost::shared_ptr<interface>) is not implemented - DEBUG ME!");
  throw std::runtime_error("error is already set");
}

void bobskin::set (const bob::io::base::array::typeinfo&) {
  PyErr_SetString(PyExc_NotImplementedError, "setting C++ bobskin with (const typeinfo&) implemented - DEBUG ME!");
  throw std::runtime_error("error is already set");
}

boost::shared_ptr<void> bobskin::owner() {
  PyErr_SetString(PyExc_NotImplementedError, "acquiring non-const owner from C++ bobskin is not implemented - DEBUG ME!");
  throw std::runtime_error("error is already set");
}

boost::shared_ptr<const void> bobskin::owner() const {
  PyErr_SetString(PyExc_NotImplementedError, "acquiring const owner from C++ bobskin is not implemented - DEBUG ME!");
  throw std::runtime_error("error is already set");
}
