/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  6 Nov 07:57:57 2013
 *
 * @brief Implementation of our bobskin class
 */

#include "bobskin.h"
#include <stdexcept>

bobskin::bobskin(PyArrayObject* array, bob::io::base::array::ElementType eltype) {

  m_type.set<npy_intp>(eltype, PyArray_NDIM((PyArrayObject*)array),
      PyArray_DIMS((PyArrayObject*)array),
      PyArray_STRIDES((PyArrayObject*)array));

  m_ptr = PyArray_DATA((PyArrayObject*)array);

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
