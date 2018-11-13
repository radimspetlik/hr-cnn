/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Uniform distributions (with integers or floating point numbers)
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>

#include <boost/make_shared.hpp>

#include <boost/random.hpp>

static auto uniform_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".uniform",
  "Models a random uniform distribution",
  "On each invocation, it returns a random value uniformly distributed in the set of numbers [min, max] (integer) and [min, max[ (real-valued)"
)
.add_constructor(bob::extension::FunctionDoc(
  "uniform",
  "Constructs a new uniform distribution object",
  "If the values ``min`` and ``max`` are not given, they are assumed to be ``min=0`` and ``max=9``, for integral distributions and ``min=0.0`` and ``max=1.0`` for real-valued distributions."
)
.add_prototype("dtype, [min], [max]", "")
.add_parameter("dtype", ":py:class:`numpy.dtype` or anything that converts to a dtype", "The data type to get the distribution for")
.add_parameter("min", "dtype", "[Default: 0] The minimum value to draw")
.add_parameter("max", "dtype", "[Default: 1. (for real-valued ``dtype``) or 9 (for integral ``dtype``)] The maximum value to be drawn")
);

/* How to create a new PyBoostUniformObject */
static PyObject* PyBoostUniform_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostUniformObject* self = (PyBoostUniformObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return Py_BuildValue("N", self);
}

/* How to delete a PyBoostUniformObject */
static void PyBoostUniform_Delete (PyBoostUniformObject* o) {
  o->distro.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);
}

static boost::shared_ptr<void> make_uniform_bool() {
  return boost::make_shared<boost::uniform_smallint<uint8_t>>(0, 1);
}

template <typename T>
boost::shared_ptr<void> make_uniform_int(PyObject* min, PyObject* max) {
  T cmin = 0;
  if (min) cmin = PyBlitzArrayCxx_AsCScalar<T>(min);
  T cmax = 9;
  if (max) cmax = PyBlitzArrayCxx_AsCScalar<T>(max);
  return boost::make_shared<boost::uniform_int<T>>(cmin, cmax);
}

template <typename T>
boost::shared_ptr<void> make_uniform_real(PyObject* min, PyObject* max) {
  T cmin = 0;
  if (min) cmin = PyBlitzArrayCxx_AsCScalar<T>(min);
  T cmax = 1;
  if (max) cmax = PyBlitzArrayCxx_AsCScalar<T>(max);
  return boost::make_shared<boost::uniform_real<T>>(cmin, cmax);
}

PyObject* PyBoostUniform_SimpleNew (int type_num, PyObject* min, PyObject* max) {
BOB_TRY
  if (type_num == NPY_BOOL && (min || max)) {
    PyErr_Format(PyExc_ValueError, "uniform distributions of boolean scalars cannot have a maximum or minimum");
    return 0;
  }

  PyBoostUniformObject* retval = (PyBoostUniformObject*)PyBoostUniform_New(&PyBoostUniform_Type, 0, 0);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_BOOL:
      retval->distro = make_uniform_bool();
      break;
    case NPY_UINT8:
      retval->distro = make_uniform_int<uint8_t>(min, max);
      break;
    case NPY_UINT16:
      retval->distro = make_uniform_int<uint16_t>(min, max);
      break;
    case NPY_UINT32:
      retval->distro = make_uniform_int<uint32_t>(min, max);
      break;
    case NPY_UINT64:
      retval->distro = make_uniform_int<uint64_t>(min, max);
      break;
    case NPY_INT8:
      retval->distro = make_uniform_int<int8_t>(min, max);
      break;
    case NPY_INT16:
      retval->distro = make_uniform_int<int16_t>(min, max);
      break;
    case NPY_INT32:
      retval->distro = make_uniform_int<int32_t>(min, max);
      break;
    case NPY_INT64:
      retval->distro = make_uniform_int<int64_t>(min, max);
      break;
    case NPY_FLOAT32:
      retval->distro = make_uniform_real<float>(min, max);
      break;
    case NPY_FLOAT64:
      retval->distro = make_uniform_real<double>(min, max);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d", Py_TYPE(retval)->tp_name, retval->type_num);
      return 0;
  }

  if (!retval->distro) { // a problem occurred
    return 0;
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_FUNCTION("SimpleNew", 0)
}

/* Implements the __init__(self) function */
static int PyBoostUniform_Init(PyBoostUniformObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = uniform_doc.kwlist();

  PyObject* min = 0;
  PyObject* max = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|OO", kwlist, &PyBlitzArray_TypenumConverter, &self->type_num, &min, &max)) return -1; ///< FAILURE

  if (self->type_num == NPY_BOOL && (min || max)) {
    PyErr_Format(PyExc_ValueError, "uniform distributions of boolean scalars cannot have a maximum or minimum");
    return -1;
  }

  switch(self->type_num) {
    case NPY_BOOL:
      self->distro = make_uniform_bool();
      break;
    case NPY_UINT8:
      self->distro = make_uniform_int<uint8_t>(min, max);
      break;
    case NPY_UINT16:
      self->distro = make_uniform_int<uint16_t>(min, max);
      break;
    case NPY_UINT32:
      self->distro = make_uniform_int<uint32_t>(min, max);
      break;
    case NPY_UINT64:
      self->distro = make_uniform_int<uint64_t>(min, max);
      break;
    case NPY_INT8:
      self->distro = make_uniform_int<int8_t>(min, max);
      break;
    case NPY_INT16:
      self->distro = make_uniform_int<int16_t>(min, max);
      break;
    case NPY_INT32:
      self->distro = make_uniform_int<int32_t>(min, max);
      break;
    case NPY_INT64:
      self->distro = make_uniform_int<int64_t>(min, max);
      break;
    case NPY_FLOAT32:
      self->distro = make_uniform_real<float>(min, max);
      break;
    case NPY_FLOAT64:
      self->distro = make_uniform_real<double>(min, max);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d", Py_TYPE(self)->tp_name, self->type_num);
      return -1;
  }

  if (!self->distro) { // a problem occurred
    return -1;
  }

  return 0; ///< SUCCESS
BOB_CATCH_MEMBER("constructor", -1)
}

int PyBoostUniform_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostUniform_Type));
}

int PyBoostUniform_Converter(PyObject* o, PyBoostUniformObject** a) {
  if (!PyBoostUniform_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostUniformObject*>(o);
  return 1;
}


static auto min_doc = bob::extension::VariableDoc(
  "min",
  "dtype",
  "The smallest value that the distribution can produce"
);
template <typename T> PyObject* get_minimum_int(PyBoostUniformObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_int<T>>(self->distro)->min());
}

template <typename T> PyObject* get_minimum_real(PyBoostUniformObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_real<T>>(self->distro)->min());
}

/**
 * Accesses the min value
 */
static PyObject* PyBoostUniform_GetMin(PyBoostUniformObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_BOOL:
      Py_RETURN_FALSE;
    case NPY_UINT8:
      return get_minimum_int<uint8_t>(self);
    case NPY_UINT16:
      return get_minimum_int<uint16_t>(self);
    case NPY_UINT32:
      return get_minimum_int<uint32_t>(self);
    case NPY_UINT64:
      return get_minimum_int<uint64_t>(self);
    case NPY_INT8:
      return get_minimum_int<int8_t>(self);
    case NPY_INT16:
      return get_minimum_int<int16_t>(self);
    case NPY_INT32:
      return get_minimum_int<int32_t>(self);
    case NPY_INT64:
      return get_minimum_int<int64_t>(self);
    case NPY_FLOAT32:
      return get_minimum_real<float>(self);
    case NPY_FLOAT64:
      return get_minimum_real<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get minimum of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("min", 0)
}


static auto max_doc = bob::extension::VariableDoc(
  "max",
  "dtype",
  "The largest value that the distributioncan produce",
  "Integer uniform distributions are bound at [min, max], while real-valued distributions are bound at [min, max[."
);
template <typename T> PyObject* get_maximum_int(PyBoostUniformObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_int<T>>(self->distro)->max());
}

template <typename T> PyObject* get_maximum_real(PyBoostUniformObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_real<T>>(self->distro)->max());
}

/**
 * Accesses the max value
 */
static PyObject* PyBoostUniform_GetMax(PyBoostUniformObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_BOOL:
      Py_RETURN_TRUE;
    case NPY_UINT8:
      return get_maximum_int<uint8_t>(self);
    case NPY_UINT16:
      return get_maximum_int<uint16_t>(self);
    case NPY_UINT32:
      return get_maximum_int<uint32_t>(self);
    case NPY_UINT64:
      return get_maximum_int<uint64_t>(self);
    case NPY_INT8:
      return get_maximum_int<int8_t>(self);
    case NPY_INT16:
      return get_maximum_int<int16_t>(self);
    case NPY_INT32:
      return get_maximum_int<int32_t>(self);
    case NPY_INT64:
      return get_maximum_int<int64_t>(self);
    case NPY_FLOAT32:
      return get_maximum_real<float>(self);
    case NPY_FLOAT64:
      return get_maximum_real<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get maximum of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("max", 0)
}


static auto dtype_doc = bob::extension::VariableDoc(
  "dtype",
  ":py:class:`numpy.dtype`",
  "The type of scalars produced by this uniform distribution"
);
static PyObject* PyBoostUniform_GetDtype(PyBoostUniformObject* self) {
BOB_TRY
  return Py_BuildValue("N", PyArray_DescrFromType(self->type_num));
BOB_CATCH_MEMBER("dtype", 0)
}



static PyGetSetDef PyBoostUniform_getseters[] = {
    {
      dtype_doc.name(),
      (getter)PyBoostUniform_GetDtype,
      0,
      dtype_doc.doc(),
      0,
    },
    {
      min_doc.name(),
      (getter)PyBoostUniform_GetMin,
      0,
      min_doc.doc(),
      0,
    },
    {
      max_doc.name(),
      (getter)PyBoostUniform_GetMax,
      0,
      max_doc.doc(),
      0,
    },
    {0}  /* Sentinel */
};



static auto reset_doc = bob::extension::FunctionDoc(
  "reset",
  "Resets this distribution",
  "After calling this method, subsequent uses of the distribution do not depend on values produced by any random number generator prior to invoking reset",
  true
)
.add_prototype("")
;
template <typename T> PyObject* reset_smallint(PyBoostUniformObject* self) {
  boost::static_pointer_cast<boost::uniform_smallint<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

template <typename T> PyObject* reset_int(PyBoostUniformObject* self) {
  boost::static_pointer_cast<boost::uniform_int<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

template <typename T> PyObject* reset_real(PyBoostUniformObject* self) {
  boost::static_pointer_cast<boost::uniform_real<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

static PyObject* PyBoostUniform_Reset(PyBoostUniformObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_BOOL:
      return reset_smallint<uint8_t>(self);
    case NPY_UINT8:
      return reset_int<uint8_t>(self);
    case NPY_UINT16:
      return reset_int<uint16_t>(self);
    case NPY_UINT32:
      return reset_int<uint32_t>(self);
    case NPY_UINT64:
      return reset_int<uint64_t>(self);
    case NPY_INT8:
      return reset_int<int8_t>(self);
    case NPY_INT16:
      return reset_int<int16_t>(self);
    case NPY_INT32:
      return reset_int<int32_t>(self);
    case NPY_INT64:
      return reset_int<int64_t>(self);
    case NPY_FLOAT32:
      return reset_real<float>(self);
    case NPY_FLOAT64:
      return reset_real<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot reset %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("reset", 0)
}

static auto call_doc = bob::extension::FunctionDoc(
  "draw",
  "Draws one random number from this distribution using the given ``rng``",
  ".. note:: The :py:meth:`__call__` function is a synonym for this ``draw``.",
  true
)
.add_prototype("rng", "value")
.add_parameter("rng", ":py:class:`mt19937`", "The random number generator to use")
.add_return("value", "dtype", "A random value that follows the uniform distribution")
;
static PyObject* call_bool(PyBoostUniformObject* self, PyBoostMt19937Object* rng) {
  if (boost::static_pointer_cast<boost::uniform_smallint<uint8_t>>(self->distro)->operator()(*rng->rng)) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

template <typename T> PyObject* call_int(PyBoostUniformObject* self, PyBoostMt19937Object* rng) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_int<T>>(self->distro)->operator()(*rng->rng));
}

template <typename T> PyObject* call_real(PyBoostUniformObject* self, PyBoostMt19937Object* rng) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_real<T>>(self->distro)->operator()(*rng->rng));
}

/**
 * Calling a PyBoostUniformObject to generate a random number
 */
static PyObject* PyBoostUniform_Call(PyBoostUniformObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = call_doc.kwlist();

  PyBoostMt19937Object* rng;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyBoostMt19937_Type, &rng)) return 0; ///< FAILURE

  switch(self->type_num) {
    case NPY_BOOL:
      return call_bool(self, rng);
      break;
    case NPY_UINT8:
      return call_int<uint8_t>(self, rng);
      break;
    case NPY_UINT16:
      return call_int<uint16_t>(self, rng);
      break;
    case NPY_UINT32:
      return call_int<uint32_t>(self, rng);
      break;
    case NPY_UINT64:
      return call_int<uint64_t>(self, rng);
      break;
    case NPY_INT8:
      return call_int<int8_t>(self, rng);
      break;
    case NPY_INT16:
      return call_int<int16_t>(self, rng);
      break;
    case NPY_INT32:
      return call_int<int32_t>(self, rng);
      break;
    case NPY_INT64:
      return call_int<int64_t>(self, rng);
      break;
    case NPY_FLOAT32:
      return call_real<float>(self, rng);
      break;
    case NPY_FLOAT64:
      return call_real<double>(self, rng);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot call %s(T) with T having an unsupported numpy type number of %d", Py_TYPE(self)->tp_name, self->type_num);
  }

  return 0; ///< FAILURE
BOB_CATCH_MEMBER("call", 0)
}

static PyMethodDef PyBoostUniform_methods[] = {
    {
      call_doc.name(),
      (PyCFunction)PyBoostUniform_Call,
      METH_VARARGS|METH_KEYWORDS,
      call_doc.doc(),
    },
    {
      reset_doc.name(),
      (PyCFunction)PyBoostUniform_Reset,
      METH_NOARGS,
      reset_doc.doc(),
    },
    {0}  /* Sentinel */
};



extern PyObject* scalar_to_bytes(PyObject* s);

/**
 * String representation and print out
 */
static PyObject* PyBoostUniform_Repr(PyBoostUniformObject* self) {
BOB_TRY
  PyObject* smin = scalar_to_bytes(PyBoostUniform_GetMin(self));
  if (!smin) return 0;
  auto smin_ = make_safe(smin);
  PyObject* smax = scalar_to_bytes(PyBoostUniform_GetMax(self));
  if (!smax) return 0;
  auto smax_ = make_safe(smax);

  return
    PyString_FromFormat
      (
       "%s(dtype='%s', min=%s, max=%s)",
       Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num),
       PyString_AS_STRING(smin), PyString_AS_STRING(smax)
      );
BOB_CATCH_MEMBER("repr", 0)
}


PyTypeObject PyBoostUniform_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BoostUniform(PyObject* module)
{
  // initialize the type struct
  PyBoostUniform_Type.tp_name = uniform_doc.name();
  PyBoostUniform_Type.tp_basicsize = sizeof(PyBoostUniformObject);
  PyBoostUniform_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBoostUniform_Type.tp_doc = uniform_doc.doc();
  PyBoostUniform_Type.tp_str = reinterpret_cast<reprfunc>(PyBoostUniform_Repr);
  PyBoostUniform_Type.tp_repr = reinterpret_cast<reprfunc>(PyBoostUniform_Repr);

  // set the functions
  PyBoostUniform_Type.tp_new = PyBoostUniform_New;
  PyBoostUniform_Type.tp_init = reinterpret_cast<initproc>(PyBoostUniform_Init);
  PyBoostUniform_Type.tp_dealloc = reinterpret_cast<destructor>(PyBoostUniform_Delete);
  PyBoostUniform_Type.tp_methods = PyBoostUniform_methods;
  PyBoostUniform_Type.tp_getset = PyBoostUniform_getseters;
  PyBoostUniform_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBoostUniform_Call);

  // check that everything is fine
  if (PyType_Ready(&PyBoostUniform_Type) < 0) return false;

  // add the type to the module
  return PyModule_AddObject(module, "uniform", Py_BuildValue("O", &PyBoostUniform_Type)) >= 0;
}
