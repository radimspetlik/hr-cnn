/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Discrete distributions (with integers or floating point numbers)
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>
#include <boost/make_shared.hpp>
#include <boost/version.hpp>

#include <bob.core/random.h>

static auto discrete_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".discrete",
  "Models a random discrete distribution",
  "A discrete distribution can only assume certain values, which for this class is defined as a number ``i`` in the range ``[0, len(probabilities)[``. "
  "Note that the condition :math:`\\sum(P) = 1`, with ``P = probabilities``, is enforced by normalizing the input values so that the sum over all probabilities always equals 1."
)
.add_constructor(bob::extension::FunctionDoc(
  "discrete",
  "Constructs a new discrete distribution object"
)
.add_prototype("dtype, probabilities", "")
.add_parameter("dtype", ":py:class:`numpy.dtype` or anything that converts to a dtype", "The data type to get the distribution for; only integral types are supported")
.add_parameter("probabilities", "[float] or iterable of floats", "The probabilities for drawing index ``i``; this also defines the number of values that are drawn")
);

/* How to create a new PyBoostDiscreteObject */
static PyObject* PyBoostDiscrete_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostDiscreteObject* self = (PyBoostDiscreteObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return Py_BuildValue("N", self);
}

/* How to delete a PyBoostDiscreteObject */
static void PyBoostDiscrete_Delete (PyBoostDiscreteObject* o) {
  o->distro.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);
}

template <typename T>
boost::shared_ptr<void> make_discrete(PyObject* probabilities) {

  std::vector<double> cxx_probabilities;

  PyObject* iterator = PyObject_GetIter(probabilities);
  if (!iterator) return boost::shared_ptr<void>();
  auto iterator_ = make_safe(iterator);

  while (PyObject* item = PyIter_Next(iterator)) {
    auto item_ = make_safe(item);
    double v = PyFloat_AsDouble(item);
    if (PyErr_Occurred()) return boost::shared_ptr<void>();
    cxx_probabilities.push_back(v);
  }

  return boost::make_shared<bob::core::random::discrete_distribution<T,double>>(cxx_probabilities);
}

PyObject* PyBoostDiscrete_SimpleNew (int type_num, PyObject* probabilities) {
BOB_TRY
  PyBoostDiscreteObject* retval = (PyBoostDiscreteObject*)PyBoostDiscrete_New(&PyBoostDiscrete_Type, 0, 0);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_UINT8:
      retval->distro = make_discrete<uint8_t>(probabilities);
      break;
    case NPY_UINT16:
      retval->distro = make_discrete<uint16_t>(probabilities);
      break;
    case NPY_UINT32:
      retval->distro = make_discrete<uint32_t>(probabilities);
      break;
    case NPY_UINT64:
      retval->distro = make_discrete<uint64_t>(probabilities);
      break;
    case NPY_INT8:
      retval->distro = make_discrete<int8_t>(probabilities);
      break;
    case NPY_INT16:
      retval->distro = make_discrete<int16_t>(probabilities);
      break;
    case NPY_INT32:
      retval->distro = make_discrete<int32_t>(probabilities);
      break;
    case NPY_INT64:
      retval->distro = make_discrete<int64_t>(probabilities);
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
static int PyBoostDiscrete_Init(PyBoostDiscreteObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = discrete_doc.kwlist();

  PyObject* probabilities;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O", kwlist, &PyBlitzArray_TypenumConverter, &self->type_num, &probabilities)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_UINT8:
      self->distro = make_discrete<uint8_t>(probabilities);
      break;
    case NPY_UINT16:
      self->distro = make_discrete<uint16_t>(probabilities);
      break;
    case NPY_UINT32:
      self->distro = make_discrete<uint32_t>(probabilities);
      break;
    case NPY_UINT64:
      self->distro = make_discrete<uint64_t>(probabilities);
      break;
    case NPY_INT8:
      self->distro = make_discrete<int8_t>(probabilities);
      break;
    case NPY_INT16:
      self->distro = make_discrete<int16_t>(probabilities);
      break;
    case NPY_INT32:
      self->distro = make_discrete<int32_t>(probabilities);
      break;
    case NPY_INT64:
      self->distro = make_discrete<int64_t>(probabilities);
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

int PyBoostDiscrete_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostDiscrete_Type));
}

int PyBoostDiscrete_Converter(PyObject* o, PyBoostDiscreteObject** a) {
  if (!PyBoostDiscrete_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostDiscreteObject*>(o);
  return 1;
}

static auto probabilities_doc = bob::extension::VariableDoc(
  "probabilities",
  "[float]",
  "The values have been set for the discrete probabilities of every entry in this distribution"
);
template <typename T>
PyObject* get_probabilities(PyBoostDiscreteObject* self) {
  std::vector<double> w = boost::static_pointer_cast<bob::core::random::discrete_distribution<T,double>>(self->distro)->probabilities();
  PyObject* retval = PyTuple_New(w.size());
  if (!retval) return 0;
  for (size_t k=0; k<w.size(); ++k) {
    PyTuple_SET_ITEM(retval, k, Py_BuildValue("d", w[k]));
  }
  return retval;
}

static PyObject* PyBoostDiscrete_GetProbabilities(PyBoostDiscreteObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_UINT8:
      return get_probabilities<uint8_t>(self);
    case NPY_UINT16:
      return get_probabilities<uint16_t>(self);
    case NPY_UINT32:
      return get_probabilities<uint32_t>(self);
    case NPY_UINT64:
      return get_probabilities<uint64_t>(self);
    case NPY_INT8:
      return get_probabilities<int8_t>(self);
    case NPY_INT16:
      return get_probabilities<int16_t>(self);
    case NPY_INT32:
      return get_probabilities<int32_t>(self);
    case NPY_INT64:
      return get_probabilities<int64_t>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get minimum of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("probabilities", 0)
}

static auto dtype_doc = bob::extension::VariableDoc(
  "dtype",
  ":py:class:`numpy.dtype`",
  "The type of scalars produced by this discrete distribution"
);
static PyObject* PyBoostDiscrete_GetDtype(PyBoostDiscreteObject* self) {
  return reinterpret_cast<PyObject*>(PyArray_DescrFromType(self->type_num));
}

static PyGetSetDef PyBoostDiscrete_getseters[] = {
    {
      dtype_doc.name(),
      (getter)PyBoostDiscrete_GetDtype,
      0,
      dtype_doc.doc(),
      0,
    },
    {
      probabilities_doc.name(),
      (getter)PyBoostDiscrete_GetProbabilities,
      0,
      probabilities_doc.doc(),
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
template <typename T> PyObject* reset(PyBoostDiscreteObject* self) {
  boost::static_pointer_cast<bob::core::random::discrete_distribution<T,double>>(self->distro)->reset();
  Py_RETURN_NONE;
}

static PyObject* PyBoostDiscrete_Reset(PyBoostDiscreteObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_FLOAT32:
      return reset<float>(self);
    case NPY_FLOAT64:
      return reset<double>(self);
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
.add_return("value", "dtype", "A random value that follows the discrete distribution")
;
template <typename T> PyObject* call(PyBoostDiscreteObject* self, PyBoostMt19937Object* rng) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::discrete_distribution<T,double>>(self->distro)->operator()(*rng->rng));
}

static PyObject* PyBoostDiscrete_Call(PyBoostDiscreteObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = call_doc.kwlist();

  PyBoostMt19937Object* rng;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyBoostMt19937_Type, &rng)) return 0; ///< FAILURE

  switch(self->type_num) {
    case NPY_UINT8:
      return call<uint8_t>(self, rng);
      break;
    case NPY_UINT16:
      return call<uint16_t>(self, rng);
      break;
    case NPY_UINT32:
      return call<uint32_t>(self, rng);
      break;
    case NPY_UINT64:
      return call<uint64_t>(self, rng);
      break;
    case NPY_INT8:
      return call<int8_t>(self, rng);
      break;
    case NPY_INT16:
      return call<int16_t>(self, rng);
      break;
    case NPY_INT32:
      return call<int32_t>(self, rng);
      break;
    case NPY_INT64:
      return call<int64_t>(self, rng);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot call %s(T) with T having an unsupported numpy type number of %d", Py_TYPE(self)->tp_name, self->type_num);
  }

  return 0; ///< FAILURE
BOB_CATCH_MEMBER("call", 0)
}

static PyMethodDef PyBoostDiscrete_methods[] = {
    {
      call_doc.name(),
      (PyCFunction)PyBoostDiscrete_Call,
      METH_VARARGS|METH_KEYWORDS,
      call_doc.doc(),
    },
    {
      reset_doc.name(),
      (PyCFunction)PyBoostDiscrete_Reset,
      METH_NOARGS,
      reset_doc.doc(),
    },
    {0}  /* Sentinel */
};


/**
 * String representation and print out
 */
static PyObject* PyBoostDiscrete_Repr(PyBoostDiscreteObject* self) {

  PyObject* probabilities = PyBoostDiscrete_GetProbabilities(self);
  if (!probabilities) return 0;
  auto probabilities_ = make_safe(probabilities);
  PyObject* prob_str = PyObject_Str(probabilities);
  if (!prob_str) return 0;
  auto prob_str_ = make_safe(prob_str);

  return PyString_FromFormat(
      "%s(dtype='%s' , probabilities=%s)",
      Py_TYPE(self)->tp_name,
      PyBlitzArray_TypenumAsString(self->type_num),
      PyString_AS_STRING(prob_str)
      );
}

PyTypeObject PyBoostDiscrete_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BoostDiscrete(PyObject* module)
{
  // initialize the type struct
  PyBoostDiscrete_Type.tp_name = discrete_doc.name();
  PyBoostDiscrete_Type.tp_basicsize = sizeof(PyBoostDiscreteObject);
  PyBoostDiscrete_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBoostDiscrete_Type.tp_doc = discrete_doc.doc();
  PyBoostDiscrete_Type.tp_str = reinterpret_cast<reprfunc>(PyBoostDiscrete_Repr);
  PyBoostDiscrete_Type.tp_repr = reinterpret_cast<reprfunc>(PyBoostDiscrete_Repr);

  // set the functions
  PyBoostDiscrete_Type.tp_new = PyBoostDiscrete_New;
  PyBoostDiscrete_Type.tp_init = reinterpret_cast<initproc>(PyBoostDiscrete_Init);
  PyBoostDiscrete_Type.tp_dealloc = reinterpret_cast<destructor>(PyBoostDiscrete_Delete);
  PyBoostDiscrete_Type.tp_methods = PyBoostDiscrete_methods;
  PyBoostDiscrete_Type.tp_getset = PyBoostDiscrete_getseters;
  PyBoostDiscrete_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBoostDiscrete_Call);

  // check that everything is fine
  if (PyType_Ready(&PyBoostDiscrete_Type) < 0) return false;

  // add the type to the module
  return PyModule_AddObject(module, "discrete", Py_BuildValue("O", &PyBoostDiscrete_Type)) >= 0;
}
