/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Binomial distributions (with integers or floating point numbers)
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>

#include <boost/make_shared.hpp>
#include <bob.core/random.h>

static auto binomial_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".binomial",
  "Models a random binomial distribution",
  "This distribution produces random numbers :math:`x` distributed with the probability density function\n\n"
  ".. math::\n\n   {{t}\\choose{k}}p^k(1-p)^{t-k}\n\n"
  "where ``t`` and ``p`` are parameters of the distribution.\n\n"
  ".. warning::\n\n"
  "   This distribution requires that :math:`t >= 0` and that :math:`0 <= p <= 1`."
)
.add_constructor(bob::extension::FunctionDoc(
  "binomial",
  "Creates a new binomial distribution object"
)
.add_prototype("dtype, [t], [p]", "")
.add_parameter("dtype", ":py:class:`numpy.dtype`", "The data type for the drawn random numbers; only integral types are supported")
.add_parameter("t", "float", "[Default: ``1.``] The :math:`t` parameter of the binomial distribution")
.add_parameter("p", "float", "[Default: ``0.5``] The :math:`p` parameter of the binomial distribution")
);

/* How to create a new PyBoostBinomialObject */
static PyObject* PyBoostBinomial_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostBinomialObject* self = (PyBoostBinomialObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return Py_BuildValue("N", self);
}

/* How to delete a PyBoostBinomialObject */
static void PyBoostBinomial_Delete (PyBoostBinomialObject* o) {
  o->distro.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);
}

template <typename T>
boost::shared_ptr<void> make_binomial(PyObject* t, PyObject* p) {
  T ct = 1.;
  if (t) ct = PyBlitzArrayCxx_AsCScalar<T>(t);
  if (ct < 0) {
    PyErr_SetString(PyExc_ValueError, "parameter t must be >= 0");
    return boost::shared_ptr<void>();
  }
  T cp = 0.5;
  if (p) cp = PyBlitzArrayCxx_AsCScalar<T>(p);
  if (cp < 0.0 || cp > 1.0) {
    PyErr_SetString(PyExc_ValueError, "parameter p must lie in the interval [0.0, 1.0]");
    return boost::shared_ptr<void>();
  }
  return boost::make_shared<bob::core::random::binomial_distribution<int64_t,T>>(ct, cp);
}

PyObject* PyBoostBinomial_SimpleNew (int type_num, PyObject* t, PyObject* p) {
BOB_TRY
  PyBoostBinomialObject* retval = (PyBoostBinomialObject*)PyBoostBinomial_New(&PyBoostBinomial_Type, 0, 0);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_FLOAT32:
      retval->distro = make_binomial<float>(t, p);
      break;
    case NPY_FLOAT64:
      retval->distro = make_binomial<double>(t, p);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d (it only supports numpy.float32 or numpy.float64)", Py_TYPE(retval)->tp_name, retval->type_num);
      return 0;
  }

  if (!retval->distro) { // a problem occurred
    return 0;
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_FUNCTION("SimpleNew", 0)
}

/* Implements the __init__(self) function */
static int PyBoostBinomial_Init(PyBoostBinomialObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = binomial_doc.kwlist();

  PyObject* t = 0;
  PyObject* p = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|OO", kwlist, &PyBlitzArray_TypenumConverter, &self->type_num, &t, &p)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      self->distro = make_binomial<float>(t, p);
      break;
    case NPY_FLOAT64:
      self->distro = make_binomial<double>(t, p);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d (it only supports numpy.float32 or numpy.float64)", Py_TYPE(self)->tp_name, self->type_num);
      return -1;
  }

  if (!self->distro) { // a problem occurred
    return -1;
  }

  return 0; ///< SUCCESS
BOB_CATCH_MEMBER("constructor", -1)
}

int PyBoostBinomial_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostBinomial_Type));
}

int PyBoostBinomial_Converter(PyObject* o, PyBoostBinomialObject** a) {
  if (!PyBoostBinomial_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostBinomialObject*>(o);
  return 1;
}


static auto t_doc = bob::extension::VariableDoc(
  "t",
  "float",
  "The parameter ``t`` of the distribution"
);
template <typename T> PyObject* get_t(PyBoostBinomialObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::binomial_distribution<int64_t,T>>(self->distro)->t());
}

static PyObject* PyBoostBinomial_GetT(PyBoostBinomialObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_t<float>(self);
    case NPY_FLOAT64:
      return get_t<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get parameter `t` of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("t", 0)
}


static auto p_doc = bob::extension::VariableDoc(
  "p",
  "float",
  "The parameter ``p`` of the distribution"
);
template <typename T> PyObject* get_p(PyBoostBinomialObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::binomial_distribution<int64_t,T>>(self->distro)->p());
}

static PyObject* PyBoostBinomial_GetP(PyBoostBinomialObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_p<float>(self);
    case NPY_FLOAT64:
      return get_p<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get parameter `p` of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("p", 0)
}


static auto dtype_doc = bob::extension::VariableDoc(
  "dtype",
  ":py:class:`numpy.dtype`",
  "The type of scalars produced by this binomial distribution"
);
static PyObject* PyBoostBinomial_GetDtype(PyBoostBinomialObject* self) {
BOB_TRY
  return reinterpret_cast<PyObject*>(PyArray_DescrFromType(self->type_num));
BOB_CATCH_MEMBER("dtype", 0)
}


static PyGetSetDef PyBoostBinomial_getseters[] = {
    {
      dtype_doc.name(),
      (getter)PyBoostBinomial_GetDtype,
      0,
      dtype_doc.doc(),
      0,
    },
    {
      t_doc.name(),
      (getter)PyBoostBinomial_GetT,
      0,
      t_doc.doc(),
      0,
    },
    {
      p_doc.name(),
      (getter)PyBoostBinomial_GetP,
      0,
      p_doc.doc(),
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
template <typename T> PyObject* reset(PyBoostBinomialObject* self) {
  boost::static_pointer_cast<bob::core::random::binomial_distribution<int64_t,T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

static PyObject* PyBoostBinomial_Reset(PyBoostBinomialObject* self) {
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
.add_return("value", "dtype", "A random value that follows the binomial distribution")
;
template <typename T> PyObject* call(PyBoostBinomialObject* self, PyBoostMt19937Object* rng) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::binomial_distribution<int64_t,T>>(self->distro)->operator()(*rng->rng));
}

static PyObject* PyBoostBinomial_Call(PyBoostBinomialObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = call_doc.kwlist();

  PyBoostMt19937Object* rng;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyBoostMt19937_Type, &rng)) return 0; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      return call<float>(self, rng);
      break;
    case NPY_FLOAT64:
      return call<double>(self, rng);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot call %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
  }

  return 0; ///< FAILURE
BOB_CATCH_MEMBER("call", 0)
}

static PyMethodDef PyBoostBinomial_methods[] = {
    {
      call_doc.name(),
      (PyCFunction)PyBoostBinomial_Call,
      METH_VARARGS|METH_KEYWORDS,
      call_doc.doc(),
    },
    {
      reset_doc.name(),
      (PyCFunction)PyBoostBinomial_Reset,
      METH_NOARGS,
      reset_doc.doc(),
    },
    {0}  /* Sentinel */
};


extern PyObject* scalar_to_bytes(PyObject* s);

/**
 * String representation and print out
 */
static PyObject* PyBoostBinomial_Repr(PyBoostBinomialObject* self) {
BOB_TRY
  PyObject* st = scalar_to_bytes(PyBoostBinomial_GetT(self));
  if (!st) return 0;
  auto st_ = make_safe(st);
  PyObject* sp = scalar_to_bytes(PyBoostBinomial_GetP(self));
  if (!sp) return 0;
  auto sp_ = make_safe(sp);

  return
    PyString_FromFormat
      (
       "%s(dtype='%s', t=%s, p=%s)",
       Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num),
       PyString_AS_STRING(st), PyString_AS_STRING(sp)
      );
BOB_CATCH_MEMBER("repr", 0)
}


PyTypeObject PyBoostBinomial_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BoostBinomial(PyObject* module)
{
  // initialize the type struct
  PyBoostBinomial_Type.tp_name = binomial_doc.name();
  PyBoostBinomial_Type.tp_basicsize = sizeof(PyBoostBinomialObject);
  PyBoostBinomial_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBoostBinomial_Type.tp_doc = binomial_doc.doc();
  PyBoostBinomial_Type.tp_str = reinterpret_cast<reprfunc>(PyBoostBinomial_Repr);
  PyBoostBinomial_Type.tp_repr = reinterpret_cast<reprfunc>(PyBoostBinomial_Repr);

  // set the functions
  PyBoostBinomial_Type.tp_new = PyBoostBinomial_New;
  PyBoostBinomial_Type.tp_init = reinterpret_cast<initproc>(PyBoostBinomial_Init);
  PyBoostBinomial_Type.tp_dealloc = reinterpret_cast<destructor>(PyBoostBinomial_Delete);
  PyBoostBinomial_Type.tp_methods = PyBoostBinomial_methods;
  PyBoostBinomial_Type.tp_getset = PyBoostBinomial_getseters;
  PyBoostBinomial_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBoostBinomial_Call);

  // check that everything is fine
  if (PyType_Ready(&PyBoostBinomial_Type) < 0) return false;

  // add the type to the module
  return PyModule_AddObject(module, "binomial", Py_BuildValue("O", &PyBoostBinomial_Type)) >= 0;
}
