/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Gamma distributions (with integers or floating point numbers)
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>
#include <boost/make_shared.hpp>

#include <bob.core/random.h>
#include <boost/random.hpp>


static auto gamma_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".gamma",
  "Models a random gamma distribution",
  "This distribution produces random numbers :math:`x` distributed with the probability density function\n\n"
  ".. math::\n\n   p(x) = x^{\\alpha-1}\\frac{e^{-x}}{\\Gamma(\\alpha)}\n\n"
  "where ``alpha`` (:math:`\\alpha`) is a parameter of this distribution class."
)
.add_constructor(bob::extension::FunctionDoc(
  "gamma",
  "Constructs a new gamma distribution object"
)
.add_prototype("dtype, [alpha]", "")
.add_parameter("dtype", ":py:class:`numpy.dtype` or anything that converts to a dtype", "The data type to get the distribution for; only real-valued types are supported")
.add_parameter("alpha", "dtype", "[Default: 1.] The mean :math:`\\alpha` of the gamma distibution")
);

/* How to create a new PyBoostGammaObject */
static PyObject* PyBoostGamma_New(PyTypeObject* type, PyObject*, PyObject*) {
  /* Allocates the python object itself */
  PyBoostGammaObject* self = (PyBoostGammaObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return Py_BuildValue("N", self);
}

/* How to delete a PyBoostGammaObject */
static void PyBoostGamma_Delete (PyBoostGammaObject* o) {
  o->distro.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);
}

template <typename T>
boost::shared_ptr<void> make_gamma(PyObject* alpha) {
  T calpha = 1.;
  if (alpha) calpha = PyBlitzArrayCxx_AsCScalar<T>(alpha);
  return boost::make_shared<bob::core::random::gamma_distribution<T>>(calpha);
}

PyObject* PyBoostGamma_SimpleNew (int type_num, PyObject* alpha) {
BOB_TRY
  PyBoostGammaObject* retval = (PyBoostGammaObject*)PyBoostGamma_New(&PyBoostGamma_Type, 0, 0);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_FLOAT32:
      retval->distro = make_gamma<float>(alpha);
      break;
    case NPY_FLOAT64:
      retval->distro = make_gamma<double>(alpha);
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
static int PyBoostGamma_Init(PyBoostGammaObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  static char** kwlist = gamma_doc.kwlist();

  PyObject* alpha = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist, &PyBlitzArray_TypenumConverter, &self->type_num, &alpha)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      self->distro = make_gamma<float>(alpha);
      break;
    case NPY_FLOAT64:
      self->distro = make_gamma<double>(alpha);
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

int PyBoostGamma_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostGamma_Type));
}

int PyBoostGamma_Converter(PyObject* o, PyBoostGammaObject** a) {
  if (!PyBoostGamma_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostGammaObject*>(o);
  return 1;
}

static auto alpha_doc = bob::extension::VariableDoc(
  "alpha",
  "dtype",
  "The alpha parameter that the distribution currently has"
);
template <typename T> PyObject* get_alpha(PyBoostGammaObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::gamma_distribution<T>>(self->distro)->alpha());
}

static PyObject* PyBoostGamma_GetAlpha(PyBoostGammaObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_alpha<float>(self);
    case NPY_FLOAT64:
      return get_alpha<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get alpha parameter of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("alpha", 0)
}

static auto dtype_doc = bob::extension::VariableDoc(
  "dtype",
  ":py:class:`numpy.dtype`",
  "The type of scalars produced by this normal distribution"
);
static PyObject* PyBoostGamma_GetDtype(PyBoostGammaObject* self) {
BOB_TRY
  return reinterpret_cast<PyObject*>(PyArray_DescrFromType(self->type_num));
BOB_CATCH_MEMBER("dtype", 0)
}


static PyGetSetDef PyBoostGamma_getseters[] = {
    {
      dtype_doc.name(),
      (getter)PyBoostGamma_GetDtype,
      0,
      dtype_doc.doc(),
      0,
    },
    {
      alpha_doc.name(),
      (getter)PyBoostGamma_GetAlpha,
      0,
      alpha_doc.doc(),
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
template <typename T> PyObject* reset(PyBoostGammaObject* self) {
  boost::static_pointer_cast<bob::core::random::gamma_distribution<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

static PyObject* PyBoostGamma_Reset(PyBoostGammaObject* self) {
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
.add_return("value", "dtype", "A random value that follows the gamma distribution")
;
template <typename T> PyObject* call(PyBoostGammaObject* self, PyBoostMt19937Object* rng) {
  typedef bob::core::random::gamma_distribution<T> distro_t;
  return PyBlitzArrayCxx_FromCScalar((*boost::static_pointer_cast<distro_t>(self->distro))(*rng->rng));
}

/**
 * Calling a PyBoostGammaObject to generate a random number
 */
static PyObject* PyBoostGamma_Call(PyBoostGammaObject* self, PyObject *args, PyObject* kwds) {
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


static PyMethodDef PyBoostGamma_methods[] = {
    {
      call_doc.name(),
      (PyCFunction)PyBoostGamma_Call,
      METH_VARARGS|METH_KEYWORDS,
      call_doc.doc(),
    },
    {
      reset_doc.name(),
      (PyCFunction)PyBoostGamma_Reset,
      METH_NOARGS,
      reset_doc.doc(),
    },
    {0}  /* Sentinel */
};

extern PyObject* scalar_to_bytes(PyObject* s);

/**
 * String representation and print out
 */
static PyObject* PyBoostGamma_Repr(PyBoostGammaObject* self) {

  PyObject* salpha = scalar_to_bytes(PyBoostGamma_GetAlpha(self));
  if (!salpha) return 0;
  auto salpha_ = make_safe(salpha);

  return
    PyString_FromFormat
      (
       "%s(dtype='%s', alpha=%s)",
       Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num),
       PyString_AS_STRING(salpha)
       );
}

PyTypeObject PyBoostGamma_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BoostGamma(PyObject* module)
{
  // initialize the type struct
  PyBoostGamma_Type.tp_name = gamma_doc.name();
  PyBoostGamma_Type.tp_basicsize = sizeof(PyBoostGammaObject);
  PyBoostGamma_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBoostGamma_Type.tp_doc = gamma_doc.doc();
  PyBoostGamma_Type.tp_str = reinterpret_cast<reprfunc>(PyBoostGamma_Repr);
  PyBoostGamma_Type.tp_repr = reinterpret_cast<reprfunc>(PyBoostGamma_Repr);

  // set the functions
  PyBoostGamma_Type.tp_new = PyBoostGamma_New;
  PyBoostGamma_Type.tp_init = reinterpret_cast<initproc>(PyBoostGamma_Init);
  PyBoostGamma_Type.tp_dealloc = reinterpret_cast<destructor>(PyBoostGamma_Delete);
  PyBoostGamma_Type.tp_methods = PyBoostGamma_methods;
  PyBoostGamma_Type.tp_getset = PyBoostGamma_getseters;
  PyBoostGamma_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBoostGamma_Call);

  // check that everything is fine
  if (PyType_Ready(&PyBoostGamma_Type) < 0) return false;

  // add the type to the module
  return PyModule_AddObject(module, "gamma", Py_BuildValue("O", &PyBoostGamma_Type)) >= 0;
}
