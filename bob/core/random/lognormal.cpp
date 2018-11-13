/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief LogNormal distributions (with floating point numbers)
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>
#include <boost/make_shared.hpp>

#include <bob.core/random.h>

static auto lognormal_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".lognormal",
  "Models a random lognormal distribution",
  "This distribution produces random numbers :math:`x` distributed with the probability density function\n\n"
  ".. math::\n\n   p(x) = \\frac{1}{x \\sigma_N \\sqrt{2\\pi}} e^{\\frac{-\\left(\\log(x)-\\mu_N\\right)^2}{2\\sigma_N^2}}\n\n"
  "for :math:`x > 0` and :math:`\\sigma_N = \\sqrt{\\log\\left(1 + \\frac{\\sigma^2}{\\mu^2}\\right)}`, "
  "where the ``mean`` (:math:`\\mu`) and ``sigma`` (:math:`\\sigma`, the standard deviation) are the parameters of this distribution class."
)
.add_constructor(bob::extension::FunctionDoc(
  "lognormal",
  "Constructs a new lognormal distribution object"
)
.add_prototype("dtype, [mean], [sigma]", "")
.add_parameter("dtype", ":py:class:`numpy.dtype` or anything that converts to a dtype", "The data type to get the distribution for; only real-valued types are supported")
.add_parameter("mean", "dtype", "[Default: 0.] The mean :math:`\\mu` of the lognormal distibution")
.add_parameter("sigma", "dtype", "[Default: 1.] The standard deviation :math:`\\sigma` of the lognormal distributiuon")
);

/* How to create a new PyBoostLogNormalObject */
static PyObject* PyBoostLogNormal_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostLogNormalObject* self = (PyBoostLogNormalObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return Py_BuildValue("N", self);
}

/* How to delete a PyBoostLogNormalObject */
static void PyBoostLogNormal_Delete (PyBoostLogNormalObject* o) {
  o->distro.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);
}

template <typename T>
boost::shared_ptr<void> make_lognormal(PyObject* mean, PyObject* sigma) {
  T cmean = 0.;
  if (mean) cmean = PyBlitzArrayCxx_AsCScalar<T>(mean);
  T csigma = 1.;
  if (sigma) csigma = PyBlitzArrayCxx_AsCScalar<T>(sigma);
  return boost::make_shared<bob::core::random::lognormal_distribution<T>>(cmean, csigma);
}

PyObject* PyBoostLogNormal_SimpleNew (int type_num, PyObject* mean, PyObject* sigma) {
BOB_TRY
  PyBoostLogNormalObject* retval = (PyBoostLogNormalObject*)PyBoostLogNormal_New(&PyBoostLogNormal_Type, 0, 0);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_FLOAT32:
      retval->distro = make_lognormal<float>(mean, sigma);
      break;
    case NPY_FLOAT64:
      retval->distro = make_lognormal<double>(mean, sigma);
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
static int PyBoostLogNormal_Init(PyBoostLogNormalObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = lognormal_doc.kwlist();

  PyObject* m = 0;
  PyObject* s = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|OO", kwlist, &PyBlitzArray_TypenumConverter, &self->type_num, &m, &s)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      self->distro = make_lognormal<float>(m, s);
      break;
    case NPY_FLOAT64:
      self->distro = make_lognormal<double>(m, s);
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

int PyBoostLogNormal_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostLogNormal_Type));
}

int PyBoostLogNormal_Converter(PyObject* o, PyBoostLogNormalObject** a) {
  if (!PyBoostLogNormal_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostLogNormalObject*>(o);
  return 1;
}


static auto mean_doc = bob::extension::VariableDoc(
  "mean",
  "dtype",
  "The mean value the distribution will produce."
);
template <typename T> PyObject* get_mean(PyBoostLogNormalObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::lognormal_distribution<T>>(self->distro)->m());
}

static PyObject* PyBoostLogNormal_GetMean(PyBoostLogNormalObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_mean<float>(self);
    case NPY_FLOAT64:
      return get_mean<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get m of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("mean", 0)
}


static auto sigma_doc = bob::extension::VariableDoc(
  "sigma",
  "dtype",
  "The standard deviation the distribution will have"
);
template <typename T> PyObject* get_sigma(PyBoostLogNormalObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::lognormal_distribution<T>>(self->distro)->s());
}

static PyObject* PyBoostLogNormal_GetSigma(PyBoostLogNormalObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_sigma<float>(self);
    case NPY_FLOAT64:
      return get_sigma<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get s of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("sigma", 0)
}


static auto dtype_doc = bob::extension::VariableDoc(
  "dtype",
  ":py:class:`numpy.dtype`",
  "The type of scalars produced by this normal distribution"
);
static PyObject* PyBoostLogNormal_GetDtype(PyBoostLogNormalObject* self) {
BOB_TRY
  return Py_BuildValue("N", PyArray_DescrFromType(self->type_num));
BOB_CATCH_MEMBER("dtype", 0)
}



static PyGetSetDef PyBoostLogNormal_getseters[] = {
    {
      dtype_doc.name(),
      (getter)PyBoostLogNormal_GetDtype,
      0,
      dtype_doc.doc(),
      0,
    },
    {
      mean_doc.name(),
      (getter)PyBoostLogNormal_GetMean,
      0,
      mean_doc.doc(),
      0,
    },
    {
      sigma_doc.name(),
      (getter)PyBoostLogNormal_GetSigma,
      0,
      sigma_doc.doc(),
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
template <typename T> PyObject* reset(PyBoostLogNormalObject* self) {
  boost::static_pointer_cast<bob::core::random::lognormal_distribution<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

static PyObject* PyBoostLogNormal_Reset(PyBoostLogNormalObject* self) {
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
.add_return("value", "dtype", "A random value that follows the lognormal distribution")
;
template <typename T> PyObject* call(PyBoostLogNormalObject* self, PyBoostMt19937Object* rng) {
  typedef bob::core::random::lognormal_distribution<T> distro_t;
  return PyBlitzArrayCxx_FromCScalar((*boost::static_pointer_cast<distro_t>(self->distro))(*rng->rng));
}

/**
 * Calling a PyBoostLogNormalObject to generate a random number
 */
static PyObject* PyBoostLogNormal_Call(PyBoostLogNormalObject* self, PyObject *args, PyObject* kwds) {
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


static PyMethodDef PyBoostLogNormal_methods[] = {
    {
      call_doc.name(),
      (PyCFunction)PyBoostLogNormal_Call,
      METH_VARARGS|METH_KEYWORDS,
      call_doc.doc(),
    },
    {
      reset_doc.name(),
      (PyCFunction)PyBoostLogNormal_Reset,
      METH_NOARGS,
      reset_doc.doc(),
    },
    {0}  /* Sentinel */
};

extern PyObject* scalar_to_bytes(PyObject* s);

/**
 * String representation and print out
 */
static PyObject* PyBoostLogNormal_Repr(PyBoostLogNormalObject* self) {
  PyObject* smean = scalar_to_bytes(PyBoostLogNormal_GetMean(self));
  if (!smean) return 0;
  auto smean_ = make_safe(smean);

  PyObject* ssigma = scalar_to_bytes(PyBoostLogNormal_GetSigma(self));
  if (!ssigma) return 0;
  auto ssigma_ = make_safe(ssigma);

  return
    PyString_FromFormat
      (
       "%s(dtype='%s', mean=%s, sigma=%s)",
       Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num),
       PyString_AS_STRING(smean), PyString_AS_STRING(ssigma)
      );
}

PyTypeObject PyBoostLogNormal_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BoostLogNormal(PyObject* module)
{
  // initialize the type struct
  PyBoostLogNormal_Type.tp_name = lognormal_doc.name();
  PyBoostLogNormal_Type.tp_basicsize = sizeof(PyBoostLogNormalObject);
  PyBoostLogNormal_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBoostLogNormal_Type.tp_doc = lognormal_doc.doc();
  PyBoostLogNormal_Type.tp_str = reinterpret_cast<reprfunc>(PyBoostLogNormal_Repr);
  PyBoostLogNormal_Type.tp_repr = reinterpret_cast<reprfunc>(PyBoostLogNormal_Repr);

  // set the functions
  PyBoostLogNormal_Type.tp_new = PyBoostLogNormal_New;
  PyBoostLogNormal_Type.tp_init = reinterpret_cast<initproc>(PyBoostLogNormal_Init);
  PyBoostLogNormal_Type.tp_dealloc = reinterpret_cast<destructor>(PyBoostLogNormal_Delete);
  PyBoostLogNormal_Type.tp_methods = PyBoostLogNormal_methods;
  PyBoostLogNormal_Type.tp_getset = PyBoostLogNormal_getseters;
  PyBoostLogNormal_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBoostLogNormal_Call);

  // check that everything is fine
  if (PyType_Ready(&PyBoostLogNormal_Type) < 0) return false;

  // add the type to the module
  return PyModule_AddObject(module, "lognormal", Py_BuildValue("O", &PyBoostLogNormal_Type)) >= 0;
}
