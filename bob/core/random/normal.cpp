/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Normal distributions (with integers or floating point numbers)
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>
#include <boost/make_shared.hpp>

#include <bob.core/random.h>

static auto normal_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".normal",
  "Models a random normal distribution",
  "This distribution produces random numbers :math:`x` distributed with the probability density function\n\n"
  ".. math::\n\n   p(x) = \\frac{1}{\\sqrt{2\\pi\\sigma}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}\n\n"
  "where the ``mean`` (:math:`\\mu`) and ``sigma`` (:math:`\\sigma`, the standard deviation) are the parameters of this distribution class."
)
.add_constructor(bob::extension::FunctionDoc(
  "normal",
  "Constructs a new normal distribution object"
)
.add_prototype("dtype, [mean], [sigma]", "")
.add_parameter("dtype", ":py:class:`numpy.dtype` or anything that converts to a dtype", "The data type to get the distribution for; only real-valued types are supported")
.add_parameter("mean", "dtype", "[Default: 0.] The mean :math:`\\mu` of the normal distibution")
.add_parameter("sigma", "dtype", "[Default: 1.] The standard deviation :math:`\\sigma` of the normal distributiuon")
);

/* How to create a new PyBoostNormalObject */
static PyObject* PyBoostNormal_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostNormalObject* self = (PyBoostNormalObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return Py_BuildValue("N", self);
}

/* How to delete a PyBoostNormalObject */
static void PyBoostNormal_Delete (PyBoostNormalObject* o) {
  o->distro.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);
}

template <typename T>
boost::shared_ptr<void> make_normal(PyObject* mean, PyObject* sigma) {
  T cmean = 0.;
  if (mean) cmean = PyBlitzArrayCxx_AsCScalar<T>(mean);
  T csigma = 1.;
  if (sigma) csigma = PyBlitzArrayCxx_AsCScalar<T>(sigma);
  return boost::make_shared<bob::core::random::normal_distribution<T>>(cmean, csigma);
}

PyObject* PyBoostNormal_SimpleNew (int type_num, PyObject* mean, PyObject* sigma) {
BOB_TRY
  PyBoostNormalObject* retval = (PyBoostNormalObject*)PyBoostNormal_New(&PyBoostNormal_Type, 0, 0);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_FLOAT32:
      retval->distro = make_normal<float>(mean, sigma);
      break;
    case NPY_FLOAT64:
      retval->distro = make_normal<double>(mean, sigma);
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
static int PyBoostNormal_Init(PyBoostNormalObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = normal_doc.kwlist();

  PyObject* mean = 0;
  PyObject* sigma = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|OO", kwlist, &PyBlitzArray_TypenumConverter, &self->type_num, &mean, &sigma)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      self->distro = make_normal<float>(mean, sigma);
      break;
    case NPY_FLOAT64:
      self->distro = make_normal<double>(mean, sigma);
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

int PyBoostNormal_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostNormal_Type));
}

int PyBoostNormal_Converter(PyObject* o, PyBoostNormalObject** a) {
  if (!PyBoostNormal_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostNormalObject*>(o);
  return 1;
}


static auto mean_doc = bob::extension::VariableDoc(
  "mean",
  "dtype",
  "The mean value the distribution will produce."
);
template <typename T> PyObject* get_mean(PyBoostNormalObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::normal_distribution<T>>(self->distro)->mean());
}

static PyObject* PyBoostNormal_GetMean(PyBoostNormalObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_mean<float>(self);
    case NPY_FLOAT64:
      return get_mean<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get mean of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("mean", 0)
}


static auto sigma_doc = bob::extension::VariableDoc(
  "sigma",
  "dtype",
  "The standard deviation the distribution will have"
);
template <typename T> PyObject* get_sigma(PyBoostNormalObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::normal_distribution<T>>(self->distro)->sigma());
}

static PyObject* PyBoostNormal_GetSigma(PyBoostNormalObject* self) {
BOB_TRY
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_sigma<float>(self);
    case NPY_FLOAT64:
      return get_sigma<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get sigma of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
BOB_CATCH_MEMBER("sigma", 0)
}

static auto dtype_doc = bob::extension::VariableDoc(
  "dtype",
  ":py:class:`numpy.dtype`",
  "The type of scalars produced by this normal distribution"
);
static PyObject* PyBoostNormal_GetDtype(PyBoostNormalObject* self) {
BOB_TRY
  return Py_BuildValue("N", PyArray_DescrFromType(self->type_num));
BOB_CATCH_MEMBER("dtype", 0)
}


static PyGetSetDef PyBoostNormal_getseters[] = {
    {
      dtype_doc.name(),
      (getter)PyBoostNormal_GetDtype,
      0,
      dtype_doc.doc(),
      0,
    },
    {
      mean_doc.name(),
      (getter)PyBoostNormal_GetMean,
      0,
      mean_doc.doc(),
      0,
    },
    {
      sigma_doc.name(),
      (getter)PyBoostNormal_GetSigma,
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
template <typename T> PyObject* reset(PyBoostNormalObject* self) {
  boost::static_pointer_cast<bob::core::random::normal_distribution<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

/**
 * Resets the distribution - this is a noop for normal distributions, here
 * only for compatibility reasons
 */
static PyObject* PyBoostNormal_Reset(PyBoostNormalObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return reset<float>(self);
    case NPY_FLOAT64:
      return reset<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot reset %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
}


static auto call_doc = bob::extension::FunctionDoc(
  "draw",
  "Draws one random number from this distribution using the given ``rng``",
  ".. note:: The :py:meth:`__call__` function is a synonym for this ``draw``.",
  true
)
.add_prototype("rng", "value")
.add_parameter("rng", ":py:class:`mt19937`", "The random number generator to use")
.add_return("value", "dtype", "A random value that follows the normal distribution")
;
template <typename T> PyObject* call(PyBoostNormalObject* self, PyBoostMt19937Object* rng) {
  typedef bob::core::random::normal_distribution<T> distro_t;
  return PyBlitzArrayCxx_FromCScalar((*boost::static_pointer_cast<distro_t>(self->distro))(*rng->rng));
}

/**
 * Calling a PyBoostNormalObject to generate a random number
 */
static PyObject* PyBoostNormal_Call(PyBoostNormalObject* self, PyObject *args, PyObject* kwds) {
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

static PyMethodDef PyBoostNormal_methods[] = {
    {
      call_doc.name(),
      (PyCFunction)PyBoostNormal_Call,
      METH_VARARGS|METH_KEYWORDS,
      call_doc.doc(),
    },
    {
      reset_doc.name(),
      (PyCFunction)PyBoostNormal_Reset,
      METH_NOARGS,
      reset_doc.doc(),
    },
    {0}  /* Sentinel */
};


extern PyObject* scalar_to_bytes(PyObject* s);

/**
 * String representation and print out
 */
static PyObject* PyBoostNormal_Repr(PyBoostNormalObject* self) {
BOB_TRY
  PyObject* smean = scalar_to_bytes(PyBoostNormal_GetMean(self));
  if (!smean) return 0;
  auto smean_ = make_safe(smean);
  PyObject* ssigma = scalar_to_bytes(PyBoostNormal_GetSigma(self));
  if (!ssigma) return 0;
  auto ssigma_ = make_safe(ssigma);

  return
    PyString_FromFormat
      (
       "%s(dtype='%s', mean=%s, sigma=%s)",
       Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num),
       PyString_AS_STRING(smean), PyString_AS_STRING(ssigma)
      );
BOB_CATCH_MEMBER("repr", 0)
}

PyTypeObject PyBoostNormal_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BoostNormal(PyObject* module)
{
  // initialize the type struct
  PyBoostNormal_Type.tp_name = normal_doc.name();
  PyBoostNormal_Type.tp_basicsize = sizeof(PyBoostNormalObject);
  PyBoostNormal_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBoostNormal_Type.tp_doc = normal_doc.doc();
  PyBoostNormal_Type.tp_str = reinterpret_cast<reprfunc>(PyBoostNormal_Repr);
  PyBoostNormal_Type.tp_repr = reinterpret_cast<reprfunc>(PyBoostNormal_Repr);

  // set the functions
  PyBoostNormal_Type.tp_new = PyBoostNormal_New;
  PyBoostNormal_Type.tp_init = reinterpret_cast<initproc>(PyBoostNormal_Init);
  PyBoostNormal_Type.tp_dealloc = reinterpret_cast<destructor>(PyBoostNormal_Delete);
  PyBoostNormal_Type.tp_methods = PyBoostNormal_methods;
  PyBoostNormal_Type.tp_getset = PyBoostNormal_getseters;
  PyBoostNormal_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBoostNormal_Call);

  // check that everything is fine
  if (PyType_Ready(&PyBoostNormal_Type) < 0) return false;

  // add the type to the module
  return PyModule_AddObject(module, "normal", Py_BuildValue("O", &PyBoostNormal_Type)) >= 0;
}
