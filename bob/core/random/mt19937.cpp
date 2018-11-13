/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:01:04 2013
 *
 * @brief Bindings for the MT19937 random number generator
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>

static auto mt19937_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".mt19937",
  "A Mersenne-Twister Random Number Generator (RNG)",
  "A Random Number Generator (RNG) based on the work "
  "*Mersenne Twister: A 623-dimensionally equidistributed uniform pseudo-random number generator, Makoto Matsumoto and Takuji Nishimura, ACM Transactions on Modeling and Computer Simulation: Special Issue on Uniform Random Number Generation, Vol. 8, No. 1, January 1998, pp. 3-30*\n\n"
  "Objects of this class support comparison operators such as ``==`` or ``!=`` and setting the seed with the method :py:meth:`seed`. "
  "Two random number generators are equal if they are at the same state -- i.e. they have been initialized with the same seed and have been called the same number of times for number generation."
)
.add_constructor(bob::extension::FunctionDoc(
  "mt19937",
  "Constructs and initializes a random number generator",
  "If no ``seed`` is specified, the default seed (http://www.boost.org/doc/libs/1_59_0/doc/html/boost/random/mersenne_twister_engine.html) is used."
)
.add_prototype("[seed]", "")
.add_parameter("seed", "int", "[optional] An integral value determining the initial seed")
);

/* How to create a new PyBoostMt19937Object */
static PyObject* PyBoostMt19937_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostMt19937Object* self = (PyBoostMt19937Object*)type->tp_alloc(type, 0);

  self->rng.reset();

  return Py_BuildValue("N", self);
}

PyObject* PyBoostMt19937_SimpleNew () {
BOB_TRY
  PyBoostMt19937Object* retval = (PyBoostMt19937Object*)PyBoostMt19937_New(&PyBoostMt19937_Type, 0, 0);

  if (!retval) return 0;

  retval->rng.reset(new boost::mt19937);

  return Py_BuildValue("N", retval);
BOB_CATCH_FUNCTION("SimpleNew", 0)
}

PyObject* PyBoostMt19937_NewWithSeed (Py_ssize_t seed) {
BOB_TRY
  PyBoostMt19937Object* retval = (PyBoostMt19937Object*)PyBoostMt19937_New(&PyBoostMt19937_Type, 0, 0);

  if (!retval) return 0;

  retval->rng.reset(new boost::mt19937(seed));

  return Py_BuildValue("N", retval);
BOB_CATCH_FUNCTION("NewWithSeed", 0)

}

int PyBoostMt19937_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostMt19937_Type));
}

int PyBoostMt19937_Converter(PyObject* o, PyBoostMt19937Object** a) {
  if (!PyBoostMt19937_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostMt19937Object*>(o);
  return 1;
}

static void PyBoostMt19937_Delete (PyBoostMt19937Object* o) {
  o->rng.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);
}

/* The __init__(self) method */
static int PyBoostMt19937_Init(PyBoostMt19937Object* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = mt19937_doc.kwlist();

  PyObject* seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &seed)) return -1;

  /* Checks seed and, if it is set, try to convert it into a know format in
   * which the RNG can be initialized with */
  if (seed) {
    Py_ssize_t s_seed = PyNumber_AsSsize_t(seed, PyExc_ValueError);
    if (PyErr_Occurred()) return -1;
    self->rng.reset(new boost::mt19937(s_seed));
  }
  else {
    self->rng.reset(new boost::mt19937);
  }

  return 0; ///< SUCCESS
BOB_CATCH_MEMBER("constructor", -1)
}


static auto seed_doc = bob::extension::FunctionDoc(
  "seed",
  "Sets the seed for this random number generator",
  0,
  true
)
.add_prototype("seed")
.add_parameter("seed", "int", "A new seed value for this RNG")
;
static PyObject* PyBoostMt19937_seed(PyBoostMt19937Object* self,
    PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = seed_doc.kwlist();

  int seed;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &seed)) return 0;

  self->rng->seed(seed);

  Py_RETURN_NONE;
BOB_CATCH_MEMBER("seed", 0)
}

static PyMethodDef PyBoostMt19937_methods[] = {
    {
      seed_doc.name(),
      (PyCFunction)PyBoostMt19937_seed,
      METH_VARARGS|METH_KEYWORDS,
      seed_doc.doc(),
    },
    {0}  /* Sentinel */
};

static PyObject* PyBoostMt19937_RichCompare(PyBoostMt19937Object* self,
    PyObject* other, int op) {
BOB_TRY
  if (!PyBoostMt19937_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", seed_doc.name(), other->ob_type->tp_name);
    return 0;
  }

  PyBoostMt19937Object* other_ = reinterpret_cast<PyBoostMt19937Object*>(other);

  switch (op) {
    case Py_EQ:
      if (*(self->rng) == *(other_->rng)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (*(self->rng) != *(other_->rng)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
BOB_CATCH_MEMBER("RichCompare", 0)
}

static PyObject* PyBoostMt19937_Repr(PyBoostMt19937Object* self) {
BOB_TRY
  return PyUnicode_FromFormat("%s()", Py_TYPE(self)->tp_name);
BOB_CATCH_MEMBER("repr", 0)
}


PyTypeObject PyBoostMt19937_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BoostMt19937(PyObject* module)
{
  // initialize the type struct
  PyBoostMt19937_Type.tp_name = mt19937_doc.name();
  PyBoostMt19937_Type.tp_basicsize = sizeof(PyBoostMt19937Object);
  PyBoostMt19937_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBoostMt19937_Type.tp_doc = mt19937_doc.doc();
  PyBoostMt19937_Type.tp_str = reinterpret_cast<reprfunc>(PyBoostMt19937_Repr);
  PyBoostMt19937_Type.tp_repr = reinterpret_cast<reprfunc>(PyBoostMt19937_Repr);

  // set the functions
  PyBoostMt19937_Type.tp_new = PyBoostMt19937_New;
  PyBoostMt19937_Type.tp_init = reinterpret_cast<initproc>(PyBoostMt19937_Init);
  PyBoostMt19937_Type.tp_dealloc = reinterpret_cast<destructor>(PyBoostMt19937_Delete);
  PyBoostMt19937_Type.tp_methods = PyBoostMt19937_methods;
  PyBoostMt19937_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBoostMt19937_RichCompare);
  // check that everything is fine
  if (PyType_Ready(&PyBoostMt19937_Type) < 0) return false;

  // add the type to the module
  return PyModule_AddObject(module, "mt19937", Py_BuildValue("O", &PyBoostMt19937_Type)) >= 0;
}
