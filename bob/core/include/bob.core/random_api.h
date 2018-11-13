/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 30 Oct 07:40:47 2013
 *
 * @brief C/C++-API for the random module
 */

#ifndef BOB_CORE_RANDOM_API_H
#define BOB_CORE_RANDOM_API_H

#include <Python.h>
#include <bob.core/config.h>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>

/* Define Module Name and Prefix for other Modules
   Note: We cannot use BOB_EXT_* macros here, unfortunately */
#define BOB_CORE_RANDOM_PREFIX    "bob.core.random"
#define BOB_CORE_RANDOM_FULL_NAME "bob.core.random._library"

/*******************
 * C API functions *
 *******************/

/**************
 * Versioning *
 **************/

#define PyBobCoreRandom_APIVersion_NUM 0
#define PyBobCoreRandom_APIVersion_TYPE int

/*****************************************
 * Bindings for bob.core.random.mt19937 *
 *****************************************/

/* Type definition for PyBoostMt19937Object */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<boost::mt19937> rng;

} PyBoostMt19937Object;

#define PyBoostMt19937_Type_NUM 1
#define PyBoostMt19937_Type_TYPE PyTypeObject

#define PyBoostMt19937_Check_NUM 2
#define PyBoostMt19937_Check_RET int
#define PyBoostMt19937_Check_PROTO (PyObject* o)

#define PyBoostMt19937_Converter_NUM 3
#define PyBoostMt19937_Converter_RET int
#define PyBoostMt19937_Converter_PROTO (PyObject* o, PyBoostMt19937Object** a)

#define PyBoostMt19937_SimpleNew_NUM 4
#define PyBoostMt19937_SimpleNew_RET PyObject*
#define PyBoostMt19937_SimpleNew_PROTO ()

#define PyBoostMt19937_NewWithSeed_NUM 5
#define PyBoostMt19937_NewWithSeed_RET PyObject*
#define PyBoostMt19937_NewWithSeed_PROTO (Py_ssize_t seed)

/*****************************************
 * Bindings for bob.core.random.uniform *
 *****************************************/

/* Type definition for PyBoostUniformObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostUniformObject;

#define PyBoostUniform_Type_NUM 6
#define PyBoostUniform_Type_TYPE PyTypeObject

#define PyBoostUniform_Check_NUM 7
#define PyBoostUniform_Check_RET int
#define PyBoostUniform_Check_PROTO (PyObject* o)

#define PyBoostUniform_Converter_NUM 8
#define PyBoostUniform_Converter_RET int
#define PyBoostUniform_Converter_PROTO (PyObject* o, PyBoostUniformObject** a)

#define PyBoostUniform_SimpleNew_NUM 9
#define PyBoostUniform_SimpleNew_RET PyObject*
#define PyBoostUniform_SimpleNew_PROTO (int type_num, PyObject* min, PyObject* max)

/****************************************
 * Bindings for bob.core.random.normal *
 ****************************************/

/* Type definition for PyBoostNormalObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostNormalObject;

#define PyBoostNormal_Type_NUM 10
#define PyBoostNormal_Type_TYPE PyTypeObject

#define PyBoostNormal_Check_NUM 11
#define PyBoostNormal_Check_RET int
#define PyBoostNormal_Check_PROTO (PyObject* o)

#define PyBoostNormal_Converter_NUM 12
#define PyBoostNormal_Converter_RET int
#define PyBoostNormal_Converter_PROTO (PyObject* o, PyBoostNormalObject** a)

#define PyBoostNormal_SimpleNew_NUM 13
#define PyBoostNormal_SimpleNew_RET PyObject*
#define PyBoostNormal_SimpleNew_PROTO (int type_num, PyObject* mean, PyObject* sigma)

/*******************************************
 * Bindings for bob.core.random.lognormal *
 *******************************************/

/* Type definition for PyBoostLogNormalObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostLogNormalObject;

#define PyBoostLogNormal_Type_NUM 14
#define PyBoostLogNormal_Type_TYPE PyTypeObject

#define PyBoostLogNormal_Check_NUM 15
#define PyBoostLogNormal_Check_RET int
#define PyBoostLogNormal_Check_PROTO (PyObject* o)

#define PyBoostLogNormal_Converter_NUM 16
#define PyBoostLogNormal_Converter_RET int
#define PyBoostLogNormal_Converter_PROTO (PyObject* o, PyBoostLogNormalObject** a)

#define PyBoostLogNormal_SimpleNew_NUM 17
#define PyBoostLogNormal_SimpleNew_RET PyObject*
#define PyBoostLogNormal_SimpleNew_PROTO (int type_num, PyObject* mean, PyObject* sigma)

/***************************************
 * Bindings for bob.core.random.gamma *
 ***************************************/

/* Type definition for PyBoostGammaObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostGammaObject;

#define PyBoostGamma_Type_NUM 18
#define PyBoostGamma_Type_TYPE PyTypeObject

#define PyBoostGamma_Check_NUM 19
#define PyBoostGamma_Check_RET int
#define PyBoostGamma_Check_PROTO (PyObject* o)

#define PyBoostGamma_Converter_NUM 20
#define PyBoostGamma_Converter_RET int
#define PyBoostGamma_Converter_PROTO (PyObject* o, PyBoostGammaObject** a)

#define PyBoostGamma_SimpleNew_NUM 21
#define PyBoostGamma_SimpleNew_RET PyObject*
#define PyBoostGamma_SimpleNew_PROTO (int type_num, PyObject* alpha)

/******************************************
 * Bindings for bob.core.random.binomial *
 ******************************************/

/* Type definition for PyBoostBinomialObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostBinomialObject;

#define PyBoostBinomial_Type_NUM 22
#define PyBoostBinomial_Type_TYPE PyTypeObject

#define PyBoostBinomial_Check_NUM 23
#define PyBoostBinomial_Check_RET int
#define PyBoostBinomial_Check_PROTO (PyObject* o)

#define PyBoostBinomial_Converter_NUM 24
#define PyBoostBinomial_Converter_RET int
#define PyBoostBinomial_Converter_PROTO (PyObject* o, PyBoostBinomialObject** a)

#define PyBoostBinomial_SimpleNew_NUM 25
#define PyBoostBinomial_SimpleNew_RET PyObject*
#define PyBoostBinomial_SimpleNew_PROTO (int type_num, PyObject* alpha, PyObject* beta)

/******************************************
 * Bindings for bob.core.random.discrete *
 ******************************************/

/* Type definition for PyBoostDiscreteObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostDiscreteObject;

#define PyBoostDiscrete_Type_NUM 26
#define PyBoostDiscrete_Type_TYPE PyTypeObject

#define PyBoostDiscrete_Check_NUM 27
#define PyBoostDiscrete_Check_RET int
#define PyBoostDiscrete_Check_PROTO (PyObject* o)

#define PyBoostDiscrete_Converter_NUM 28
#define PyBoostDiscrete_Converter_RET int
#define PyBoostDiscrete_Converter_PROTO (PyObject* o, PyBoostDiscreteObject** a)

#define PyBoostDiscrete_SimpleNew_NUM 29
#define PyBoostDiscrete_SimpleNew_RET PyObject*
#define PyBoostDiscrete_SimpleNew_PROTO (int type_num, PyObject* probabilities)

/* Total number of C API pointers */
#define PyBobCoreRandom_API_pointers 30

#ifdef BOB_CORE_RANDOM_MODULE

  /* This section is used when compiling `bob.core.random' itself */

  /**************
   * Versioning *
   **************/

  extern int PyBobCoreRandom_APIVersion;

  /*****************************************
   * Bindings for bob.core.random.mt19937 *
   *****************************************/

  extern PyBoostMt19937_Type_TYPE PyBoostMt19937_Type;

  PyBoostMt19937_Check_RET PyBoostMt19937_Check PyBoostMt19937_Check_PROTO;

  PyBoostMt19937_Converter_RET PyBoostMt19937_Converter PyBoostMt19937_Converter_PROTO;

  PyBoostMt19937_SimpleNew_RET PyBoostMt19937_SimpleNew PyBoostMt19937_SimpleNew_PROTO;

  PyBoostMt19937_NewWithSeed_RET PyBoostMt19937_NewWithSeed PyBoostMt19937_NewWithSeed_PROTO;

  /*****************************************
   * Bindings for bob.core.random.uniform *
   *****************************************/

  extern PyBoostUniform_Type_TYPE PyBoostUniform_Type;

  PyBoostUniform_Check_RET PyBoostUniform_Check PyBoostUniform_Check_PROTO;

  PyBoostUniform_Converter_RET PyBoostUniform_Converter PyBoostUniform_Converter_PROTO;

  PyBoostUniform_SimpleNew_RET PyBoostUniform_SimpleNew PyBoostUniform_SimpleNew_PROTO;

  /****************************************
   * Bindings for bob.core.random.normal *
   ****************************************/

  extern PyBoostNormal_Type_TYPE PyBoostNormal_Type;

  PyBoostNormal_Check_RET PyBoostNormal_Check PyBoostNormal_Check_PROTO;

  PyBoostNormal_Converter_RET PyBoostNormal_Converter PyBoostNormal_Converter_PROTO;

  PyBoostNormal_SimpleNew_RET PyBoostNormal_SimpleNew PyBoostNormal_SimpleNew_PROTO;

  /*******************************************
   * Bindings for bob.core.random.lognormal *
   *******************************************/

  extern PyBoostLogNormal_Type_TYPE PyBoostLogNormal_Type;

  PyBoostLogNormal_Check_RET PyBoostLogNormal_Check PyBoostLogNormal_Check_PROTO;

  PyBoostLogNormal_Converter_RET PyBoostLogNormal_Converter PyBoostLogNormal_Converter_PROTO;

  PyBoostLogNormal_SimpleNew_RET PyBoostLogNormal_SimpleNew PyBoostLogNormal_SimpleNew_PROTO;

  /***************************************
   * Bindings for bob.core.random.gamma *
   ***************************************/

  extern PyBoostGamma_Type_TYPE PyBoostGamma_Type;

  PyBoostGamma_Check_RET PyBoostGamma_Check PyBoostGamma_Check_PROTO;

  PyBoostGamma_Converter_RET PyBoostGamma_Converter PyBoostGamma_Converter_PROTO;

  PyBoostGamma_SimpleNew_RET PyBoostGamma_SimpleNew PyBoostGamma_SimpleNew_PROTO;

  /******************************************
   * Bindings for bob.core.random.binomial *
   ******************************************/

  extern PyBoostBinomial_Type_TYPE PyBoostBinomial_Type;

  PyBoostBinomial_Check_RET PyBoostBinomial_Check PyBoostBinomial_Check_PROTO;

  PyBoostBinomial_Converter_RET PyBoostBinomial_Converter PyBoostBinomial_Converter_PROTO;

  PyBoostBinomial_SimpleNew_RET PyBoostBinomial_SimpleNew PyBoostBinomial_SimpleNew_PROTO;

  /******************************************
   * Bindings for bob.core.random.discrete *
   ******************************************/

  extern PyBoostDiscrete_Type_TYPE PyBoostDiscrete_Type;

  PyBoostDiscrete_Check_RET PyBoostDiscrete_Check PyBoostDiscrete_Check_PROTO;

  PyBoostDiscrete_Converter_RET PyBoostDiscrete_Converter PyBoostDiscrete_Converter_PROTO;

  PyBoostDiscrete_SimpleNew_RET PyBoostDiscrete_SimpleNew PyBoostDiscrete_SimpleNew_PROTO;

#else

  /* This section is used in modules that use `bob.core.random's' C-API */

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyBobCoreRandom_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyBobCoreRandom_API;
#    else
  static void **PyBobCoreRandom_API=NULL;
#    endif
#  endif

  /**************
   * Versioning *
   **************/

# define PyBobCoreRandom_APIVersion (*(PyBobCoreRandom_APIVersion_TYPE *)PyBobCoreRandom_API[PyBobCoreRandom_APIVersion_NUM])

  /*****************************************
   * Bindings for bob.core.random.mt19937 *
   *****************************************/

# define PyBoostMt19937_Type (*(PyBoostMt19937_Type_TYPE *)PyBobCoreRandom_API[PyBoostMt19937_Type_NUM])

# define PyBoostMt19937_Check (*(PyBoostMt19937_Check_RET (*)PyBoostMt19937_Check_PROTO) PyBobCoreRandom_API[PyBoostMt19937_Check_NUM])

# define PyBoostMt19937_Converter (*(PyBoostMt19937_Converter_RET (*)PyBoostMt19937_Converter_PROTO) PyBobCoreRandom_API[PyBoostMt19937_Converter_NUM])

# define PyBoostMt19937_SimpleNew (*(PyBoostMt19937_SimpleNew_RET (*)PyBoostMt19937_SimpleNew_PROTO) PyBobCoreRandom_API[PyBoostMt19937_SimpleNew_NUM])

# define PyBoostMt19937_NewWithSeed (*(PyBoostMt19937_NewWithSeed_RET (*)PyBoostMt19937_NewWithSeed_PROTO) PyBobCoreRandom_API[PyBoostMt19937_NewWithSeed_NUM])

  /*****************************************
   * Bindings for bob.core.random.uniform *
   *****************************************/

# define PyBoostUniform_Type (*(PyBoostUniform_Type_TYPE *)PyBobCoreRandom_API[PyBoostUniform_Type_NUM])

# define PyBoostUniform_Check (*(PyBoostUniform_Check_RET (*)PyBoostUniform_Check_PROTO) PyBobCoreRandom_API[PyBoostUniform_Check_NUM])

# define PyBoostUniform_Converter (*(PyBoostUniform_Converter_RET (*)PyBoostUniform_Converter_PROTO) PyBobCoreRandom_API[PyBoostUniform_Converter_NUM])

# define PyBoostUniform_SimpleNew (*(PyBoostUniform_SimpleNew_RET (*)PyBoostUniform_SimpleNew_PROTO) PyBobCoreRandom_API[PyBoostUniform_SimpleNew_NUM])

  /****************************************
   * Bindings for bob.core.random.normal *
   ****************************************/

# define PyBoostNormal_Type (*(PyBoostNormal_Type_TYPE *)PyBobCoreRandom_API[PyBoostNormal_Type_NUM])

# define PyBoostNormal_Check (*(PyBoostNormal_Check_RET (*)PyBoostNormal_Check_PROTO) PyBobCoreRandom_API[PyBoostNormal_Check_NUM])

# define PyBoostNormal_Converter (*(PyBoostNormal_Converter_RET (*)PyBoostNormal_Converter_PROTO) PyBobCoreRandom_API[PyBoostNormal_Converter_NUM])

# define PyBoostNormal_SimpleNew (*(PyBoostNormal_SimpleNew_RET (*)PyBoostNormal_SimpleNew_PROTO) PyBobCoreRandom_API[PyBoostNormal_SimpleNew_NUM])

  /*******************************************
   * Bindings for bob.core.random.lognormal *
   *******************************************/

# define PyBoostLogNormal_Type (*(PyBoostLogNormal_Type_TYPE *)PyBobCoreRandom_API[PyBoostLogNormal_Type_NUM])

# define PyBoostLogNormal_Check (*(PyBoostLogNormal_Check_RET (*)PyBoostLogNormal_Check_PROTO) PyBobCoreRandom_API[PyBoostLogNormal_Check_NUM])

# define PyBoostLogNormal_Converter (*(PyBoostLogNormal_Converter_RET (*)PyBoostLogNormal_Converter_PROTO) PyBobCoreRandom_API[PyBoostLogNormal_Converter_NUM])

# define PyBoostLogNormal_SimpleNew (*(PyBoostLogNormal_SimpleNew_RET (*)PyBoostLogNormal_SimpleNew_PROTO) PyBobCoreRandom_API[PyBoostLogNormal_SimpleNew_NUM])

  /***************************************
   * Bindings for bob.core.random.gamma *
   ***************************************/

# define PyBoostGamma_Type (*(PyBoostGamma_Type_TYPE *)PyBobCoreRandom_API[PyBoostGamma_Type_NUM])

# define PyBoostGamma_Check (*(PyBoostGamma_Check_RET (*)PyBoostGamma_Check_PROTO) PyBobCoreRandom_API[PyBoostGamma_Check_NUM])

# define PyBoostGamma_Converter (*(PyBoostGamma_Converter_RET (*)PyBoostGamma_Converter_PROTO) PyBobCoreRandom_API[PyBoostGamma_Converter_NUM])

# define PyBoostGamma_SimpleNew (*(PyBoostGamma_SimpleNew_RET (*)PyBoostGamma_SimpleNew_PROTO) PyBobCoreRandom_API[PyBoostGamma_SimpleNew_NUM])

  /******************************************
   * Bindings for bob.core.random.binomial *
   ******************************************/

# define PyBoostBinomial_Type (*(PyBoostBinomial_Type_TYPE *)PyBobCoreRandom_API[PyBoostBinomial_Type_NUM])

# define PyBoostBinomial_Check (*(PyBoostBinomial_Check_RET (*)PyBoostBinomial_Check_PROTO) PyBobCoreRandom_API[PyBoostBinomial_Check_NUM])

# define PyBoostBinomial_Converter (*(PyBoostBinomial_Converter_RET (*)PyBoostBinomial_Converter_PROTO) PyBobCoreRandom_API[PyBoostBinomial_Converter_NUM])

# define PyBoostBinomial_SimpleNew (*(PyBoostBinomial_SimpleNew_RET (*)PyBoostBinomial_SimpleNew_PROTO) PyBobCoreRandom_API[PyBoostBinomial_SimpleNew_NUM])

  /******************************************
   * Bindings for bob.core.random.discrete *
   ******************************************/

# define PyBoostDiscrete_Type (*(PyBoostDiscrete_Type_TYPE *)PyBobCoreRandom_API[PyBoostDiscrete_Type_NUM])

# define PyBoostDiscrete_Check (*(PyBoostDiscrete_Check_RET (*)PyBoostDiscrete_Check_PROTO) PyBobCoreRandom_API[PyBoostDiscrete_Check_NUM])

# define PyBoostDiscrete_Converter (*(PyBoostDiscrete_Converter_RET (*)PyBoostDiscrete_Converter_PROTO) PyBobCoreRandom_API[PyBoostDiscrete_Converter_NUM])

# define PyBoostDiscrete_SimpleNew (*(PyBoostDiscrete_SimpleNew_RET (*)PyBoostDiscrete_SimpleNew_PROTO) PyBobCoreRandom_API[PyBoostDiscrete_SimpleNew_NUM])

# if !defined(NO_IMPORT_ARRAY)

  /**
   * Returns -1 on error, 0 on success.
   */
  static int import_bob_core_random(void) {

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOB_CORE_RANDOM_FULL_NAME);

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

#   if PY_VERSION_HEX >= 0x02070000
    if (PyCapsule_CheckExact(c_api_object)) {
      PyBobCoreRandom_API = (void **)PyCapsule_GetPointer(c_api_object,
          PyCapsule_GetName(c_api_object));
    }
#   else
    if (PyCObject_Check(c_api_object)) {
      PyBobCoreRandom_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }
#   endif

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    if (!PyBobCoreRandom_API) {
      PyErr_SetString(PyExc_ImportError, "cannot find C/C++ API "
#   if PY_VERSION_HEX >= 0x02070000
          "capsule"
#   else
          "cobject"
#   endif
          " at `" BOB_CORE_RANDOM_FULL_NAME "._C_API'");
      return -1;
    }

    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyBobCoreRandom_API[PyBobCoreRandom_APIVersion_NUM];

    if (BOB_CORE_API_VERSION != imported_version) {
      PyErr_Format(PyExc_ImportError, BOB_CORE_RANDOM_FULL_NAME " import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOB_CORE_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

# endif //!defined(NO_IMPORT_ARRAY)

#endif /* BOB_CORE_RANDOM_MODULE */

#endif /* BOB_CORE_RANDOM_API_H */
