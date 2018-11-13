/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to boost::random
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

/**
 * Converts a scalar, that will be stolen, into a str/bytes
 * This function will be used in the bindings of all Boost classes
 */
PyObject* scalar_to_bytes(PyObject* s) {
  if (!s) return s;
  auto s_ = make_safe(s);
  PyObject* b = PyObject_Str(s);
  return b;
}


PyDoc_STRVAR(module_docstr,
"boost::random classes and methods"
);

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

int PyBobCoreRandom_APIVersion = BOB_CORE_API_VERSION;

extern bool init_BoostMt19937(PyObject* module);
extern bool init_BoostUniform(PyObject* module);
extern bool init_BoostNormal(PyObject* module);
extern bool init_BoostLogNormal(PyObject* module);
extern bool init_BoostGamma(PyObject* module);
extern bool init_BoostBinomial(PyObject* module);
extern bool init_BoostDiscrete(PyObject* module);

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
  auto m_ = make_xsafe(m);
  const char* ret = "O";
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!m) return 0;

  /* register the types to python */
  if (!init_BoostMt19937(m)) return 0;
  if (!init_BoostUniform(m)) return 0;
  if (!init_BoostNormal(m)) return 0;
  if (!init_BoostLogNormal(m)) return 0;
  if (!init_BoostGamma(m)) return 0;
  if (!init_BoostBinomial(m)) return 0;
  if (!init_BoostDiscrete(m)) return 0;

  static void* PyBobCoreRandom_API[PyBobCoreRandom_API_pointers];

  /* exhaustive list of C APIs */
  PyBobCoreRandom_API[PyBobCoreRandom_APIVersion_NUM] = (void *)&PyBobCoreRandom_APIVersion;

  /*****************************************
   * Bindings for bob.core.random.mt19937 *
   *****************************************/

  PyBobCoreRandom_API[PyBoostMt19937_Type_NUM] = (void *)&PyBoostMt19937_Type;
  PyBobCoreRandom_API[PyBoostMt19937_Check_NUM] = (void *)PyBoostMt19937_Check;
  PyBobCoreRandom_API[PyBoostMt19937_Converter_NUM] = (void *)PyBoostMt19937_Converter;
  PyBobCoreRandom_API[PyBoostMt19937_SimpleNew_NUM] = (void *)PyBoostMt19937_SimpleNew;
  PyBobCoreRandom_API[PyBoostMt19937_NewWithSeed_NUM] = (void *)PyBoostMt19937_NewWithSeed;

  /*****************************************
   * Bindings for bob.core.random.uniform *
   *****************************************/

  PyBobCoreRandom_API[PyBoostUniform_Type_NUM] = (void *)&PyBoostUniform_Type;
  PyBobCoreRandom_API[PyBoostUniform_Check_NUM] = (void *)PyBoostUniform_Check;
  PyBobCoreRandom_API[PyBoostUniform_Converter_NUM] = (void *)PyBoostUniform_Converter;
  PyBobCoreRandom_API[PyBoostUniform_SimpleNew_NUM] = (void *)PyBoostUniform_SimpleNew;

  /****************************************
   * Bindings for bob.core.random.normal *
   ****************************************/

  PyBobCoreRandom_API[PyBoostNormal_Type_NUM] = (void *)&PyBoostNormal_Type;
  PyBobCoreRandom_API[PyBoostNormal_Check_NUM] = (void *)PyBoostNormal_Check;
  PyBobCoreRandom_API[PyBoostNormal_Converter_NUM] = (void *)PyBoostNormal_Converter;
  PyBobCoreRandom_API[PyBoostNormal_SimpleNew_NUM] = (void *)PyBoostNormal_SimpleNew;

  /*******************************************
   * Bindings for bob.core.random.lognormal *
   *******************************************/

  PyBobCoreRandom_API[PyBoostLogNormal_Type_NUM] = (void *)&PyBoostLogNormal_Type;
  PyBobCoreRandom_API[PyBoostLogNormal_Check_NUM] = (void *)PyBoostLogNormal_Check;
  PyBobCoreRandom_API[PyBoostLogNormal_Converter_NUM] = (void *)PyBoostLogNormal_Converter;
  PyBobCoreRandom_API[PyBoostLogNormal_SimpleNew_NUM] = (void *)PyBoostLogNormal_SimpleNew;

  /***************************************
   * Bindings for bob.core.random.gamma *
   ***************************************/

  PyBobCoreRandom_API[PyBoostGamma_Type_NUM] = (void *)&PyBoostGamma_Type;
  PyBobCoreRandom_API[PyBoostGamma_Check_NUM] = (void *)PyBoostGamma_Check;
  PyBobCoreRandom_API[PyBoostGamma_Converter_NUM] = (void *)PyBoostGamma_Converter;
  PyBobCoreRandom_API[PyBoostGamma_SimpleNew_NUM] = (void *)PyBoostGamma_SimpleNew;

  /******************************************
   * Bindings for bob.core.random.binomial *
   ******************************************/

  PyBobCoreRandom_API[PyBoostBinomial_Type_NUM] = (void *)&PyBoostBinomial_Type;
  PyBobCoreRandom_API[PyBoostBinomial_Check_NUM] = (void *)PyBoostBinomial_Check;
  PyBobCoreRandom_API[PyBoostBinomial_Converter_NUM] = (void *)PyBoostBinomial_Converter;
  PyBobCoreRandom_API[PyBoostBinomial_SimpleNew_NUM] = (void *)PyBoostBinomial_SimpleNew;

  /******************************************
   * Bindings for bob.core.random.discrete *
   ******************************************/

  PyBobCoreRandom_API[PyBoostDiscrete_Type_NUM] = (void *)&PyBoostDiscrete_Type;
  PyBobCoreRandom_API[PyBoostDiscrete_Check_NUM] = (void *)PyBoostDiscrete_Check;
  PyBobCoreRandom_API[PyBoostDiscrete_Converter_NUM] = (void *)PyBoostDiscrete_Converter;
  PyBobCoreRandom_API[PyBoostDiscrete_SimpleNew_NUM] = (void *)PyBoostDiscrete_SimpleNew;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyBobCoreRandom_API,
      BOB_EXT_MODULE_PREFIX "." BOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyBobCoreRandom_API, 0);

#endif

  if (c_api_object) PyModule_AddObject(m, "_C_API", c_api_object);

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;

  return Py_BuildValue(ret, m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
