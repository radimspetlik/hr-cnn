/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu  7 Nov 13:50:16 2013
 *
 * @brief Binds configuration information available from bob
 */

// include our own configuration
#include <bob.example.library/config.h>

#define BOB_IMPORT_VERSION
#include <bob.blitz/config.h>
#include <bob.blitz/cleanup.h>

// builds the dictionary of versions
static PyObject* build_version_dictionary() {

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  if (!dict_steal(retval, "Blitz++", blitz_version())) return 0;
  if (!dict_steal(retval, "Boost", boost_version())) return 0;
  if (!dict_steal(retval, "Compiler", compiler_version())) return 0;
  if (!dict_steal(retval, "Python", python_version())) return 0;
  if (!dict_steal(retval, "NumPy", numpy_version())) return 0;
  if (!dict_steal(retval, "bob.blitz", bob_blitz_version())) return 0;

  return Py_BuildValue("O", retval);
}

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr,
"Information about software used to compile the C++ Bob API"
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

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  /* register version numbers and constants */
  if (PyModule_AddIntConstant(module, "api", BOB_EXAMPLE_LIBRARY_API_VERSION) < 0) return 0;
  if (PyModule_AddStringConstant(module, "module", BOB_EXT_MODULE_VERSION) < 0) return 0;
  if (PyModule_AddObject(module, "externals", build_version_dictionary()) < 0) return 0;

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
