// include directly and indirectly dependent libraries
#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>

// include our own library
#include <bob.example.library/Function.h>

// use the documentation classes to document the function
static bob::extension::FunctionDoc reverse_doc = bob::extension::FunctionDoc(
  "reverse",
  "This is a simple example of bridging between blitz arrays (C++) and numpy.ndarrays (Python)",
  "Detailed documentation of the function goes here."
)
.add_prototype("array", "reversed")
.add_parameter("array", "array_like (1D, float)", "The array to reverse")
.add_return("reversed", "array_like (1D, float)", "A copy of the ``array`` with reversed order of entries")
;

// declare the function
// we use the default Python C-API here.
static PyObject* PyBobExampleLibrary_Reverse(PyObject*, PyObject* args, PyObject* kwargs) {

  BOB_TRY

  // declare the expected parameter names
  char** kwlist = reverse_doc.kwlist(0);

  // declare an object of the bridging type
  PyBlitzArrayObject* array;
  // ... and get the command line argument
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &array)) return 0;

  // since PyBlitzArray_Converter increased the reference count of array,
  // assure that the reference is decreased when the function exits (either way)
  auto array_ = make_safe(array);

  // check that the array has the expected properties
  if (array->type_num != NPY_FLOAT64 || array->ndim != 1){
    PyErr_Format(PyExc_TypeError, "%s : Only 1D arrays of type float are allowed", reverse_doc.name());
    return 0;
  }

  // extract the actual blitz array from the Python type
  blitz::Array<double, 1> bz = *PyBlitzArrayCxx_AsBlitz<double, 1>(array);

  // call the C++ function
  blitz::Array<double, 1> reversed = bob::example::library::reverse(bz);

  // convert the blitz array back to numpy and return it
  return PyBlitzArrayCxx_AsNumpy(reversed);

  // handle exceptions that occurred in this function
  BOB_CATCH_FUNCTION("reverse", 0)
}


//////////////////////////////////////////////////////////////////////////
/////// Python module declaration ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// module-wide methods
static PyMethodDef module_methods[] = {
  {
    reverse_doc.name(),
    (PyCFunction)PyBobExampleLibrary_Reverse,
    METH_VARARGS|METH_KEYWORDS,
    reverse_doc.doc()
  },
  {NULL}  // Sentinel
};

// module documentation
PyDoc_STRVAR(module_docstr, "Exemplary Python Bindings");

// module definition
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

// create the module
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

  if (PyModule_AddStringConstant(module, "__version__", BOB_EXT_MODULE_VERSION) < 0) return 0;

  /* imports bob.blitz C-API + dependencies */
  if (import_bob_blitz() < 0) return 0;

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
