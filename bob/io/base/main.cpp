/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to bob::io
 */

#define BOB_IO_BASE_MODULE
#include <bob.io.base/api.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>

extern bool init_File(PyObject* module);
extern bool init_HDF5File(PyObject* module);

/**
 * Creates an str object, from a C or C++ string. Returns a **new
 * reference**.
 */
static PyObject* make_object(const char* s) {
  return Py_BuildValue("s", s);
}

static auto s_extensions = bob::extension::FunctionDoc(
  "extensions",
  "Returns a dictionary containing all extensions and descriptions currently stored on the global codec registry",
  "The extensions are returned as a dictionary from the filename extension to a description of the data format."
)
.add_prototype("", "extensions")
.add_return("extensions", "{str : str}", "A dictionary of supported extensions");
static PyObject* PyBobIo_Extensions(PyObject*) {
BOB_TRY
  typedef std::map<std::string, std::string> map_type;
  const map_type& table = bob::io::base::CodecRegistry::getExtensions();

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (auto it=table.begin(); it!=table.end(); ++it) {
    PyObject* pyvalue = make_object(it->second.c_str());
    if (!pyvalue) return 0;
    auto p_ = make_safe(pyvalue);
    if (PyDict_SetItemString(retval, it->first.c_str(), pyvalue) != 0) {
      return 0;
    }
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_FUNCTION("extensions", 0);
}

static PyMethodDef module_methods[] = {
    {
      s_extensions.name(),
      (PyCFunction)PyBobIo_Extensions,
      METH_NOARGS,
      s_extensions.doc(),
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Core bob::io classes and methods");

int PyBobIo_APIVersion = BOB_IO_BASE_API_VERSION;

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
  PyObject* m = PyModule_Create(&module_definition);
  auto m_ = make_xsafe(m);
  const char* ret = "O";
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!m) return 0;

  /* register some constants */
  if (!init_File(m)) return 0;
  if (!init_HDF5File(m)) return 0;

  static void* PyBobIo_API[PyBobIo_API_pointers];

  /* exhaustive list of C APIs */

  /**************
   * Versioning *
   **************/

  PyBobIo_API[PyBobIo_APIVersion_NUM] = (void *)&PyBobIo_APIVersion;

  /**********************************
   * Bindings for bob.io.base.File *
   **********************************/

  PyBobIo_API[PyBobIoFile_Type_NUM] = (void *)&PyBobIoFile_Type;

  PyBobIo_API[PyBobIoFileIterator_Type_NUM] = (void *)&PyBobIoFileIterator_Type;

  /************************
   * I/O generic bindings *
   ************************/

  PyBobIo_API[PyBobIo_AsTypenum_NUM] = (void *)PyBobIo_AsTypenum;

  PyBobIo_API[PyBobIo_TypeInfoAsTuple_NUM] = (void *)PyBobIo_TypeInfoAsTuple;

  PyBobIo_API[PyBobIo_FilenameConverter_NUM] = (void *)PyBobIo_FilenameConverter;

  /*****************
   * HDF5 bindings *
   *****************/

  PyBobIo_API[PyBobIoHDF5File_Type_NUM] = (void *)&PyBobIoHDF5File_Type;

  PyBobIo_API[PyBobIoHDF5File_Check_NUM] = (void *)&PyBobIoHDF5File_Check;

  PyBobIo_API[PyBobIoHDF5File_Converter_NUM] = (void *)&PyBobIoHDF5File_Converter;

/*****************************************
 * Code Registration and De-registration *
 *****************************************/

  PyBobIo_API[PyBobIoCodec_Register_NUM] = (void *)&PyBobIoCodec_Register;

  PyBobIo_API[PyBobIoCodec_Deregister_NUM] = (void *)&PyBobIoCodec_Deregister;

  PyBobIo_API[PyBobIoCodec_IsRegistered_NUM] = (void *)&PyBobIoCodec_IsRegistered;

  PyBobIo_API[PyBobIoCodec_GetDescription_NUM] = (void *)&PyBobIoCodec_GetDescription;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyBobIo_API,
      BOB_EXT_MODULE_PREFIX "." BOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyBobIo_API, 0);

#endif

  if (!c_api_object) return 0;

  if (PyModule_AddObject(m, "_C_API", c_api_object) < 0) return 0;

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
