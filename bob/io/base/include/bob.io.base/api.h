/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  5 Nov 12:22:48 2013
 *
 * @brief Python API for bob::io::base
 */

#ifndef BOB_IO_BASE_H
#define BOB_IO_BASE_H

/* Define Module Name and Prefix for other Modules
   Note: We cannot use BOB_EXT_* macros here, unfortunately */
#define BOB_IO_BASE_PREFIX    "bob.io.base"
#define BOB_IO_BASE_FULL_NAME "bob.io.base._library"

#include <Python.h>

#include <bob.io.base/config.h>
#include <bob.io.base/File.h>
#include <bob.io.base/CodecRegistry.h>
#include <bob.io.base/HDF5File.h>

#include <boost/shared_ptr.hpp>

/*******************
 * C API functions *
 *******************/

/* Enum defining entries in the function table */
enum _PyBobIo_ENUM{
  PyBobIo_APIVersion_NUM = 0,
  // Bindings for bob.io.base.File
  PyBobIoFile_Type_NUM,
  PyBobIoFileIterator_Type_NUM,
  // I/O generic bindings
  PyBobIo_AsTypenum_NUM,
  PyBobIo_TypeInfoAsTuple_NUM,
  PyBobIo_FilenameConverter_NUM,
  // HDF5 bindings
  PyBobIoHDF5File_Type_NUM,
  PyBobIoHDF5File_Check_NUM,
  PyBobIoHDF5File_Converter_NUM,
  // Codec registration and de-registration
  PyBobIoCodec_Register_NUM,
  PyBobIoCodec_Deregister_NUM,
  PyBobIoCodec_IsRegistered_NUM,
  PyBobIoCodec_GetDescription_NUM,
  // Total number of C API pointers
  PyBobIo_API_pointers
};

/**************
 * Versioning *
 **************/

#define PyBobIo_APIVersion_TYPE int

/**********************************
 * Bindings for bob.io.base.File *
 **********************************/

/* Type definition for PyBobIoFileObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<bob::io::base::File> f;

} PyBobIoFileObject;

#define PyBobIoFile_Type_TYPE PyTypeObject

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  PyBobIoFileObject* pyfile;
  Py_ssize_t curpos;

} PyBobIoFileIteratorObject;

#define PyBobIoFileIterator_Type_TYPE PyTypeObject

/************************
 * I/O generic bindings *
 ************************/

#define PyBobIo_AsTypenum_RET int
#define PyBobIo_AsTypenum_PROTO (bob::io::base::array::ElementType et)

#define PyBobIo_TypeInfoAsTuple_RET PyObject*
#define PyBobIo_TypeInfoAsTuple_PROTO (const bob::io::base::array::typeinfo& ti)

#define PyBobIo_FilenameConverter_RET int
#define PyBobIo_FilenameConverter_PROTO (PyObject* o, const char** b)

/*****************
 * HDF5 bindings *
 *****************/

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<bob::io::base::HDF5File> f;

} PyBobIoHDF5FileObject;

#define PyBobIoHDF5File_Type_TYPE PyTypeObject

#define PyBobIoHDF5File_Check_RET int
#define PyBobIoHDF5File_Check_PROTO (PyObject* o)

#define PyBobIoHDF5File_Converter_RET int
#define PyBobIoHDF5File_Converter_PROTO (PyObject* o, PyBobIoHDF5FileObject** a)

/*****************************************
 * Code Registration and De-registration *
 *****************************************/

#define PyBobIoCodec_Register_RET int
#define PyBobIoCodec_Register_PROTO (const char* extension, const char* description, bob::io::base::file_factory_t factory)

#define PyBobIoCodec_Deregister_RET int
#define PyBobIoCodec_Deregister_PROTO (const char* extension)

#define PyBobIoCodec_IsRegistered_RET int
#define PyBobIoCodec_IsRegistered_PROTO (const char* extension)

#define PyBobIoCodec_GetDescription_RET const char*
#define PyBobIoCodec_GetDescription_PROTO (const char* extension)

#ifdef BOB_IO_BASE_MODULE

  /* This section is used when compiling `bob.io.base' itself */

  /**************
   * Versioning *
   **************/

  extern int PyBobIo_APIVersion;

  /**********************************
   * Bindings for bob.io.base.File *
   **********************************/

  extern PyBobIoFile_Type_TYPE PyBobIoFile_Type;
  extern PyBobIoFileIterator_Type_TYPE PyBobIoFileIterator_Type;

  /************************
   * I/O generic bindings *
   ************************/

  PyBobIo_AsTypenum_RET PyBobIo_AsTypenum PyBobIo_AsTypenum_PROTO;

  PyBobIo_TypeInfoAsTuple_RET PyBobIo_TypeInfoAsTuple PyBobIo_TypeInfoAsTuple_PROTO;

  PyBobIo_FilenameConverter_RET PyBobIo_FilenameConverter PyBobIo_FilenameConverter_PROTO;

/*****************
 * HDF5 bindings *
 *****************/

  extern PyBobIoHDF5File_Type_TYPE PyBobIoHDF5File_Type;

  PyBobIoHDF5File_Check_RET PyBobIoHDF5File_Check PyBobIoHDF5File_Check_PROTO;

  PyBobIoHDF5File_Converter_RET PyBobIoHDF5File_Converter PyBobIoHDF5File_Converter_PROTO;

/*****************************************
 * Code Registration and De-registration *
 *****************************************/

 PyBobIoCodec_Register_RET PyBobIoCodec_Register PyBobIoCodec_Register_PROTO;

 PyBobIoCodec_Deregister_RET PyBobIoCodec_Deregister PyBobIoCodec_Deregister_PROTO;

 PyBobIoCodec_IsRegistered_RET PyBobIoCodec_IsRegistered PyBobIoCodec_IsRegistered_PROTO;

 PyBobIoCodec_GetDescription_RET PyBobIoCodec_GetDescription PyBobIoCodec_GetDescription_PROTO;

#else // BOB_IO_BASE_MODULE

  /* This section is used in modules that use `bob.io.base's' C-API */

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyBobIo_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyBobIo_API;
#    else
  static void **PyBobIo_API=NULL;
#    endif
#  endif

  /**************
   * Versioning *
   **************/

# define PyBobIo_APIVersion (*(PyBobIo_APIVersion_TYPE *)PyBobIo_API[PyBobIo_APIVersion_NUM])

  /*****************************
   * Bindings for bob.io.File *
   *****************************/

# define PyBobIoFile_Type (*(PyBobIoFile_Type_TYPE *)PyBobIo_API[PyBobIoFile_Type_NUM])
# define PyBobIoFileIterator_Type (*(PyBobIoFileIterator_Type_TYPE *)PyBobIo_API[PyBobIoFileIterator_Type_NUM])

  /************************
   * I/O generic bindings *
   ************************/

# define PyBobIo_AsTypenum (*(PyBobIo_AsTypenum_RET (*)PyBobIo_AsTypenum_PROTO) PyBobIo_API[PyBobIo_AsTypenum_NUM])

# define PyBobIo_TypeInfoAsTuple (*(PyBobIo_TypeInfoAsTuple_RET (*)PyBobIo_TypeInfoAsTuple_PROTO) PyBobIo_API[PyBobIo_TypeInfoAsTuple_NUM])

# define PyBobIo_FilenameConverter (*(PyBobIo_FilenameConverter_RET (*)PyBobIo_FilenameConverter_PROTO) PyBobIo_API[PyBobIo_FilenameConverter_NUM])

  /*****************
   * HDF5 bindings *
   *****************/

# define PyBobIoHDF5File_Type (*(PyBobIoHDF5File_Type_TYPE *)PyBobIo_API[PyBobIoHDF5File_Type_NUM])

# define PyBobIoHDF5File_Check (*(PyBobIoHDF5File_Check_RET (*)PyBobIoHDF5File_Check_PROTO) PyBobIo_API[PyBobIoHDF5File_Check_NUM])

# define PyBobIoHDF5File_Converter (*(PyBobIoHDF5File_Converter_RET (*)PyBobIoHDF5File_Converter_PROTO) PyBobIo_API[PyBobIoHDF5File_Converter_NUM])

/*****************************************
 * Code Registration and De-registration *
 *****************************************/

# define PyBobIoCodec_Register (*(PyBobIoCodec_Register_RET (*)PyBobIoCodec_Register_PROTO) PyBobIo_API[PyBobIoCodec_Register_NUM])

# define PyBobIoCodec_Deregister (*(PyBobIoCodec_Deregister_RET (*)PyBobIoCodec_Deregister_PROTO) PyBobIo_API[PyBobIoCodec_Deregister_NUM])

# define PyBobIoCodec_IsRegistered (*(PyBobIoCodec_IsRegistered_RET (*)PyBobIoCodec_IsRegistered_PROTO) PyBobIo_API[PyBobIoCodec_IsRegistered_NUM])

# define PyBobIoCodec_GetDescription (*(PyBobIoCodec_GetDescription_RET (*)PyBobIoCodec_GetDescription_PROTO) PyBobIo_API[PyBobIoCodec_GetDescription_NUM])

# if !defined(NO_IMPORT_ARRAY)

  /**
   * Returns -1 on error, 0 on success.
   */
  static int import_bob_io_base(void) {

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOB_IO_BASE_FULL_NAME);

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

#   if PY_VERSION_HEX >= 0x02070000
    if (PyCapsule_CheckExact(c_api_object)) {
      PyBobIo_API = (void **)PyCapsule_GetPointer(c_api_object,
          PyCapsule_GetName(c_api_object));
    }
#   else
    if (PyCObject_Check(c_api_object)) {
      PyBobIo_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }
#   endif

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    if (!PyBobIo_API) {
      PyErr_SetString(PyExc_ImportError, "cannot find C/C++ API "
#   if PY_VERSION_HEX >= 0x02070000
          "capsule"
#   else
          "cobject"
#   endif
          " at `" BOB_IO_BASE_FULL_NAME "._C_API'");
      return -1;
    }

    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyBobIo_API[PyBobIo_APIVersion_NUM];

    if (BOB_IO_BASE_API_VERSION != imported_version) {
      PyErr_Format(PyExc_ImportError, BOB_IO_BASE_FULL_NAME " import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOB_IO_BASE_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

# endif //!defined(NO_IMPORT_ARRAY)

#endif /* BOB_IO_BASE_MODULE */

#endif /* BOB_IO_BASE_H */
