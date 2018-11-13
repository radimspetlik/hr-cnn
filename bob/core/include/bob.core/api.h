/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 04 Aug 2014 13:17:07 CEST
 *
 * @brief C/C++-API for the logging module
 */

#ifndef BOB_CORE_LOGGING_H
#define BOB_CORE_LOGGING_H

#include <Python.h>
#include <bob.core/config.h>
#include <boost/shared_ptr.hpp>
#include <bob.core/logging.h>

/* Define Module Name and Prefix for other Modules
   Note: We cannot use BOB_EXT_* macros here, unfortunately */
#define BOB_CORE_LOGGING_PREFIX    "bob.core"
#define BOB_CORE_LOGGING_FULL_NAME "bob.core._logging"

/*******************
 * C API functions *
 *******************/

/**************
 * Versioning *
 **************/

#define PyBobCoreLogging_APIVersion_NUM 0
#define PyBobCoreLogging_APIVersion_TYPE int

/*********************************
 * Bindings for bob.core.logging *
 *********************************/

#define PyBobCoreLogging_Debug_NUM 1
#define PyBobCoreLogging_Debug_RET boost::iostreams::stream<bob::core::AutoOutputDevice>&
#define PyBobCoreLogging_Debug_PROTO ()

#define PyBobCoreLogging_Info_NUM 2
#define PyBobCoreLogging_Info_RET boost::iostreams::stream<bob::core::AutoOutputDevice>&
#define PyBobCoreLogging_Info_PROTO ()

#define PyBobCoreLogging_Warn_NUM 3
#define PyBobCoreLogging_Warn_RET boost::iostreams::stream<bob::core::AutoOutputDevice>&
#define PyBobCoreLogging_Warn_PROTO ()

#define PyBobCoreLogging_Error_NUM 4
#define PyBobCoreLogging_Error_RET boost::iostreams::stream<bob::core::AutoOutputDevice>&
#define PyBobCoreLogging_Error_PROTO ()

/* Total number of C API pointers */
#define PyBobCoreLogging_API_pointers 5

#ifdef BOB_CORE_LOGGING_MODULE

  /* This section is used when compiling `bob.core.logging' itself */

  /**************
   * Versioning *
   **************/

  extern int PyBobCoreLogging_APIVersion;

  /*********************************
   * Bindings for bob.core.logging *
   *********************************/

  PyBobCoreLogging_Debug_RET PyBobCoreLogging_Debug PyBobCoreLogging_Debug_PROTO;

  PyBobCoreLogging_Info_RET PyBobCoreLogging_Info PyBobCoreLogging_Info_PROTO;

  PyBobCoreLogging_Warn_RET PyBobCoreLogging_Warn PyBobCoreLogging_Warn_PROTO;

  PyBobCoreLogging_Error_RET PyBobCoreLogging_Error PyBobCoreLogging_Error_PROTO;

#else

  /* This section is used in modules that use `bob.core.logging's' C-API */

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyBobCoreLogging_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyBobCoreLogging_API;
#    else
  static void **PyBobCoreLogging_API=NULL;
#    endif
#  endif

  /**************
   * Versioning *
   **************/

# define PyBobCoreLogging_APIVersion (*(PyBobCoreLogging_APIVersion_TYPE *)PyBobCoreLogging_API[PyBobCoreLogging_APIVersion_NUM])

  /*********************************
   * Bindings for bob.core.logging *
   *********************************/

# define PyBobCoreLogging_Debug (*(PyBobCoreLogging_Debug_RET (*)PyBobCoreLogging_Debug_PROTO) PyBobCoreLogging_API[PyBobCoreLogging_Debug_NUM])

# define PyBobCoreLogging_Info (*(PyBobCoreLogging_Info_RET (*)PyBobCoreLogging_Info_PROTO) PyBobCoreLogging_API[PyBobCoreLogging_Info_NUM])

# define PyBobCoreLogging_Warn (*(PyBobCoreLogging_Warn_RET (*)PyBobCoreLogging_Warn_PROTO) PyBobCoreLogging_API[PyBobCoreLogging_Warn_NUM])

# define PyBobCoreLogging_Error (*(PyBobCoreLogging_Error_RET (*)PyBobCoreLogging_Error_PROTO) PyBobCoreLogging_API[PyBobCoreLogging_Error_NUM])

# if !defined(NO_IMPORT_ARRAY)

  /**
   * Returns -1 on error, 0 on success.
   */
  static int import_bob_core_logging(void) {

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOB_CORE_LOGGING_FULL_NAME);

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

#   if PY_VERSION_HEX >= 0x02070000
    if (PyCapsule_CheckExact(c_api_object)) {
      PyBobCoreLogging_API = (void **)PyCapsule_GetPointer(c_api_object,
          PyCapsule_GetName(c_api_object));
    }
#   else
    if (PyCObject_Check(c_api_object)) {
      PyBobCoreLogging_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }
#   endif

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    if (!PyBobCoreLogging_API) {
      PyErr_SetString(PyExc_ImportError, "cannot find C/C++ API "
#   if PY_VERSION_HEX >= 0x02070000
          "capsule"
#   else
          "cobject"
#   endif
          " at `" BOB_CORE_LOGGING_FULL_NAME "._C_API'");
      return -1;
    }

    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyBobCoreLogging_API[PyBobCoreLogging_APIVersion_NUM];

    if (BOB_CORE_API_VERSION != imported_version) {
      PyErr_Format(PyExc_ImportError, BOB_CORE_LOGGING_FULL_NAME " import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOB_CORE_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

# endif //!defined(NO_IMPORT_ARRAY)

#endif /* BOB_CORE_LOGGING_MODULE */

#endif /* BOB_CORE_LOGGING_H */
