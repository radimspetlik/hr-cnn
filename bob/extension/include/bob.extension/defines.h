/**
 * @file bob/extension/include/bob.extension/documentation.h
 * @date Fri Nov 21 10:27:38 CET 2014
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 *
 * @brief Implements a few functions to generate doc strings
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


/** By including this file, you will be able to
*
* 1. Use the Python2 functions PyInt_Check, PyInt_AS_LONG, PyString_Check, PyString_FromString and PyString_AS_STRING within the bindings for python3
* 2. Add try{ ... } catch {...} blocks around your bindings so that you make sure that **all** exceptions in the C++ code are handled correctly.
*
*/

#ifndef BOB_EXTENSION_DEFINES_H_INCLUDED
#define BOB_EXTENSION_DEFINES_H_INCLUDED


#if PY_VERSION_HEX >= 0x03000000
#define PyInt_Check PyLong_Check
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyString_Check PyUnicode_Check
#define PyString_FromString PyUnicode_FromString
#define PyString_FromFormat PyUnicode_FromFormat
#define PyString_AS_STRING(x) PyBytes_AS_STRING(make_safe(PyUnicode_AsUTF8String(x)).get())
#define PyString_AsString(x) PyUnicode_AsUTF8(x)
#endif

#define PyBob_NumberCheck(x) (PyInt_Check(x) || PyLong_Check(x) || PyFloat_Check(x) || PyComplex_Check(x))


// BOB_TRY is simply a try{
#define BOB_TRY try{

// for catching exceptions, you can define a message, and you have to select
// the error return value (i.e., -1 for constructors, and 0 for other
// functions)

// There exist two macros that will print slightly different messages.
// BOB_CATCH_MEMBER is to be used within the binding of a class, and it will
// use the "self" pointer
// BOB_CATCH_FUNCTION is to be used to bind functions outside a class
#define BOB_CATCH_MEMBER(message,ret) }\
  catch (std::exception& e) {\
    PyErr_Format(PyExc_RuntimeError, "%s - %s: C++ exception caught: '%s'", Py_TYPE(self)->tp_name, message, e.what());\
    return ret;\
  } \
  catch (...) {\
    PyErr_Format(PyExc_RuntimeError, "%s - %s: unknown exception caught", Py_TYPE(self)->tp_name, message);\
    return ret;\
  }

#define BOB_CATCH_FUNCTION(message, ret) }\
  catch (std::exception& e) {\
    PyErr_Format(PyExc_RuntimeError, "%s: C++ exception caught: '%s'", message, e.what());\
    return ret;\
  } \
  catch (...) {\
    PyErr_Format(PyExc_RuntimeError, "%s: unknown exception caught", message);\
    return ret;\
  }

#endif // BOB_EXTENSION_DEFINES_H_INCLUDED
