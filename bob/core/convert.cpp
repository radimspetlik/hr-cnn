/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 16 Oct 17:40:24 2013
 *
 * @brief Pythonic bindings to C++ constructs on bob.core
 */

#include <bob.core/config.h>
#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>

#include <bob.core/array_convert.h>
#include <bob.core/array_sort.h>

template <typename Tdst, typename Tsrc, int N>
PyObject* inner_convert (PyBlitzArrayObject* src,
    PyObject* dst_min, PyObject* dst_max,
    PyObject* src_min, PyObject* src_max) {

  using bob::core::array::convert;
  using bob::core::array::convertFromRange;
  using bob::core::array::convertToRange;

  Tdst c_dst_min = dst_min ? PyBlitzArrayCxx_AsCScalar<Tdst>(dst_min) : 0;
  Tdst c_dst_max = dst_max ? PyBlitzArrayCxx_AsCScalar<Tdst>(dst_max) : 0;
  Tsrc c_src_min = src_min ? PyBlitzArrayCxx_AsCScalar<Tsrc>(src_min) : 0;
  Tsrc c_src_max = src_max ? PyBlitzArrayCxx_AsCScalar<Tsrc>(src_max) : 0;
  auto bz_src = PyBlitzArrayCxx_AsBlitz<Tsrc,N>(src);

  if (src_min) {

    if (dst_min) { //both src_range and dst_range are valid
      auto bz_dst = convert<Tdst,Tsrc>(*bz_src, c_dst_min, c_dst_max, c_src_min, c_src_max);
      return PyBlitzArrayCxx_AsNumpy(bz_dst);
    }

    //only src_range is valid
    auto bz_dst = convertFromRange<Tdst,Tsrc>(*bz_src, c_src_min, c_src_max);
    return PyBlitzArrayCxx_AsNumpy(bz_dst);
  }

  else if (dst_min) { //only dst_range is valid
    auto bz_dst = convertToRange<Tdst,Tsrc>(*bz_src, c_dst_min, c_dst_max);
    return PyBlitzArrayCxx_AsNumpy(bz_dst);
  }

  //use all defaults
  auto bz_dst = convert<Tdst,Tsrc>(*bz_src);
  return PyBlitzArrayCxx_AsNumpy(bz_dst);
}


template <typename Tdst, typename Tsrc>
PyObject* convert_dim (PyBlitzArrayObject* src,
    PyObject* dst_min, PyObject* dst_max,
    PyObject* src_min, PyObject* src_max) {

  switch (src->ndim) {
    case 1: return inner_convert<Tdst, Tsrc, 1>(src, dst_min, dst_max, src_min, src_max);
    case 2: return inner_convert<Tdst, Tsrc, 2>(src, dst_min, dst_max, src_min, src_max);
    case 3: return inner_convert<Tdst, Tsrc, 3>(src, dst_min, dst_max, src_min, src_max);
    case 4: return inner_convert<Tdst, Tsrc, 4>(src, dst_min, dst_max, src_min, src_max);
    default:
      PyErr_Format(PyExc_TypeError, "conversion does not support %" PY_FORMAT_SIZE_T "d dimensions", src->ndim);
  }
  return 0;
}

template <typename T> PyObject* convert_to(PyBlitzArrayObject* src,
    PyObject* dst_min, PyObject* dst_max,
    PyObject* src_min, PyObject* src_max) {

  switch (src->type_num) {
    case NPY_BOOL: return convert_dim<T, bool>(src, dst_min, dst_max, src_min, src_max);
    case NPY_INT8: return convert_dim<T, int8_t>(src, dst_min, dst_max, src_min, src_max);
    case NPY_INT16: return convert_dim<T, int16_t>(src, dst_min, dst_max, src_min, src_max);
    case NPY_INT32: return convert_dim<T, int32_t>(src, dst_min, dst_max, src_min, src_max);
    case NPY_INT64: return convert_dim<T, int64_t>(src, dst_min, dst_max, src_min, src_max);
    case NPY_UINT8: return convert_dim<T, uint8_t>(src, dst_min, dst_max, src_min, src_max);
    case NPY_UINT16: return convert_dim<T, uint16_t>(src, dst_min, dst_max, src_min, src_max);
    case NPY_UINT32: return convert_dim<T, uint32_t>(src, dst_min, dst_max, src_min, src_max);
    case NPY_UINT64: return convert_dim<T, uint64_t>(src, dst_min, dst_max, src_min, src_max);
    case NPY_FLOAT32: return convert_dim<T, float>(src, dst_min, dst_max, src_min, src_max);
    case NPY_FLOAT64: return convert_dim<T, double>(src, dst_min, dst_max, src_min, src_max);
    default:
      PyErr_Format(PyExc_TypeError, "conversion from `%s' (%d) is not supported", PyBlitzArray_TypenumAsString(src->type_num), src->type_num);
  }
  return 0;
}

static auto convert_doc = bob::extension::FunctionDoc(
  "convert",
  "Converts array data type, with optional range squash/expansion",
  "This function allows to convert/rescale a array of a given type into another array of a possibly different type. "
  "Typically, this can be used to rescale a 16 bit precision grayscale image (2D array) into an 8 bit precision grayscale image."
)
.add_prototype("src, dtype, [dest_range], [source_range]", "converted")
.add_parameter("src", "array_like", "Input array")
.add_parameter("dtype", ":py:class:`numpy.dtype` or anything convertible", "The element data type for the returned ``converted`` array")
.add_parameter("dest_range", "(dtype, dtype)", "[Default: full range of ``dtype``] The range ``[min, max]`` to be deployed at the ``converted`` array")
.add_parameter("source_range", "(X, X)", "[Default: full range of ``src`` data type]  Determines the input range ``[min,max]`` that will be used for scaling")
.add_return("converted", "array_like", "A new array with the same shape as ``src``, but re-scaled and with its element type as given by the ``dtype``")
;
static PyObject* py_convert(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = convert_doc.kwlist();

  PyBlitzArrayObject* src;
  int type_num;
  PyObject* dst_min = 0;
  PyObject* dst_max = 0;
  PyObject* src_min = 0;
  PyObject* src_max = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|(OO)(OO)",
        kwlist,
        &PyBlitzArray_Converter, &src,
        &PyBlitzArray_TypenumConverter, &type_num,
        &dst_min, &dst_max,
        &src_min, &src_max
        )) return 0;
  auto src_ = make_safe(src);

  switch (type_num) {
    case NPY_UINT8: return convert_to<uint8_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_UINT16: return convert_to<uint16_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_FLOAT64: return convert_to<double>(src, dst_min, dst_max, src_min, src_max);
      break;

    default:
      PyErr_Format(PyExc_TypeError, "conversion to `%s' (%d) is not supported", PyBlitzArray_TypenumAsString(type_num), type_num);
  }
  return 0;
BOB_CATCH_FUNCTION("convert", 0)
}



static auto sort_doc = bob::extension::FunctionDoc(
  "_sort",
  "Sorts a blitz::Array.",
  "This function should only be used in the C++ code. "
  "The binding is only for test purposes."
)
.add_prototype("array")
.add_parameter("array", "array_like(float,1D)", "The unsorted array, which will be sorted afterwards")
;

static PyObject* sort(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = sort_doc.kwlist();

  PyBlitzArrayObject* array;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBlitzArray_OutputConverter, &array)) return 0;
  auto array_ = make_safe(array);

  if (array->ndim != 1 || array->type_num != NPY_FLOAT64){
    PyErr_SetString(PyExc_TypeError, "Invalid input");
    return 0;
  }

  bob::core::array::sort(*PyBlitzArrayCxx_AsBlitz<double,1>(array));

  Py_RETURN_NONE;
BOB_CATCH_FUNCTION("sort", 0)
}



static PyMethodDef module_methods[] = {
    {
      convert_doc.name(),
      (PyCFunction)py_convert,
      METH_VARARGS|METH_KEYWORDS,
      convert_doc.doc()
    },
    {
      sort_doc.name(),
      (PyCFunction)sort,
      METH_VARARGS|METH_KEYWORDS,
      sort_doc.doc()
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "bob::core::array::convert bindings");

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
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m);

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;

  return Py_BuildValue("O", m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
