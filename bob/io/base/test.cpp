/**
 * @author Manuel Gunther
 * @date Tue Sep 13 13:01:31 MDT 2016
 *
 * @brief Tests for bob::io::base
 */

#include <bob.io.base/api.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

static auto s_test_api = bob::extension::FunctionDoc(
  "_test_api",
  "Some tests for API functions"
)
.add_prototype("tempdir")
.add_parameter("tempdir", "str", "A temporary directory to write data to")
;
static PyObject* _test_api(PyObject*, PyObject *args, PyObject* kwds){
BOB_TRY
  static char** kwlist = s_test_api.kwlist();

  const char* tempdir;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &tempdir)) return 0;

  blitz::Array<uint8_t, 1> test_data(5);
  for (int i = 0; i < 5; ++i){
    test_data(i) = i+1;
  }

  auto h5file = bob::io::base::CodecRegistry::instance()->findByExtension(".hdf5");

  boost::filesystem::path hdf5(tempdir); hdf5 /= std::string("test.h5");

  auto output = h5file(hdf5.string().c_str(), 'w');
  output->write(test_data);
  output.reset();

  auto input = h5file(hdf5.string().c_str(), 'r');
  blitz::Array<uint8_t,1> read_data(input->read<uint8_t,1>(0));

  // Does not compile at the moment
  blitz::Array<uint16_t,1> read_data_2(input->cast<uint16_t,1>(0));

  input.reset();

  if (blitz::any(test_data - read_data))
    throw std::runtime_error("The CSV IO test did not succeed");

  if (blitz::any(test_data - read_data_2))
    throw std::runtime_error("The CSV IO test did not succeed");

  Py_RETURN_NONE;
BOB_CATCH_FUNCTION("_test_api", 0)
}

static PyMethodDef module_methods[] = {
  {
    s_test_api.name(),
    (PyCFunction)_test_api,
    METH_VARARGS|METH_KEYWORDS,
    s_test_api.doc(),
  },
  {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Tests for bob::io::base");

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

  return Py_BuildValue(ret, m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
