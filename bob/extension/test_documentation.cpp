#include <bob.extension/documentation.h>

int test_documenation(){

  auto function_doc = bob::extension::FunctionDoc(
    "TestClass",
    "This is the constructor of the TestClass",
    ".. todo:: Add more information here",
    true
  )
  .add_prototype("para1", "")
  .add_parameter("para1", "int", "A parameter of type int");

  auto class_doc = bob::extension::ClassDoc(
    "TestClass",
    "This is the documentation for a test class.",
    "Just to check that it works"
  )
  .add_constructor(function_doc);

  auto var_doc = bob::extension::VariableDoc(
    "test_variable",
    "float",
    "A float variable",
    "With more documentation"
  );

  // create documentation, just to see if it works
  class_doc.name();
  class_doc.doc(72);

  var_doc.name();
  var_doc.doc(72);

  return 0;
}
