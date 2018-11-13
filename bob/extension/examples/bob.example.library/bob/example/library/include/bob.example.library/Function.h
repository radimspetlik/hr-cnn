/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Fri Oct 10 12:49:11 CEST 2014
 *
 * @brief General directives for all modules in bob.example.library
 */

#ifndef BOB_EXAMPLE_LIBRARY_FUNCTION_H
#define BOB_EXAMPLE_LIBRARY_FUNCTION_H

#include <blitz/array.h>

namespace bob { namespace example { namespace library {

  // Reverses the order of the elements in the given array
  blitz::Array<double,1> reverse (const blitz::Array<double,1>& array);

} } } // namespaces

# endif // BOB_EXAMPLE_LIBRARY_FUNCTION_H
