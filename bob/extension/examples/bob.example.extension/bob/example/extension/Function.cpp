#include <blitz/array.h>

/**
  Simple example of a function dealing with a blitz array
*/
blitz::Array<double,1> reverse (const blitz::Array<double,1>& array){
  // create new array in the desired shape
  blitz::Array<double,1> retval(array.shape());
  // copy data
  for (int i = 0, j = array.extent(0)-1; i < array.extent(0); ++i, --j){
    retval(j) = array(i);
  }
  // return the copied data
  return retval;
}
