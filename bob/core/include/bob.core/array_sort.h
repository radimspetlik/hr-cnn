/**
 * @date Thu Oct 20 11:25:46 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines deep copy functions for blitz++ arrays
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_CORE_ARRAY_SORT_H
#define BOB_CORE_ARRAY_SORT_H

#include <blitz/array.h>
#include <vector>
#include <algorithm>

namespace bob {
  namespace core { namespace array {
    /**
     * @ingroup CORE_ARRAY
     * @{
     */

    /**
     * @brief Sorts an array.
     * By default, the array is sorted ascendantly.
     * If you wish a different sort order, please select a different predicate, which must be a functor that implements an order over T.
     * For an example, have a look at the documentation of std::less<T> (the default in this function)
     *
     * @warning: The array sort is performed out of place, which means that the data is copied, sorted and copied back.
     */
    template <typename T, typename predicate = std::less<T>>
    void sort(blitz::Array<T,1>& a)
    {
      std::vector<T> b(a.extent(0));
      std::copy(a.begin(), a.end(), b.begin());
      std::sort(b.begin(), b.end(), predicate());
      std::copy(b.begin(), b.end(), a.begin());
    }

    /**
     * @}
     */
  }}
}

#endif /* BOB_CORE_ARRAY_SORT_H */
