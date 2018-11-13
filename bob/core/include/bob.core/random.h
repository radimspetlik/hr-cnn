/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 30 Oct 07:40:47 2013
 *
 * @brief C/C++-API for the random module
 */

#ifndef BOB_CORE_RANDOM_H
#define BOB_CORE_RANDOM_H

#include <boost/version.hpp>

#if BOOST_VERSION >= 105600

// Use the default implementations of BOOST
namespace bob { namespace core { namespace random {

  template <class RealType=double>
    using normal_distribution = boost::random::normal_distribution<RealType>;

  template <class RealType=double>
    using lognormal_distribution = boost::random::lognormal_distribution<RealType>;

  template<class RealType=double>
    using gamma_distribution = boost::random::gamma_distribution<RealType>;

  template<class IntType=int, class RealType=double>
    using binomial_distribution = boost::random::binomial_distribution<IntType, RealType>;

  template<class IntType=int, class WeightType=double>
    using discrete_distribution = boost::random::discrete_distribution<IntType, WeightType>;


} } } // namespaces

#else

// Use the copied implementations of boost 1.56
// where the bugs have been fixed
#include <bob.core/boost/normal_distribution.hpp>
#include <bob.core/boost/lognormal_distribution.hpp>
#include <bob.core/boost/binomial_distribution.hpp>
#include <bob.core/boost/discrete_distribution.hpp>
#include <bob.core/boost/gamma_distribution.hpp>

#endif // BOOST VERSION

#endif /* BOB_CORE_RANDOM_H */
