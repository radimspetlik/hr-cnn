/* boost random/detail/operators.hpp header file
 *
 * Copyright Steven Watanabe 2010-2011
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * See http://www.boost.org for most recent version including documentation.
 *
 * $Id$
 */

#ifndef BOB_BOOST_RANDOM_DETAIL_OPERATORS_HPP
#define BOB_BOOST_RANDOM_DETAIL_OPERATORS_HPP

#include <boost/random/detail/config.hpp>
#include <boost/detail/workaround.hpp>

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1310)   \
    || BOOST_WORKAROUND(__SUNPRO_CC, BOOST_TESTED_AT(0x5100))

#define BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR(os, T, t)                  \
    template<class CharT, class Traits>                                 \
    friend std::basic_ostream<CharT,Traits>&                            \
    operator<<(std::basic_ostream<CharT,Traits>& os, const T& t) {      \
        t.print(os, t);                                                 \
        return os;                                                      \
    }                                                                   \
    template<class CharT, class Traits>                                 \
    static std::basic_ostream<CharT,Traits>&                            \
    print(std::basic_ostream<CharT,Traits>& os, const T& t)

#define BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR(is, T, t)                  \
    template<class CharT, class Traits>                                 \
    friend std::basic_istream<CharT,Traits>&                            \
    operator>>(std::basic_istream<CharT,Traits>& is, T& t) {            \
        t.read(is, t);                                                  \
        return is;                                                      \
    }                                                                   \
    template<class CharT, class Traits>                                 \
    static std::basic_istream<CharT,Traits>&                            \
    read(std::basic_istream<CharT,Traits>& is, T& t)

#endif

#if defined(__BORLANDC__)

#define BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR(T, lhs, rhs)              \
    bool operator==(const T& rhs) const                                 \
    { return T::is_equal(*this, rhs); }                                 \
    static bool is_equal(const T& lhs, const T& rhs)

#define BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR(T)                      \
    bool operator!=(const T& rhs) const                                 \
    { return !T::is_equal(*this, rhs); }

#endif

#ifndef BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR
#define BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR(os, T, t)                  \
    template<class CharT, class Traits>                                 \
    friend std::basic_ostream<CharT,Traits>&                            \
    operator<<(std::basic_ostream<CharT,Traits>& os, const T& t)
#endif

#ifndef BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR
#define BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR(is, T, t)                  \
    template<class CharT, class Traits>                                 \
    friend std::basic_istream<CharT,Traits>&                            \
    operator>>(std::basic_istream<CharT,Traits>& is, T& t)
#endif

#ifndef BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR
#define BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR(T, lhs, rhs)              \
    friend bool operator==(const T& lhs, const T& rhs)
#endif

#ifndef BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR
#define BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR(T)                      \
    friend bool operator!=(const T& lhs, const T& rhs)                  \
    { return !(lhs == rhs); }
#endif


// from vector_io.hpp

namespace bob {
namespace core {
namespace random {
namespace detail {

template<class CharT, class Traits, class T>
void print_vector(std::basic_ostream<CharT, Traits>& os,
                  const std::vector<T>& vec)
{
    typename std::vector<T>::const_iterator
        iter = vec.begin(),
        end =  vec.end();
    os << os.widen('[');
    if(iter != end) {
        os << *iter;
        ++iter;
        for(; iter != end; ++iter)
        {
            os << os.widen(' ') << *iter;
        }
    }
    os << os.widen(']');
}

template<class CharT, class Traits, class T>
void read_vector(std::basic_istream<CharT, Traits>& is, std::vector<T>& vec)
{
    CharT ch;
    if(!(is >> ch)) {
        return;
    }
    if(ch != is.widen('[')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
        return;
    }
    T val;
    while(is >> std::ws >> val) {
        vec.push_back(val);
    }
    if(is.fail()) {
        is.clear();
        if(!(is >> ch)) {
            return;
        }
        if(ch != is.widen(']')) {
            is.putback(ch);
            is.setstate(std::ios_base::failbit);
        }
    }
}

} } } } // namespaces

#endif
