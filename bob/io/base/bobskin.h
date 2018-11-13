/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  5 Nov 22:09:07 2013
 *
 * @brief A pythonic version of bob::io::base::array::interface, with minimal
 * functionality.
 */

#ifndef PYTHON_BOB_IO_BOBSKIN_H
#define PYTHON_BOB_IO_BOBSKIN_H

#include <Python.h>

#include <bob.io.base/array.h>

#include <bob.blitz/capi.h>


/**
 * Wraps a PyArrayObject such that we can access it from bob::io
 */
class bobskin: public bob::io::base::array::interface {

  public: //api

    /**
     * @brief Builds a new skin from an array like object
     */
    bobskin(PyObject* array, bob::io::base::array::ElementType eltype);

    /**
     * @brief Builds a new skin from a numpy array object
     */
    bobskin(PyArrayObject* array, bob::io::base::array::ElementType eltype);

    /**
     * @brief Builds a new skin around a blitz array object
     */
    bobskin(PyBlitzArrayObject* array);

    /**
     * @brief By default, the interface is never freed. You must override
     * this method to do something special for your class type.
     */
    virtual ~bobskin();

    /**
     * @brief Copies the data from another interface.
     */
    virtual void set(const interface& other);

    /**
     * @brief Refers to the data of another interface.
     */
    virtual void set(boost::shared_ptr<interface> other);

    /**
     * @brief Re-allocates this interface taking into consideration new
     * requirements. The internal memory should be considered uninitialized.
     */
    virtual void set (const bob::io::base::array::typeinfo& req);

    /**
     * @brief Type information for this interface.
     */
    virtual const bob::io::base::array::typeinfo& type() const { return m_type; }

    /**
     * @brief Borrows a reference from the underlying memory. This means
     * this object continues to be responsible for deleting the memory and
     * you should make sure that it outlives the usage of the returned
     * pointer.
     */
    virtual void* ptr() { return m_ptr; }
    virtual const void* ptr() const { return m_ptr; }

    /**
     * @brief Returns a representation of the internal cache using shared
     * pointers.
     */
    virtual boost::shared_ptr<void> owner();
    virtual boost::shared_ptr<const void> owner() const;

  private: //representation

    bob::io::base::array::typeinfo m_type; ///< type information
    void* m_ptr; ///< pointer to the data

};

#endif /* PYTHON_BOB_IO_BOBSKIN_H */
