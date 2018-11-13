/**
 * @author Manuel Guenther <siebenkopf@googlemail.com>
 * @date Wed Jun  7 17:24:09 MDT 2017
 *
 * @brief Header file for bindings to bob::io::video
 */


#ifndef BOB_IO_VIDEO_MAIN_H
#define BOB_IO_VIDEO_MAIN_H

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

#include "cpp/utils.h"
#include "cpp/reader.h"
#include "cpp/writer.h"
#include "bobskin.h"
#include "file.h"

// Reader
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::io::video::Reader> v;
} PyBobIoVideoReaderObject;

extern PyTypeObject PyBobIoVideoReader_Type;
bool init_BobIoVideoReader(PyObject* module);
int PyBobIoVideoReader_Check(PyObject* o);

// Iterator
typedef struct {
  PyObject_HEAD
  PyBobIoVideoReaderObject* pyreader;
  boost::shared_ptr<bob::io::video::Reader::const_iterator> iter;
} PyBobIoVideoReaderIteratorObject;
extern PyTypeObject PyBobIoVideoReaderIterator_Type;

// Writer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::io::video::Writer> v;
} PyBobIoVideoWriterObject;

extern PyTypeObject PyBobIoVideoWriter_Type;
bool init_BobIoVideoWriter(PyObject* module);
int PyBobIoVideoWriter_Check(PyObject* o);

#endif // BOB_IO_VIDEO_MAIN_H
