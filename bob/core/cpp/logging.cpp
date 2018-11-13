/**
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements all logging infrastructure.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/shared_array.hpp>
#include <boost/make_shared.hpp>

#ifdef _WIN32
#include <io.h> //definition of mktemp
#endif

/**
 * MT "lock" support was only introduced in Boost 1.35. Before copying this
 * very ugly hack, make sure we are still using Boost 1.34. This will no longer
 * be the case starting January 2011.
 */
#include <boost/version.hpp>
#include <boost/thread/mutex.hpp>
#if ((BOOST_VERSION / 100) % 1000) > 34
#include <boost/thread/locks.hpp>
#else
#warning Disabling MT locks because Boost < 1.35!
#endif

#include <bob.core/logging.h>

// This static variable holds the current log level setting for all modules
static bob::core::LOG_LEVEL _CURRENT_LOG_LEVEL = bob::core::WARNING;

void bob::core::log_level(bob::core::LOG_LEVEL level) {
  _CURRENT_LOG_LEVEL = level;
}

bob::core::LOG_LEVEL bob::core::log_level() {
  return _CURRENT_LOG_LEVEL;
}


bob::core::OutputDevice::~OutputDevice() {}


struct NullOutputDevice: public bob::core::OutputDevice {
  virtual ~NullOutputDevice() {}
  virtual std::streamsize write(const char*, std::streamsize n) {
    return n;
  }
};


struct StdoutOutputDevice: public bob::core::OutputDevice {
  virtual ~StdoutOutputDevice() {}
  virtual std::streamsize write(const char* s, std::streamsize n) {
    static boost::mutex mutex;
#if ((BOOST_VERSION / 100) % 1000) > 35
    boost::lock_guard<boost::mutex> lock(mutex);
#endif
    std::cout.write(s, n);
    return n;
  }
};


struct StderrOutputDevice: public bob::core::OutputDevice {
  virtual ~StderrOutputDevice() {}
  virtual std::streamsize write(const char* s, std::streamsize n) {
    static boost::mutex mutex;
#if ((BOOST_VERSION / 100) % 1000) > 35
    boost::lock_guard<boost::mutex> lock(mutex);
#endif
    std::cerr.write(s, n);
    return n;
  }
};


/**
 * Determines if the input filename ends in ".gz"
 *
 * @param filename The name of the file to be analyzed.
 */
inline static bool is_dot_gz(const std::string& filename) {
  return boost::filesystem::path(filename).extension() == ".gz";
}

struct FileOutputDevice: public bob::core::OutputDevice {
  FileOutputDevice(const std::string& filename)
    : m_filename(filename),
      m_file(),
      m_ostream(new boost::iostreams::filtering_ostream),
      m_mutex(new boost::mutex)
  {
    //this first bit creates the output file handle
    std::ios_base::openmode mode = std::ios_base::out | std::ios_base::trunc;
    if (is_dot_gz(filename)) mode |= std::ios_base::binary;
    m_file = boost::make_shared<std::ofstream>(filename.c_str(), mode);
    //this second part configures gzip'ing if necessary and associates the
    //output file with the filtering stream.
    if (is_dot_gz(filename))
      m_ostream->push(boost::iostreams::basic_gzip_compressor<>());
    m_ostream->push(*m_file);
  }

  FileOutputDevice(const FileOutputDevice& other)
    : m_filename(other.m_filename),
      m_file(other.m_file),
      m_ostream(other.m_ostream),
      m_mutex(other.m_mutex)
  {
  }

  virtual ~FileOutputDevice() {}

  virtual std::streamsize write(const char* s, std::streamsize n) {
#if ((BOOST_VERSION / 100) % 1000) > 35
    boost::lock_guard<boost::mutex> lock(*m_mutex);
#endif
    m_ostream->write(s, n);
    return n;
  }

  //internal representation
  private:
    std::string m_filename; ///< the name of the file I'm writing to
    boost::shared_ptr<std::ofstream> m_file; ///< the file output stream
    boost::shared_ptr<boost::iostreams::filtering_ostream> m_ostream; ///< the output stream
    boost::shared_ptr<boost::mutex> m_mutex; ///< multi-threading guardian

};


bob::core::AutoOutputDevice::AutoOutputDevice()
: m_level(bob::core::DISABLED), m_device(new NullOutputDevice)
{
}

bob::core::AutoOutputDevice::AutoOutputDevice(const std::string& configuration,
    bob::core::LOG_LEVEL level)
: m_level(level), m_device()
{
  reset(configuration, level);
}

bob::core::AutoOutputDevice::AutoOutputDevice(boost::shared_ptr<OutputDevice> d,
    bob::core::LOG_LEVEL level)
: m_level(level), m_device(d)
{
}

bob::core::AutoOutputDevice::~AutoOutputDevice() {
}

void bob::core::AutoOutputDevice::reset(const std::string& configuration, bob::core::LOG_LEVEL level)
{
  m_level = level;
  std::string str(configuration);
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  if (str == "null" || str.size()==0) m_device.reset(new NullOutputDevice);
  else if (str == "stdout") m_device.reset(new StdoutOutputDevice);
  else if (str == "stderr") m_device.reset(new StderrOutputDevice);
  else m_device.reset(new FileOutputDevice(configuration));
}

std::streamsize bob::core::AutoOutputDevice::write(const char* s, std::streamsize n) {
  if (m_level >= _CURRENT_LOG_LEVEL) return m_device->write(s, n);
  /* else */ return n;
}

void bob::core::AutoOutputDevice::close() {
  m_level = bob::core::DISABLED;
  m_device->close();
}


boost::iostreams::stream<bob::core::AutoOutputDevice>
  bob::core::debug("stdout", bob::core::DEBUG);
boost::iostreams::stream<bob::core::AutoOutputDevice>
  bob::core::info("stdout", bob::core::INFO);
boost::iostreams::stream<bob::core::AutoOutputDevice>
  bob::core::warn("stderr", bob::core::WARNING);
boost::iostreams::stream<bob::core::AutoOutputDevice>
  bob::core::error("stderr", bob::core::ERROR);


bool bob::core::debug_level(unsigned int i) {
  const char* value = getenv("BOB_DEBUG");
  if (!value) return false;
  unsigned long v = strtoul(value, 0, 0);
  if (v < 1 || v > 3) v = 0;
  return (i <= v);
}


