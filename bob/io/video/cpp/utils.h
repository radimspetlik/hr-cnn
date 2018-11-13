#ifndef BOB_IO_VIDEO_UTILS_H
#define BOB_IO_VIDEO_UTILS_H

#include <map>
#include <vector>
#include <string>
#include <blitz/array.h>
#include <stdint.h>

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
}

namespace bob { namespace io { namespace video {

  /************************************************************************
   * General Utilities
   ************************************************************************/

  /**
   * Breaks a list of words separated by commands into a word list
   */
  void tokenize_csv(const char* what, std::vector<std::string>& values);

  /**
   * Returns a list of supported codecs, with their capabilities
   */
  void codecs_supported (std::map<std::string, const AVCodec*>& installed);

  /**
   * Returns a list of installed codecs, with their capabilities
   */
  void codecs_installed (std::map<std::string, const AVCodec*>& installed);

  /**
   * Returns 'true' if a given codec is supported
   */
  bool codec_is_supported (const std::string& codecname);

  /**
   * Returns a list of input formats this installation can handle, with details
   */
  void iformats_supported (std::map<std::string, AVInputFormat*>& installed);

  /**
   * Returns a list of input formats this installation can handle, with details
   */
  void iformats_installed (std::map<std::string, AVInputFormat*>& installed);

  /**
   * Returns 'true' if a given input format is supported
   */
  bool iformat_is_supported (const std::string& formatname);

  /**
   * Returns a list of output formats this installation can handle, with
   * details
   */
  void oformats_supported (std::map<std::string, AVOutputFormat*>& installed);

  /**
   * Returns a list of output formats this installation can handle, with
   * details
   */
  void oformats_installed (std::map<std::string, AVOutputFormat*>& installed);

  /**
   * Returns 'true' if a given output format is supported
   */
  bool oformat_is_supported (const std::string& formatname);

  /**
   * List the codecs supported by a given output format
   */
  void oformat_supported_codecs (const std::string& formatname,
      std::vector<const AVCodec*>& installed);

  /**
   * Tells if a certain output format supports a given codec.
   */
  bool oformat_supports_codec (const std::string& formatname,
      const std::string& codecname);

  /************************************************************************
   * Video reading and writing utilities (shared)
   ************************************************************************/

  /**
   * Creates a new codec encoding context and verify all is good.
   *
   * @note The returned object knows how to correctly delete itself, freeing
   * all acquired resources. Nonetheless, when this object is used in
   * conjunction with other objects required for file decoding, order must be
   * respected.
   */
  boost::shared_ptr<AVCodecContext> make_decoder_context(
      const std::string& filename, AVStream* stream, AVCodec* codec);

  /**
   * Creates a new codec encoding context and verify all is good.
   *
   * @note The returned object knows how to correctly delete itself, freeing
   * all acquired resources. Nonetheless, when this object is used in
   * conjunction with other objects required for file encoding, order must be
   * respected.
   */
  boost::shared_ptr<AVCodecContext> make_encoder_context(
      const std::string& filename, AVFormatContext* fmtctxt, AVStream* stream,
      AVCodec* codec, size_t height, size_t width, double framerate,
      double bitrate, size_t gop);

  /**
   * Allocates the software scaler that handles size and pixel format
   * conversion.
   *
   * @note The returned object knows how to correctly delete itself, freeing
   * all acquired resources. Nonetheless, when this object is used in
   * conjunction with other objects required for file encoding, order must be
   * respected.
   *
   * @note This scaler constructor is used both in encoding and decoding,
   * therefore needs to know source and destination pixel formats, which may
   * different in each circumstance.
   */
  boost::shared_ptr<SwsContext> make_scaler(const std::string& filename,
      boost::shared_ptr<AVCodecContext> stream,
      AVPixelFormat source_pixel_format, AVPixelFormat dest_pixel_format);

  /**
   * Allocates a frame for a particular context. The frame space will be
   * allocated to accomodate the type of encoding you defined, upon the
   * selection of the pixel format (last parameter).
   *
   * @note The returned object knows how to correctly delete itself, freeing
   * all acquired resources. Nonetheless, when this object is used in
   * conjunction with other objects required for file encoding, order must be
   * respected.
   */
  boost::shared_ptr<AVFrame> make_frame(const std::string& filename,
      boost::shared_ptr<AVCodecContext> stream);

  /************************************************************************
   * Video reading specific utilities
   ************************************************************************/

  /**
   * Opens a video file for input, makes sure it finds the stream information
   * on that file. Otherwise, raises.
   *
   * @note The returned object knows how to correctly delete itself, freeing
   * all acquired resources. Nonetheless, when this object is used in
   * conjunction with other objects required for file encoding, order must be
   * respected.
   */
  boost::shared_ptr<AVFormatContext> make_input_format_context
    (const std::string& filename);

  /**
   * Finds the location of the video stream in the file or raises, if no video
   * stream can be found.
   */
  int find_video_stream(const std::string& filename,
      boost::shared_ptr<AVFormatContext> format_context);

  /**
   * Finds a proper decoder (codec) for the video stream.
   */
  AVCodec* find_decoder(const std::string& filename,
      boost::shared_ptr<AVFormatContext> format_context, int stream_index);

  /**
   * Allocates an empty frame.
   *
   * @note The returned object knows how to correctly delete itself, freeing
   * all acquired resources. Nonetheless, when this object is used in
   * conjunction with other objects required for file encoding, order must be
   * respected.
   */
  boost::shared_ptr<AVFrame> make_empty_frame(const std::string& filename);

  /**
   * Reads a single video frame from the stream. Input data must be previously
   * allocated and be of the right type and size for holding the frame
   * contents. It is an error to try to read past the end of the file.
   *
   * @return true if it manages to load a video frame or false otherwise.
   */
  bool read_video_frame (const std::string& filename, int current_frame,
      int stream_index, boost::shared_ptr<AVFormatContext> format_context,
      boost::shared_ptr<AVCodecContext> codec_context,
      boost::shared_ptr<SwsContext> swscaler,
      boost::shared_ptr<AVFrame> context_frame, uint8_t* data,
      bool throw_on_error);

  /**
   * Reads a single video frame from the stream, but skip it in the fastest
   * possible way. This method can be used for a somewhat fast forward strategy
   * with read iterators.
   *
   * @return true if it manages to skip a video frame or false otherwise.
   */
  bool skip_video_frame (const std::string& filename, int current_frame,
      int stream_index, boost::shared_ptr<AVFormatContext> format_context,
      boost::shared_ptr<AVCodecContext> codec_context,
      boost::shared_ptr<AVFrame> context_frame, bool throw_on_error);

  /************************************************************************
   * Video writing specific utilities
   ************************************************************************/

  /**
   * Creates a new AVFormatContext object based on a filename and a desired
   * multimedia format this file will contain.
   *
   * @note The returned object knows how to correctly delete itself, freeing
   * all acquired resources. Nonetheless, when this object is used in
   * conjunction with other objects required for file encoding, order must be
   * respected.
   */
  boost::shared_ptr<AVFormatContext> make_output_format_context
    (const std::string& filename, const std::string& formatname);

  /**
   * Finds the encoder that best suits the filename/codecname combination. You
   * don't need to delete the returned object.
   */
  AVCodec* find_encoder(const std::string& filename,
      boost::shared_ptr<AVFormatContext> fmtctxt, const std::string& codecname);

  /**
   * Creates a new AVStream on the output file given by the format context
   * pointer and the codec
   *
   * @note The returned object knows how to correctly delete itself, freeing
   * all acquired resources. Nonetheless, when this object is used in
   * conjunction with other objects required for file encoding, order must be
   * respected.
   */
  boost::shared_ptr<AVStream> make_stream(const std::string& filename,
      boost::shared_ptr<AVFormatContext> fmtctxt, AVCodec* codec);

  /**
   * Opens the output file using the given context, writes a header, if the
   * format requires.
   */
  void open_output_file(const std::string& filename,
      boost::shared_ptr<AVFormatContext> format_context);

  /**
   * Closes the output file using the given context, writes a trailer, if the
   * format requires.
   */
  void close_output_file(const std::string& filename,
      boost::shared_ptr<AVFormatContext> format_context);

  /**
   * Flushes frames which are buffered on the given encoder stream. This is
   * only required if (codec->capabilities & CODEC_CAP_DELAY) is true.
   */
  void flush_encoder(const std::string& filename,
      boost::shared_ptr<AVFormatContext> format_context,
      boost::shared_ptr<AVStream> stream,
      boost::shared_ptr<AVCodecContext> codec_context);

  /**
   * Writes a data frame into the encoder stream.
   *
   * @note The encoder may have a CODEC_CAP_DELAY capability, which means that
   * the frame insertion is not always instantaneous. You must use
   * flush_encoder() in such conditions.
   */
  void write_video_frame (const blitz::Array<uint8_t,3>& data,
    const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context,
    boost::shared_ptr<AVStream> stream,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<AVFrame> context_frame,
    boost::shared_ptr<AVFrame> tmp_frame,
    boost::shared_ptr<SwsContext> swscaler);


}}}

#endif /* BOB_IO_VIDEO_UTILS_H */
