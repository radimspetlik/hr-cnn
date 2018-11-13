#include <set>
#include <boost/token_iterator.hpp>
#include <boost/format.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/avstring.h>
#include <libavutil/mathematics.h>
#include <libavutil/imgutils.h>
}

#include <bob.core/logging.h>
#include <bob.core/assert.h>

#include "utils.h"

/**
 * Tries to find an encoder name through a decoder
 */
static AVCodec* try_find_through_decoder(const char* codecname) {
  AVCodec* tmp = avcodec_find_decoder_by_name(codecname);
  if (tmp) return avcodec_find_encoder(tmp->id);
  return 0;
}

/**
 * Returns a list of available codecs from what I wish to support
 */
static void check_codec_support(std::map<std::string, const AVCodec*>& retval) {

  std::string tmp[] = {
    "libvpx",
    "vp8",
    "wmv1",
    "wmv2",
    //"wmv3", /* no encoding support */
    "mjpeg",
    "mpegvideo", // the same as mpeg2video
    "mpeg1video",
    //"mpeg1video_vdpau", //hw accelerated mpeg1video decoding
    "mpeg2video", // the same as mpegvideo
    //"mpegvideo_vdpau", //hw accelerated mpegvideo decoding
    "mpeg4",
    "msmpeg4",
    //"msmpeg4v1", /* no encoding support */
    "msmpeg4v2", // the same as msmpeg4
    "ffv1", // buggy on ffmpeg >= 2.0
    //"h263p", //bogus on libav-0.8.4
    "h264",
    //"h264_vdpau", //hw accelerated h264 decoding
    //"theora", //buggy on some platforms
    //"libtheora", //buggy on some platforms
    "libopenh264",
    "libx264",
    "zlib",
  };

  std::set<std::string> wishlist(tmp, tmp + (sizeof(tmp)/sizeof(tmp[0])));

  for (AVCodec* it = av_codec_next(0); it != 0; it = av_codec_next(it) ) {
    if (wishlist.find(it->name) == wishlist.end()) continue; ///< ignore this codec
    if (it->type == AVMEDIA_TYPE_VIDEO) {
      auto exists = retval.find(std::string(it->name));
      if (exists != retval.end() && exists->second->id != it->id) {
        bob::core::warn << "Not overriding video codec \"" << it->long_name
          << "\" (" << it->name << ")" << std::endl;
      }
      else {
        // a codec is potentially available, check encoder and decoder
        bool has_decoder = (bool)(avcodec_find_decoder(it->id));
        bool has_encoder = (bool)(avcodec_find_encoder(it->id));
        if (!has_encoder) {
          has_encoder = (bool)try_find_through_decoder(it->name);
        }
        if (has_encoder && has_decoder) retval[it->name] = it;
        // else, skip this one (cannot test encoding loop)
      }
    }
  }
}

static void check_iformat_support(std::map<std::string, AVInputFormat*>& retval) {

  std::string tmp[] = {
    "avi",
    "mov",
    //"flv", //buggy
    "mp4",
  };

  std::set<std::string> wishlist(tmp, tmp + (sizeof(tmp)/sizeof(tmp[0])));

  for (AVInputFormat* it = av_iformat_next(0); it != 0; it = av_iformat_next(it) ) {
    std::vector<std::string> names;
    bob::io::video::tokenize_csv(it->name, names);
    for (auto k = names.begin(); k != names.end(); ++k) {
      if (wishlist.find(*k) == wishlist.end()) continue; ///< ignore this format
      auto exists = retval.find(*k);
      if (exists != retval.end()) {
        bob::core::warn << "Not overriding input video format \""
          << it->long_name << "\" (" << *k
          << ") which is already assigned to \"" << exists->second->long_name
          << "\"" << std::endl;
      }
      else retval[*k] = it;
    }
  }
}

static void check_oformat_support(std::map<std::string, AVOutputFormat*>& retval) {

  std::string tmp[] = {
    "avi",
    "mov",
    //"flv", //buggy
    "mp4",
  };

  std::set<std::string> wishlist(tmp, tmp + (sizeof(tmp)/sizeof(tmp[0])));

  for (AVOutputFormat* it = av_oformat_next(0); it != 0; it = av_oformat_next(it) ) {
    std::vector<std::string> names;
    bob::io::video::tokenize_csv(it->name, names);
    for (auto k = names.begin(); k != names.end(); ++k) {
      if (wishlist.find(*k) == wishlist.end()) continue; ///< ignore this format
      auto exists = retval.find(*k);
      if (exists != retval.end()) {
        bob::core::warn << "Not overriding input video format \""
          << it->long_name << "\" (" << *k
          << ") which is already assigned to \"" << exists->second->long_name
          << "\"" << std::endl;
      }
      else retval[*k] = it;
    }
  }
}

/**
 * Defines which combinations of codecs and output formats are valid.
 */
static void define_output_support_map(std::map<AVOutputFormat*, std::vector<const AVCodec*> >& retval) {
  std::map<std::string, const AVCodec*> cdict;
  check_codec_support(cdict);
  std::map<std::string, AVOutputFormat*> odict;
  check_oformat_support(odict);

  auto it = odict.find("avi");
  if (it != odict.end()) { // ".avi" format is available
    retval[it->second].clear();
    for (auto jt = cdict.begin(); jt != cdict.end(); ++jt) {
      retval[it->second].push_back(jt->second); // all formats are possible
    }
  }

  it = odict.find("mov");
  if (it != odict.end()) { // ".mov" format is available
    retval[it->second].clear();
    for (auto jt = cdict.begin(); jt != cdict.end(); ++jt) {
      retval[it->second].push_back(jt->second); // all formats are possible
    }
  }

  it = odict.find("mp4");
  if (it != odict.end()) { // ".mp4" format is available
    retval[it->second].clear();
    std::string tmp[] = {
      "libx264",
      "libopenh264",
      "h264",
      //"h264_vdpau",
      "mjpeg",
      "mpeg1video",
      //"mpegvideo_vdpau"
      "mpeg2video",
      "mpegvideo",
      "mpeg4",
    };
    std::vector<std::string> codecs(tmp, tmp + (sizeof(tmp)/sizeof(tmp[0])));
    for (auto jt = codecs.begin(); jt != codecs.end(); ++jt) {
      auto kt = cdict.find(*jt);
      if (kt != cdict.end()) retval[it->second].push_back(kt->second);
    }
  }
}

void bob::io::video::tokenize_csv(const char* what, std::vector<std::string>& values) {
  if (!what) return;
  boost::char_separator<char> sep(",");
  std::string w(what);
  boost::tokenizer< boost::char_separator<char> > tok(w, sep);
  for (auto k = tok.begin(); k != tok.end(); ++k) values.push_back(*k);
}

void bob::io::video::codecs_installed (std::map<std::string, const AVCodec*>& installed) {
  for (AVCodec* it = av_codec_next(0); it != 0; it = av_codec_next(it) ) {
    if (it->type == AVMEDIA_TYPE_VIDEO) {
      /**
      auto exists = installed.find(std::string(it->name));
      if (exists != installed.end() && exists->second->id != it->id) {
        bob::core::warn << "Not overriding video codec \"" << it->long_name
          << "\" (" << it->name << ")" << std::endl;
      }
      else **/
      installed[it->name] = it;
    }
  }
}

void bob::io::video::codecs_supported (std::map<std::string, const AVCodec*>& installed) {
  check_codec_support(installed);
}

bool bob::io::video::codec_is_supported (const std::string& name) {
  std::map<std::string, const AVCodec*> cdict;
  bob::io::video::codecs_supported(cdict);
  return (cdict.find(name) != cdict.end());
}

void bob::io::video::iformats_supported (std::map<std::string, AVInputFormat*>& installed) {
  check_iformat_support(installed);
}

bool bob::io::video::iformat_is_supported (const std::string& name) {
  std::map<std::string, AVInputFormat*> idict;
  bob::io::video::iformats_supported(idict);
  std::vector<std::string> names;
  bob::io::video::tokenize_csv(name.c_str(), names);
  for (auto k = names.begin(); k != names.end(); ++k) {
    if (idict.find(*k) != idict.end()) return true;
  }
  return false;
}

void bob::io::video::iformats_installed (std::map<std::string, AVInputFormat*>& installed) {
  for (AVInputFormat* it = av_iformat_next(0); it != 0; it = av_iformat_next(it) ) {
    std::vector<std::string> names;
    bob::io::video::tokenize_csv(it->name, names);
    for (auto k = names.begin(); k != names.end(); ++k) {
      auto exists = installed.find(*k);
      if (exists != installed.end()) {
        bob::core::warn << "Not overriding input video format \""
          << it->long_name << "\" (" << *k
          << ") which is already assigned to \"" << exists->second->long_name
          << "\"" << std::endl;
      }
      else installed[*k] = it;
    }
  }
}

void bob::io::video::oformats_supported (std::map<std::string, AVOutputFormat*>& installed) {
  check_oformat_support(installed);
}

bool bob::io::video::oformat_is_supported (const std::string& name) {
  std::map<std::string, AVOutputFormat*> odict;
  bob::io::video::oformats_supported(odict);
  std::vector<std::string> names;
  bob::io::video::tokenize_csv(name.c_str(), names);
  for (auto k = names.begin(); k != names.end(); ++k) {
    if (odict.find(*k) != odict.end()) return true;
  }
  return false;
}

void bob::io::video::oformats_installed (std::map<std::string, AVOutputFormat*>& installed) {
  for (AVOutputFormat* it = av_oformat_next(0); it != 0; it = av_oformat_next(it) ) {
    if (!it->video_codec) continue;
    std::vector<std::string> names;
    bob::io::video::tokenize_csv(it->name, names);
    for (auto k = names.begin(); k != names.end(); ++k) {
      auto exists = installed.find(*k);
      if (exists != installed.end()) {
        bob::core::warn << "Not overriding output video format \""
          << it->long_name << "\" (" << *k
          << ") which is already assigned to \"" << exists->second->long_name
          << "\"" << std::endl;
      }
      else installed[*k] = it;
    }
  }
}

void bob::io::video::oformat_supported_codecs (const std::string& name,
    std::vector<const AVCodec*>& installed) {
  std::map<AVOutputFormat*, std::vector<const AVCodec*> > format2codec;
  define_output_support_map(format2codec);
  std::map<std::string, AVOutputFormat*> odict;
  bob::io::video::oformats_supported(odict);
  auto it = odict.find(name);
  if (it == odict.end()) {
    boost::format f("output format `%s' is not supported by this build");
    f % name;
    throw std::runtime_error(f.str());
  }
  installed = format2codec[it->second];
}

bool bob::io::video::oformat_supports_codec (const std::string& name,
    const std::string& codecname) {
  std::vector<const AVCodec*> codecs;
  oformat_supported_codecs(name, codecs);
  for (auto k=codecs.begin(); k!=codecs.end(); ++k) {
    if (codecname == (*k)->name) return true;
  }
  return false;
}

static std::string ffmpeg_error(int num) {
  static const int ERROR_SIZE = 1024;
  char message[ERROR_SIZE];
  int ok = av_strerror(num, message, ERROR_SIZE);
  if (ok < 0) {
    throw std::runtime_error("bob::io::video::av_strerror() failed to report - maybe you have a memory issue?");
  }
  return std::string(message);
}

static void deallocate_input_format_context(AVFormatContext* c) {
  avformat_close_input(&c);
}

boost::shared_ptr<AVFormatContext> bob::io::video::make_input_format_context(
    const std::string& filename) {

  AVFormatContext* retval = 0;

  int ok = avformat_open_input(&retval, filename.c_str(), 0, 0);
  if (ok != 0) {
    boost::format m("bob::io::video::avformat_open_input(filename=`%s') failed: ffmpeg reported %d == `%s'");
    m % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  // creates and protects the return value
  boost::shared_ptr<AVFormatContext> shared_retval(retval, std::ptr_fun(deallocate_input_format_context));

  // retrieve stream information, throws if cannot find it
  ok = avformat_find_stream_info(retval, 0);

  if (ok < 0) {
    boost::format m("bob::io::video::avformat_find_stream_info(filename=`%s') failed: ffmpeg reported %d == `%s'");
    m % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  return shared_retval;
}

int bob::io::video::find_video_stream(const std::string& filename, boost::shared_ptr<AVFormatContext> format_context) {

  int retval = av_find_best_stream(format_context.get(), AVMEDIA_TYPE_VIDEO,
      -1, -1, 0, 0);

  if (retval < 0) {
    boost::format m("bob::io::video::av_find_stream_info(`%s') failed: cannot find any video streams on this file - ffmpeg reports error %d == `%s'");
    m % filename % retval % ffmpeg_error(retval);
    throw std::runtime_error(m.str());
  }

  return retval;

}

AVCodec* bob::io::video::find_decoder(const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context, int stream_index) {

  AVCodec* retval = avcodec_find_decoder(format_context->streams[stream_index]->codecpar->codec_id);

  if (!retval) {
    boost::format m("bob::io::video::avcodec_find_decoder(0x%x) failed: cannot find a suitable codec to read stream %d of file `%s'");
    m % format_context->streams[stream_index]->codecpar->codec_id
      % stream_index % filename;
    throw std::runtime_error(m.str());
  }

  return retval;
}

static void deallocate_output_format_context(AVFormatContext* f) {
  if (f) avformat_free_context(f);
}

boost::shared_ptr<AVFormatContext> bob::io::video::make_output_format_context(
    const std::string& filename, const std::string& formatname) {

  AVFormatContext* retval;
  const char* filename_c = filename.c_str();
  const char* formatname_c = formatname.c_str();

  if (formatname.size() != 0) {
    int ok = avformat_alloc_output_context2(&retval, 0, formatname_c,
        filename_c);
    if (ok < 0) {
      boost::format m("bob::io::video::avformat_alloc_output_context2() failed: could not allocate output context based on format name == `%s', filename == `%s' - ffmpeg reports error %d == `%s'");
      m % formatname_c % filename_c % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
  }
  else {
    int ok = avformat_alloc_output_context2(&retval, 0, 0, filename_c);
    if (ok < 0) {
      boost::format m("bob::io::video::avformat_alloc_output_context2() failed: could not allocate output context based only on filename == `%s' - ffmpeg reports error %d == `%s'");
      m % formatname_c % filename_c % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
  }

  return boost::shared_ptr<AVFormatContext>(retval, std::ptr_fun(deallocate_output_format_context));
}

AVCodec* bob::io::video::find_encoder(const std::string& filename,
    boost::shared_ptr<AVFormatContext> fmtctxt, const std::string& codecname) {

  AVCodec* retval = 0;

  /* find the video encoder */
  if (codecname.size() != 0) {
    retval = avcodec_find_encoder_by_name(codecname.c_str());
    if (!retval) retval = try_find_through_decoder(codecname.c_str());
    if (!retval) {
      boost::format m("bob::io::video::avcodec_find_encoder_by_name(`%s') failed: could not find a suitable codec for encoding video file `%s' using the output format `%s' == `%s'");
      m % codecname % filename % fmtctxt->oformat->name
        % fmtctxt->oformat->long_name;
      throw std::runtime_error(m.str());
    }
  }
  else {
    if (fmtctxt->oformat->video_codec == AV_CODEC_ID_NONE) {
      boost::format m("could not identify codec for encoding video file `%s'; tried codec with name `%s' first and then tried output format's `%s' == `%s' video_codec entry, which was also null");
      m % filename % fmtctxt->oformat->name % fmtctxt->oformat->long_name;
      throw std::runtime_error(m.str());
    }
    retval = avcodec_find_encoder(fmtctxt->oformat->video_codec);

    if (!retval) {
      boost::format m("bob::io::video::avcodec_find_encoder(0x%x) failed: could not find encoder for codec with identifier for encoding video file `%s' using the output format `%s' == `%s'");
      m % fmtctxt->oformat->video_codec % filename
        % fmtctxt->oformat->name % fmtctxt->oformat->long_name;
      throw std::runtime_error(m.str());
    }
  }

  return retval;
}

static void deallocate_stream(AVStream* s) {
  //nop
}

boost::shared_ptr<AVStream> bob::io::video::make_stream(
    const std::string& filename,
    boost::shared_ptr<AVFormatContext> fmtctxt,
    AVCodec* codec) {

  AVStream* retval = avformat_new_stream(fmtctxt.get(), codec);

  if (!retval) {
    boost::format m("bob::io::video::avformat_new_stream(format=`%s' == `%s', codec=`%s[0x%x]' == `%s') failed: could not allocate video stream container for encoding video to file `%s'");
    m % fmtctxt->oformat->name % fmtctxt->oformat->long_name
      % codec->id % codec->name % codec->long_name % filename;
    throw std::runtime_error(m.str());
  }

  /* Some adjustments on the newly created stream */
  retval->id = fmtctxt->nb_streams-1; ///< this should be 0, normally

  return boost::shared_ptr<AVStream>(retval, std::ptr_fun(deallocate_stream));
}

static void deallocate_frame(AVFrame* f) {
  if (f) {
    if (f->data[0]) av_free(f->data[0]);
    av_free(f);
  }
}

boost::shared_ptr<AVFrame>
bob::io::video::make_frame(const std::string& filename,
    boost::shared_ptr<AVCodecContext> codec) {

  /* allocate and init a re-usable frame */
  AVFrame* retval = av_frame_alloc();
  if (!retval) {
    boost::format m("bob::io::video::av_frame_alloc() failed: cannot allocate frame to start encoding video file `%s'");
    m % filename;
    throw std::runtime_error(m.str());
  }

  retval->format = codec->pix_fmt;
  retval->width  = codec->width;
  retval->height = codec->height;

  int ok = av_image_alloc(retval->data, retval->linesize, codec->width, codec->height, codec->pix_fmt, 1);
  if (ok < 0) {
    av_free(retval);
    boost::format m("bob::io::video::av_image_alloc(data, linesize, width=%d, height=%d, 1) failed: cannot allocate frame/picture buffer start reading or writing video file `%s'");
    m % codec->width % codec->height % filename;
    throw std::runtime_error(m.str());
  }

  return boost::shared_ptr<AVFrame>(retval, std::ptr_fun(deallocate_frame));
}

static void deallocate_empty_frame(AVFrame* f) {
  if (f) av_free(f);
}

boost::shared_ptr<AVFrame> bob::io::video::make_empty_frame(const std::string& filename) {

  /* allocate and init a re-usable frame */
  AVFrame* retval = av_frame_alloc();
  if (!retval) {
    boost::format m("bob::io::video::av_frame_alloc() failed: cannot allocate frame to start encoding video file `%s'");
    m % filename;
    throw std::runtime_error(m.str());
  }

  return boost::shared_ptr<AVFrame>(retval, std::ptr_fun(deallocate_empty_frame));
}

static void deallocate_swscaler(SwsContext* s) {
  if (s) sws_freeContext(s);
}

boost::shared_ptr<SwsContext> bob::io::video::make_scaler
(const std::string& filename, boost::shared_ptr<AVCodecContext> ctxt,
 AVPixelFormat source_pixel_format, AVPixelFormat dest_pixel_format) {

  /* check pixel format before scaler gets allocated */
  if (source_pixel_format == AV_PIX_FMT_NONE) {
    boost::format m("bob::io::video::make_scaler() cannot be called with source_pixel_format == `AV_PIX_FMT_NONE'");
    throw std::runtime_error(m.str());
  }

  if (dest_pixel_format == AV_PIX_FMT_NONE) {
    boost::format m("bob::io::video::make_scaler() cannot be called with dest_pixel_format == `AV_PIX_FMT_NONE'");
    throw std::runtime_error(m.str());
  }

  /**
   * Initializes the software scaler (SWScale) so we can convert images to
   * the movie native format from RGB. You can define which kind of
   * interpolation to perform. Some options from libswscale are:
   * SWS_FAST_BILINEAR, SWS_BILINEAR, SWS_BICUBIC, SWS_X, SWS_POINT, SWS_AREA
   * SWS_BICUBLIN, SWS_GAUSS, SWS_SINC, SWS_LANCZOS, SWS_SPLINE
   */
  SwsContext* retval = sws_getContext(
      ctxt->width, ctxt->height, source_pixel_format,
      ctxt->width, ctxt->height, dest_pixel_format,
      SWS_BICUBIC, 0, 0, 0);

  if (!retval) {
    boost::format m("bob::io::video::sws_getContext(src_width=%d, src_height=%d, src_pix_format=`%s', dest_width=%d, dest_height=%d, dest_pix_format=`%s', flags=SWS_BICUBIC, 0, 0, 0) failed: cannot get software scaler context to start encoding or decoding video file `%s'");
    m % ctxt->width % ctxt->height % av_get_pix_fmt_name(source_pixel_format)
      % ctxt->width % ctxt->height % av_get_pix_fmt_name(dest_pixel_format)
      % filename;
    throw std::runtime_error(m.str());
  }
  return boost::shared_ptr<SwsContext>(retval, std::ptr_fun(deallocate_swscaler));
}

/**
 * Transforms from Bob's planar 8-bit RGB representation to whatever is
 * required by the FFmpeg encoder output context (peeked from the AVStream
 * object passed).
 */
static void image_to_context(const blitz::Array<uint8_t,3>& data,
    boost::shared_ptr<AVStream> stream,
    boost::shared_ptr<SwsContext> scaler,
    boost::shared_ptr<AVFrame> output_frame) {

  /** The ffmpeg sws scaler requires contiguous planes of data **/
  if (!bob::core::array::isCContiguous(data)) {
    throw std::runtime_error("sws_scale() check failed: cannot encode blitz::Array<uint8_t,3> in video stream - ffmpeg/libav requires contiguous color planes");
  }

  int width = stream->codecpar->width;
  int height = stream->codecpar->height;

  const uint8_t* datap = data.data();
  int plane_size = width * height;
  const uint8_t* planes[] = {datap+plane_size, datap+2*plane_size, datap, 0};
  int linesize[] = {width, width, width, 0};

  int ok = sws_scale(scaler.get(), planes, linesize, 0, height, output_frame->data, output_frame->linesize);
  if (ok < 0) {
    boost::format m("bob::io::video::sws_scale() failed: could not scale frame while encoding - ffmpeg reports error %d = `%s'");
    m % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }
}

/**
 * Transforms from Bob's planar 8-bit RGB representation to whatever is
 * required by the FFmpeg encoder output context (peeked from the AVStream
 * object passed).
 */
static void image_to_context(const blitz::Array<uint8_t,3>& data,
    boost::shared_ptr<AVStream> stream,
    boost::shared_ptr<SwsContext> scaler,
    boost::shared_ptr<AVFrame> output_frame,
    boost::shared_ptr<AVFrame> tmp_frame) {

  int width = stream->codecpar->width;
  int height = stream->codecpar->height;

  // replace data in the buffer frame by the pixmap to encode
  tmp_frame->linesize[0] = width*3;
  blitz::Array<uint8_t,3> ordered(tmp_frame->data[0],
		  blitz::shape(height, width, 3), blitz::neverDeleteData);
  ordered = const_cast<blitz::Array<uint8_t,3>&>(data).transpose(1,2,0); //organize for ffmpeg

  int ok = sws_scale(scaler.get(), tmp_frame->data, tmp_frame->linesize,
		  0, height, output_frame->data, output_frame->linesize);
  if (ok < 0) {
    boost::format m("bob::io::video::sws_scale() failed: could not scale frame while encoding - ffmpeg reports error %d = `%s'");
    m % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }
}

static void deallocate_codec_context(AVCodecContext* c) {
  int ok = avcodec_close(c);
  avcodec_free_context(&c);
  if (ok < 0) {
    boost::format m("bob::io::video::avcodec_close() failed: cannot close codec context (ffmpeg reports error %d = `%s')");
    m % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }
}

boost::shared_ptr<AVCodecContext> bob::io::video::make_decoder_context(
    const std::string& filename, AVStream* stream, AVCodec* codec) {

  AVCodecContext* retval = avcodec_alloc_context3(codec);

  /* copy the parameters from the demuxer to the codec context */
  int ok = avcodec_parameters_to_context(retval, stream->codecpar);
  if (ok < 0) {
    deallocate_codec_context(retval);
    boost::format m("bob::io::video::avcodec_parameters_to_context(codec=`%s'(0x%x) == `%s') failed: cannot open codec context to start reading video file `%s' - ffmpeg reports error %d == `%s'");
    m % codec->name % codec->id % codec->long_name % filename
      % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  // In the case we opened for writing, this should initialize the context
  ok = avcodec_open2(retval, codec, 0);
  if (ok < 0) {
    boost::format m("bob::io::video::avcodec_open2(codec=`%s'(0x%x) == `%s') failed: cannot open codec context to start writing video file `%s' - ffmpeg reports error %d == `%s'");
    m % codec->name % codec->id % codec->long_name % filename
      % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  return boost::shared_ptr<AVCodecContext>(retval,
      std::ptr_fun(deallocate_codec_context));
}

boost::shared_ptr<AVCodecContext> bob::io::video::make_encoder_context(
    const std::string& filename, AVFormatContext* fmtctxt, AVStream* stream,
    AVCodec* codec, size_t height, size_t width, double framerate, double
    bitrate, size_t gop) {

  AVCodecContext* retval = avcodec_alloc_context3(codec);

  /* Set user parameters */
  retval->bit_rate = bitrate;

  /* Resolution must be a multiple of two. */
  if (height%2 != 0 || height == 0 || width%2 != 0 || width == 0) {
    boost::format m("ffmpeg only accepts video height and width if they are, both, multiples of two, but you supplied %d x %d while configuring video output for file `%s' - correct these and re-run");
    m % height % width % filename;
    deallocate_codec_context(retval);
    throw std::runtime_error(m.str());
  }

  retval->width    = width;
  retval->height   = height;

  /* timebase: This is the fundamental unit of time (in seconds) in terms
   * of which frame timestamps are represented. For fixed-fps content,
   * timebase should be 1/framerate and timestamp increments should be
   * identical to 1. */
  stream->time_base = av_make_q(1, framerate);
  retval->time_base = stream->time_base;
  retval->framerate = av_make_q(framerate, 1);

  retval->gop_size      = gop; /* emit one intra frame every X at most */
  retval->pix_fmt       = AV_PIX_FMT_YUV420P;

  // checks if the wanted pixel format can be digested by codec
  if (codec->pix_fmts && codec->pix_fmts[0] != AV_PIX_FMT_NONE) {
    //override with preference for native formats supported by codec
    retval->pix_fmt = codec->pix_fmts[0];
  }

  if (retval->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
    /* just for testing, we also add B frames */
    retval->max_b_frames = 2;
  }

  if (retval->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
    /* Needed to avoid using macroblocks in which some coeffs overflow.
     * This does not happen with normal video, it just happens here as
     * the motion of the chroma plane does not match the luma plane. */
    retval->mb_decision = 2;
  }

  if (retval->codec_id == AV_CODEC_ID_MJPEG) {
    /* set jpeg color range */
    retval->color_range = AVCOL_RANGE_JPEG;
  }

  /* Some formats want stream headers to be separate. */
  if (fmtctxt->oformat->flags & AVFMT_GLOBALHEADER) {
    retval->flags |= CODEC_FLAG_GLOBAL_HEADER;
  }

  // In the case we opened for writing, this should initialize the context
  int ok = avcodec_open2(retval, codec, 0);
  if (ok < 0) {
    boost::format m("bob::io::video::avcodec_open2(codec=`%s'(0x%x) == `%s') failed: cannot open codec context to start reading or writing video file `%s' - ffmpeg reports error %d == `%s'");
    m % codec->name % codec->id % codec->long_name % filename
      % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  /* copy the parameters from the codec context to the muxer */
  ok = avcodec_parameters_from_context(stream->codecpar, retval);
  if (ok < 0) {
    deallocate_codec_context(retval);
    boost::format m("bob::io::video::avcodec_parameters_from_context(codec=`%s'(0x%x) == `%s') failed: cannot open codec context to start reading or writing video file `%s' - ffmpeg reports error %d == `%s'");
    m % codec->name % codec->id % codec->long_name % filename
      % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  return boost::shared_ptr<AVCodecContext>(retval,
      std::ptr_fun(deallocate_codec_context));
}

void bob::io::video::open_output_file(const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context) {

  /* open the output file, if needed */
  if (!(format_context->oformat->flags & AVFMT_NOFILE)) {
    if (avio_open(&format_context->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0) {
      boost::format m("bob::io::video::avio_open(filename=`%s', AVIO_FLAG_WRITE) failed: cannot open output file for writing");
      m % filename.c_str();
      throw std::runtime_error(m.str());
    }
  }

  /* Write the stream header, if any. */
  int error = avformat_write_header(format_context.get(), 0);
  if (error < 0) {
    boost::format m("bob::io::video::avformat_write_header(filename=`%s') failed: cannot write header to output file for some reason - ffmpeg reports error %d == `%s'");
    m % filename.c_str() % error % ffmpeg_error(error);
    throw std::runtime_error(m.str());
  }
}

void bob::io::video::close_output_file(const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context) {

  /* Write the trailer, if any. The trailer must be written before you
   * close the CodecContexts open when you wrote the header; otherwise
   * av_write_trailer() may try to use memory that was freed on
   * av_codec_close(). */
  int error = av_write_trailer(format_context.get());
  if (error < 0) {
    boost::format m("bob::io::video::av_write_trailer(filename=`%s') failed: cannot write trailer to output file for some reason - ffmpeg reports error %d == `%s')");
    m % filename % error % ffmpeg_error(error);
    throw std::runtime_error(m.str());
  }

  /* Closes the output file */
  avio_closep(&format_context->pb);

}

static AVPacket* allocate_packet() {
  AVPacket* retval = av_packet_alloc();
  if (!retval) {
    boost::format m("bob::io::video::av_packet_alloc() failed to allocate a new packet");
  }
  av_init_packet(retval);
  retval->data = 0;
  retval->size = 0;
  return retval;
}

static void deallocate_packet(AVPacket* p) {
  av_packet_free(&p);
}

static boost::shared_ptr<AVPacket> make_packet() {
  return boost::shared_ptr<AVPacket>(allocate_packet(),
      std::ptr_fun(deallocate_packet));
}

static void write_packet_to_stream(const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context,
    boost::shared_ptr<AVStream> stream,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<AVPacket> pkt) {

  //effectively writes the encoded packet to the stream
  /* writes the compressed frame to the media file. */
  pkt->stream_index = stream->index;
  pkt->duration = av_rescale_q(1, codec_context->time_base,
      stream->time_base);
  int ok = av_interleaved_write_frame(format_context.get(), pkt.get());
  if (ok && (ok != AVERROR(EINVAL))) {
    boost::format m("bob::io::video::av_interleaved_write_frame() failed: failed to write video frame while encoding file `%s' - ffmpeg reports error %d == `%s'");
    m % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

}

void bob::io::video::flush_encoder (const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context,
    boost::shared_ptr<AVStream> stream,
    boost::shared_ptr<AVCodecContext> codec_context) {

  if (format_context->oformat->flags & AVFMT_RAWPICTURE) return;

  boost::shared_ptr<AVPacket> pkt = make_packet();

  int ok = avcodec_send_frame(codec_context.get(), 0); //flush
  if (ok < 0) {
    boost::format m("bob::io::video::avcodec_send_frame() failed: failed to encode video frame while flushing file `%s' - ffmpeg reports error %d == `%s'");
    m % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  while (ok >= 0) {
    ok = avcodec_receive_packet(codec_context.get(), pkt.get());
    if (ok < 0 && ok != AVERROR(EAGAIN) && ok != AVERROR_EOF) {
      boost::format m("bob::io::video::avcodec_receive_packet() failed: failed to flush encoder while writing to file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
    if (ok >= 0 && pkt->size != 0) { //write last packet to the stream
      write_packet_to_stream(filename, format_context, stream, codec_context,
          pkt);
    }
    av_packet_unref(pkt.get());
  }

}

void bob::io::video::write_video_frame (const blitz::Array<uint8_t,3>& data,
    const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context,
    boost::shared_ptr<AVStream> stream,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<AVFrame> context_frame,
    boost::shared_ptr<AVFrame> tmp_frame,
    boost::shared_ptr<SwsContext> swscaler) {

  if (tmp_frame)
    image_to_context(data, stream, swscaler, context_frame, tmp_frame);
  else
    image_to_context(data, stream, swscaler, context_frame);

  boost::shared_ptr<AVPacket> pkt = make_packet();

  if (format_context->oformat->flags & AVFMT_RAWPICTURE) {
    /* Raw video case - directly store the picture in the packet */
    pkt->flags        |= AV_PKT_FLAG_KEY;
    pkt->data          = context_frame->data[0];
    pkt->size          = sizeof(AVPicture);
    write_packet_to_stream(filename, format_context, stream, codec_context,
        pkt);
  }

  else {

    int ok = avcodec_send_frame(codec_context.get(), context_frame.get());
    if (ok < 0) {
      boost::format m("bob::io::video::avcodec_send_frame() failed: failed to encode video frame while writing to file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }

    while (ok >= 0) {
      ok = avcodec_receive_packet(codec_context.get(), pkt.get());
      if (ok == AVERROR(EAGAIN)) { //ready for new frame
        break;
      }
      else if (ok == AVERROR_EOF) {
        boost::format m("bob::io::video::avcodec_receive_packet() failed: failed to encode video frame while writing to file `%s' - ffmpeg reports error %d == `%s' (file is already closed)");
        m % filename % ok % ffmpeg_error(ok);
        throw std::runtime_error(m.str());
      }
      else if (ok < 0) { //real error condition
        boost::format m("bob::io::video::avcodec_receive_packet() failed: failed to encode video frame while writing to file `%s' - ffmpeg reports error %d == `%s'");
        m % filename % ok % ffmpeg_error(ok);
        throw std::runtime_error(m.str());
      }
      else if (pkt->size > 0) { //write last packet to the stream
        write_packet_to_stream(filename, format_context, stream, codec_context,
            pkt);
      }
      av_packet_unref(pkt.get());
    }

  }

  //sets the output frame PTS [Note: presentation timestamp in time_base
  //units (time when frame should be shown to user) If AV_NOPTS_VALUE then
  //frame_rate = 1/time_base will be assumed].
  context_frame->pts += av_rescale_q(1, codec_context->time_base,
      stream->time_base);

}

// The flush packet is a non-NULL packet with size 0 and data NULL
static int decode(AVCodecContext *avctx, AVFrame *frame, int *got_frame,
    AVPacket *pkt)
{
  int ret;

  *got_frame = 0;

  if (pkt) {
    ret = avcodec_send_packet(avctx, pkt);
    // In particular, we don't expect AVERROR(EAGAIN), because we read all
    // decoded frames with avcodec_receive_frame() until done.
    if (ret < 0)
      return ret == AVERROR_EOF ? 0 : ret;
  }

  ret = avcodec_receive_frame(avctx, frame);
  if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF)
    return ret;
  if (ret >= 0)
    *got_frame = 1;

  return 0;
}


static int decode_frame (const std::string& filename, int current_frame,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<SwsContext> scaler,
    boost::shared_ptr<AVFrame> context_frame, uint8_t* data,
    boost::shared_ptr<AVPacket> pkt,
    int& got_frame, bool throw_on_error) {

  // In this call, 3 things can happen:
  //
  // 1. if ok < 0, an error has been detected
  // 2. if ok >=0, something was read from the file, correctly. In this
  // condition, **only* if "got_frame" == 1, a frame is ready to be decoded.
  //
  // It is **not** an error that ok is >= 0 and got_frame == 0. This, in fact,
  // happens often with recent versions of ffmpeg.

  int ok = decode(codec_context.get(), context_frame.get(), &got_frame,
      pkt.get());
  if (ok < 0 && throw_on_error) {
    boost::format m("bob::io::video::avcodec_decode_video/2() failed: could not decode frame %d of file `%s' - ffmpeg reports error %d == `%s'");
    m % current_frame % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  if (got_frame) {

    // In this case, we call the software scaler to decode the frame data.
    // Normally, this means converting from planar YUV420 into packed RGB.

    uint8_t* planes[] = {data, 0};
    int linesize[] = {3*codec_context->width, 0};

    int conv_height = sws_scale(scaler.get(), context_frame->data,
        context_frame->linesize, 0, codec_context->height, planes, linesize);

    if (conv_height < 0) {

      if (throw_on_error) {
        boost::format m("bob::io::video::sws_scale() failed: could not scale frame %d of file `%s' - ffmpeg reports error %d");
        m % current_frame % filename % conv_height;
        throw std::runtime_error(m.str());
      }

      return -1;
    }

  }

  return ok;
}

bool bob::io::video::read_video_frame (const std::string& filename,
    int current_frame, int stream_index,
    boost::shared_ptr<AVFormatContext> format_context,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<SwsContext> swscaler,
    boost::shared_ptr<AVFrame> context_frame, uint8_t* data,
    bool throw_on_error) {

  boost::shared_ptr<AVPacket> pkt = make_packet();

  int ok = 0;
  int got_frame = 0;

  while ((ok = av_read_frame(format_context.get(), pkt.get())) >= 0) {
    if (pkt->stream_index == stream_index) {
      decode_frame(filename, current_frame, codec_context,
          swscaler, context_frame, data, pkt, got_frame,
          throw_on_error);
    }
    av_packet_unref(pkt.get());
    if (got_frame) return true; //break loop
  }

  if (ok < 0 && ok != (int)AVERROR_EOF) {
    if (throw_on_error) {
      boost::format m("bob::io::video::av_read_frame() failed: on file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
    else return false;
  }

  // it is the end of the file
  pkt->data = NULL;
  pkt->size = 0;
  //N.B.: got_frame == 0
  const unsigned int MAX_FLUSH_ITERATIONS = 128;
  unsigned int iteration_counter = MAX_FLUSH_ITERATIONS;
  do {
    if (pkt->stream_index == stream_index) {
      decode_frame(filename, current_frame, codec_context,
          swscaler, context_frame, data, pkt, got_frame,
          throw_on_error);
      --iteration_counter;
      if (iteration_counter == 0) {
        if (throw_on_error) {
          boost::format m("bob::io::video::decode_frame() failed: on file `%s' - I've been iterating for over %d times and I cannot find a new frame: this codec (%s) must be buggy!");
          m % filename % MAX_FLUSH_ITERATIONS % codec_context->codec->name;
          throw std::runtime_error(m.str());
        }
        break;
      }
    }
    else break;
  } while (got_frame == 0);

  return true;
}

static int dummy_decode_frame (const std::string& filename, int current_frame,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<AVFrame> context_frame,
    boost::shared_ptr<AVPacket> pkt,
    int& got_frame, bool throw_on_error) {

  // In this call, 3 things can happen:
  //
  // 1. if ok < 0, an error has been detected
  // 2. if ok >=0, something was read from the file, correctly. In this
  // condition, **only* if "got_frame" == 1, a frame is ready to be decoded.
  //
  // It is **not** an error that ok is >= 0 and got_frame == 0. This, in fact,
  // happens often with recent versions of ffmpeg.

  int ok = decode(codec_context.get(), context_frame.get(), &got_frame,
      pkt.get());
  if (ok < 0 && throw_on_error) {
    boost::format m("bob::io::video::avcodec_decode_video/2() failed: could not skip frame %d of file `%s' - ffmpeg reports error %d == `%s'");
    m % current_frame % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  return ok;
}

bool bob::io::video::skip_video_frame (const std::string& filename,
    int current_frame, int stream_index,
    boost::shared_ptr<AVFormatContext> format_context,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<AVFrame> context_frame,
    bool throw_on_error) {

  boost::shared_ptr<AVPacket> pkt = make_packet();

  int ok = 0;
  int got_frame = 0;

  while ((ok = av_read_frame(format_context.get(), pkt.get())) >= 0) {
    if (pkt->stream_index == stream_index) {
      dummy_decode_frame(filename, current_frame, codec_context,
          context_frame, pkt, got_frame, throw_on_error);
    }
    av_packet_unref(pkt.get());
    if (got_frame) return true; //break loop
  }

  if (ok < 0 && ok != (int)AVERROR_EOF) {
    if (throw_on_error) {
      boost::format m("bob::io::video::av_read_frame() failed: on file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
    else return false;
  }

  // it is the end of the file
  pkt->data = NULL;
  pkt->size = 0;
  //N.B.: got_frame == 0
  const unsigned int MAX_FLUSH_ITERATIONS = 128;
  unsigned int iteration_counter = MAX_FLUSH_ITERATIONS;
  do {
    if (pkt->stream_index == stream_index) {
      dummy_decode_frame(filename, current_frame, codec_context,
          context_frame, pkt, got_frame, throw_on_error);
      --iteration_counter;
      if (iteration_counter == 0) {
        if (throw_on_error) {
          boost::format m("bob::io::video::decode_frame() failed: on file `%s' - I've been iterating for over %d times and I cannot find a new frame: this codec (%s) must be buggy!");
          m % filename % MAX_FLUSH_ITERATIONS % codec_context->codec->name;
          throw std::runtime_error(m.str());
        }
        break;
      }
    }
    else break;
  } while (got_frame == 0);

  return true;
}
