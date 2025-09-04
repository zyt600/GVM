#pragma once

#include <sstream>
#include <string>

namespace gvm {

// Log levels
enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3, FATAL = 4 };

// Core logging functions
void log(LogLevel level, const char *file, int line, const char *func,
         const char *message);
void set_log_level(LogLevel level);
LogLevel get_log_level();

// Printf-style logging function
void log_printf(LogLevel level, const char *file, int line, const char *func,
                const char *format, ...);

// Stream-style logging support
class LogStream {
private:
  std::ostringstream stream_;
  LogLevel level_;
  const char *file_;
  int line_;
  const char *func_;

public:
  LogStream(LogLevel level, const char *file, int line, const char *func)
      : level_(level), file_(file), line_(line), func_(func) {}

  ~LogStream() { log(level_, file_, line_, func_, stream_.str().c_str()); }

  template <typename T> LogStream &operator<<(const T &value) {
    stream_ << value;
    return *this;
  }

  // Special handling for std::endl and other manipulators
  LogStream &operator<<(std::ostream &(*manip)(std::ostream &)) {
    stream_ << manip;
    return *this;
  }
};

} // namespace gvm

// Basic string logging macros
#define GVM_LOG_DEBUG(msg)                                                     \
  gvm::log(gvm::LogLevel::DEBUG, __FILE__, __LINE__, __func__, msg)
#define GVM_LOG_INFO(msg)                                                      \
  gvm::log(gvm::LogLevel::INFO, __FILE__, __LINE__, __func__, msg)
#define GVM_LOG_WARN(msg)                                                      \
  gvm::log(gvm::LogLevel::WARN, __FILE__, __LINE__, __func__, msg)
#define GVM_LOG_ERROR(msg)                                                     \
  gvm::log(gvm::LogLevel::ERROR, __FILE__, __LINE__, __func__, msg)
#define GVM_LOG_FATAL(msg)                                                     \
  gvm::log(gvm::LogLevel::FATAL, __FILE__, __LINE__, __func__, msg)

// Printf-style logging macros
#define GVM_LOG_DEBUG_F(fmt, ...)                                              \
  gvm::log_printf(gvm::LogLevel::DEBUG, __FILE__, __LINE__, __func__, fmt,     \
                  ##__VA_ARGS__)
#define GVM_LOG_INFO_F(fmt, ...)                                               \
  gvm::log_printf(gvm::LogLevel::INFO, __FILE__, __LINE__, __func__, fmt,      \
                  ##__VA_ARGS__)
#define GVM_LOG_WARN_F(fmt, ...)                                               \
  gvm::log_printf(gvm::LogLevel::WARN, __FILE__, __LINE__, __func__, fmt,      \
                  ##__VA_ARGS__)
#define GVM_LOG_ERROR_F(fmt, ...)                                              \
  gvm::log_printf(gvm::LogLevel::ERROR, __FILE__, __LINE__, __func__, fmt,     \
                  ##__VA_ARGS__)
#define GVM_LOG_FATAL_F(fmt, ...)                                              \
  gvm::log_printf(gvm::LogLevel::FATAL, __FILE__, __LINE__, __func__, fmt,     \
                  ##__VA_ARGS__)

// Stream-style logging macros (simplified syntax)
#define GVM_LOG_DEBUG_S                                                        \
  gvm::LogStream(gvm::LogLevel::DEBUG, __FILE__, __LINE__, __func__)
#define GVM_LOG_INFO_S                                                         \
  gvm::LogStream(gvm::LogLevel::INFO, __FILE__, __LINE__, __func__)
#define GVM_LOG_WARN_S                                                         \
  gvm::LogStream(gvm::LogLevel::WARN, __FILE__, __LINE__, __func__)
#define GVM_LOG_ERROR_S                                                        \
  gvm::LogStream(gvm::LogLevel::ERROR, __FILE__, __LINE__, __func__)
#define GVM_LOG_FATAL_S                                                        \
  gvm::LogStream(gvm::LogLevel::FATAL, __FILE__, __LINE__, __func__)
