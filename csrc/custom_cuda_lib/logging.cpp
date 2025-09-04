#include <cstdarg>
#include <cstdio>

#include "logging.hpp"

namespace gvm {

// Global log level
static LogLevel current_log_level = LogLevel::INFO;

static const char *level_names[] = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"};

static const char *level_colors[] = {
    "\033[36m", // Cyan for DEBUG
    "\033[32m", // Green for INFO
    "\033[33m", // Yellow for WARN
    "\033[31m", // Red for ERROR
    "\033[35m"  // Magenta for FATAL
};

static const char *reset_color = "\033[0m";

// Extract filename from path
static const char *get_filename(const char *path) {
  const char *filename = path;
  for (const char *p = path; *p; p++) {
    if (*p == '/') {
      filename = p + 1;
    }
  }
  return filename;
}

void log(LogLevel level, const char *file, int line, const char *func,
         const char *message) {
  if (level < current_log_level) {
    return;
  }

  const bool verbose = false;
  const char *filename = get_filename(file);
  const char *level_name = level_names[static_cast<int>(level)];
  const char *color = level_colors[static_cast<int>(level)];

  if (verbose) {
    std::fprintf(stderr, "%s[%s]%s[GVM][%s:%d:%s()]: %s\n", color, level_name,
                 reset_color, filename, line, func, message);
  } else {
    std::fprintf(stderr, "%s[%s]%s[GVM][%s:%d]: %s\n", color, level_name,
                 reset_color, filename, line, message);
  }
}

void set_log_level(LogLevel level) { current_log_level = level; }

LogLevel get_log_level() { return current_log_level; }

void log_printf(LogLevel level, const char *file, int line, const char *func,
                const char *format, ...) {
  if (level < current_log_level) {
    return;
  }

  // Format the message using printf-style formatting
  char buffer[4096]; // Should be sufficient for most log messages
  va_list args;
  va_start(args, format);
  std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);

  // Use the existing log function
  log(level, file, line, func, buffer);
}

} // namespace gvm