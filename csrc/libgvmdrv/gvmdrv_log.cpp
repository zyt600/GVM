#include "gvmdrv_log.h"
#include <stdio.h>
#include <string.h>

namespace gvmdrv {

// Global log level
static LogLevel current_log_level = LOG_WARN;

// Log level names
static const char *level_names[] = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"};

// Log level colors (for terminal output)
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

} // namespace gvmdrv

extern "C" {

void gvmdrv_log(LogLevel level, const char *file, int line, const char *func,
                const char *message) {
  if (level < gvmdrv::current_log_level) {
    return;
  }

  const bool verbose = false;
  const char *filename = gvmdrv::get_filename(file);
  const char *level_name = gvmdrv::level_names[level];
  const char *color = gvmdrv::level_colors[level];

  if (verbose) {
    fprintf(stderr, "%s[%s]%s[GVM][%s:%d:%s()]: %s\n", color, level_name,
            gvmdrv::reset_color, filename, line, func, message);
  } else {
    fprintf(stderr, "%s[%s]%s[GVM][%s:%d]: %s\n", color, level_name,
            gvmdrv::reset_color, filename, line, message);
  }
}

void gvmdrv_set_log_level(LogLevel level) { gvmdrv::current_log_level = level; }

LogLevel gvmdrv_get_log_level(void) { return gvmdrv::current_log_level; }
}
