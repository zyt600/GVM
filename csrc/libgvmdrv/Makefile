# Makefile for gvmdrv library

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -fPIC
DEBUG_CXXFLAGS = -std=c++17 -Wall -Wextra -g -DDEBUG -fPIC

# Library name and version
LIBRARY_NAME = libgvmdrv
LIBRARY_VERSION = 1.0.0
SONAME = $(LIBRARY_NAME).so.1
SHARED_LIB = $(LIBRARY_NAME).so.$(LIBRARY_VERSION)
STATIC_LIB = $(LIBRARY_NAME).a

# Build directories
BUILD_DIR = build
TEST_BUILD_DIR = $(BUILD_DIR)/test

# Source files
SOURCES = gvmdrv.cpp gvmdrv_log.cpp
OBJECTS = $(addprefix $(BUILD_DIR)/, $(SOURCES:.cpp=.o))

# Test files
TEST_SOURCES = test/test.cpp test/test_basic.cpp test/test_advanced.cpp
TEST_OBJECTS = $(addprefix $(BUILD_DIR)/, $(TEST_SOURCES:.cpp=.o))
TEST_EXECUTABLES = $(addprefix $(TEST_BUILD_DIR)/, $(notdir $(TEST_SOURCES:.cpp=)))

# Header files
HEADERS = gvmdrv.h gvmdrv_ioctl.h gvmdrv_log.h

# Installation directories
PREFIX = /usr/local
LIBDIR = $(PREFIX)/lib
INCLUDEDIR = $(PREFIX)/include
PKGCONFIGDIR = $(PREFIX)/lib/pkgconfig

# Default target
all: shared static

# Debug build
debug: CXXFLAGS = $(DEBUG_CXXFLAGS)
debug: shared static

# Create build directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_BUILD_DIR):
	mkdir -p $(TEST_BUILD_DIR)

# Shared library
shared: $(BUILD_DIR)/$(SHARED_LIB)

$(BUILD_DIR)/$(SHARED_LIB): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) -shared -Wl,-soname,$(SONAME) -o $@ $^
	cd $(BUILD_DIR) && ln -sf $(SHARED_LIB) $(LIBRARY_NAME).so
	cd $(BUILD_DIR) && ln -sf $(LIBRARY_NAME).so $(SONAME)

# Static library
static: $(BUILD_DIR)/$(STATIC_LIB)

$(BUILD_DIR)/$(STATIC_LIB): $(OBJECTS) | $(BUILD_DIR)
	ar rcs $@ $^

# Object files
$(BUILD_DIR)/%.o: %.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Test programs
test: $(TEST_EXECUTABLES)

# Individual test executables
$(TEST_BUILD_DIR)/test: $(BUILD_DIR)/test/test.o $(BUILD_DIR)/$(SHARED_LIB) | $(TEST_BUILD_DIR)
	$(CXX) -o $@ $< -L$(BUILD_DIR) -lgvmdrv

$(TEST_BUILD_DIR)/test_basic: $(BUILD_DIR)/test/test_basic.o $(BUILD_DIR)/$(SHARED_LIB) | $(TEST_BUILD_DIR)
	$(CXX) -o $@ $< -L$(BUILD_DIR) -lgvmdrv

$(TEST_BUILD_DIR)/test_advanced: $(BUILD_DIR)/test/test_advanced.o $(BUILD_DIR)/$(SHARED_LIB) | $(TEST_BUILD_DIR)
	$(CXX) -o $@ $< -L$(BUILD_DIR) -lgvmdrv

# Test object files
$(BUILD_DIR)/test/%.o: test/%.cpp $(HEADERS) | $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/test
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Installation
install: shared static
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(INCLUDEDIR)
	install -m 644 $(BUILD_DIR)/$(SHARED_LIB) $(DESTDIR)$(LIBDIR)/
	install -m 644 $(BUILD_DIR)/$(STATIC_LIB) $(DESTDIR)$(LIBDIR)/
	install -m 644 $(HEADERS) $(DESTDIR)$(INCLUDEDIR)/
	cd $(DESTDIR)$(LIBDIR) && ln -sf $(SHARED_LIB) $(LIBRARY_NAME).so
	cd $(DESTDIR)$(LIBDIR) && ln -sf $(LIBRARY_NAME).so $(SONAME)
	ldconfig

# Uninstall
uninstall:
	rm -f $(DESTDIR)$(LIBDIR)/$(LIBRARY_NAME).so*
	rm -f $(DESTDIR)$(LIBDIR)/$(STATIC_LIB)
	rm -f $(DESTDIR)$(INCLUDEDIR)/gvmdrv*.h
	ldconfig

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Distclean (remove everything)
distclean: clean
	rm -f *.o *.a *.so*

# Show help
help:
	@echo "Available targets:"
	@echo "  all        - Build shared and static libraries (default)"
	@echo "  shared     - Build shared library only"
	@echo "  static     - Build static library only"
	@echo "  debug      - Build with debug flags"
	@echo "  test       - Build all test programs in $(TEST_BUILD_DIR)/"
	@echo "  test-basic - Build and run basic tests"
	@echo "  test-advanced - Build and run advanced tests"
	@echo "  test-all   - Build and run all tests"
	@echo "  install    - Install library and headers"
	@echo "  uninstall  - Remove installed files"
	@echo "  clean      - Remove build/ directory"
	@echo "  distclean  - Remove all generated files"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Build output:"
	@echo "  Libraries: $(BUILD_DIR)/"
	@echo "  Test executables: $(TEST_BUILD_DIR)/"

# Convenience test targets
test-basic: $(TEST_BUILD_DIR)/test_basic
	@echo "Running basic tests..."
	@$(TEST_BUILD_DIR)/test_basic

test-advanced: $(TEST_BUILD_DIR)/test_advanced
	@echo "Running advanced tests..."
	@$(TEST_BUILD_DIR)/test_advanced

test-all: test
	@echo "Running all tests..."
	@for test_exe in $(TEST_EXECUTABLES); do \
		echo "Running $$test_exe..."; \
		$$test_exe; \
	done

# Phony targets
.PHONY: all debug shared static test test-basic test-advanced test-all install uninstall clean distclean help