#!/bin/bash

echo "Checking GLib build dependencies..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
python_package_exists() {
    python3 -c "import $1" >/dev/null 2>&1
}

missing_deps=()

# Check for essential tools
if ! command_exists python3; then
    missing_deps+=("python3")
fi

if ! command_exists pip3; then
    missing_deps+=("python3-pip")
fi

if ! command_exists git; then
    missing_deps+=("git")
fi

if ! command_exists wget; then
    missing_deps+=("wget")
fi

if ! command_exists pkg-config; then
    missing_deps+=("pkg-config")
fi

# Check for meson and ninja
meson_available=false
ninja_available=false

if command_exists meson; then
    meson_available=true
    echo "✓ meson found: $(meson --version)"
else
    echo "✗ meson not found"
fi

if command_exists ninja; then
    ninja_available=true
    echo "✓ ninja found"
else
    echo "✗ ninja not found"
fi

# Print missing system dependencies
if [ ${#missing_deps[@]} -gt 0 ]; then
    echo ""
    echo "Missing system dependencies:"
    for dep in "${missing_deps[@]}"; do
        echo "  - $dep"
    done
    echo ""
    echo "These can usually be installed with your package manager."
    echo "However, since you don't have root access, you might need to:"
    echo "1. Ask your system administrator to install them"
    echo "2. Use a user-local package manager like conda/miniconda"
    echo "3. Compile from source in your home directory"
fi

# Check Python build tools
if ! $meson_available || ! $ninja_available; then
    echo ""
    echo "Installing meson and ninja via pip3 (user install):"
    echo "  pip3 install --user meson ninja"
    echo ""
    echo "Make sure ~/.local/bin is in your PATH:"
    echo "  export PATH=\$HOME/.local/bin:\$PATH"
    echo "  # Add the above line to your ~/.bashrc for persistence"
fi

# Check for development libraries that might be needed
echo ""
echo "Note: GLib build has been configured to disable optional dependencies"
echo "that might not be available without root access (libmount, selinux, etc.)"

if [ ${#missing_deps[@]} -eq 0 ] && $meson_available && $ninja_available; then
    echo ""
    echo "✓ All dependencies are available! You can run ./setup_glib.sh"
else
    echo ""
    echo "Please install the missing dependencies first."
fi