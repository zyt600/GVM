#!/bin/bash

set -e

GLIB_VERSION="2.78.0"
GLIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up GLib ${GLIB_VERSION} in ${GLIB_DIR}"

# Check if meson is available
if ! command -v meson &> /dev/null; then
    echo "ERROR: meson is required to build GLib"
    echo "Install it with: pip3 install meson ninja"
    exit 1
fi

# Check if ninja is available
if ! command -v ninja &> /dev/null; then
    echo "ERROR: ninja is required to build GLib"
    echo "Install it with: pip3 install ninja (or use your package manager)"
    exit 1
fi

cd "${GLIB_DIR}"

# Download GLib source if not already present
if [ ! -f "meson.build" ]; then
    echo "Downloading GLib ${GLIB_VERSION}..."

    # Try to download tarball first
    if wget "https://download.gnome.org/sources/glib/2.78/glib-${GLIB_VERSION}.tar.xz" -O glib.tar.xz; then
        tar -xf glib.tar.xz --strip-components=1
        rm glib.tar.xz
        echo "Downloaded and extracted tarball successfully"
    else
        echo "Tarball download failed, trying to clone from git..."
        if git clone https://gitlab.gnome.org/GNOME/glib.git tmp_glib 2>/dev/null; then
            echo "Cloned from git successfully"
            cd tmp_glib
            git checkout ${GLIB_VERSION} || git checkout main
            cd ..
            mv tmp_glib/* .
            mv tmp_glib/.* . 2>/dev/null || true
            rmdir tmp_glib
        else
            echo "ERROR: Failed to download GLib source via both tarball and git."
            exit 1
        fi
    fi
fi

# Create install directory
mkdir -p install

# Configure build
echo "Configuring build..."
meson setup builddir --prefix="${GLIB_DIR}/install" \
    --buildtype=release \
    -Dlibmount=disabled \
    -Dselinux=disabled \
    -Dxattr=false \
    -Dlibelf=disabled \
    -Dsysprof=disabled

# Build
echo "Building GLib..."
meson compile -C builddir

# Install
echo "Installing GLib to ${GLIB_DIR}/install..."
meson install -C builddir

echo "GLib setup complete!"