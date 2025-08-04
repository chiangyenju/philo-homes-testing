#!/bin/bash
# Quick install/reinstall script for macOS

ADDON_NAME="philo_interior_addon"
BLENDER_VERSION="4.4"  # Change this to your Blender version
ADDON_DIR="$HOME/Library/Application Support/Blender/$BLENDER_VERSION/scripts/addons"
SOURCE_DIR="$(dirname "$0")/$ADDON_NAME"

# Remove old installation and cache
echo "Removing old installation..."
rm -rf "$ADDON_DIR/$ADDON_NAME"
find "$HOME/Library/Application Support/Blender" -name "__pycache__" -path "*/$ADDON_NAME/*" -exec rm -rf {} + 2>/dev/null

# Install
echo "Installing to: $ADDON_DIR/$ADDON_NAME"
mkdir -p "$ADDON_DIR"
cp -r "$SOURCE_DIR" "$ADDON_DIR/"

echo "Done! Now:"
echo "1. Restart Blender"
echo "2. Enable 'Philo Interior Generator' in Preferences > Add-ons"
echo ""
echo "Or paste this in Blender's Python Console after restart:"
echo "import bpy; bpy.ops.preferences.addon_enable(module='philo_interior_addon')"