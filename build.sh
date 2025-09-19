#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit  # Exit on error

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Build completed successfully!"