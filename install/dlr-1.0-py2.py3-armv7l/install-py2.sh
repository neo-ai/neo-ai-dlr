#!/bin/bash

set -e

check_command () {
  if command -v $1 &>/dev/null; then
    echo $1 is installed
  else
    echo Error: $1 is not installed
    exit -1
  fi
}

# Check required commnads exist
check_command python
check_command easy_install
check_command pip

# Install numpy
echo
echo "Installing numpy..."
pip install numpy --user --upgrade

# Install dlr egg
echo
echo "Installing dlr for python..."
easy_install --user -Z dlr-1.0-py2.7-linux-armv7l.egg
echo

# Test dlr runtime
echo
echo "Testing dlr using python..."
python test-dlr.py

echo
echo "dlr was installed successfully!"
echo
