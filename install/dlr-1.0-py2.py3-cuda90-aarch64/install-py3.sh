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
check_command python3
check_command easy_install3
check_command pip3

# Install numpy
echo
echo "Installing numpy..."
pip3 install numpy decorator --user --upgrade

# Install dlr egg
echo
echo "Installing dlr for python3..."
easy_install3 --user -Z dlr-1.0-py3.5-linux-aarch64.egg
echo

# Test dlr runtime
echo
echo "Testing dlr using python3..."
python3 test-dlr.py

echo
echo "dlr was installed successfully!"
echo
