#!/usr/bin/env bash
set -e
set -x

rm -rf build
mkdir build
cd python
python setup.py install
cd ..
