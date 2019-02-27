#!/usr/bin/env bash
set -e

cd python
rm -f dist/*
python setup.py bdist_wheel --universal
for file in dist/*.whl
do
  mv ${file} ${file%-any.whl}-manylinux1_x86_64.whl
done
