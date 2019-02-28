#!/usr/bin/env bash
set -e

suffix="$1"

cd python
rm -f dist/*
python setup.py bdist_wheel --universal
for file in dist/*.whl
do
  mv ${file} ${file%-any.whl}-${suffix}.whl
done
