#!/usr/bin/env bash
set -e

suffix="$1"

if [[ -z "${PYTHON_COMMAND}" ]]
then
  PYTHON_COMMAND=python
fi

cd python
rm -f dist/*
"${PYTHON_COMMAND}" setup.py bdist_wheel --universal
for file in dist/*.whl
do
  mv ${file} ${file%-any.whl}-${suffix}.whl
done
