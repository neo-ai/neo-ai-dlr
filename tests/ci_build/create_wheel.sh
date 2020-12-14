#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
  echo "Error: <plat-name> should be provided, e.g. manylinux1_x86_64"
  exit 1
fi

if [[ -z "${PYTHON_COMMAND}" ]]
then
  PYTHON_COMMAND=python
fi

cd python
rm -f dist/*
"${PYTHON_COMMAND}" setup.py bdist_wheel --plat-name=$1
