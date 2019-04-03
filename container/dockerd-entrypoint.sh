#!/bin/bash
set -e
set -x

if [[ "$1" = "serve" ]]; then
    cp -v /opt/ml/model/* /home/model-server/model
    cp -v /home/model-server/neo_template.py /home/model-server/model
    model-archiver --handler neo_template:predict --model-name neomodel --model-path /home/model-server/model -f --export-path /home/model-server
    mv /home/model-server/neomodel.mar /home/model-server/model

    shift 1
    mxnet-model-server --start --mms-config config.properties
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
