#!/bin/bash
set -e
set -x

source /home/model-server/mms_config.sh

if [[ "$1" = "serve" ]]; then
    shift 1
    mkdir -p /opt/ml/errors
    touch /opt/ml/errors/errors.log
    chown model-server /opt/ml/errors/errors.log
    su - model-server
    cp -v -r /opt/ml/model/* /home/model-server/model
    cp -v -r /home/model-server/neo_template.py /home/model-server/model
    model-archiver --handler neo_template:handle --model-name neomodel --model-path /home/model-server/model -f --export-path /home/model-server --runtime python3
    mv /home/model-server/neomodel.mar /home/model-server/model

    if [ ! -z "${MMS_NUM_WORKER}" ]; then
      sed -e "s/#default_workers_per_model=35/default_workers_per_model=${MMS_NUM_WORKER}/" \
        /home/model-server/config.properties > /home/model-server/config.properties.new
      export OMP_NUM_THREADS=${MMS_NUM_THREAD_PER_WORKER}
      mxnet-model-server --start --mms-config config.properties.new
    else
      mxnet-model-server --start --mms-config config.properties
    fi

else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
