FROM ubuntu:18.04
ARG APP=xgboost

ENV PYTHONUNBUFFERED TRUE

# Disable thread pinning in TVM and Treelite
ENV TVM_BIND_THREADS 0
ENV TREELITE_BIND_THREADS 0

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    openjdk-8-jdk-headless \
    curl \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp

RUN pip3 install --pre --no-cache-dir mxnet-model-server

RUN pip3 install --no-cache-dir numpy scipy xlrd Pillow boto3 six requests

#RUN pip3 install --no-cache-dir dlr

RUN mkdir -p /home/model-server && cd /home/model-server \
    && git clone --recursive https://github.com/neo-ai/neo-ai-dlr \
    && cd neo-ai-dlr && git checkout 7dd33f829062e19df3c0f175912199ad176da970 \
    && git submodule update --init --recursive \
    && mkdir build && cd build && cmake .. \
    && make -j15 && cd ../python && python3 setup.py bdist_wheel \
    && pip3 install dist/*.whl

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp \
    && mkdir -p /home/model-server/model

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY config.properties /home/model-server/config.properties

COPY neo_template_$APP.py /home/model-server/neo_template.py

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server

EXPOSE 8080 8081

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="chyunsu@amazon.com, dantu@amazon.com, rakvas@amazon.com, lufen@amazon.com, dden@amazon.com"
