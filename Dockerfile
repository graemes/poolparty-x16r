#
# Dockerfile for ccminer
#
# requires: nvidia-docker (https://github.com/NVIDIA/nvidia-docker)
#
# usage: docker build -t cccminer:latest .
# run: docker run -it --rm --runtime=nvidia graemes/poolparty:latest [ARGS]
# ex: docker run -it --rm --runtime=nvidia graemes/poolparty:latest -o cryptopool.party:3636 -u RH4KkDFJV7FuURwrZDyQZoPWAKc4hSHuDU -p x
#

ARG REPOSITORY 
# Build
FROM ${REPOSITORY}:9.2-devel as builder

ARG COMPUTE

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libgmp-dev \
    libcurl4-openssl-dev \
    libjansson-dev \
    automake \ 
  && rm -rf /var/lib/apt/lists/*

COPY . /app/
#RUN cd /app/ && ./build.sh
RUN cd /app/ && \
    make distclean || echo clean && \
    rm -f Makefile.in && \
    rm -f config.status && \
    ./autogen.sh || echo done && \
    ./configure.sh --enable-compute=$COMPUTE && \
    make -j 8 && \
    gcc --version

# App
FROM ${REPOSITORY}:9.2-base

LABEL maintainer="graemes@github"

ENV UID 8777
ENV GID 8777

ENV UID 8777
ENV GID 8777

RUN apt-get update && apt-get install -y \
    libcurl4 \
    libjansson4 \
    libgomp1 \
    cuda-nvml-dev-9-2 \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir /conf
VOLUME ["/conf"]

RUN groupadd -r -g ${GID} miner && useradd --no-log-init -m -u ${UID} -g miner miner 
RUN mkdir /log
RUN chown -R miner:miner /log
VOLUME ["/log"]

USER miner
WORKDIR /home/miner

COPY --from=builder /app/ccminer /home/miner
ENTRYPOINT ["/home/miner/ccminer"]
CMD ["--help"]
