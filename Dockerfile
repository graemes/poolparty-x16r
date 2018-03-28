#
# Dockerfile for ccminer
#
# requires: nvidia-docker (https://github.com/NVIDIA/nvidia-docker)
#
# usage: docker build -t cccminer:latest .
# run: docker run -it --rm --runtime=nvidia ccminer:latest [ARGS]
# ex: docker run -it --rm --runtime=nvidia ccminer:latest -o cryptopool.party:3636 -u RH4KkDFJV7FuURwrZDyQZoPWAKc4hSHuDU -p x
#

# Build
FROM nvidia/cuda:9.1-devel as builder

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
    make -j 8

# App
FROM nvidia/cuda:9.1-base

LABEL maintainer="graemes@graemes.com"

RUN apt-get update && apt-get install -y \
    libcurl3 \
    libjansson4 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir /conf
VOLUME ["/conf"]

RUN groupadd -r miner && useradd --no-log-init -m -g miner miner
USER miner
WORKDIR /home/miner

COPY --from=builder /app/ccminer /home/miner
ENTRYPOINT ["/home/miner/ccminer"]
CMD ["--help"]
