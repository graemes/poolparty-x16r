#
# Dockerfile for ccminer
# usage: docker build -t cccminer:latest .
# run: docker run -it --rm ccminer:latest [ARGS]
# ex: docker run -it --rm ccminer:latest -a cryptonight -o cryptonight.eu.nicehash.com:3355 -u 1MiningDW2GKzf4VQfmp4q2XoUvR6iy6PD.worker1 -p x -t 3
#

# Build
FROM nvidia/cuda:9.1-devel as builder


RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libgmp-dev \
    libcurl4-openssl-dev \
    libjansson-dev \
    automake \
  && rm -rf /var/lib/apt/lists/*

COPY . /app/
RUN cd /app/ && ./build.sh

# App
FROM nvidia/cuda:9.1-base

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
