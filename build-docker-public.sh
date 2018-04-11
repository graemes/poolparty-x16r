#!/bin/sh

docker pull nvidia/cuda:9.1-base
docker pull nvidia/cuda:9.1-runtime
docker pull nvidia/cuda:9.1-devel

docker build . -t graemes/poolparty-x16r:latest 
docker push graemes/poolparty-x16r:latest

for COMPUTEARCH in 52 61
do
    docker build . --build-arg COMPUTE=${COMPUTEARCH} \
          -t graemes/poolparty-x16r:compute${COMPUTEARCH}
    docker push graemes/poolparty-x16r:compute${COMPUTEARCH}
done
