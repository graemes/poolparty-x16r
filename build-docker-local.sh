#!/bin/sh

docker pull nvidia/cuda:9.1-base
docker pull nvidia/cuda:9.1-runtime
docker pull nvidia/cuda:9.1-devel

for COMPUTEARCH in 61 52
do
    docker build . --build-arg COMPUTE=${COMPUTEARCH} \
          -t registry.graemes.com/graemes/cryptopool-x16r:compute${COMPUTEARCH} \
          -t graemes/cryptopool-x16r:compute${COMPUTEARCH}
    docker push registry.graemes.com/graemes/cryptopool-x16r:compute${COMPUTEARCH}
done

docker build . -t registry.graemes.com/graemes/cryptopool-x16r:latest  
docker push registry.graemes.com/graemes/cryptopool-x16r:latest
