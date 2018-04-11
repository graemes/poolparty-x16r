#!/bin/sh

docker pull nvidia/cuda:9.1-base
docker pull nvidia/cuda:9.1-runtime
docker pull nvidia/cuda:9.1-devel

mv configure.sh configure-public.sh
mv configure-local.sh configure.sh

for COMPUTEARCH in 61 52
do
    cat configure.sh
    docker build . --build-arg COMPUTE=${COMPUTEARCH} \
          -t registry.graemes.com/graemes/poolparty-x16r:compute${COMPUTEARCH} 
    docker push registry.graemes.com/graemes/poolparty-x16r:compute${COMPUTEARCH}
done

#docker build . -t registry.graemes.com/graemes/poolparty-x16r:latest  
#docker push registry.graemes.com/graemes/poolparty-x16r:latest

mv configure.sh configure-local.sh
mv configure-public.sh configure.sh
