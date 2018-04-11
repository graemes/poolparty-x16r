#!/bin/sh

#docker pull nvidia/cuda:9.1-base
#docker pull nvidia/cuda:9.1-runtime
#docker pull nvidia/cuda:9.1-devel
mv configure.sh configure-public.sh
mv configure-bench.sh configure.sh

cat configure.sh

docker build . --build-arg COMPUTE=61 -t registry.graemes.com/graemes/poolparty-x16r-bench:compute61
docker push registry.graemes.com/graemes/poolparty-x16r-bench:compute61

mv configure.sh configure-bench.sh
mv configure-public.sh configure.sh

