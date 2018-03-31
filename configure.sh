# To change the cuda arch, edit Makefile.am and run ./build.sh

#extracflags="-march=native -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16"
#extracflags="-std=c++11 -march=haswell -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16"
#extracflags="-march=haswell -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16"
extracflags="-mtune=generic -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16"

#CUDA_CFLAGS="-std=c++11 -O3 -lineno -Xcompiler -Wall  -D_FORCE_INLINES" \
CUDA_CFLAGS="-O3 -lineno -Xcompiler -Wall  -D_FORCE_INLINES" \
	./configure CXXFLAGS="-O3 $extracflags" --with-cuda=/usr/local/cuda --with-nvml=libnvidia-ml.so $@

