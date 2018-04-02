# allow to use Host cuda functions in C/C++
DEF_INCLUDES = @CUDA_INCLUDES@

if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.md README.txt LICENSE.txt \
			  ccminer.sln ccminer.vcxproj ccminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c

SUBDIRS = compat

bin_PROGRAMS = ccminer

ccminer_SOURCES	= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  crc32.c \
			  ccminer.cpp pools.cpp util.cpp bench.cpp bignum.cpp \
			  api.cpp hashlog.cpp nvml.cpp stats.cpp sysinfos.cpp cuda.cpp \
			  nvsettings.cpp \
			  sph/fugue.c uint256.h \
			  cuda_checkhash.cu \
			  sph/bmw.c sph/blake.c sph/groestl.c sph/jh.c sph/keccak.c sph/skein.c \
			  sph/cubehash.c sph/echo.c sph/luffa.c sph/sha2.c sph/shavite.c sph/simd.c \
			  sph/hamsi.c sph/hamsi_helper.c sph/streebog.c \
			  sph/shabal.c sph/whirlpool.c sph/sha2big.c sph/haval.c \
			  sph/ripemd.c sph/sph_sha2.c \
			  quark/cuda_quark_blake512.cu \
			  quark/cuda_bmw512.cu \
			  quark/cuda_quark_groestl512.cu \
			  quark/cuda_jh512.cu \
			  quark/cuda_quark_keccak512.cu JHA/cuda_jha_keccak512.cu \
			  quark/cuda_skein512.cu \
			  qubit/qubit_luffa512_alexis.cu x11/cuda_x11_luffa512.cu \
                          x11/cuda_x11_cubehash512_alexis.cu x11/cuda_x11_cubehash512_80.cu \
			  x11/cuda_x11_shavite512_alexis.cu x11/cuda_x11_shavite512_80.cu \
                          x11/cuda_x11_simd512_alexis.cu x16r/cuda_x16_simd512_80.cu \
                          x11/cuda_x11_echo_alexis.cu x16r/cuda_x16_echo512.cu \
			  x13/cuda_x13_hamsi512_alexis.cu x13/cuda_x13_hamsi512_80.cu \
                          x13/cuda_x13_fugue512_alexis.cu x16r/cuda_x16_fugue512.cu \
			  x15/cuda_x14_shabal512_alexis.cu x16r/cuda_x16_shabal512.cu \
                          x15/cuda_x15_whirlpool_alexis.cu x15/cuda_x15_whirlpool_sm3.cu \
			  x17/cuda_x17_sha512_alexis.cu x17/cuda_x17_sha512_80.cu \
			  x16r/x16r.cu 

if HAVE_NVML
nvml_defs = -DUSE_WRAPNVML
nvml_libs = -ldl
endif

if HAVE_WINDOWS
ccminer_SOURCES += compat/winansi.c
endif

ccminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD    = @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ $(nvml_libs)
ccminer_CPPFLAGS = @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(CPPFLAGS) $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) $(DEF_INCLUDES) $(nvml_defs)

if HAVE_OSX
ccminer_CPPFLAGS += -I/usr/local/llvm/lib/clang/4.0.0/include
ccminer_LDFLAGS += -L/usr/local/llvm/lib
ccminer_LDADD += -lomp
endif

#ccminer_CPPFLAGS += -DUSE_LIBSODIUM
#ccminer_LDFLAGS += -Lequi/lib
#ccminer_LDADD += -lsodium
ccminer_LDADD += -lcuda

nvcc_ARCH :=
if COMPUTE61
nvcc_ARCH += -gencode=arch=compute_61,code=\"sm_61,compute_61\"
else
if COMPUTE52
nvcc_ARCH += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
else
nvcc_ARCH += -gencode=arch=compute_61,code=\"sm_61,compute_61\"
nvcc_ARCH += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
nvcc_ARCH += -gencode=arch=compute_50,code=\"sm_50,compute_50\"
nvcc_ARCH += -gencode=arch=compute_35,code=\"sm_35,compute_35\"
endif
endif

nvcc_FLAGS = $(nvcc_ARCH) @CUDA_INCLUDES@ -I. @CUDA_CFLAGS@
nvcc_FLAGS += $(JANSSON_INCLUDES) --ptxas-options="-v"

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<

# Luffa and Echo are faster with 80 registers than 128
x11/cuda_x11_luffa512_alexis.o: x11/cuda_x11_luffa512_alexis.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

x11/cuda_x11_simd512.o: x11/cuda_x11_simd512.cu
	$(NVCC) $(nvcc_FLAGS) -Xcompiler -Wno-unused-variable -o $@ -c $<

#x13/cuda_x13_hamsi512.o: x13/cuda_x13_hamsi512_alexis.cu
#	$(NVCC) $(nvcc_FLAGS) --maxrregcount=72 -o $@ -c $<

x17/cuda_x17_sha512.o: x17/cuda_x17_sha512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

quark/cuda_quark_blake512.o: quark/cuda_quark_blake512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

quark/cuda_jh512.o: quark/cuda_jh512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

quark/cuda_quark_keccak512.o: quark/cuda_quark_keccak512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=88 -o $@ -c $<

#x15/cuda_x15_whirlpool_alexis.o: x15/cuda_x15_whirlpool_alexis.cu
#	$(NVCC) $(nvcc_FLAGS) --maxrregcount=72 -o $@ -c $<
