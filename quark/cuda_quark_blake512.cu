﻿#include <stdio.h>
#include <memory.h>
#include <sys/types.h> // off_t

#include "miner.h"
#include "cuda_helper.h"

#define ROTR(x,n) ROTR64(x,n)

// use sp kernel on SM 5+
#define SP_KERNEL

#define USE_SHUFFLE 0

__constant__
static uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)

// ---------------------------- BEGIN CUDA quark_blake512 functions ------------------------------------

__device__ __constant__
static const uint8_t c_sigma_big[16][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },

	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13 , 0 },

	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
};

__device__ __constant__
static const uint64_t c_u512[16] =
{
	0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
	0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
	0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,
	0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
	0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL,
	0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
	0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL,
	0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
};

#define G(a,b,c,d,x) { \
	uint32_t idx1 = sigma[i][x]; \
	uint32_t idx2 = sigma[i][x+1]; \
	v[a] += (m[idx1] ^ u512[idx2]) + v[b]; \
	v[d] = SWAPDWORDS(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROTR( v[b] ^ v[c], 25); \
	v[a] += (m[idx2] ^ u512[idx1]) + v[b]; \
	v[d] = ROTR( v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR( v[b] ^ v[c], 11); \
}

__device__ __forceinline__
void quark_blake512_compress(uint64_t *h, const uint64_t *block, const uint8_t ((*sigma)[16]), const uint64_t *u512, const int T0)
{
	uint64_t v[16];
	uint64_t m[16];

	#pragma unroll
	for(int i=0; i < 16; i++) {
		m[i] = cuda_swab64(block[i]);
	}

	//#pragma unroll 8
	for(int i=0; i < 8; i++)
		v[i] = h[i];

	v[ 8] = u512[0];
	v[ 9] = u512[1];
	v[10] = u512[2];
	v[11] = u512[3];
	v[12] = u512[4] ^ T0;
	v[13] = u512[5] ^ T0;
	v[14] = u512[6];
	v[15] = u512[7];

	//#pragma unroll 16
	for(int i=0; i < 16; i++)
	{
		/* column step */
		G( 0, 4, 8, 12, 0 );
		G( 1, 5, 9, 13, 2 );
		G( 2, 6, 10, 14, 4 );
		G( 3, 7, 11, 15, 6 );
		/* diagonal step */
		G( 0, 5, 10, 15, 8 );
		G( 1, 6, 11, 12, 10 );
		G( 2, 7, 8, 13, 12 );
		G( 3, 4, 9, 14, 14 );
	}

	h[0] ^= v[0] ^ v[8];
	h[1] ^= v[1] ^ v[9];
	h[2] ^= v[2] ^ v[10];
	h[3] ^= v[3] ^ v[11];
	h[4] ^= v[4] ^ v[12];
	h[5] ^= v[5] ^ v[13];
	h[6] ^= v[6] ^ v[14];
	h[7] ^= v[7] ^ v[15];
}

__global__ __launch_bounds__(256, 4)
void quark_blake512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *g_nonceVector, uint64_t *g_hash)
{
}

__global__ __launch_bounds__(256,4)
void quark_blake512_gpu_hash_80(uint32_t threads, uint32_t startNounce, void *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t buf[16];
		#pragma unroll
		for (int i=0; i < 16; ++i)
			buf[i] = c_PaddedMessage80[i];

		// The test Nonce
		const uint32_t nounce = startNounce + thread;
		((uint32_t*)buf)[19] = cuda_swab32(nounce);

		uint64_t h[8] = {
			0x6a09e667f3bcc908ULL,
			0xbb67ae8584caa73bULL,
			0x3c6ef372fe94f82bULL,
			0xa54ff53a5f1d36f1ULL,
			0x510e527fade682d1ULL,
			0x9b05688c2b3e6c1fULL,
			0x1f83d9abfb41bd6bULL,
			0x5be0cd19137e2179ULL
		};

		quark_blake512_compress(h, buf, c_sigma_big, c_u512, 640);

		uint64_t *outHash = (uint64_t*)outputHash + (thread * 8U);
		for (uint32_t i=0; i < 8; i++) {
			outHash[i] = cuda_swab64( h[i] );
		}
	}
}

#include "cuda_quark_blake512_sp.cuh"

__host__
void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order)
{
	quark_blake512_cpu_hash_64_sp(threads, startNounce, d_nonceVector, d_outputHash);
	
	//MyStreamSynchronize(NULL, order, thr_id);
}

__host__
void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash)
{
	quark_blake512_cpu_hash_80_sp(threads, startNounce, d_outputHash);
}

// ---------------------------- END CUDA quark_blake512 functions ------------------------------------

__host__
void quark_blake512_cpu_init(int thr_id, uint32_t threads)
{
	cuda_get_arch(thr_id);
}

__host__
void quark_blake512_cpu_free(int thr_id)
{
}

// ----------------------------- Host midstate for 80-bytes input ------------------------------------

#undef SPH_C32
#undef SPH_T32
#undef SPH_C64
#undef SPH_T64

extern "C" {
#include "sph/sph_blake.h"
}

__host__
void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *endiandata)
{
	quark_blake512_cpu_setBlock_80_sp(thr_id, (uint64_t*) endiandata);
	
	CUDA_LOG_ERROR();
}
