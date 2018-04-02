/*
 * sha-512 cuda kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014 djm34
 *               2016 tpruvot
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 */
#include <stdio.h>

#define NEED_HASH_512

#include "cuda_helper.h"

#define SWAP64(u64) cuda_swab64(u64)

static __constant__ uint64_t c_WB[80];

#define BSG5_0(x) xor3(ROTR64(x,28), ROTR64(x,34), ROTR64(x,39))
#define SSG5_0(x) xor3(ROTR64(x, 1), ROTR64(x ,8), shr_t64(x,7))
#define SSG5_1(x) xor3(ROTR64(x,19), ROTR64(x,61), shr_t64(x,6))

//#define MAJ(X, Y, Z)   (((X) & (Y)) | (((X) | (Y)) & (Z)))
#define MAJ(x, y, z)   andor(x,y,z)

__device__ __forceinline__
uint64_t Tone(uint64_t* K, uint64_t* r, uint64_t* W, const int a, const int i)
{
	//asm("// TONE \n");
	const uint64_t e = r[(a+4) & 7];
	uint64_t BSG51 = xor3(ROTR64(e, 14), ROTR64(e, 18), ROTR64(e, 41));
	const uint64_t f = r[(a+5) & 7];
	const uint64_t g = r[(a+6) & 7];
	uint64_t CHl = ((f ^ g) & e) ^ g; // xandx(e, f, g);
	return (r[(a+7) & 7] + BSG51 + CHl + K[i] + W[i]);
}

#define SHA3_STEP(K, r, W, ord, i) { \
	const int a = (8 - ord) & 7; \
	uint64_t T1 = Tone(K, r, W, a, i); \
	r[(a+3) & 7] += T1; \
	uint64_t T2 = (BSG5_0(r[a]) + MAJ(r[a], r[(a+1) & 7], r[(a+2) & 7])); \
	r[(a+7) & 7] = T1 + T2; \
}

__constant__
static uint64_t c_PaddedMessage80[10];

__global__
/*__launch_bounds__(256, 4)*/
void x16_sha512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t W[80];
		#pragma unroll
		for (int i = 0; i < 9; i ++) {
			W[i] = SWAP64(c_PaddedMessage80[i]);
		}
		const uint32_t nonce = startNonce + thread;
		//((uint32_t*)W)[19] = cuda_swab32(nonce);
		W[9] = REPLACE_HIDWORD(c_PaddedMessage80[9], cuda_swab32(nonce));
		W[9] = cuda_swab64(W[9]);
		W[10] = 0x8000000000000000;

		#pragma unroll
		for (int i = 11; i<15; i++) {
			W[i] = 0U;
		}
		W[15] = 0x0000000000000280;

		#pragma unroll 64
		for (int i = 16; i < 80; i ++) {
			W[i] = SSG5_1(W[i-2]) + W[i-7];
			W[i] += SSG5_0(W[i-15]) + W[i-16];
		}

		const uint64_t IV512[8] = {
			0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
			0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
			0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
			0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
		};

		uint64_t r[8];
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			r[i] = IV512[i];
		}

		#pragma unroll
		for (int i = 0; i < 80; i++) {
			SHA3_STEP(c_WB, r, W, i&7, i);
		}

		const uint64_t hashPosition = thread;
		uint64_t *pHash = &g_hash[hashPosition << 3];
		#pragma unroll
		for (int u = 0; u < 8; u ++) {
			pHash[u] = SWAP64(r[u] + IV512[u]);
		}
	}
}

__host__
void x16_sha512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x16_sha512_gpu_hash_80 <<<grid, block >>> (threads, startNounce, (uint64_t*)d_hash);
}

__host__
void x16_sha512_setBlock_80(void *pdata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}
