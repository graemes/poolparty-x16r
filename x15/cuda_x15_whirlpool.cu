/**
 * Whirlpool-512 CUDA implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014-2016 djm34, tpruvot, SP, Provos Alexis
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
 * @author djm34 (initial draft)
 * @author tpruvot (dual old/whirlpool modes, midstate)
 * @author SP ("final" function opt and tuning)
 * @author Provos Alexis (Applied partial shared memory utilization, precomputations, merging & tuning for 970/750ti under CUDA7.5 -> +93% increased throughput of whirlpool)
 */


// Change with caution, used by shared mem fetch
#define TPB80 384
#define TPB64 384

extern "C"
{
#include "sph/sph_whirlpool.h"
#include "miner.h"
}

#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "cuda_whirlpool_tables.cuh"

__device__ static uint64_t b0[256];
__device__ static uint64_t b7[256];

__constant__ static uint2 precomputed_round_key_64[72];
//__constant__ static uint2 precomputed_round_key_80[80];

//__device__ static uint2 c_PaddedMessage80[16];


/**
 * Round constants.
 */
__device__ uint2 InitVector_RC[10];

//--------START OF WHIRLPOOL DEVICE MACROS---------------------------------------------------------------------------
__device__ __forceinline__
void static TRANSFER(uint2 *const __restrict__ dst,const uint2 *const __restrict__ src){
	dst[0] = src[ 0];
	dst[1] = src[ 1];
	dst[2] = src[ 2];
	dst[3] = src[ 3];
	dst[4] = src[ 4];
	dst[5] = src[ 5];
	dst[6] = src[ 6];
	dst[7] = src[ 7];
}

__device__ __forceinline__
static uint2 d_ROUND_ELT_LDG(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7){
	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= ROR24(__ldg((uint2*)&b0[__byte_perm(in[i5].y, 0, 0x4441)]));
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7){

	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= sharedMemory[5][__byte_perm(in[i5].y, 0, 0x4441)];
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT1_LDG(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const uint2 c0){

	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= ROR24(__ldg((uint2*)&b0[__byte_perm(in[i5].y, 0, 0x4441)]));
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	ret ^= c0;
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT1(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const uint2 c0){
	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= sharedMemory[5][__byte_perm(in[i5].y, 0, 0x4441)];
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));//sharedMemory[6][__byte_perm(in[i6].y, 0, 0x4442)]
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);//sharedMemory[7][__byte_perm(in[i7].y, 0, 0x4443)]
	ret ^= c0;
	return ret;
}

//--------END OF WHIRLPOOL DEVICE MACROS-----------------------------------------------------------------------------

//--------START OF WHIRLPOOL HOST MACROS-----------------------------------------------------------------------------

#define table_skew(val,num) SPH_ROTL64(val,8*num)
#define BYTE(x, n)     ((unsigned)((x) >> (8 * (n))) & 0xFF)

#define ROUND_ELT(table, in, i0, i1, i2, i3, i4, i5, i6, i7) \
	(table[BYTE(in[i0], 0)] \
	^ table_skew(table[BYTE(in[i1], 1)], 1) \
	^ table_skew(table[BYTE(in[i2], 2)], 2) \
	^ table_skew(table[BYTE(in[i3], 3)], 3) \
	^ table_skew(table[BYTE(in[i4], 4)], 4) \
	^ table_skew(table[BYTE(in[i5], 5)], 5) \
	^ table_skew(table[BYTE(in[i6], 6)], 6) \
	^ table_skew(table[BYTE(in[i7], 7)], 7))

#define ROUND(table, in, out, c0, c1, c2, c3, c4, c5, c6, c7)   do { \
		out[0] = ROUND_ELT(table, in, 0, 7, 6, 5, 4, 3, 2, 1) ^ c0; \
		out[1] = ROUND_ELT(table, in, 1, 0, 7, 6, 5, 4, 3, 2) ^ c1; \
		out[2] = ROUND_ELT(table, in, 2, 1, 0, 7, 6, 5, 4, 3) ^ c2; \
		out[3] = ROUND_ELT(table, in, 3, 2, 1, 0, 7, 6, 5, 4) ^ c3; \
		out[4] = ROUND_ELT(table, in, 4, 3, 2, 1, 0, 7, 6, 5) ^ c4; \
		out[5] = ROUND_ELT(table, in, 5, 4, 3, 2, 1, 0, 7, 6) ^ c5; \
		out[6] = ROUND_ELT(table, in, 6, 5, 4, 3, 2, 1, 0, 7) ^ c6; \
		out[7] = ROUND_ELT(table, in, 7, 6, 5, 4, 3, 2, 1, 0) ^ c7; \
	} while (0)

/*
__host__
static void ROUND_KSCHED(const uint64_t *in,uint64_t *out,const uint64_t c){
	const uint64_t *a = in;
	uint64_t *b = out;
	ROUND(old1_T0, a, b, c, 0, 0, 0, 0, 0, 0, 0);
}
*/

//--------END OF WHIRLPOOL HOST MACROS-------------------------------------------------------------------------------

__host__
extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int mode){

	uint64_t* table0 = NULL;

	switch (mode) {
	case 0: /* x15 with rotated T1-T7 (based on T0) */
		table0 = (uint64_t*)plain_T0;
		cudaMemcpyToSymbol(InitVector_RC, plain_RC, 10*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(precomputed_round_key_64, plain_precomputed_round_key_64, 72*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
		break;
	case 1: /* old whirlpool */
		table0 = (uint64_t*)old1_T0;
		cudaMemcpyToSymbol(InitVector_RC, old1_RC, 10*sizeof(uint64_t),0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(precomputed_round_key_64, old1_precomputed_round_key_64, 72*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
		break;
	default:
		applog(LOG_ERR,"Bad whirlpool mode");
		exit(0);
	}
	cudaMemcpyToSymbol(b0, table0, 256*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
	uint64_t table7[256];
	for(int i=0;i<256;i++){
		table7[i] = ROTR64(table0[i],8);
	}
	cudaMemcpyToSymbol(b7, table7, 256*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
}

void whirl_midstate(void *state, const void *input)
{
	sph_whirlpool_context ctx;

	sph_whirlpool1_init(&ctx);
	sph_whirlpool1(&ctx, input, 64);

	memcpy(state, ctx.state, 64);
}

__host__
extern void x15_whirlpool_cpu_free(int thr_id){
	cudaFree(InitVector_RC);
	cudaFree(b0);
	cudaFree(b7);
}

__global__ __launch_bounds__(TPB64,2)
void x15_whirlpool_gpu_hash_64(uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint2 sharedMemory[7][256];

	if (threadIdx.x < 256) {
		const uint2 tmp = __ldg((uint2*)&b0[threadIdx.x]);
		sharedMemory[0][threadIdx.x] = tmp;
		sharedMemory[1][threadIdx.x] = ROL8(tmp);
		sharedMemory[2][threadIdx.x] = ROL16(tmp);
		sharedMemory[3][threadIdx.x] = ROL24(tmp);
		sharedMemory[4][threadIdx.x] = SWAPUINT2(tmp);
		sharedMemory[5][threadIdx.x] = ROR24(tmp);
		sharedMemory[6][threadIdx.x] = ROR16(tmp);
	}

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads){

		uint2 hash[8], n[8], h[ 8];
		uint2 tmp[8] = {
			{0xC0EE0B30,0x672990AF},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},
			{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828}
		};
	
		*(uint2x4*)&hash[ 0] = __ldg4((uint2x4*)&g_hash[(thread<<3) + 0]);
		*(uint2x4*)&hash[ 4] = __ldg4((uint2x4*)&g_hash[(thread<<3) + 4]);

		__syncthreads();

		#pragma unroll 8
		for(int i=0;i<8;i++)
			n[i]=hash[i];

		tmp[ 0]^= d_ROUND_ELT(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1);
		tmp[ 1]^= d_ROUND_ELT_LDG(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[ 2]^= d_ROUND_ELT(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[ 3]^= d_ROUND_ELT_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[ 4]^= d_ROUND_ELT(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[ 5]^= d_ROUND_ELT_LDG(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[ 6]^= d_ROUND_ELT(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[ 7]^= d_ROUND_ELT_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0);
		for (int i=1; i <10; i++){
			TRANSFER(n, tmp);
			tmp[ 0] = d_ROUND_ELT1_LDG(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, precomputed_round_key_64[(i-1)*8+0]);
			tmp[ 1] = d_ROUND_ELT1(    sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, precomputed_round_key_64[(i-1)*8+1]);
			tmp[ 2] = d_ROUND_ELT1(    sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, precomputed_round_key_64[(i-1)*8+2]);
			tmp[ 3] = d_ROUND_ELT1_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, precomputed_round_key_64[(i-1)*8+3]);
			tmp[ 4] = d_ROUND_ELT1(    sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, precomputed_round_key_64[(i-1)*8+4]);
			tmp[ 5] = d_ROUND_ELT1(    sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, precomputed_round_key_64[(i-1)*8+5]);
			tmp[ 6] = d_ROUND_ELT1(    sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, precomputed_round_key_64[(i-1)*8+6]);
			tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, precomputed_round_key_64[(i-1)*8+7]);
		}

		TRANSFER(h, tmp);
		#pragma unroll 8
		for (int i=0; i<8; i++)
			hash[ i] = h[i] = h[i] ^ hash[i];

		#pragma unroll 6
		for (int i=1; i<7; i++)
			n[i]=vectorize(0);

		n[0] = vectorize(0x80);
		n[7] = vectorize(0x2000000000000);

		#pragma unroll 8
		for (int i=0; i < 8; i++) {
			n[i] = n[i] ^ h[i];
		}

//		#pragma unroll 10
		for (int i=0; i < 10; i++) {
			tmp[ 0] = InitVector_RC[i];
			tmp[ 0]^= d_ROUND_ELT(sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1);
			tmp[ 1] = d_ROUND_ELT(sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
			tmp[ 2] = d_ROUND_ELT_LDG(sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
			tmp[ 3] = d_ROUND_ELT(sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
			tmp[ 4] = d_ROUND_ELT_LDG(sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
			tmp[ 5] = d_ROUND_ELT(sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
			tmp[ 6] = d_ROUND_ELT(sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
			tmp[ 7] = d_ROUND_ELT(sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
			TRANSFER(h, tmp);
			tmp[ 0] = d_ROUND_ELT1(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
			tmp[ 1] = d_ROUND_ELT1_LDG(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
			tmp[ 2] = d_ROUND_ELT1(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
			tmp[ 3] = d_ROUND_ELT1(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
			tmp[ 4] = d_ROUND_ELT1_LDG(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
			tmp[ 5] = d_ROUND_ELT1(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
			tmp[ 6] = d_ROUND_ELT1_LDG(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
			tmp[ 7] = d_ROUND_ELT1(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);
			TRANSFER(n, tmp);
		}

		hash[0] = xor3x(hash[0], n[0], vectorize(0x80));
		hash[1] = hash[1]^ n[1];
		hash[2] = hash[2]^ n[2];
		hash[3] = hash[3]^ n[3];
		hash[4] = hash[4]^ n[4];
		hash[5] = hash[5]^ n[5];
		hash[6] = hash[6]^ n[6];
		hash[7] = xor3x(hash[7], n[7], vectorize(0x2000000000000));

		*(uint2x4*)&g_hash[(thread<<3)+ 0]    = *(uint2x4*)&hash[ 0];
		*(uint2x4*)&g_hash[(thread<<3)+ 4]    = *(uint2x4*)&hash[ 4];
	}
}

__host__
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPB64-1) / TPB64);
	dim3 block(TPB64);

	x15_whirlpool_gpu_hash_64 <<<grid, block>>> (threads, (uint64_t*)d_hash);
}
