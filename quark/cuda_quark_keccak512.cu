/*
	Based upon Tanguy Pruvot's repo
	
	Provos Alexis - 2016
	graemes - 2018
*/

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "miner.h"

#define TPB50 256
#define TPB52 128
#define TPF50 3
#define TPF52 7

__constant__ 
uint2 keccak_round_constants[24] = {
		{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 },	{ 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
		{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 },	{ 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
		{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 },	{ 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
		{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 },	{ 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
		{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 },	{ 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
		{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 },	{ 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

#if __CUDA_ARCH__ > 500
__global__ __launch_bounds__(TPB52,TPF52)
#else
__global__ __launch_bounds__(TPB50,TPF50)
#endif
void quark_keccak512_gpu_hash_64(const uint32_t threads, uint2 *const __restrict__ g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint2 t[5], u[5], v, w;
	uint2 s[25];
	if (thread < threads){
	
		uint2x4* d_hash = (uint2x4 *)&g_hash[thread * 8];

		#if __CUDA_ARCH__ > 500
		*(uint2x4*)&s[ 0] = __ldg4(&d_hash[ 0]);
		*(uint2x4*)&s[ 4] = __ldg4(&d_hash[ 1]);
		#else
		*(uint2x4*)&s[ 0] = d_hash[ 0];
		*(uint2x4*)&s[ 4] = d_hash[ 1];		
		#endif
		
		s[8] = make_uint2(1,0x80000000);

		/*theta*/
		t[ 0] = vectorize(devectorize(s[ 0])^devectorize(s[ 5]));
		t[ 1] = vectorize(devectorize(s[ 1])^devectorize(s[ 6]));
		t[ 2] = vectorize(devectorize(s[ 2])^devectorize(s[ 7]));
		t[ 3] = vectorize(devectorize(s[ 3])^devectorize(s[ 8]));
		t[ 4] = s[4];
		
		/*theta*/
		#pragma unroll 5
		for(int j=0;j<5;j++){
			u[ j] = ROL2(t[ j], 1);
		}
		
		s[ 4] = xor3x(s[ 4], t[3], u[ 0]);
		s[24] = s[19] = s[14] = s[ 9] = t[ 3] ^ u[ 0];

		s[ 0] = xor3x(s[ 0], t[4], u[ 1]);
		s[ 5] = xor3x(s[ 5], t[4], u[ 1]);
		s[20] = s[15] = s[10] = t[4] ^ u[ 1];

		s[ 1] = xor3x(s[ 1], t[0], u[ 2]);
		s[ 6] = xor3x(s[ 6], t[0], u[ 2]);
		s[21] = s[16] = s[11] = t[0] ^ u[ 2];
		
		s[ 2] = xor3x(s[ 2], t[1], u[ 3]);
		s[ 7] = xor3x(s[ 7], t[1], u[ 3]);
		s[22] = s[17] = s[12] = t[1] ^ u[ 3];
		
		s[ 3] = xor3x(s[ 3], t[2], u[ 4]);s[ 8] = xor3x(s[ 8], t[2], u[ 4]);
		s[23] = s[18] = s[13] = t[2] ^ u[ 4];
		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1]  = ROL2(s[6], 44);
		s[6]  = ROL2(s[9], 20);
		s[9]  = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2]  = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL8(s[19]);
		s[19] = ROR8(s[23]);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4]  = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8]  = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5]  = ROL2(s[3], 28);
		s[3]  = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7]  = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);
		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		#pragma unroll 5
		for(int j=0;j<25;j+=5){
			v=s[j];w=s[j + 1];s[j] = chi(v,w,s[j+2]);s[j+1] = chi(w,s[j+2],s[j+3]);s[j+2]=chi(s[j+2],s[j+3],s[j+4]);s[j+3]=chi(s[j+3],s[j+4],v);s[j+4]=chi(s[j+4],v,w);
		}
		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[ 0];

		#if __CUDA_ARCH__ > 500
		#pragma unroll 4
		#else
		#pragma unroll 3
		#endif
		for (int i = 1; i < 23; i++) {
			/*theta*/
			#pragma unroll 5
			for(int j=0;j<5;j++){
				t[ j] = vectorize(xor5(devectorize(s[ j]),devectorize(s[j+5]),devectorize(s[j+10]),devectorize(s[j+15]),devectorize(s[j+20])));
			}

			/*theta*/
			#pragma unroll 5
			for(int j=0;j<5;j++){
				u[ j] = ROL2(t[ j], 1);
			}
			s[ 4] = xor3x(s[ 4], t[3], u[ 0]);s[ 9] = xor3x(s[ 9], t[3], u[ 0]);s[14] = xor3x(s[14], t[3], u[ 0]);s[19] = xor3x(s[19], t[3], u[ 0]);s[24] = xor3x(s[24], t[3], u[ 0]);
			s[ 0] = xor3x(s[ 0], t[4], u[ 1]);s[ 5] = xor3x(s[ 5], t[4], u[ 1]);s[10] = xor3x(s[10], t[4], u[ 1]);s[15] = xor3x(s[15], t[4], u[ 1]);s[20] = xor3x(s[20], t[4], u[ 1]);
			s[ 1] = xor3x(s[ 1], t[0], u[ 2]);s[ 6] = xor3x(s[ 6], t[0], u[ 2]);s[11] = xor3x(s[11], t[0], u[ 2]);s[16] = xor3x(s[16], t[0], u[ 2]);s[21] = xor3x(s[21], t[0], u[ 2]);
			s[ 2] = xor3x(s[ 2], t[1], u[ 3]);s[ 7] = xor3x(s[ 7], t[1], u[ 3]);s[12] = xor3x(s[12], t[1], u[ 3]);s[17] = xor3x(s[17], t[1], u[ 3]);s[22] = xor3x(s[22], t[1], u[ 3]);
			s[ 3] = xor3x(s[ 3], t[2], u[ 4]);s[ 8] = xor3x(s[ 8], t[2], u[ 4]);s[13] = xor3x(s[13], t[2], u[ 4]);s[18] = xor3x(s[18], t[2], u[ 4]);s[23] = xor3x(s[23], t[2], u[ 4]);

			/* rho pi: b[..] = rotl(a[..], ..) */
			v = s[1];
			s[1]  = ROL2(s[6], 44);
			s[6]  = ROL2(s[9], 20);
			s[9]  = ROL2(s[22], 61);
			s[22] = ROL2(s[14], 39);
			s[14] = ROL2(s[20], 18);
			s[20] = ROL2(s[2], 62);
			s[2]  = ROL2(s[12], 43);
			s[12] = ROL2(s[13], 25);
			s[13] = ROL8(s[19]);
			s[19] = ROR8(s[23]);
			s[23] = ROL2(s[15], 41);
			s[15] = ROL2(s[4], 27);
			s[4]  = ROL2(s[24], 14);
			s[24] = ROL2(s[21], 2);
			s[21] = ROL2(s[8], 55);
			s[8]  = ROL2(s[16], 45);
			s[16] = ROL2(s[5], 36);
			s[5]  = ROL2(s[3], 28);
			s[3]  = ROL2(s[18], 21);
			s[18] = ROL2(s[17], 15);
			s[17] = ROL2(s[11], 10);
			s[11] = ROL2(s[7], 6);
			s[7]  = ROL2(s[10], 3);
			s[10] = ROL2(v, 1);

			/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
			#pragma unroll 5
			for(int j=0;j<25;j+=5){
				v=s[j];w=s[j + 1];s[j] = chi(v,w,s[j+2]);s[j+1] = chi(w,s[j+2],s[j+3]);s[j+2]=chi(s[j+2],s[j+3],s[j+4]);s[j+3]=chi(s[j+3],s[j+4],v);s[j+4]=chi(s[j+4],v,w);
			}

			/* iota: a[0,0] ^= round constant */
			s[0] ^= keccak_round_constants[i];
		}
		/*theta*/
		#pragma unroll 5
		for(int j=0;j<5;j++){
			t[ j] = xor3x(xor3x(s[j+0],s[j+5],s[j+10]),s[j+15],s[j+20]);
		}
		/*theta*/
		#pragma unroll 5
		for(int j=0;j<5;j++){
			u[ j] = ROL2(t[ j], 1);
		}
		s[ 9] = xor3x(s[ 9], t[3], u[ 0]);
		s[24] = xor3x(s[24], t[3], u[ 0]);
		s[ 0] = xor3x(s[ 0], t[4], u[ 1]);
		s[10] = xor3x(s[10], t[4], u[ 1]);
		s[ 6] = xor3x(s[ 6], t[0], u[ 2]);
		s[16] = xor3x(s[16], t[0], u[ 2]);
		s[12] = xor3x(s[12], t[1], u[ 3]);
		s[22] = xor3x(s[22], t[1], u[ 3]);
		s[ 3] = xor3x(s[ 3], t[2], u[ 4]);
		s[18] = xor3x(s[18], t[2], u[ 4]);
		/* rho pi: b[..] = rotl(a[..], ..) */
		s[ 1]  = ROL2(s[ 6], 44);
		s[ 2]  = ROL2(s[12], 43);
		s[ 5]  = ROL2(s[ 3], 28);
		s[ 7]  = ROL2(s[10], 3);
		s[ 3]  = ROL2(s[18], 21);
		s[ 4]  = ROL2(s[24], 14);
		s[ 6]  = ROL2(s[ 9], 20);
		s[ 8]  = ROL2(s[16], 45);
		s[ 9]  = ROL2(s[22], 61);
		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v=s[ 0];w=s[ 1];s[ 0] = chi(v,w,s[ 2]);s[ 1] = chi(w,s[ 2],s[ 3]);s[ 2]=chi(s[ 2],s[ 3],s[ 4]);s[ 3]=chi(s[ 3],s[ 4],v);s[ 4]=chi(s[ 4],v,w);		
		v=s[ 5];w=s[ 6];s[ 5] = chi(v,w,s[ 7]);s[ 6] = chi(w,s[ 7],s[ 8]);s[ 7]=chi(s[ 7],s[ 8],s[ 9]);
		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[23];
		
		d_hash[0] = *(uint2x4*)&s[0];
		d_hash[1] = *(uint2x4*)&s[4];

	}
}

__host__
void quark_keccak512_cpu_hash_64(const int thr_id, const uint32_t threads, uint32_t *d_hash, const uint32_t tpb)
{
	const dim3 grid((threads+tpb-1)/tpb);
	const dim3 block(tpb);

	quark_keccak512_gpu_hash_64<<<grid, block>>>(threads, (uint2*)d_hash);
}

__host__
void quark_keccak512_cpu_init_64(const int thr_id, uint32_t threads) {}

__host__
void quark_keccak512_cpu_free_64(const int thr_id) {}

__host__
uint32_t quark_keccak512_calc_tpb_64(const int thr_id) {

    int blockSize = 0;
    int minGridSize = 0;
    int maxActiveBlocks, device;
	cudaDeviceProp props;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, quark_keccak512_gpu_hash_64, 0,	0);

	// calculate theoretical occupancy
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, quark_keccak512_gpu_hash_64, blockSize, 0);
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
	float occupancy = (maxActiveBlocks * blockSize / props.warpSize)
			/ (float) (props.maxThreadsPerMultiProcessor / props.warpSize);

	if (!opt_quiet) gpulog(LOG_INFO, thr_id, "keccak512_64 tpb calc - block size %d. Theoretical occupancy: %f", blockSize, occupancy);

	return (uint32_t)blockSize;
}
