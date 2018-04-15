// Auf QuarkCoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>
#include <sys/types.h> // off_t

#include <cuda_helper.h>

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#endif

#define TPB 256
#define THF 4U

#include "groestl_functions_quad_80.cuh"
#include "groestl_transf_quad_80.cuh"

__constant__ static uint32_t c_Message80[20];

__host__
void quark_groestl512_setBlock_80(int thr_id, uint32_t *endiandata)
{
	cudaMemcpyToSymbol(c_Message80, endiandata, sizeof(c_Message80), 0, cudaMemcpyHostToDevice);
}

__global__ __launch_bounds__(TPB, THF)
void quark_groestl512_gpu_hash_80_quad(const uint32_t threads, const uint32_t startNounce, uint32_t * g_outhash)
{
	// BEWARE : 4-WAY CODE (one hash need 4 threads)
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
	if (thread < threads)
	{
		const uint32_t thr = threadIdx.x & 0x3; // % THF

		/*| M0 M1 M2 M3 M4 | M5 M6 M7 | (input)
		--|----------------|----------|
		T0|  0  4  8 12 16 | 80       |
		T1|  1  5       17 |          |
		T2|  2  6       18 |          |
		T3|  3  7       Nc |       01 |
		--|----------------|----------| TPR */

		uint32_t message[8];

		#pragma unroll 5
		for(int k=0; k<5; k++) message[k] = c_Message80[thr + (k * THF)];

		#pragma unroll 3
		for(int k=5; k<8; k++) message[k] = 0;

		if (thr == 0) message[5] = 0x80U;
		if (thr == 3) {
			message[4] = cuda_swab32(startNounce + thread);
			message[7] = 0x01000000U;
		}

		uint32_t msgBitsliced[8];
		to_bitslice_quad_80(message, msgBitsliced);

		uint32_t state[8];
		groestl512_progressMessage_quad_80(state, msgBitsliced);

		uint32_t hash[16];
		from_bitslice_quad_80(state, hash);

		if (thr == 0) { /* 4 threads were done */
			const off_t hashPosition = thread;
			//if (!thread) hash[15] = 0xFFFFFFFF;
			uint4 *outpt = (uint4*) &g_outhash[hashPosition << 4];
			uint4 *phash = (uint4*) hash;
			outpt[0] = phash[0];
			outpt[1] = phash[1];
			outpt[2] = phash[2];
			outpt[3] = phash[3];
		}
	}
}

__host__
void quark_groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash, const uint32_t tpb)
{
	//tbp = TPB;
	const uint32_t factor = THF;

	const dim3 grid(factor*((tpb+tpb-1)/tpb));
	const dim3 block(tpb);

	quark_groestl512_gpu_hash_80_quad<<<grid, block>>>(threads, startNounce, d_hash);
}

__host__
void quark_groestl512_cpu_init_80(int thr_id, uint32_t threads) {}

__host__
void quark_groestl512_cpu_free_80(int thr_id) {}

#include "miner.h"

__host__
int quark_groestl512_calc_tpb_80(int thr_id) {

	int blockSize, minGridSize, maxActiveBlocks, device;
	cudaDeviceProp props;


	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, quark_groestl512_gpu_hash_80_quad, 0,	0);

	// calculate theoretical occupancy
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, quark_groestl512_gpu_hash_80_quad, blockSize, 0);
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
	float occupancy = (maxActiveBlocks * blockSize / props.warpSize)
			/ (float) (props.maxThreadsPerMultiProcessor / props.warpSize);

	if (!opt_quiet) gpulog(LOG_INFO, thr_id, "groestl512_80 tpb calc - block size %d. Theoretical occupancy: %f", blockSize, occupancy);

	return (uint32_t)blockSize;
}
