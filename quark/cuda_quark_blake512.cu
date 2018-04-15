/*
	Based upon Tanguy Pruvot's and SP's work
			
	Provos Alexis - 2016
	graemes - 2018
*/
#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h"

#define TPB52_64 192
#define TPB50_64 192

__constant__ uint2 _ALIGN(16) c_m[16]; // padded message (80 bytes + padding)

__constant__ uint2 _ALIGN(16) c_v[16]; //state

__constant__ uint2 _ALIGN(16) c_x[128]; //precomputed xors

// ---------------------------- BEGIN CUDA quark_blake512 functions ------------------------------------

__constant__ _ALIGN(16) uint2 z[16] =
{
	{0x85a308d3,0x243f6a88},{0x03707344,0x13198a2e},{0x299f31d0,0xa4093822},{0xec4e6c89,0x082efa98},
	{0x38d01377,0x452821e6},{0x34e90c6c,0xbe5466cf},{0xc97c50dd,0xc0ac29b7},{0xb5470917,0x3f84d5b5},
	{0x8979fb1b,0x9216d5d9},{0x98dfb5ac,0xd1310ba6},{0xd01adfb7,0x2ffd72db},{0x6a267e96,0xb8e1afed},
	{0xf12c7f99,0xba7c9045},{0xb3916cf7,0x24a19947},{0x858efc16,0x0801f2e2},{0x71574e69,0x636920d8}
};

__constant__ const uint2 h[8] = {
		{ 0xf3bcc908UL, 0x6a09e667UL },
		{ 0x84caa73bUL, 0xbb67ae85UL },
		{ 0xfe94f82bUL, 0x3c6ef372UL },
		{ 0x5f1d36f1UL, 0xa54ff53aUL },
		{ 0xade682d1UL, 0x510e527fUL },
		{ 0x2b3e6c1fUL, 0x9b05688cUL },
		{ 0xfb41bd6bUL, 0x1f83d9abUL },
		{ 0x137e2179UL, 0x5be0cd19UL }
	};

#define G4(x, a,b,c,d,a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3) { \
	v[a] += (m[c_sigma[i][x]] ^ z[c_sigma[i][x+1]]) + v[b]; \
	v[a1] += (m[c_sigma[i][x+2]] ^ z[c_sigma[i][x+3]]) + v[b1]; \
	v[a2] += (m[c_sigma[i][x+4]] ^ z[c_sigma[i][x+5]]) + v[b2]; \
	v[a3] += (m[c_sigma[i][x+6]] ^ z[c_sigma[i][x+7]]) + v[b3]; \
	v[d] = xorswap32(v[d] , v[a]); \
	v[d1] = xorswap32(v[d1] , v[a1]); \
	v[d2] = xorswap32(v[d2] , v[a2]); \
	v[d3] = xorswap32(v[d3] , v[a3]); \
	v[c] += v[d]; \
	v[c1] += v[d1]; \
	v[c2] += v[d2]; \
	v[c3] += v[d3]; \
	v[b] = ROR2( v[b] ^ v[c], 25); \
	v[b1] = ROR2( v[b1] ^ v[c1], 25); \
	v[b2] = ROR2( v[b2] ^ v[c2], 25); \
	v[b3] = ROR2( v[b3] ^ v[c3], 25); \
	v[a] += (m[c_sigma[i][x+1]] ^ z[c_sigma[i][x]]) + v[b]; \
	v[a1] += (m[c_sigma[i][x+3]] ^ z[c_sigma[i][x+2]]) + v[b1]; \
	v[a2] += (m[c_sigma[i][x+5]] ^ z[c_sigma[i][x+4]]) + v[b2]; \
	v[a3] += (m[c_sigma[i][x+7]] ^ z[c_sigma[i][x+6]]) + v[b3]; \
	v[d] = ROR16( v[d] ^ v[a]); \
	v[d1] = ROR16( v[d1] ^ v[a1]); \
	v[d2] = ROR16( v[d2] ^ v[a2]); \
	v[d3] = ROR16( v[d3] ^ v[a3]); \
	v[c] += v[d]; \
	v[c1] += v[d1]; \
	v[c2] += v[d2]; \
	v[c3] += v[d3]; \
	v[b] = ROR2( v[b] ^ v[c], 11); \
	v[b1] = ROR2( v[b1] ^ v[c1], 11); \
	v[b2] = ROR2( v[b2] ^ v[c2], 11); \
	v[b3] = ROR2( v[b3] ^ v[c3], 11); \
}

#define GS4(a,b,c,d,e,f,a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3){\
	v[ a]+= (m[ e] ^ z[ f]) + v[ b];	v[a1]+= (m[e1] ^ z[f1]) + v[b1];	v[a2]+= (m[e2] ^ z[f2]) + v[b2];	v[a3]+= (m[e3] ^ z[f3]) + v[b3];\
	v[ d] = SWAPDWORDS2(v[ d] ^ v[ a]);	v[d1] = SWAPDWORDS2(v[d1] ^ v[a1]);	v[d2] = SWAPDWORDS2(v[d2] ^ v[a2]);	v[d3] = SWAPDWORDS2(v[d3] ^ v[a3]);\
	v[ c]+= v[ d];				v[c1]+= v[d1];				v[c2]+= v[d2];				v[c3]+= v[d3];\
	v[ b] = ROR2(v[b] ^ v[c], 25);		v[b1] = ROR2(v[b1] ^ v[c1], 25);	v[b2] = ROR2(v[b2] ^ v[c2], 25);	v[b3] = ROR2(v[b3] ^ v[c3], 25); \
	v[ a]+= (m[ f] ^ z[ e]) + v[ b];	v[a1]+= (m[f1] ^ z[e1]) + v[b1];	v[a2]+= (m[f2] ^ z[e2]) + v[b2];	v[a3]+= (m[f3] ^ z[e3]) + v[b3];\
	v[ d] = ROR16(v[d] ^ v[a]);		v[d1] = ROR16(v[d1] ^ v[a1]);		v[d2] = ROR16(v[d2] ^ v[a2]);		v[d3] = ROR16(v[d3] ^ v[a3]);\
	v[ c]+= v[ d];				v[c1]+= v[d1];				v[c2]+= v[d2];				v[c3]+= v[d3];\
	v[ b] = ROR2(v[b] ^ v[c], 11);		v[b1] = ROR2(v[b1] ^ v[c1], 11);	v[b2] = ROR2(v[b2] ^ v[c2], 11);	v[b3] = ROR2(v[b3] ^ v[c3], 11);\
}

#define GSn4(a,b,c,d,e,f,a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3){\
	v[ a] = v[ a] + e + v[ b];		v[a1] = v[a1] + e1 + v[b1];		v[a2] = v[a2] + e2 + v[b2];		v[a3] = v[a3] + e3 + v[b3];\
	v[ d] = SWAPDWORDS2(v[ d] ^ v[ a]);	v[d1] = SWAPDWORDS2(v[d1] ^ v[a1]);	v[d2] = SWAPDWORDS2(v[d2] ^ v[a2]);	v[d3] = SWAPDWORDS2(v[d3] ^ v[a3]);\
	v[ c] = v[ c] + v[ d];			v[c1] = v[c1] + v[d1];			v[c2] = v[c2] + v[d2];			v[c3] = v[c3] + v[d3];\
	v[ b] = ROR2(v[b] ^ v[c],25);		v[b1] = ROR2(v[b1] ^ v[c1],25);		v[b2] = ROR2(v[b2] ^ v[c2],25);		v[b3] = ROR2(v[b3] ^ v[c3],25); \
	v[ a] = v[ a] + f + v[ b];		v[a1] = v[a1] + f1 + v[b1];		v[a2] = v[a2] + f2 + v[b2];		v[a3] = v[a3] + f3 + v[b3];\
	v[ d] = ROR16(v[d] ^ v[a]);		v[d1] = ROR16(v[d1] ^ v[a1]);		v[d2] = ROR16(v[d2] ^ v[a2]);		v[d3] = ROR16(v[d3] ^ v[a3]);\
	v[ c] = v[ c] + v[ d];			v[c1] = v[c1] + v[d1];			v[c2] = v[c2] + v[d2];			v[c3] = v[c3] + v[d3];\
	v[ b] = ROR2(v[b] ^ v[c],11);		v[b1] = ROR2(v[b1] ^ v[c1],11);		v[b2] = ROR2(v[b2] ^ v[c2],11);		v[b3] = ROR2(v[b3] ^ v[c3],11);\
}

#define GShost(a,b,c,d,e,f) { \
	v[a] += (m[e] ^ z[f]) + v[b]; \
	v[d] = ROTR64(v[d] ^ v[a],32); \
	v[c] += v[d]; \
	v[b] = ROTR64( v[b] ^ v[c], 25); \
	v[a] += (m[f] ^ z[e]) + v[b]; \
	v[d] = ROTR64( v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR64( v[b] ^ v[c], 11); \
}

__global__ __launch_bounds__(TPB50_64, 1)
void quark_blake512_gpu_hash_64(uint32_t threads, const uint32_t *const __restrict__ g_nonceVector, uint2* g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){
		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread];

		uint2 msg[16];

		uint2x4 *phash = (uint2x4*)&g_hash[hashPosition<<3];
		uint2x4 *outpt = (uint2x4*)msg;
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		uint2 m[16];
		m[ 0] = cuda_swab64_U2(msg[0]);
		m[ 1] = cuda_swab64_U2(msg[1]);
		m[ 2] = cuda_swab64_U2(msg[2]);
		m[ 3] = cuda_swab64_U2(msg[3]);
		m[ 4] = cuda_swab64_U2(msg[4]);
		m[ 5] = cuda_swab64_U2(msg[5]);
		m[ 6] = cuda_swab64_U2(msg[6]);
		m[ 7] = cuda_swab64_U2(msg[7]);
		m[ 8] = make_uint2(0,0x80000000);
		m[ 9] = make_uint2(0,0);
		m[10] = make_uint2(0,0);
		m[11] = make_uint2(0,0);
		m[12] = make_uint2(0,0);
		m[13] = make_uint2(1,0);
		m[14] = make_uint2(0,0);
		m[15] = make_uint2(0x200,0);

		uint2 v[16] = {
			h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
			z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]
		};
		v[12].x ^= 512U;
		v[13].x ^= 512U;

		GS4(0, 4, 8,12, 0, 1,		1, 5, 9,13, 2, 3,		2, 6,10,14, 4, 5,		3, 7,11,15, 6, 7);
		GS4(0, 5,10,15, 8, 9,		1, 6,11,12,10,11,		2, 7, 8,13,12,13,		3, 4, 9,14,14,15);

		GS4(0, 4, 8, 12, 14, 10,	1, 5, 9, 13, 4, 8,		2, 6, 10, 14, 9, 15,		3, 7, 11, 15, 13, 6);
		GS4(0, 5, 10, 15, 1, 12,	1, 6, 11, 12, 0, 2,		2, 7, 8, 13, 11, 7,		3, 4, 9, 14, 5, 3);

		GS4(0, 4, 8, 12, 11, 8,		1, 5, 9, 13, 12, 0,		2, 6, 10, 14, 5, 2,		3, 7, 11, 15, 15, 13);
		GS4(0, 5, 10, 15, 10, 14,	1, 6, 11, 12, 3, 6,		2, 7, 8, 13, 7, 1,		3, 4, 9, 14, 9, 4);

		GS4(0, 4, 8, 12, 7, 9,		1, 5, 9, 13, 3, 1,		2, 6, 10, 14, 13, 12,		3, 7, 11, 15, 11, 14);
		GS4(0, 5, 10, 15, 2, 6,		1, 6, 11, 12, 5, 10,		2, 7, 8, 13, 4, 0,		3, 4, 9, 14, 15, 8);

		GS4(0, 4, 8, 12, 9, 0,		1, 5, 9, 13, 5, 7,		2, 6, 10, 14, 2, 4,		3, 7, 11, 15, 10, 15);
		GS4(0, 5, 10, 15, 14, 1,	1, 6, 11, 12, 11, 12,		2, 7, 8, 13, 6, 8,		3, 4, 9, 14, 3, 13);

		GS4(0, 4, 8, 12, 2, 12,		1, 5, 9, 13, 6, 10,		2, 6, 10, 14, 0, 11,		3, 7, 11, 15, 8, 3);
		GS4(0, 5, 10, 15, 4, 13,	1, 6, 11, 12, 7, 5,		2, 7, 8, 13, 15, 14,		3, 4, 9, 14, 1, 9);

		GS4(0, 4, 8, 12, 12, 5,		1, 5, 9, 13, 1, 15,		2, 6, 10, 14, 14, 13,		3, 7, 11, 15, 4, 10);
		GS4(0, 5, 10, 15, 0, 7,		1, 6, 11, 12, 6, 3,		2, 7, 8, 13, 9, 2,		3, 4, 9, 14, 8, 11);

		GS4(0, 4, 8, 12, 13, 11,	1, 5, 9, 13, 7, 14,		2, 6, 10, 14, 12, 1,		3, 7, 11, 15, 3, 9);
		GS4(0, 5, 10, 15, 5, 0,		1, 6, 11, 12, 15, 4,		2, 7, 8, 13, 8, 6,		3, 4, 9, 14, 2, 10);

		GS4(0, 4, 8, 12, 6, 15,		1, 5, 9, 13, 14, 9,		2, 6, 10, 14, 11, 3,		3, 7, 11, 15, 0, 8);
		GS4(0, 5, 10, 15, 12, 2,	1, 6, 11, 12, 13, 7,		2, 7, 8, 13, 1, 4,		3, 4, 9, 14, 10, 5);

		GS4(0, 4, 8, 12, 10, 2,		1, 5, 9, 13, 8, 4,		2, 6, 10, 14, 7, 6,		3, 7, 11, 15, 1, 5);
		GS4(0, 5, 10, 15,15,11,		1, 6, 11, 12, 9, 14,		2, 7, 8, 13, 3, 12,		3, 4, 9, 14, 13, 0);

//		#if __CUDA_ARCH__ == 500

		GS4(0, 4, 8,12, 0, 1,		1, 5, 9,13, 2, 3,		2, 6,10,14, 4, 5,		3, 7,11,15, 6, 7);
		GS4(0, 5,10,15, 8, 9,		1, 6,11,12,10,11,		2, 7, 8,13,12,13,		3, 4, 9,14,14,15);

		GS4(0, 4, 8, 12, 14, 10,	1, 5, 9, 13, 4, 8,		2, 6, 10, 14, 9, 15,		3, 7, 11, 15, 13, 6);
		GS4(0, 5, 10, 15, 1, 12,	1, 6, 11, 12, 0, 2,		2, 7, 8, 13, 11, 7,		3, 4, 9, 14, 5, 3);

		GS4(0, 4, 8, 12, 11, 8,		1, 5, 9, 13, 12, 0,		2, 6, 10, 14, 5, 2,		3, 7, 11, 15, 15, 13);
		GS4(0, 5, 10, 15, 10, 14,	1, 6, 11, 12, 3, 6,		2, 7, 8, 13, 7, 1,		3, 4, 9, 14, 9, 4);

		GS4(0, 4, 8, 12, 7, 9,		1, 5, 9, 13, 3, 1,		2, 6, 10, 14, 13, 12,		3, 7, 11, 15, 11, 14);
		GS4(0, 5, 10, 15, 2, 6,		1, 6, 11, 12, 5, 10,		2, 7, 8, 13, 4, 0,		3, 4, 9, 14, 15, 8);

		GS4(0, 4, 8, 12, 9, 0,		1, 5, 9, 13, 5, 7,		2, 6, 10, 14, 2, 4,		3, 7, 11, 15, 10, 15);
		GS4(0, 5, 10, 15, 14, 1,	1, 6, 11, 12, 11, 12,		2, 7, 8, 13, 6, 8,		3, 4, 9, 14, 3, 13);

		GS4(0, 4, 8, 12, 2, 12,		1, 5, 9, 13, 6, 10,		2, 6, 10, 14, 0, 11,		3, 7, 11, 15, 8, 3);
		GS4(0, 5, 10, 15, 4, 13,	1, 6, 11, 12, 7, 5,		2, 7, 8, 13, 15, 14,		3, 4, 9, 14, 1, 9);

//		#else*/
/*
		for (int i = 0; i < 6; i++)
		{
			G4(0,	0, 4, 8,12,	1, 5, 9,13,	2, 6,10,14,	3, 7,11,15);
			G4(8,	0, 5,10,15,	1, 6,11,12,	2, 7, 8,13,	3, 4, 9,14);
		}
*/
//		#endif
		v[0] = cuda_swab64_U2(xor3x(v[0],h[0],v[ 8]));
		v[1] = cuda_swab64_U2(xor3x(v[1],h[1],v[ 9]));
		v[2] = cuda_swab64_U2(xor3x(v[2],h[2],v[10]));
		v[3] = cuda_swab64_U2(xor3x(v[3],h[3],v[11]));
		v[4] = cuda_swab64_U2(xor3x(v[4],h[4],v[12]));
		v[5] = cuda_swab64_U2(xor3x(v[5],h[5],v[13]));
		v[6] = cuda_swab64_U2(xor3x(v[6],h[6],v[14]));
		v[7] = cuda_swab64_U2(xor3x(v[7],h[7],v[15]));

/*		uint2* outHash = &g_hash[hashPosition<<3];
		#pragma unroll 8
		for(uint32_t i=0;i<8;i++){
			outHash[i] = v[i];
		}*/
		phash[0] = *(uint2x4*)&v[ 0];
		phash[1] = *(uint2x4*)&v[ 4];
	}
}

// ---------------------------- END CUDA quark_blake512 functions ------------------------------------
/*
__host__
void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_outputHash){
	uint32_t tpb = TPB52_64;
	int dev_id = device_map[thr_id];
	
	if (device_sm[dev_id] <= 500) tpb = TPB50_64;
*/
__host__
void quark_blake512_cpu_hash_64(int thr_id, const uint32_t threads, uint32_t *d_outputHash, const uint32_t tpb){

	const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);

	quark_blake512_gpu_hash_64<<<grid, block>>>(threads, NULL, (uint2*)d_outputHash);
}

__host__
void quark_blake512_cpu_init_64(int thr_id, uint32_t threads) {}

__host__
void quark_blake512_cpu_free_64(int thr_id) {}

__host__
int quark_blake512_calc_tpb_64(int thr_id) {

	int blockSize, minGridSize, maxActiveBlocks, device;
	cudaDeviceProp props;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, quark_blake512_gpu_hash_64, 0,	0);

	// calculate theoretical occupancy
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, quark_blake512_gpu_hash_64, blockSize, 0);
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
	float occupancy = (maxActiveBlocks * blockSize / props.warpSize)
			/ (float) (props.maxThreadsPerMultiProcessor / props.warpSize);

	if (!opt_quiet) gpulog(LOG_INFO, thr_id, "blake512_64 tpb calc - block size %d. Theoretical occupancy: %f", blockSize, minGridSize, occupancy);

	return (uint32_t)blockSize;
}
