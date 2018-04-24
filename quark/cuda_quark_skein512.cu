/* Based on SP's work
 * 
 * Provos Alexis - 2016
 * graemes 2018
 */

#include "miner.h"
#include "cuda_vectors.h"
#include "skein_header.h"

#define TPB 512
#define TPF 3

/* ************************ */
__constant__ const uint2 buffer[152] = {
	{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC33,0xAE18A40B},
	{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC73,0x9E18A40B},{0x98173EC5,0xCAB2076D},
	{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC73,0x9E18A40B},{0x98173F04,0xCAB2076D},{0x749C51D0,0x4903ADFF},
	{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173F04,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF06,0x0D95DE39},
	{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79BD2,0x8FD19341},
	{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CB6,0x9A255629},
	{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7B6,0x5DB62599},
	{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C3FB,0xEABE394C},
	{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B52B,0x991112C7},
	{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC3C,0xAE18A40B},
	{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC73,0x9E18A40B},{0x98173ece,0xcab2076d},
	{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC73,0x9E18A40B},{0x98173F04,0xCAB2076D},{0x749C51D9,0x4903ADFF},
	{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173F04,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF0F,0x0D95DE39},
	{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79BDB,0x8FD19341},
	{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CBF,0x9A255629},
	{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7BF,0x5DB62599},
	{0x660FCC33,0xAE18A40B},{0x98173ec4,0xcab2076d},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C404,0xEABE394C},
	{0x98173ec4,0xcab2076d},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B534,0x991112C7},
	{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC45,0xAE18A40B}
};

__global__
__launch_bounds__(TPB, TPF)
void quark_skein512_gpu_hash_64(const uint32_t threads, uint64_t* g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		// Skein
		uint2 p[8], h[9];

		uint64_t *Hash = &g_hash[thread<<3];

		uint2x4 *phash = (uint2x4*)Hash;
		*(uint2x4*)&p[0] = __ldg4(&phash[0]);
		*(uint2x4*)&p[4] = __ldg4(&phash[1]);
		
		h[0] = p[0];	h[1] = p[1];	h[2] = p[2];	h[3] = p[3];
		h[4] = p[4];	h[5] = p[5];	h[6] = p[6];	h[7] = p[7];

		p[0] += buffer[0];	p[1] += buffer[1];	p[2] += buffer[2];	p[3] += buffer[3];	p[4] += buffer[4];	p[5] += buffer[5];	p[6] += buffer[6];	p[7] += buffer[7];
		TFBIGMIX8e();
		p[0] += buffer[8];	p[1] += buffer[9];	p[2] += buffer[10];	p[3] += buffer[11];	p[4] += buffer[12];	p[5] += buffer[13];	p[6] += buffer[14];	p[7] += buffer[15];
		TFBIGMIX8o();
		p[0] += buffer[16];	p[1] += buffer[17];	p[2] += buffer[18];	p[3] += buffer[19];	p[4] += buffer[20];	p[5] += buffer[21];	p[6] += buffer[22];	p[7] += buffer[23];
		TFBIGMIX8e();
		p[0] += buffer[24];	p[1] += buffer[25];	p[2] += buffer[26];	p[3] += buffer[27];	p[4] += buffer[28];	p[5] += buffer[29];	p[6] += buffer[30];	p[7] += buffer[31];
		TFBIGMIX8o();
		p[0] += buffer[32];	p[1] += buffer[33];	p[2] += buffer[34];	p[3] += buffer[35];	p[4] += buffer[36];	p[5] += buffer[37];	p[6] += buffer[38];	p[7] += buffer[39];
		TFBIGMIX8e();
		p[0] += buffer[40];	p[1] += buffer[41];	p[2] += buffer[42];	p[3] += buffer[43];	p[4] += buffer[44];	p[5] += buffer[45];	p[6] += buffer[46];	p[7] += buffer[47];
		TFBIGMIX8o();
		p[0] += buffer[48];	p[1] += buffer[49];	p[2] += buffer[50];	p[3] += buffer[51];	p[4] += buffer[52];	p[5] += buffer[53];	p[6] += buffer[54];	p[7] += buffer[55];
		TFBIGMIX8e();
		p[0] += buffer[56];	p[1] += buffer[57];	p[2] += buffer[58];	p[3] += buffer[59];	p[4] += buffer[60];	p[5] += buffer[61];	p[6] += buffer[62];	p[7] += buffer[63];
		TFBIGMIX8o();
		p[0] += buffer[64];	p[1] += buffer[65];	p[2] += buffer[66];	p[3] += buffer[67];	p[4] += buffer[68];	p[5] += buffer[69];	p[6] += buffer[70];	p[7] += buffer[71];
		TFBIGMIX8e();
		p[0] += buffer[72];	p[1] += buffer[73];	p[2] += buffer[74];	p[3] += buffer[75];	p[4] += buffer[76];	p[5] += buffer[77];	p[6] += buffer[78];	p[7] += buffer[79];
		TFBIGMIX8o();
		p[0] += buffer[80];	p[1] += buffer[81];	p[2] += buffer[82];	p[3] += buffer[83];	p[4] += buffer[84];	p[5] += buffer[85];	p[6] += buffer[86];	p[7] += buffer[87];
		TFBIGMIX8e();
		p[0] += buffer[88];	p[1] += buffer[89];	p[2] += buffer[90];	p[3] += buffer[91];	p[4] += buffer[92];	p[5] += buffer[93];	p[6] += buffer[94];	p[7] += buffer[95];
		TFBIGMIX8o();
		p[0] += buffer[96];	p[1] += buffer[97];	p[2] += buffer[98];	p[3] += buffer[99];	p[4] += buffer[100];	p[5] += buffer[101];	p[6] += buffer[102];	p[7] += buffer[103];
		TFBIGMIX8e();
		p[0] += buffer[104];	p[1] += buffer[105];	p[2] += buffer[106];	p[3] += buffer[107];	p[4] += buffer[108];	p[5] += buffer[109];	p[6] += buffer[110];	p[7] += buffer[111];
		TFBIGMIX8o();
		p[0] += buffer[112];	p[1] += buffer[113];	p[2] += buffer[114];	p[3] += buffer[115];	p[4] += buffer[116];	p[5] += buffer[117];	p[6] += buffer[118];	p[7] += buffer[119];
		TFBIGMIX8e();
		p[0] += buffer[120];	p[1] += buffer[121];	p[2] += buffer[122];	p[3] += buffer[123];	p[4] += buffer[124];	p[5] += buffer[125];	p[6] += buffer[126];	p[7] += buffer[127];
		TFBIGMIX8o();
		p[0] += buffer[128];	p[1] += buffer[129];	p[2] += buffer[130];	p[3] += buffer[131];	p[4] += buffer[132];	p[5] += buffer[133];	p[6] += buffer[134];	p[7] += buffer[135];
		TFBIGMIX8e();
		p[0] += buffer[136];	p[1] += buffer[137];	p[2] += buffer[138];	p[3] += buffer[139];	p[4] += buffer[140];	p[5] += buffer[141];	p[6] += buffer[142];	p[7] += buffer[143];
		TFBIGMIX8o();
		p[0] += buffer[144];	p[1] += buffer[145];	p[2] += buffer[146];	p[3] += buffer[147];	p[4] += buffer[148];	p[5] += buffer[149];	p[6] += buffer[150];	p[7] += buffer[151];

		h[0]^= p[0];
		h[1]^= p[1];
		h[2]^= p[2];
		h[3]^= p[3];
		h[4]^= p[4];
		h[5]^= p[5];
		h[6]^= p[6];
		h[7]^= p[7];
		
		h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ vectorize(0x1BD11BDAA9FC1A22);

		uint32_t t0;
		uint2 t1,t2;
		t0 = 8;
		t1 = vectorize(0xFF00000000000000);
		t2 = t1+t0;

		p[5] = h[5] + 8U;

		p[0] = h[0] + h[1];
		p[1] = ROL2(h[1], 46) ^ p[0];
		p[2] = h[2] + h[3];
		p[3] = ROL2(h[3], 36) ^ p[2];
		p[4] = h[4] + p[5];
		p[5] = ROL2(p[5], 19) ^ p[4];
		p[6] = (h[6] + h[7] + t1);
		p[7] = ROL2(h[7], 37) ^ p[6];
		p[2]+= p[1];
		p[1] = ROL2(p[1], 33) ^ p[2];
		p[4]+= p[7];
		p[7] = ROL2(p[7], 27) ^ p[4];
		p[6]+= p[5];
		p[5] = ROL2(p[5], 14) ^ p[6];
		p[0]+= p[3];
		p[3] = ROL2(p[3], 42) ^ p[0];
		p[4]+= p[1];
		p[1] = ROL2(p[1], 17) ^ p[4];
		p[6]+= p[3];
		p[3] = ROL2(p[3], 49) ^ p[6];
		p[0]+= p[5];
		p[5] = ROL2(p[5], 36) ^ p[0];
		p[2]+= p[7];
		p[7] = ROL2(p[7], 39) ^ p[2];
		p[6]+= p[1];
		p[1] = ROL2(p[1], 44) ^ p[6];
		p[0]+= p[7];
		p[7] = ROL2(p[7], 9) ^ p[0];
		p[2]+= p[5];
		p[5] = ROL2(p[5], 54) ^ p[2];
		p[4]+= p[3];
		p[3] = ROR8(p[3]) ^ p[4];
		
		p[0]+= h[1];		p[1]+= h[2];
		p[2]+= h[3];		p[3]+= h[4];
		p[4]+= h[5];		p[5]+= h[6] + t1;
		p[6]+= h[7] + t2;	p[7]+= h[8] + 1U;
		TFBIGMIX8o();
		p[0]+= h[2];		p[1]+= h[3];
		p[2]+= h[4];		p[3]+= h[5];
		p[4]+= h[6];		p[5]+= h[7] + t2;
		p[6]+= h[8] + t0;	p[7]+= h[0] + 2U;
		TFBIGMIX8e();
		p[0]+= h[3];		p[1]+= h[4];
		p[2]+= h[5];		p[3]+= h[6];
		p[4]+= h[7];		p[5]+= h[8] + t0;
		p[6]+= h[0] + t1;	p[7]+= h[1] + 3U;
		TFBIGMIX8o();
		p[0]+= h[4];		p[1]+= h[5];
		p[2]+= h[6];		p[3]+= h[7];
		p[4]+= h[8];		p[5]+= h[0] + t1;
		p[6]+= h[1] + t2;	p[7]+= h[2] + 4U;
		TFBIGMIX8e();
		p[0]+= h[5];		p[1]+= h[6];
		p[2]+= h[7];		p[3]+= h[8];
		p[4]+= h[0];		p[5]+= h[1] + t2;
		p[6]+= h[2] + t0;	p[7]+= h[3] + 5U;
		TFBIGMIX8o();
		p[0]+= h[6];		p[1]+= h[7];
		p[2]+= h[8];		p[3]+= h[0];
		p[4]+= h[1];		p[5]+= h[2] + t0;
		p[6]+= h[3] + t1;	p[7]+= h[4] + 6U;
		TFBIGMIX8e();
		p[0]+= h[7];		p[1]+= h[8];
		p[2]+= h[0];		p[3]+= h[1];
		p[4]+= h[2];		p[5]+= h[3] + t1;
		p[6]+= h[4] + t2;	p[7]+= h[5] + 7U;
		TFBIGMIX8o();
		p[0]+= h[8];		p[1]+= h[0];
		p[2]+= h[1];		p[3]+= h[2];
		p[4]+= h[3];		p[5]+= h[4] + t2;
		p[6]+= h[5] + t0;	p[7]+= h[6] + 8U;
		TFBIGMIX8e();
		p[0]+= h[0];		p[1]+= h[1];
		p[2]+= h[2];		p[3]+= h[3];
		p[4]+= h[4];		p[5]+= h[5] + t0;
		p[6]+= h[6] + t1;	p[7]+= h[7] + 9U;
		TFBIGMIX8o();
		p[0] = p[0] + h[1];	p[1] = p[1] + h[2];
		p[2] = p[2] + h[3];	p[3] = p[3] + h[4];
		p[4] = p[4] + h[5];	p[5] = p[5] + h[6] + t1;
		p[6] = p[6] + h[7] + t2;p[7] = p[7] + h[8] + 10U;
		TFBIGMIX8e();
		p[0] = p[0] + h[2];	p[1] = p[1] + h[3];
		p[2] = p[2] + h[4];	p[3] = p[3] + h[5];
		p[4] = p[4] + h[6];	p[5] = p[5] + h[7] + t2;
		p[6] = p[6] + h[8] + t0;p[7] = p[7] + h[0] + 11U;
		TFBIGMIX8o();
		p[0] = p[0] + h[3];	p[1] = p[1] + h[4];
		p[2] = p[2] + h[5];	p[3] = p[3] + h[6];
		p[4] = p[4] + h[7];	p[5] = p[5] + h[8] + t0;
		p[6] = p[6] + h[0] + t1;p[7] = p[7] + h[1] + 12U;
		TFBIGMIX8e();
		p[0] = p[0] + h[4];	p[1] = p[1] + h[5];
		p[2] = p[2] + h[6];	p[3] = p[3] + h[7];
		p[4] = p[4] + h[8];	p[5] = p[5] + h[0] + t1;
		p[6] = p[6] + h[1] + t2;p[7] = p[7] + h[2] + 13U;
		TFBIGMIX8o();
		p[0] = p[0] + h[5];	p[1] = p[1] + h[6];
		p[2] = p[2] + h[7];	p[3] = p[3] + h[8];
		p[4] = p[4] + h[0];	p[5] = p[5] + h[1] + t2;
		p[6] = p[6] + h[2] + t0;p[7] = p[7] + h[3] + 14U;
		TFBIGMIX8e();
		p[0] = p[0] + h[6];	p[1] = p[1] + h[7];
		p[2] = p[2] + h[8];	p[3] = p[3] + h[0];
		p[4] = p[4] + h[1];	p[5] = p[5] + h[2] + t0;
		p[6] = p[6] + h[3] + t1;p[7] = p[7] + h[4] + 15U;
		TFBIGMIX8o();
		p[0] = p[0] + h[7];	p[1] = p[1] + h[8];
		p[2] = p[2] + h[0];	p[3] = p[3] + h[1];
		p[4] = p[4] + h[2];	p[5] = p[5] + h[3] + t1;
		p[6] = p[6] + h[4] + t2;p[7] = p[7] + h[5] + 16U;
		TFBIGMIX8e();
		p[0] = p[0] + h[8];	p[1] = p[1] + h[0];
		p[2] = p[2] + h[1];	p[3] = p[3] + h[2];
		p[4] = p[4] + h[3];	p[5] = p[5] + h[4] + t2;
		p[6] = p[6] + h[5] + t0;p[7] = p[7] + h[6] + 17U;
		TFBIGMIX8o();
		p[0] = p[0] + h[0];	p[1] = p[1] + h[1];
		p[2] = p[2] + h[2];	p[3] = p[3] + h[3];
		p[4] = p[4] + h[4];	p[5] = p[5] + h[5] + t0;
		p[6] = p[6] + h[6] + t1;p[7] = p[7] + h[7] + 18U;
		
		phash = (uint2x4*)p;
		uint2x4 *outpt = (uint2x4*)Hash;
		outpt[0] = phash[0];
		outpt[1] = phash[1];

	}
}

__host__
void quark_skein512_cpu_hash_64(const int thr_id, const uint32_t threads, uint32_t *d_hash, const uint32_t tpb)
{
	const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);

	quark_skein512_gpu_hash_64 << <grid, block >> >(threads, (uint64_t*)d_hash);
}

__host__
void quark_skein512_cpu_init_64(const int thr_id, uint32_t threads) {}

__host__
void quark_skein512_cpu_free_64(const int thr_id) {}

__host__
uint32_t quark_skein512_calc_tpb_64(const int thr_id) {

	int blockSize = 0;
	int minGridSize = 0;
	int maxActiveBlocks, device;
	cudaDeviceProp props;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, quark_skein512_gpu_hash_64, 0,	0);

	// calculate theoretical occupancy
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, quark_skein512_gpu_hash_64, blockSize, 0);
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
	float occupancy = (maxActiveBlocks * blockSize / props.warpSize)
			/ (float) (props.maxThreadsPerMultiProcessor / props.warpSize);

	if (!opt_quiet) gpulog(LOG_INFO, thr_id, "skein512_64 tpb calc - block size %d. Theoretical occupancy: %f", blockSize, minGridSize, occupancy);

	return (uint32_t)blockSize;
}
