/*
 * Quick Hamsi-512 for X13 by tsiv - 2014
 * + Hamsi-512 80 by tpruvot - 2018
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda_helper.h"

typedef unsigned char BitSequence;

static __constant__ uint32_t d_alpha_n[32];
static __constant__ uint32_t d_alpha_f[32];
static __constant__ uint32_t d_T512[64][16];

#define hamsi_s00   m0
#define hamsi_s01   m1
#define hamsi_s02   c0
#define hamsi_s03   c1
#define hamsi_s04   m2
#define hamsi_s05   m3
#define hamsi_s06   c2
#define hamsi_s07   c3
#define hamsi_s08   c4
#define hamsi_s09   c5
#define hamsi_s0A   m4
#define hamsi_s0B   m5
#define hamsi_s0C   c6
#define hamsi_s0D   c7
#define hamsi_s0E   m6
#define hamsi_s0F   m7
#define hamsi_s10   m8
#define hamsi_s11   m9
#define hamsi_s12   c8
#define hamsi_s13   c9
#define hamsi_s14   mA
#define hamsi_s15   mB
#define hamsi_s16   cA
#define hamsi_s17   cB
#define hamsi_s18   cC
#define hamsi_s19   cD
#define hamsi_s1A   mC
#define hamsi_s1B   mD
#define hamsi_s1C   cE
#define hamsi_s1D   cF
#define hamsi_s1E   mE
#define hamsi_s1F   mF

#define SBOX(a, b, c, d) { \
		uint32_t t; \
		t = (a); \
		(a) &= (c); \
		(a) ^= (d); \
		(c) ^= (b); \
		(c) ^= (a); \
		(d) |= t; \
		(d) ^= (b); \
		t ^= (c); \
		(b) = (d); \
		(d) |= t; \
		(d) ^= (a); \
		(a) &= (b); \
		t ^= (a); \
		(b) ^= (d); \
		(b) ^= t; \
		(a) = (c); \
		(c) = (b); \
		(b) = (d); \
		(d) = SPH_T32(~t); \
	}

#define HAMSI_L(a, b, c, d) { \
		(a) = ROTL32(a, 13); \
		(c) = ROTL32(c, 3); \
		(b) ^= (a) ^ (c); \
		(d) ^= (c) ^ ((a) << 3); \
		(b) = ROTL32(b, 1); \
		(d) = ROTL32(d, 7); \
		(a) ^= (b) ^ (d); \
		(c) ^= (d) ^ ((b) << 7); \
		(a) = ROTL32(a, 5); \
		(c) = ROTL32(c, 22); \
	}

#define ROUND_BIG(rc, alpha) { \
		hamsi_s00 ^= alpha[0x00]; \
		hamsi_s08 ^= alpha[0x08]; \
		hamsi_s10 ^= alpha[0x10]; \
		hamsi_s18 ^= alpha[0x18]; \
		hamsi_s01 ^= alpha[0x01] ^ (uint32_t)(rc); \
		hamsi_s09 ^= alpha[0x09]; \
		hamsi_s11 ^= alpha[0x11]; \
		hamsi_s19 ^= alpha[0x19]; \
		hamsi_s02 ^= alpha[0x02]; \
		hamsi_s0A ^= alpha[0x0A]; \
		hamsi_s12 ^= alpha[0x12]; \
		hamsi_s1A ^= alpha[0x1A]; \
		hamsi_s03 ^= alpha[0x03]; \
		hamsi_s0B ^= alpha[0x0B]; \
		hamsi_s13 ^= alpha[0x13]; \
		hamsi_s1B ^= alpha[0x1B]; \
		hamsi_s04 ^= alpha[0x04]; \
		hamsi_s0C ^= alpha[0x0C]; \
		hamsi_s14 ^= alpha[0x14]; \
		hamsi_s1C ^= alpha[0x1C]; \
		hamsi_s05 ^= alpha[0x05]; \
		hamsi_s0D ^= alpha[0x0D]; \
		hamsi_s15 ^= alpha[0x15]; \
		hamsi_s1D ^= alpha[0x1D]; \
		hamsi_s06 ^= alpha[0x06]; \
		hamsi_s0E ^= alpha[0x0E]; \
		hamsi_s16 ^= alpha[0x16]; \
		hamsi_s1E ^= alpha[0x1E]; \
		hamsi_s07 ^= alpha[0x07]; \
		hamsi_s0F ^= alpha[0x0F]; \
		hamsi_s17 ^= alpha[0x17]; \
		hamsi_s1F ^= alpha[0x1F]; \
		SBOX(hamsi_s00, hamsi_s08, hamsi_s10, hamsi_s18); \
		SBOX(hamsi_s01, hamsi_s09, hamsi_s11, hamsi_s19); \
		SBOX(hamsi_s02, hamsi_s0A, hamsi_s12, hamsi_s1A); \
		SBOX(hamsi_s03, hamsi_s0B, hamsi_s13, hamsi_s1B); \
		SBOX(hamsi_s04, hamsi_s0C, hamsi_s14, hamsi_s1C); \
		SBOX(hamsi_s05, hamsi_s0D, hamsi_s15, hamsi_s1D); \
		SBOX(hamsi_s06, hamsi_s0E, hamsi_s16, hamsi_s1E); \
		SBOX(hamsi_s07, hamsi_s0F, hamsi_s17, hamsi_s1F); \
		HAMSI_L(hamsi_s00, hamsi_s09, hamsi_s12, hamsi_s1B); \
		HAMSI_L(hamsi_s01, hamsi_s0A, hamsi_s13, hamsi_s1C); \
		HAMSI_L(hamsi_s02, hamsi_s0B, hamsi_s14, hamsi_s1D); \
		HAMSI_L(hamsi_s03, hamsi_s0C, hamsi_s15, hamsi_s1E); \
		HAMSI_L(hamsi_s04, hamsi_s0D, hamsi_s16, hamsi_s1F); \
		HAMSI_L(hamsi_s05, hamsi_s0E, hamsi_s17, hamsi_s18); \
		HAMSI_L(hamsi_s06, hamsi_s0F, hamsi_s10, hamsi_s19); \
		HAMSI_L(hamsi_s07, hamsi_s08, hamsi_s11, hamsi_s1A); \
		HAMSI_L(hamsi_s00, hamsi_s02, hamsi_s05, hamsi_s07); \
		HAMSI_L(hamsi_s10, hamsi_s13, hamsi_s15, hamsi_s16); \
		HAMSI_L(hamsi_s09, hamsi_s0B, hamsi_s0C, hamsi_s0E); \
		HAMSI_L(hamsi_s19, hamsi_s1A, hamsi_s1C, hamsi_s1F); \
	}


#define P_BIG  { \
		for( int r = 0; r < 6; r++ ) \
			ROUND_BIG(r, d_alpha_n); \
	}

#define PF_BIG { \
		for( int r = 0; r < 12; r++ ) \
			ROUND_BIG(r, d_alpha_f); \
	}

#define T_BIG  { \
		/* order is important */ \
		cF = (h[0xF] ^= hamsi_s17); \
		cE = (h[0xE] ^= hamsi_s16); \
		cD = (h[0xD] ^= hamsi_s15); \
		cC = (h[0xC] ^= hamsi_s14); \
		cB = (h[0xB] ^= hamsi_s13); \
		cA = (h[0xA] ^= hamsi_s12); \
		c9 = (h[0x9] ^= hamsi_s11); \
		c8 = (h[0x8] ^= hamsi_s10); \
		c7 = (h[0x7] ^= hamsi_s07); \
		c6 = (h[0x6] ^= hamsi_s06); \
		c5 = (h[0x5] ^= hamsi_s05); \
		c4 = (h[0x4] ^= hamsi_s04); \
		c3 = (h[0x3] ^= hamsi_s03); \
		c2 = (h[0x2] ^= hamsi_s02); \
		c1 = (h[0x1] ^= hamsi_s01); \
		c0 = (h[0x0] ^= hamsi_s00); \
	}


__constant__ static uint64_t c_PaddedMessage80[10];

__host__
void x16_hamsi512_setBlock_80(void *pdata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

__global__
void x16_hamsi512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		unsigned char h1[80];
		#pragma unroll
		for (int i = 0; i < 10; i++)
			((uint2*)h1)[i] = ((uint2*)c_PaddedMessage80)[i];
		//((uint64_t*)h1)[9] = REPLACE_HIDWORD(c_PaddedMessage80[9], cuda_swab32(startNonce + thread));
		((uint32_t*)h1)[19] = cuda_swab32(startNonce + thread);

		uint32_t c0 = 0x73746565, c1 = 0x6c706172, c2 = 0x6b204172, c3 = 0x656e6265;
		uint32_t c4 = 0x72672031, c5 = 0x302c2062, c6 = 0x75732032, c7 = 0x3434362c;
		uint32_t c8 = 0x20422d33, c9 = 0x30303120, cA = 0x4c657576, cB = 0x656e2d48;
		uint32_t cC = 0x65766572, cD = 0x6c65652c, cE = 0x2042656c, cF = 0x6769756d;
		uint32_t h[16] = { c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF };
		uint32_t m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, mA, mB, mC, mD, mE, mF;
		uint32_t *tp, db, dm;

		for(int i = 0; i < 80; i += 8)
		{
			m0 = 0; m1 = 0; m2 = 0; m3 = 0; m4 = 0; m5 = 0; m6 = 0; m7 = 0;
			m8 = 0; m9 = 0; mA = 0; mB = 0; mC = 0; mD = 0; mE = 0; mF = 0;
			tp = &d_T512[0][0];

			#pragma unroll
			for (int u = 0; u < 8; u++) {
				db = h1[i + u];
				#pragma unroll 2
				for (int v = 0; v < 8; v++, db >>= 1) {
					dm = -(uint32_t)(db & 1);
					m0 ^= dm & tp[ 0]; m1 ^= dm & tp[ 1];
					m2 ^= dm & tp[ 2]; m3 ^= dm & tp[ 3];
					m4 ^= dm & tp[ 4]; m5 ^= dm & tp[ 5];
					m6 ^= dm & tp[ 6]; m7 ^= dm & tp[ 7];
					m8 ^= dm & tp[ 8]; m9 ^= dm & tp[ 9];
					mA ^= dm & tp[10]; mB ^= dm & tp[11];
					mC ^= dm & tp[12]; mD ^= dm & tp[13];
					mE ^= dm & tp[14]; mF ^= dm & tp[15];
					tp += 16;
				}
			}

			#pragma unroll
			for (int r = 0; r < 6; r++) {
				ROUND_BIG(r, d_alpha_n);
			}
			T_BIG;
		}

		#define INPUT_BIG { \
			m0 = 0; m1 = 0; m2 = 0; m3 = 0; m4 = 0; m5 = 0; m6 = 0; m7 = 0; \
			m8 = 0; m9 = 0; mA = 0; mB = 0; mC = 0; mD = 0; mE = 0; mF = 0; \
			tp = &d_T512[0][0]; \
			for (int u = 0; u < 8; u++) { \
				db = endtag[u]; \
				for (int v = 0; v < 8; v++, db >>= 1) { \
					dm = -(uint32_t)(db & 1); \
					m0 ^= dm & tp[ 0]; m1 ^= dm & tp[ 1]; \
					m2 ^= dm & tp[ 2]; m3 ^= dm & tp[ 3]; \
					m4 ^= dm & tp[ 4]; m5 ^= dm & tp[ 5]; \
					m6 ^= dm & tp[ 6]; m7 ^= dm & tp[ 7]; \
					m8 ^= dm & tp[ 8]; m9 ^= dm & tp[ 9]; \
					mA ^= dm & tp[10]; mB ^= dm & tp[11]; \
					mC ^= dm & tp[12]; mD ^= dm & tp[13]; \
					mE ^= dm & tp[14]; mF ^= dm & tp[15]; \
					tp += 16; \
				} \
			} \
		}

		// close
		uint8_t endtag[8] = { 0x80, 0x00, 0x00, 0x00,  0x00, 0x00, 0x00, 0x00 };
		INPUT_BIG;

		#pragma unroll
		for (int r = 0; r < 6; r++) {
			ROUND_BIG(r, d_alpha_n);
		}
		T_BIG;

		endtag[0] = endtag[1] = 0x00;
		endtag[6] = 0x02;
		endtag[7] = 0x80;
		INPUT_BIG;

		// PF_BIG
		#pragma unroll
		for(int r = 0; r < 12; r++) {
			ROUND_BIG(r, d_alpha_f);
		}
		T_BIG;

		uint64_t hashPosition = thread;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition << 3];
		#pragma unroll 16
		for(int i = 0; i < 16; i++)
			Hash[i] = cuda_swab32(h[i]);

		#undef INPUT_BIG
	}
}

__host__
void x16_hamsi512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x16_hamsi512_gpu_hash_80 <<<grid, block>>> (threads, startNounce, (uint64_t*)d_hash);
}
