/* SKEIN 80 based on Alexis Provos version
 *
 * tpruvot - ????
 * graemes - 2018
*/

#include <stdio.h>
#include <cuda_vectors.h>

#define TPB50 256
#define TPB52 512
#define TPF50 5
#define TPF52 3


/* ************************ */

/*
 * M9_ ## s ## _ ## i  evaluates to s+i mod 9 (0 <= s <= 18, 0 <= i <= 7).
 */

#define M9_0_0    0
#define M9_0_1    1
#define M9_0_2    2
#define M9_0_3    3
#define M9_0_4    4
#define M9_0_5    5
#define M9_0_6    6
#define M9_0_7    7

#define M9_1_0    1
#define M9_1_1    2
#define M9_1_2    3
#define M9_1_3    4
#define M9_1_4    5
#define M9_1_5    6
#define M9_1_6    7
#define M9_1_7    8

#define M9_2_0    2
#define M9_2_1    3
#define M9_2_2    4
#define M9_2_3    5
#define M9_2_4    6
#define M9_2_5    7
#define M9_2_6    8
#define M9_2_7    0

#define M9_3_0    3
#define M9_3_1    4
#define M9_3_2    5
#define M9_3_3    6
#define M9_3_4    7
#define M9_3_5    8
#define M9_3_6    0
#define M9_3_7    1

#define M9_4_0    4
#define M9_4_1    5
#define M9_4_2    6
#define M9_4_3    7
#define M9_4_4    8
#define M9_4_5    0
#define M9_4_6    1
#define M9_4_7    2

#define M9_5_0    5
#define M9_5_1    6
#define M9_5_2    7
#define M9_5_3    8
#define M9_5_4    0
#define M9_5_5    1
#define M9_5_6    2
#define M9_5_7    3

#define M9_6_0    6
#define M9_6_1    7
#define M9_6_2    8
#define M9_6_3    0
#define M9_6_4    1
#define M9_6_5    2
#define M9_6_6    3
#define M9_6_7    4

#define M9_7_0    7
#define M9_7_1    8
#define M9_7_2    0
#define M9_7_3    1
#define M9_7_4    2
#define M9_7_5    3
#define M9_7_6    4
#define M9_7_7    5

#define M9_8_0    8
#define M9_8_1    0
#define M9_8_2    1
#define M9_8_3    2
#define M9_8_4    3
#define M9_8_5    4
#define M9_8_6    5
#define M9_8_7    6

#define M9_9_0    0
#define M9_9_1    1
#define M9_9_2    2
#define M9_9_3    3
#define M9_9_4    4
#define M9_9_5    5
#define M9_9_6    6
#define M9_9_7    7

#define M9_10_0   1
#define M9_10_1   2
#define M9_10_2   3
#define M9_10_3   4
#define M9_10_4   5
#define M9_10_5   6
#define M9_10_6   7
#define M9_10_7   8

#define M9_11_0   2
#define M9_11_1   3
#define M9_11_2   4
#define M9_11_3   5
#define M9_11_4   6
#define M9_11_5   7
#define M9_11_6   8
#define M9_11_7   0

#define M9_12_0   3
#define M9_12_1   4
#define M9_12_2   5
#define M9_12_3   6
#define M9_12_4   7
#define M9_12_5   8
#define M9_12_6   0
#define M9_12_7   1

#define M9_13_0   4
#define M9_13_1   5
#define M9_13_2   6
#define M9_13_3   7
#define M9_13_4   8
#define M9_13_5   0
#define M9_13_6   1
#define M9_13_7   2

#define M9_14_0   5
#define M9_14_1   6
#define M9_14_2   7
#define M9_14_3   8
#define M9_14_4   0
#define M9_14_5   1
#define M9_14_6   2
#define M9_14_7   3

#define M9_15_0   6
#define M9_15_1   7
#define M9_15_2   8
#define M9_15_3   0
#define M9_15_4   1
#define M9_15_5   2
#define M9_15_6   3
#define M9_15_7   4

#define M9_16_0   7
#define M9_16_1   8
#define M9_16_2   0
#define M9_16_3   1
#define M9_16_4   2
#define M9_16_5   3
#define M9_16_6   4
#define M9_16_7   5

#define M9_17_0   8
#define M9_17_1   0
#define M9_17_2   1
#define M9_17_3   2
#define M9_17_4   3
#define M9_17_5   4
#define M9_17_6   5
#define M9_17_7   6

#define M9_18_0   0
#define M9_18_1   1
#define M9_18_2   2
#define M9_18_3   3
#define M9_18_4   4
#define M9_18_5   5
#define M9_18_6   6
#define M9_18_7   7

/*
 * M3_ ## s ## _ ## i  evaluates to s+i mod 3 (0 <= s <= 18, 0 <= i <= 1).
 */

#define M3_0_0    0
#define M3_0_1    1
#define M3_1_0    1
#define M3_1_1    2
#define M3_2_0    2
#define M3_2_1    0
#define M3_3_0    0
#define M3_3_1    1
#define M3_4_0    1
#define M3_4_1    2
#define M3_5_0    2
#define M3_5_1    0
#define M3_6_0    0
#define M3_6_1    1
#define M3_7_0    1
#define M3_7_1    2
#define M3_8_0    2
#define M3_8_1    0
#define M3_9_0    0
#define M3_9_1    1
#define M3_10_0   1
#define M3_10_1   2
#define M3_11_0   2
#define M3_11_1   0
#define M3_12_0   0
#define M3_12_1   1
#define M3_13_0   1
#define M3_13_1   2
#define M3_14_0   2
#define M3_14_1   0
#define M3_15_0   0
#define M3_15_1   1
#define M3_16_0   1
#define M3_16_1   2
#define M3_17_0   2
#define M3_17_1   0
#define M3_18_0   0
#define M3_18_1   1

#define XCAT(x, y)     XCAT_(x, y)
#define XCAT_(x, y)    x ## y

#define SKBI(k, s, i)   XCAT(k, XCAT(XCAT(XCAT(M9_, s), _), i))
#define SKBT(t, s, v)   XCAT(t, XCAT(XCAT(XCAT(M3_, s), _), v))

#define TFBIG_ADDKEY(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
	w0 = (w0 + SKBI(k, s, 0)); \
	w1 = (w1 + SKBI(k, s, 1)); \
	w2 = (w2 + SKBI(k, s, 2)); \
	w3 = (w3 + SKBI(k, s, 3)); \
	w4 = (w4 + SKBI(k, s, 4)); \
	w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
	w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
	w7 = (w7 + SKBI(k, s, 7) + make_uint2(s,0); \
}

#define TFBIG_MIX(x0, x1, rc) { \
	x0 = x0 + x1; \
	x1 = ROL2(x1, rc) ^ x0; \
}

#define TFBIG_MIX8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
	TFBIG_MIX(w0, w1, rc0); \
	TFBIG_MIX(w2, w3, rc1); \
	TFBIG_MIX(w4, w5, rc2); \
	TFBIG_MIX(w6, w7, rc3); \
}

#define TFBIG_4e(s)  { \
	TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
	TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
	TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
	TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
}

#define TFBIG_4o(s)  { \
	TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
	TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
	TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
	TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
}

/* uint2 variant for SM3.2+ */

#define TFBIG_KINIT_UI2(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
	k8 = ((k0 ^ k1) ^ (k2 ^ k3)) ^ ((k4 ^ k5) ^ (k6 ^ k7)) \
		^ vectorize(0x1BD11BDAA9FC1A22); \
	t2 = t0 ^ t1; \
}

#define TFBIG_ADDKEY_UI2(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
	w0 = (w0 + SKBI(k, s, 0)); \
	w1 = (w1 + SKBI(k, s, 1)); \
	w2 = (w2 + SKBI(k, s, 2)); \
	w3 = (w3 + SKBI(k, s, 3)); \
	w4 = (w4 + SKBI(k, s, 4)); \
	w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
	w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
	w7 = (w7 + SKBI(k, s, 7) + vectorize(s)); \
}

#define TFBIG_ADDKEY_PRE(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
	w0 = (w0 + SKBI(k, s, 0)); \
	w1 = (w1 + SKBI(k, s, 1)); \
	w2 = (w2 + SKBI(k, s, 2)); \
	w3 = (w3 + SKBI(k, s, 3)); \
	w4 = (w4 + SKBI(k, s, 4)); \
	w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
	w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
	w7 = (w7 + SKBI(k, s, 7) + (s)); \
}

#define TFBIG_MIX_UI2(x0, x1, rc) { \
	x0 = x0 + x1; \
	x1 = ROL2(x1, rc) ^ x0; \
}

#define TFBIG_MIX_PRE(x0, x1, rc) { \
	x0 = x0 + x1; \
	x1 = ROTL64(x1, rc) ^ x0; \
}

#define TFBIG_MIX8_UI2(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
	TFBIG_MIX_UI2(w0, w1, rc0); \
	TFBIG_MIX_UI2(w2, w3, rc1); \
	TFBIG_MIX_UI2(w4, w5, rc2); \
	TFBIG_MIX_UI2(w6, w7, rc3); \
}

#define TFBIG_MIX8_PRE(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
	TFBIG_MIX_PRE(w0, w1, rc0); \
	TFBIG_MIX_PRE(w2, w3, rc1); \
	TFBIG_MIX_PRE(w4, w5, rc2); \
	TFBIG_MIX_PRE(w6, w7, rc3); \
}

#define TFBIG_4e_UI2(s) { \
	TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
	TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
	TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
	TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
}

#define TFBIG_4e_PRE(s) { \
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
	TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
	TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
	TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
}

#define TFBIG_4o_UI2(s) { \
	TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
	TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
	TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
	TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
}

#define TFBIG_4o_PRE(s) { \
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
	TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
	TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
	TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
}

#define macro1() {\
	p[0] += p[1]; p[2] += p[3]; p[4] += p[5]; p[6] += p[7]; p[1] = ROL2(p[1],46) ^ p[0]; \
	p[3] = ROL2(p[3],36) ^ p[2]; p[5] = ROL2(p[5],19) ^ p[4]; p[7] = ROL2(p[7], 37) ^ p[6]; \
	p[2] += p[1]; p[4] += p[7]; p[6] += p[5]; p[0] += p[3]; p[1] = ROL2(p[1],33) ^ p[2]; \
	p[7] = ROL2(p[7],27) ^ p[4]; p[5] = ROL2(p[5],14) ^ p[6]; p[3] = ROL2(p[3], 42) ^ p[0]; \
	p[4] += p[1]; p[6] += p[3]; p[0] += p[5]; p[2] += p[7]; p[1] = ROL2(p[1],17) ^ p[4]; \
	p[3] = ROL2(p[3],49) ^ p[6]; p[5] = ROL2(p[5],36) ^ p[0]; p[7] = ROL2(p[7], 39) ^ p[2]; \
	p[6] += p[1]; p[0] += p[7]; p[2] += p[5]; p[4] += p[3]; p[1] = ROL2(p[1],44) ^ p[6]; \
	p[7] = ROL2(p[7], 9) ^ p[0]; p[5] = ROL2(p[5],54) ^ p[2]; p[3] = ROR8(p[3]) ^ p[4]; \
}

#define macro2() { \
	p[0] += p[1]; p[2] += p[3]; p[4] += p[5]; p[6] += p[7]; p[1] = ROL2(p[1], 39) ^ p[0]; \
	p[3] = ROL2(p[3], 30) ^ p[2]; p[5] = ROL2(p[5], 34) ^ p[4]; p[7] = ROL24(p[7]) ^ p[6]; \
	p[2] += p[1]; p[4] += p[7]; p[6] += p[5]; p[0] += p[3]; p[1] = ROL2(p[1], 13) ^ p[2]; \
	p[7] = ROL2(p[7], 50) ^ p[4]; p[5] = ROL2(p[5], 10) ^ p[6]; p[3] = ROL2(p[3], 17) ^ p[0]; \
	p[4] += p[1]; p[6] += p[3]; p[0] += p[5]; p[2] += p[7]; p[1] = ROL2(p[1], 25) ^ p[4]; \
	p[3] = ROL2(p[3], 29) ^ p[6]; p[5] = ROL2(p[5], 39) ^ p[0]; p[7] = ROL2(p[7], 43) ^ p[2]; \
	p[6] += p[1]; p[0] += p[7]; p[2] += p[5]; p[4] += p[3]; p[1] = ROL8(p[1]) ^ p[6]; \
	p[7] = ROL2(p[7], 35) ^ p[0]; p[5] = ROR8(p[5]) ^ p[2]; p[3] = ROL2(p[3], 22) ^ p[4]; \
}

#define macro3() { \
	hash64[0]+= hash64[1]; hash64[2]+= hash64[3]; hash64[4]+= hash64[5]; hash64[6]+= hash64[7]; \
	hash64[1] = ROL2(hash64[1], 39) ^ hash64[0]; \
	hash64[3] = ROL2(hash64[3], 30) ^ hash64[2]; \
	hash64[5] = ROL2(hash64[5], 34) ^ hash64[4]; \
	hash64[7] = ROL24(hash64[7]) ^ hash64[6]; \
	hash64[2]+= hash64[1]; hash64[4]+= hash64[7]; hash64[6]+= hash64[5]; hash64[0]+= hash64[3]; \
	hash64[1] = ROL2(hash64[1], 13) ^ hash64[2]; \
	hash64[7] = ROL2(hash64[7], 50) ^ hash64[4]; \
	hash64[5] = ROL2(hash64[5], 10) ^ hash64[6]; \
	hash64[3] = ROL2(hash64[3], 17) ^ hash64[0]; \
	hash64[4]+= hash64[1]; hash64[6]+= hash64[3]; hash64[0]+= hash64[5]; hash64[2]+= hash64[7]; \
	hash64[1] = ROL2(hash64[1], 25) ^ hash64[4]; \
	hash64[3] = ROL2(hash64[3], 29) ^ hash64[6]; \
	hash64[5] = ROL2(hash64[5], 39) ^ hash64[0]; \
	hash64[7] = ROL2(hash64[7], 43) ^ hash64[2]; \
	hash64[6]+= hash64[1]; hash64[0]+= hash64[7]; hash64[2]+= hash64[5]; hash64[4]+= hash64[3]; \
	hash64[1] = ROL8(hash64[1]) ^ hash64[6]; \
	hash64[7] = ROL2(hash64[7], 35) ^ hash64[0]; \
	hash64[5] = ROR8(hash64[5]) ^ hash64[2]; \
	hash64[3] = ROL2(hash64[3], 22) ^ hash64[4]; \
}

#define macro4() {\
	hash64[0]+= hash64[1]; hash64[2]+= hash64[3]; hash64[4]+= hash64[5]; hash64[6]+= hash64[7]; \
	hash64[1] = ROL2(hash64[1], 46) ^ hash64[0]; \
	hash64[3] = ROL2(hash64[3], 36) ^ hash64[2]; \
	hash64[5] = ROL2(hash64[5], 19) ^ hash64[4]; \
	hash64[7] = ROL2(hash64[7], 37) ^ hash64[6]; \
	hash64[2]+= hash64[1]; hash64[4]+= hash64[7]; hash64[6]+= hash64[5]; hash64[0]+= hash64[3]; \
	hash64[1] = ROL2(hash64[1], 33) ^ hash64[2]; \
	hash64[7] = ROL2(hash64[7], 27) ^ hash64[4]; \
	hash64[5] = ROL2(hash64[5], 14) ^ hash64[6]; \
	hash64[3] = ROL2(hash64[3], 42) ^ hash64[0]; \
	hash64[4]+= hash64[1]; hash64[6]+= hash64[3]; hash64[0]+= hash64[5]; hash64[2]+= hash64[7]; \
	hash64[1] = ROL2(hash64[1], 17) ^ hash64[4]; \
	hash64[3] = ROL2(hash64[3], 49) ^ hash64[6]; \
	hash64[5] = ROL2(hash64[5], 36) ^ hash64[0]; \
	hash64[7] = ROL2(hash64[7], 39) ^ hash64[2]; \
	hash64[6]+= hash64[1]; hash64[0]+= hash64[7]; hash64[2]+= hash64[5]; hash64[4]+= hash64[3]; \
	hash64[1] = ROL2(hash64[1], 44) ^ hash64[6]; \
	hash64[7] = ROL2(hash64[7], 9) ^ hash64[0]; \
	hash64[5] = ROL2(hash64[5], 54) ^ hash64[2]; \
	hash64[3] = ROR8(hash64[3]) ^ hash64[4]; \
}
/*
__constant__ const uint2 buffer[112] = {
	{0x749C51CE, 0x4903ADFF}, {0x9746DF03, 0x0D95DE39}, {0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629},
	{0xDF6CA7B0, 0x5DB62599}, {0xA9D5C434, 0xEABE394C}, {0x1A75B523, 0x891112C7}, {0x660FCC33, 0xAE18A40B},
	{0x9746DF03, 0x0D95DE39}, {0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599},
	{0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x891112C7}, {0x660FCC73, 0x9E18A40B}, {0x98173EC5, 0xCAB2076D},
	{0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C},
	{0x1A75B523, 0x991112C7}, {0x660FCC73, 0x9E18A40B}, {0x98173F04, 0xCAB2076D}, {0x749C51D0, 0x4903ADFF},
	{0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x991112C7},
	{0x660FCC33, 0xAE18A40B}, {0x98173F04, 0xCAB2076D}, {0x749C51CE, 0x3903ADFF}, {0x9746DF06, 0x0D95DE39},
	{0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x991112C7}, {0x660FCC33, 0xAE18A40B},
	{0x98173EC4, 0xCAB2076D}, {0x749C51CE, 0x3903ADFF}, {0x9746DF43, 0xFD95DE39}, {0x27C79BD2, 0x8FD19341},
	{0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x991112C7}, {0x660FCC33, 0xAE18A40B}, {0x98173EC4, 0xCAB2076D},
	{0x749C51CE, 0x4903ADFF}, {0x9746DF43, 0xFD95DE39}, {0x27C79C0E, 0x8FD19341}, {0xFF352CB6, 0x9A255629},
	{0x1A75B523, 0x991112C7}, {0x660FCC33, 0xAE18A40B}, {0x98173EC4, 0xCAB2076D}, {0x749C51CE, 0x4903ADFF},
	{0x9746DF03, 0x0D95DE39}, {0x27C79C0E, 0x8FD19341}, {0xFF352CB1, 0x8A255629}, {0xDF6CA7B6, 0x5DB62599},
	{0x660FCC33, 0xAE18A40B}, {0x98173EC4, 0xCAB2076D}, {0x749C51CE, 0x4903ADFF}, {0x9746DF03, 0x0D95DE39},
	{0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x8A255629}, {0xDF6CA7F0, 0x4DB62599}, {0xA9D5C3FB, 0xEABE394C},
	{0x98173EC4, 0xCAB2076D}, {0x749C51CE, 0x4903ADFF}, {0x9746DF03, 0x0D95DE39}, {0x27C79BCE, 0x8FD19341},
	{0xFF352CB1, 0x9A255629}, {0xDF6CA7F0, 0x4DB62599}, {0xA9D5C434, 0xEABE394C}, {0x1A75B52B, 0x991112C7},
	{0x749C51CE, 0x4903ADFF}, {0x9746DF03, 0x0D95DE39}, {0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629},
	{0xDF6CA7B0, 0x5DB62599}, {0xA9D5C434, 0xEABE394C}, {0x1A75B523, 0x891112C7}, {0x660FCC3C, 0xAE18A40B},
	{0x9746DF03, 0x0D95DE39}, {0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599},
	{0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x891112C7}, {0x660FCC73, 0x9E18A40B}, {0x98173ece, 0xcab2076d},
	{0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C},
	{0x1A75B523, 0x991112C7}, {0x660FCC73, 0x9E18A40B}, {0x98173F04, 0xCAB2076D}, {0x749C51D9, 0x4903ADFF},
	{0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x991112C7},
	{0x660FCC33, 0xAE18A40B}, {0x98173F04, 0xCAB2076D}, {0x749C51CE, 0x3903ADFF}, {0x9746DF0F, 0x0D95DE39},
	{0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x991112C7}, {0x660FCC33, 0xAE18A40B},
	{0x98173EC4, 0xCAB2076D}, {0x749C51CE, 0x3903ADFF}, {0x9746DF43, 0xFD95DE39}, {0x27C79BDB, 0x8FD19341}
};
*/
// 120 * 8 = 960 ... too big ?
static __constant__ uint2 c_buffer[120]; // padded message (80 bytes + 72*8 bytes midstate + align)

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52, TPF52)
#else
__launch_bounds__(TPB50, TPF50)
#endif
void quark_skein512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *output64)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;

		uint32_t nonce = cuda_swab32(startNounce + thread);
		uint2 nonce2 = make_uint2(c_buffer[0].x, nonce);

		uint2 p[8];
		p[1] = nonce2;

		h0 = c_buffer[ 1];
		h1 = c_buffer[ 2];
		h2 = c_buffer[ 3];
		h3 = c_buffer[ 4];
		h4 = c_buffer[ 5];
		h5 = c_buffer[ 6];
		h6 = c_buffer[ 7];
		h7 = c_buffer[ 8];
		h8 = c_buffer[ 9];

		t0 = vectorize(0x50ull);
		t1 = vectorize(0xB000000000000000ull);
		t2 = t0^t1;

		p[1]=nonce2 + h1;	p[0]= c_buffer[10] + p[1];
		p[2]=c_buffer[11];
		p[3]=c_buffer[12];
		p[4]=c_buffer[13];
		p[5]=c_buffer[14];
		p[6]=c_buffer[15];
		p[7]=c_buffer[16];

//		macro1();
		p[1] = ROL2(p[1], 46) ^ p[0];
		p[2] += p[1];
		p[0] += p[3];
		p[1] = ROL2(p[1], 33) ^ p[2];
		p[3] = c_buffer[17] ^ p[0];
		p[4] += p[1];
		p[6] += p[3];
		p[0] += p[5];
		p[2] += p[7];
		p[1] = ROL2(p[1], 17) ^ p[4];
		p[3] = ROL2(p[3], 49) ^ p[6];
		p[5] = c_buffer[18] ^ p[0];
		p[7] = c_buffer[19] ^ p[2];
		p[6] += p[1];
		p[0] += p[7];
		p[2] += p[5];
		p[4] += p[3];
		p[1] = ROL2(p[1], 44) ^ p[6];
		p[7] = ROL2(p[7], 9) ^ p[0];
		p[5] = ROL2(p[5], 54) ^ p[2];
		p[3] = ROR8(p[3]) ^ p[4];

		p[0]+=h1;	p[1]+=h2;	p[2]+=h3;	p[3]+=h4;	p[4]+=h5;
		p[5]+=c_buffer[20];	p[7]+=c_buffer[21];	p[6]+=c_buffer[22];
		macro2();
		p[0]+=h2;	p[1]+=h3;	p[2]+=h4;	p[3]+=h5;	p[4]+=h6;
		p[5]+=c_buffer[22];	p[7]+=c_buffer[23];	p[6]+=c_buffer[24];
		macro1();
		p[0]+=h3;	p[1]+=h4;	p[2]+=h5;	p[3]+=h6;	p[4]+=h7;
		p[5]+=c_buffer[24];	p[7]+=c_buffer[25];	p[6]+=c_buffer[26];
		macro2();
		p[0]+=h4;	p[1]+=h5;	p[2]+=h6;	p[3]+=h7;	p[4]+=h8;
		p[5]+=c_buffer[26];	p[7]+=c_buffer[27];	p[6]+=c_buffer[28];
		macro1();
		p[0]+=h5;	p[1]+=h6;	p[2]+=h7;	p[3]+=h8;	p[4]+=h0;
		p[5]+=c_buffer[28];	p[7]+=c_buffer[29];	p[6]+=c_buffer[30];
		macro2();
		p[0]+=h6;	p[1]+=h7;	p[2]+=h8;	p[3]+=h0;	p[4]+=h1;
		p[5]+=c_buffer[30];	p[7]+=c_buffer[31];	p[6]+=c_buffer[32];
		macro1();
		p[0]+=h7;	p[1]+=h8;	p[2]+=h0;	p[3]+=h1;	p[4]+=h2;
		p[5]+=c_buffer[32];	p[7]+=c_buffer[33];	p[6]+=c_buffer[34];
		macro2();
		p[0]+=h8;	p[1]+=h0;	p[2]+=h1;	p[3]+=h2;	p[4]+=h3;
		p[5]+=c_buffer[34];	p[7]+=c_buffer[35];	p[6]+=c_buffer[36];
		macro1();
		p[0]+=h0;	p[1]+=h1;	p[2]+=h2;	p[3]+=h3;	p[4]+=h4;
		p[5]+=c_buffer[36];	p[7]+=c_buffer[37];	p[6]+=c_buffer[38];
		macro2();
		p[0]+=h1;	p[1]+=h2;	p[2]+=h3;	p[3]+=h4;	p[4]+=h5;
		p[5]+=c_buffer[38];	p[7]+=c_buffer[39];	p[6]+=c_buffer[40];
		macro1();
		p[0]+=h2;	p[1]+=h3;	p[2]+=h4;	p[3]+=h5;	p[4]+=h6;
		p[5]+=c_buffer[40];	p[7]+=c_buffer[41];	p[6]+=c_buffer[42];
		macro2();
		p[0]+=h3;	p[1]+=h4;	p[2]+=h5;	p[3]+=h6;	p[4]+=h7;
		p[5]+=c_buffer[42];	p[7]+=c_buffer[43];	p[6]+=c_buffer[44];
		macro1();
		p[0]+=h4;	p[1]+=h5;	p[2]+=h6;	p[3]+=h7;	p[4]+=h8;
		p[5]+=c_buffer[44];	p[7]+=c_buffer[45];	p[6]+=c_buffer[46];
		macro2();
		p[0]+=h5;	p[1]+=h6;	p[2]+=h7;	p[3]+=h8;	p[4]+=h0;
		p[5]+=c_buffer[46];	p[7]+=c_buffer[47];	p[6]+=c_buffer[48];
		macro1();
		p[0]+=h6;	p[1]+=h7;	p[2]+=h8;	p[3]+=h0;	p[4]+=h1;
		p[5]+=c_buffer[48];	p[7]+=c_buffer[49];	p[6]+=c_buffer[50];
		macro2();
		p[0]+=h7;	p[1]+=h8;	p[2]+=h0;	p[3]+=h1;	p[4]+=h2;
		p[5]+=c_buffer[50];	p[7]+=c_buffer[51];	p[6]+=c_buffer[52];
		macro1();
		p[0]+=h8;	p[1]+=h0;	p[2]+=h1;	p[3]+=h2;	p[4]+=h3;
		p[5]+=c_buffer[52];	p[7]+=c_buffer[53];	p[6]+=c_buffer[54];
		macro2();
		p[0]+=h0;	p[1]+=h1;	p[2]+=h2;	p[3]+=h3;	p[4]+=h4;
		p[5]+=c_buffer[54];	p[7]+=c_buffer[55];	p[6]+=c_buffer[56];

		p[0]^= c_buffer[57];
		p[1]^= nonce2;

		t0 = vectorize(8); // extra
		t1 = vectorize(0xFF00000000000000ull); // etype
//		t2 = vectorize(0xB000000000000050ull);

		h0 = p[0];
		h1 = p[1];
		h2 = p[2];
		h3 = p[3];
		h4 = p[4];
		h5 = p[5];
		h6 = p[6];
		h7 = p[7];

		TFBIG_KINIT_UI2(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);

		p[0] = p[1] = p[2] = p[3] = p[4] =p[5] =p[6] = p[7] = vectorize(0);

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		uint64_t *outpHash = &output64[thread<<3];
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			outpHash[i] = devectorize(p[i]);
	}
}

__host__
void quark_skein512_cpu_setBlock_80(void *pdata)
{
	uint64_t message[20];
	memcpy(&message[0], pdata, 80);

	uint64_t p[8];
	uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
	uint64_t t0, t1, t2;

	h0 = 0x4903ADFF749C51CEull;
	h1 = 0x0D95DE399746DF03ull;
	h2 = 0x8FD1934127C79BCEull;
	h3 = 0x9A255629FF352CB1ull;
	h4 = 0x5DB62599DF6CA7B0ull;
	h5 = 0xEABE394CA9D5C3F4ull;
	h6 = 0x991112C71A75B523ull;
	h7 = 0xAE18A40B660FCC33ull;
	// h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);
	h8 = 0xcab2076d98173ec4ULL;

	t0 = 64; // ptr
	t1 = 0x7000000000000000ull;
	t2 = 0x7000000000000040ull;

	memcpy(&p[0], &message[0], 64);

	TFBIG_4e_PRE(0);
	TFBIG_4o_PRE(1);
	TFBIG_4e_PRE(2);
	TFBIG_4o_PRE(3);
	TFBIG_4e_PRE(4);
	TFBIG_4o_PRE(5);
	TFBIG_4e_PRE(6);
	TFBIG_4o_PRE(7);
	TFBIG_4e_PRE(8);
	TFBIG_4o_PRE(9);
	TFBIG_4e_PRE(10);
	TFBIG_4o_PRE(11);
	TFBIG_4e_PRE(12);
	TFBIG_4o_PRE(13);
	TFBIG_4e_PRE(14);
	TFBIG_4o_PRE(15);
	TFBIG_4e_PRE(16);
	TFBIG_4o_PRE(17);
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

	message[10] = message[0] ^ p[0];
	message[11] = message[1] ^ p[1];
	message[12] = message[2] ^ p[2];
	message[13] = message[3] ^ p[3];
	message[14] = message[4] ^ p[4];
	message[15] = message[5] ^ p[5];
	message[16] = message[6] ^ p[6];
	message[17] = message[7] ^ p[7];

	message[18] = t2;

	uint64_t buffer[128];

//	buffer[ 0] = message[ 8];
	buffer[ 0] = message[ 9];
	h0 = buffer[ 1] = message[10];
	h1 = buffer[ 2] = message[11];
	h2 = buffer[ 3] = message[12];
	h3 = buffer[ 4] = message[13];
	h4 = buffer[ 5] = message[14];
	h5 = buffer[ 6] = message[15];
	h6 = buffer[ 7] = message[16];
	h7 = buffer[ 8] = message[17];
	h8 = buffer[ 9] = h0^h1^h2^h3^h4^h5^h6^h7^0x1BD11BDAA9FC1A22ULL;

	t0 = 0x50ull;
	t1 = 0xB000000000000000ull;
	t2 = t0^t1;

	p[0] = message[ 8] + h0;
	p[2] = h2; p[3] = h3; p[4] = h4;
	p[5] = h5 + t0; p[6] = h6 + t1; p[7] = h7;
	p[2] += p[3]; p[4] += p[5]; p[6] += p[7];
	p[3] = ROTL64(p[3], 36) ^ p[2];
	p[5] = ROTL64(p[5], 19) ^ p[4];
	p[7] = ROTL64(p[7], 37) ^ p[6];
	p[4] += p[7];
	p[6] += p[5];
	p[7] = ROTL64(p[7], 27) ^ p[4];
	p[5] = ROTL64(p[5], 14) ^ p[6];

	buffer[10] = p[0];
	buffer[11] = p[2];
	buffer[12] = p[3];
	buffer[13] = p[4];
	buffer[14] = p[5];
	buffer[15] = p[6];
	buffer[16] = p[7];
	buffer[17] = ROTL64(p[3], 42);
	buffer[18] = ROTL64(p[5], 36);
	buffer[19] = ROTL64(p[7], 39);

	buffer[20] = h6+t1;
	buffer[21] = h8+1;
	buffer[22] = h7+t2;
	buffer[23] = h0+2;
	buffer[24] = h8+t0;
	buffer[25] = h1+3;
	buffer[26] = h0+t1;
	buffer[27] = h2+4;
	buffer[28] = h1+t2;
	buffer[29] = h3+5;
	buffer[30] = h2+t0;
	buffer[31] = h4+6;
	buffer[32] = h3+t1;
	buffer[33] = h5+7;
	buffer[34] = h4+t2;
	buffer[35] = h6+8;
	buffer[36] = h5+t0;
	buffer[37] = h7+9;
	buffer[38] = h6+t1;
	buffer[39] = h8+10;
	buffer[40] = h7+t2;
	buffer[41] = h0+11;
	buffer[42] = h8+t0;
	buffer[43] = h1+12;
	buffer[44] = h0+t1;
	buffer[45] = h2+13;
	buffer[46] = h1+t2;
	buffer[47] = h3+14;
	buffer[48] = h2+t0;
	buffer[49] = h4+15;
	buffer[50] = h3+t1;
	buffer[51] = h5+16;
	buffer[52] = h4+t2;
	buffer[53] = h6+17;
	buffer[54] = h5+t0;
	buffer[55] = h7+18;
	buffer[56] = h6+t1;

	buffer[57] = message[8];

	cudaMemcpyToSymbol(c_buffer, buffer, sizeof(c_buffer), 0, cudaMemcpyHostToDevice);
	CUDA_SAFE_CALL(cudaGetLastError());
}

__host__
void quark_skein512_cpu_hash_80(const int thr_id, const uint32_t threads, uint32_t startNounce, uint32_t *d_hash, const uint32_t tpb)
{
	const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);

	// hash function is cut in 2 parts to reduce kernel size
	quark_skein512_gpu_hash_80 <<< grid, block >>> (threads, startNounce, (uint64_t*)d_hash);
}

__host__
void quark_skein512_cpu_init_80(const int thr_id, uint32_t threads) {}

__host__
void quark_skein512_cpu_free_80(const int thr_id) {}

#include "miner.h"

__host__
uint32_t quark_skein512_calc_tpb_80(const int thr_id) {

	int blockSize = 0;
	int minGridSize = 0;
	int maxActiveBlocks, device;
	cudaDeviceProp props;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, quark_skein512_gpu_hash_80, 0,	0);

	// calculate theoretical occupancy
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, quark_skein512_gpu_hash_80, blockSize, 0);
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
	float occupancy = (maxActiveBlocks * blockSize / props.warpSize)
			/ (float) (props.maxThreadsPerMultiProcessor / props.warpSize);

	if (!opt_quiet) gpulog(LOG_INFO, thr_id, "skein512_80 tpb calc - block size %d. Theoretical occupancy: %f", blockSize, occupancy);

	return (uint32_t)blockSize;
}
