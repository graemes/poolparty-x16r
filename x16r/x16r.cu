/**
 * X16R algorithm (X16 with Randomized chain order)
 *
 * tpruvot 2018 - GPL code
 */

#include <stdio.h>
//#include <memory.h>
#include <unistd.h>

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "cuda_x16r.h"

#define NBN 2

//#define _DEBUG
#define _DEBUG_PREFIX "x16r-"
#include "cuda_debug.cuh"

// Internal functions
static void getAlgoString(const uint32_t* prevblock, char *output);
static void init_x16r(int thr_id, int dev_id);
static void setBenchHash();
static void calcOptimumTPBs(int thr_id);

// Local variables
static uint32_t *d_hash[MAX_GPUS];
static uint32_t thr_throughput[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { 0 };

static __thread uint32_t s_ntime = UINT32_MAX;
static __thread char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };

static uint64_t bench_hash = 0x67452301EFCDAB89;	// Default
extern char* opt_bench_hash;

extern bool opt_autotune;

// Initialise tpb arrays to default values (based on sm > 50)
static uint32_t tpb64[HASH_FUNC_COUNT + 1] = { 192, 32,512,512,128,512,384,768,384,128,128,384,256,384,384,256 } ;
static uint32_t tpb80[HASH_FUNC_COUNT + 1] = { 512,128,256,256,256,512,256,256,128,128,128,128,256,256,256,256 } ;

// Lets do it

// X16R CPU Hash (Validation)
extern "C" void x16r_hash(void *output, const void *input)
{
	unsigned char _ALIGN(64) hash[128];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_fugue512_context ctx_fugue;
	sph_shabal512_context ctx_shabal;
	sph_whirlpool_context ctx_whirlpool;
	sph_sha512_context ctx_sha512;

	void *in = (void*) input;
	int size = 80;

	uint32_t *in32 = (uint32_t*) input;
	getAlgoString(&in32[1], hashOrder);

	for (int i = 0; i < 16; i++)
	{
		const char elem = hashOrder[i];
		const uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

		switch (algo) {
		case BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, in, size);
			sph_blake512_close(&ctx_blake, hash);
			break;
		case BMW:
			sph_bmw512_init(&ctx_bmw);
			sph_bmw512(&ctx_bmw, in, size);
			sph_bmw512_close(&ctx_bmw, hash);
			break;
		case GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, in, size);
			sph_groestl512_close(&ctx_groestl, hash);
			break;
		case SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, in, size);
			sph_skein512_close(&ctx_skein, hash);
			break;
		case JH:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, in, size);
			sph_jh512_close(&ctx_jh, hash);
			break;
		case KECCAK:
			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, in, size);
			sph_keccak512_close(&ctx_keccak, hash);
			break;
		case LUFFA:
			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, in, size);
			sph_luffa512_close(&ctx_luffa, hash);
			break;
		case CUBEHASH:
			sph_cubehash512_init(&ctx_cubehash);
			sph_cubehash512(&ctx_cubehash, in, size);
			sph_cubehash512_close(&ctx_cubehash, hash);
			break;
		case SHAVITE:
			sph_shavite512_init(&ctx_shavite);
			sph_shavite512(&ctx_shavite, in, size);
			sph_shavite512_close(&ctx_shavite, hash);
			break;
		case SIMD:
			sph_simd512_init(&ctx_simd);
			sph_simd512(&ctx_simd, in, size);
			sph_simd512_close(&ctx_simd, hash);
			break;
		case ECHO:
			sph_echo512_init(&ctx_echo);
			sph_echo512(&ctx_echo, in, size);
			sph_echo512_close(&ctx_echo, hash);
			break;
		case HAMSI:
			sph_hamsi512_init(&ctx_hamsi);
			sph_hamsi512(&ctx_hamsi, in, size);
			sph_hamsi512_close(&ctx_hamsi, hash);
			break;
		case FUGUE:
			sph_fugue512_init(&ctx_fugue);
			sph_fugue512(&ctx_fugue, in, size);
			sph_fugue512_close(&ctx_fugue, hash);
			break;
		case SHABAL:
			sph_shabal512_init(&ctx_shabal);
			sph_shabal512(&ctx_shabal, in, size);
			sph_shabal512_close(&ctx_shabal, hash);
			break;
		case WHIRLPOOL:
			sph_whirlpool_init(&ctx_whirlpool);
			sph_whirlpool(&ctx_whirlpool, in, size);
			sph_whirlpool_close(&ctx_whirlpool, hash);
			break;
		case SHA512:
			sph_sha512_init(&ctx_sha512);
			sph_sha512(&ctx_sha512,(const void*) in, size);
			sph_sha512_close(&ctx_sha512,(void*) hash);
			break;
		}
		in = (void*) hash;
		size = 64;
	}
	memcpy(output, hash, 32);
}

static int algo80_fails[HASH_FUNC_COUNT] = { 0 };

extern "C" int scanhash_x16r(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	const int dev_id = device_map[thr_id];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	// Only initialise (and calculate throughput) when necessary
	if (!init[thr_id]){
		init_x16r(thr_id, dev_id);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * thr_throughput[thr_id]), 0);
		cuda_check_cpu_init(thr_id, thr_throughput[thr_id]);
	}

	uint32_t throughput = thr_throughput[thr_id];

	if (opt_benchmark) {
		((uint32_t*)ptarget)[7] = 0x003f;
		*((uint64_t*)&pdata[1]) = bench_hash;
	}

	uint32_t _ALIGN(64) endiandata[20];

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	uint32_t ntime = swab32(pdata[17]);
	if (s_ntime != ntime) {
		getAlgoString(&endiandata[1], hashOrder);
		s_ntime = ntime;
		if (!thr_id && !opt_quiet) applog(LOG_INFO, "hash order %s (%08x)", hashOrder, ntime);
	}

	cuda_check_cpu_setTarget(ptarget);

	char elem = hashOrder[0];
	const uint8_t algo80 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';
//	const uint8_t algo80 = (*(uint64_t*)&endiandata[1] >> 60 - (0 * 4)) & 0x0f ;

	switch (algo80) {
		case BLAKE:
			quark_blake512_cpu_setBlock_80(thr_id, endiandata);
			break;
		case BMW:
			quark_bmw512_cpu_setBlock_80(endiandata);
			break;
		case GROESTL:
			quark_groestl512_setBlock_80(thr_id, endiandata);
			break;
		case JH:
			quark_jh512_setBlock_80(thr_id, endiandata);
			break;
		case KECCAK:
			quark_keccak512_setBlock_80(thr_id, (void*)endiandata);
			break;
		case SKEIN:
			quark_skein512_cpu_setBlock_80((void*)endiandata);
			break;
		case LUFFA:
			qubit_luffa512_cpu_setBlock_80((void*)endiandata);
			break;
		case CUBEHASH:
			x11_cubehash512_setBlock_80(thr_id, endiandata);
			break;
		case SHAVITE:
			x11_shavite512_setBlock_80((void*)endiandata);
			break;
		case SIMD:
			x16_simd512_setBlock_80((void*)endiandata);
			break;
		case ECHO:
			x16_echo512_setBlock_80((void*)endiandata);
			break;
		case HAMSI:
			x13_hamsi512_setBlock_80((void*)endiandata);
			break;
		case FUGUE:
			x16_fugue512_setBlock_80((void*)pdata);
			break;
		case SHABAL:
			x16_shabal512_setBlock_80((void*)endiandata);
			break;
		case WHIRLPOOL:
			x15_whirlpool512_setBlock_80((void*)endiandata);
			break;
		case SHA512:
			x17_sha512_setBlock_80(endiandata);
			break;
		default: {
			if (!thr_id)
//				applog(LOG_WARNING, "kernel %s %c unimplemented, order %s", algo_strings[algo80], elem, hashOrder);
			sleep(5);
			return -1;
		}
	}

	/*
	 * TODO: a1i3nj03 has what looks to be a far more concise way of calling the algo's but when I looked at it
	 * some compiler warnings related to the pointers made me nervous - needs further investigation.
	 */
	int warn = 0;
	do {
		// Hash with CUDA
		switch (algo80) {
			case BLAKE:
				quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[BLAKE]);
				TRACE("blake80:");
				break;
			case BMW:
				quark_bmw512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[BMW]);
				TRACE("bmw80  :");
				break;
			case GROESTL:
				quark_groestl512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[GROESTL]);
				TRACE("grstl80:");
				break;
			case JH:
				quark_jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[JH]);
				TRACE("jh51280:");
				break;
			case KECCAK:
				quark_keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[KECCAK]);
				TRACE("keccak80:");
				break;
			case SKEIN:
				quark_skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[SKEIN]);
				TRACE("skein80:");
				break;
			case LUFFA:
				qubit_luffa512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[LUFFA]);
				TRACE("luffa80:");
				break;
			case CUBEHASH:
				x11_cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[CUBEHASH]);
				TRACE("cube 80:");
				break;
			case SHAVITE:
				x11_shavite512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[SHAVITE]);
				TRACE("shavite:");
				break;
			case SIMD:
				x16_simd512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[SIMD]);
				TRACE("simd512:");
				break;
			case ECHO:
				x16_echo512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[ECHO]);
				TRACE("echo   :");
				break;
			case HAMSI:
				x13_hamsi512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[HAMSI]);
				TRACE("hamsi  :");
				break;
			case FUGUE:
				x16_fugue512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[FUGUE]);
				TRACE("fugue  :");
				break;
			case SHABAL:
				x16_shabal512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[SHABAL]);
				TRACE("shabal :");
				break;
			case WHIRLPOOL:
				x15_whirlpool512_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[WHIRLPOOL]);
				TRACE("whirl  :");
				break;
			case SHA512:
				x17_sha512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], tpb80[SHA512]);
				TRACE("sha512 :");
				break;
		}

		for (int i = 1; i < 16; i++)
		{
			const char elem = hashOrder[i];
			const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';
//			const uint8_t algo64t = (*(uint64_t*)&endiandata[1] >> 60 - (1 * 4)) & 0x0f ;
//			const uint8_t algo64t;
//			memcpy(algo64t,(&endiandata[1] >> 60 - (1 * 4)) & 0x0f),sizeof endiandata[1] ;

			switch (algo64) {
			case BLAKE:
				quark_blake512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[BLAKE]);
				TRACE("blake    :");
				break;
			case BMW:
				quark_bmw512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[BMW]);
				TRACE("bmw      :");
				break;
			case GROESTL:
				quark_groestl512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[GROESTL]);
				TRACE("groestl  :");
				break;
			case JH:
				quark_jh512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[JH]);
				TRACE("jh512    :");
				break;
			case KECCAK:
				quark_keccak512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[KECCAK]);
				TRACE("keccak   :");
				break;
			case SKEIN:
				quark_skein512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[SKEIN]);
				TRACE("skein    :");
				break;
			case LUFFA:
				qubit_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[LUFFA]);
				TRACE("luffa    :");
				break;
			case CUBEHASH:
				x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[CUBEHASH]);
				TRACE("cubehash :");
				break;
			case SHAVITE:
				x11_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[SHAVITE]);
				TRACE("shavite  :");
				break;
			case SIMD:
				x11_simd512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[SIMD]);
				TRACE("simd     :");
				break;
			case ECHO:
				x11_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[ECHO]);
				TRACE("echo     :");
				break;
			case HAMSI:
				x13_hamsi512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[HAMSI]);
				TRACE("hamsi    :");
				break;
			case FUGUE:
				x13_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[FUGUE]);
				TRACE("fugue    :");
				break;
			case SHABAL:
				x14_shabal512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[SHABAL]);
				TRACE("shabal   :");
				break;
			case WHIRLPOOL:
				x15_whirlpool512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[WHIRLPOOL]);
				TRACE("whirlpool:");
				break;
			case SHA512:
				x17_sha512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], tpb64[SHA512]);
				TRACE("sha512   :");
				break;
			}
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
#ifdef _DEBUG
		uint32_t _ALIGN(64) dhash[8];
		be32enc(&endiandata[19], pdata[19]);
		x16r_hash(dhash, endiandata);
		applog_hash(dhash);
		return -1;
#endif
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			x16r_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					x16r_hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				// x11+ coins could do some random error, but not on retry
				gpu_increment_reject(thr_id);
				algo80_fails[algo80]++;
				if (!warn) {
					warn++;
					pdata[19] = work->nonces[0] + 1;
					continue;
				} else {
//					if (!opt_quiet)	gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! %s %s",
//						work->nonces[0], algo_strings[algo80], hashOrder);
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! %s %s",
							work->nonces[0], algo_strings[algo80], hashOrder);
					warn = 0;
				}
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_x16r(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	// 64 byte kernels free
	// A number could be noops but as this out of the mainline.......
	quark_blake512_cpu_free_64(thr_id);
	quark_bmw512_cpu_free_64(thr_id);
	quark_groestl512_cpu_free_64(thr_id);
	quark_jh512_cpu_free_64(thr_id); 
	quark_keccak512_cpu_free_64(thr_id);
	quark_skein512_cpu_free_64(thr_id);
	qubit_luffa512_cpu_free_64(thr_id);
	x11_cubehash512_cpu_free_64(thr_id);
	x11_shavite512_cpu_free_64(thr_id);
	x11_simd512_cpu_free_64(thr_id);
	x11_echo512_cpu_free_64(thr_id);
	x13_hamsi512_cpu_free_64(thr_id);
	x13_fugue512_cpu_free_64(thr_id);
	x14_shabal512_cpu_free_64(thr_id);
	x15_whirlpool512_cpu_free_64(thr_id);
	x17_sha512_cpu_free_64(thr_id);

	// 80 byte kernels free
	// A number could be noops but as this out of the mainline.......
	quark_blake512_cpu_free_80(thr_id);
	quark_bmw512_cpu_free_80(thr_id);
	quark_groestl512_cpu_free_80(thr_id);
	quark_jh512_cpu_free_80(thr_id); 
	quark_keccak512_cpu_free_80(thr_id);
	quark_skein512_cpu_free_80(thr_id);
	qubit_luffa512_cpu_free_80(thr_id);
	x11_cubehash512_cpu_free_80(thr_id);
	x11_shavite512_cpu_free_80(thr_id);
	x16_simd512_cpu_free_80(thr_id);
	x16_echo512_cpu_free_80(thr_id);
	x13_hamsi512_cpu_free_80(thr_id);
	x16_fugue512_cpu_free_80(thr_id);
	x16_shabal512_cpu_free_80(thr_id);
	x15_whirlpool512_cpu_free_80(thr_id);
	x17_sha512_cpu_free_80(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}

// Internal functions
static void getAlgoString(const uint32_t* prevblock, char *output)
{
//	for (int i = 0; i < 16; i++)
//	{
//			*output++ = (*(uint64_t*)prevblock >> 60 - (i * 4)) & 0x0f;
//	}

	char *sptr = output;
	uint8_t* data = (uint8_t*)prevblock;

	for (uint8_t j = 0; j < HASH_FUNC_COUNT; j++) {
		uint8_t b = (15 - j) >> 1; // 16 ascii hex chars, reversed
		uint8_t algoDigit = (j & 1) ? data[b] & 0xF : data[b] >> 4;
		if (algoDigit >= 10)
			sprintf(sptr, "%c", 'A' + (algoDigit - 10));
		else
			sprintf(sptr, "%u", (uint32_t) algoDigit);
		sptr++;
	}
	*sptr = '\0';

}

//extern "C" uint32_t init_x16r(int thr_id)
static void init_x16r(int thr_id, int dev_id)
{
	uint32_t throughput = 0;
	int intensity = 18;

	cuda_get_arch(thr_id);
	gpulog(LOG_INFO, thr_id, "Detected %s", device_name[dev_id]);

	// Simple heuristic for setting default intensity
	int gpu_model;
	char gpu_family1[10], gpu_family2[10];
	sscanf(device_name[dev_id], "%s %s %d", gpu_family1, gpu_family2, &gpu_model);  // More robust way to extract model number?
	if (gpu_model > 1070) intensity = 20;
	else if (gpu_model > 1000) intensity = 19;

	throughput = cuda_default_throughput(thr_id, 1U << intensity);

	cudaSetDevice(device_map[thr_id]);
	if (opt_cudaschedule == -1 && gpu_threads == 1) {
		cudaDeviceReset();
		// reduce cpu usage
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	}
	gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

	// 64 byte kernels initialisation
	// A number could be noops but as this out of the mainline.......
	quark_blake512_cpu_init_64(thr_id, throughput);
	quark_bmw512_cpu_init_64(thr_id, throughput);
	quark_groestl512_cpu_init_64(thr_id, throughput);
	quark_jh512_cpu_init_64(thr_id, throughput); 
	quark_keccak512_cpu_init_64(thr_id, throughput);
	quark_skein512_cpu_init_64(thr_id, throughput);
	qubit_luffa512_cpu_init_64(thr_id, throughput);
	x11_cubehash512_cpu_init_64(thr_id, throughput);
	x11_shavite512_cpu_init_64(thr_id, throughput);
	x11_simd512_cpu_init_64(thr_id, throughput);
	x11_echo512_cpu_init_64(thr_id, throughput);
	x13_hamsi512_cpu_init_64(thr_id, throughput);
	x13_fugue512_cpu_init_64(thr_id, throughput);
	x14_shabal512_cpu_init_64(thr_id, throughput);
	x15_whirlpool512_cpu_init_64(thr_id, throughput);
	x17_sha512_cpu_init_64(thr_id, throughput);

	// 80 byte kernels initialisation
	// A number could be noops but as this out of the mainline.......
	quark_blake512_cpu_init_80(thr_id, throughput);
	quark_bmw512_cpu_init_80(thr_id, throughput);
	quark_groestl512_cpu_init_80(thr_id, throughput);
	quark_jh512_cpu_init_80(thr_id, throughput); 
	quark_keccak512_cpu_init_80(thr_id, throughput);
	quark_skein512_cpu_init_80(thr_id, throughput);
	qubit_luffa512_cpu_init_80(thr_id, throughput);
	x11_cubehash512_cpu_init_80(thr_id, throughput);
	x11_shavite512_cpu_init_80(thr_id, throughput);
	x16_simd512_cpu_init_80(thr_id, throughput);
	x16_echo512_cpu_init_80(thr_id, throughput);
	x13_hamsi512_cpu_init_80(thr_id, throughput);
	x16_fugue512_cpu_init_80(thr_id, throughput);
	x16_shabal512_cpu_init_80(thr_id, throughput);
	x15_whirlpool512_cpu_init_80(thr_id, throughput);
	x17_sha512_cpu_init_80(thr_id, throughput);

	if (opt_benchmark) {
		setBenchHash();
	}

	if (device_sm[dev_id] == 500) {
			tpb64[KECCAK] = 256;
	}

	if (opt_autotune) calcOptimumTPBs(thr_id);

	thr_throughput[thr_id] = throughput;
	init[thr_id] = true;
}

static void setBenchHash() {

	if (opt_bench_hash[0]) {
		bool bench_hash_found = false;
		applog(LOG_INFO, "Looking for benchmark hashing algorithm %s", opt_bench_hash);
		for (uint8_t j = 0; j < (HASH_FUNC_COUNT); j++) {
			// full hash?
			if ((strcmp(algo_strings[j], opt_bench_hash) == 0)) {
				bench_hash = algo_hashes[j];
				bench_hash_found = true;
			}
			// hash 80 only?
			if ((strcmp(algo80_strings[j], opt_bench_hash) == 0)) {
				bench_hash = algo80_hashes[j];
				bench_hash_found = true;
			}
		}

		if (!bench_hash_found)
			applog(LOG_WARNING, "Specified benchmark hashing algorithm %s not found. Using default.", opt_bench_hash);
	}
}

static void calcOptimumTPBs(int thr_id){

	tpb64[BLAKE] = quark_blake512_calc_tpb_64(thr_id);
	tpb80[BLAKE] = quark_blake512_calc_tpb_80(thr_id);

	tpb64[BMW] = quark_bmw512_calc_tpb_64(thr_id);
	tpb80[BMW] = quark_bmw512_calc_tpb_80(thr_id);

	tpb64[GROESTL] = quark_groestl512_calc_tpb_64(thr_id);
	tpb80[GROESTL] = quark_groestl512_calc_tpb_80(thr_id);

	tpb64[JH] = quark_jh512_calc_tpb_64(thr_id);
	tpb80[JH] = quark_jh512_calc_tpb_80(thr_id);

	tpb64[KECCAK] = quark_keccak512_calc_tpb_64(thr_id);
	tpb80[KECCAK] = quark_keccak512_calc_tpb_80(thr_id);

	tpb64[SKEIN] = quark_skein512_calc_tpb_64(thr_id);
	tpb80[SKEIN] = quark_skein512_calc_tpb_80(thr_id);

	tpb64[LUFFA] = qubit_luffa512_calc_tpb_64(thr_id);
	tpb80[LUFFA] = qubit_luffa512_calc_tpb_80(thr_id);

	tpb64[CUBEHASH] = x11_cubehash512_calc_tpb_64(thr_id);
	tpb80[CUBEHASH] = x11_cubehash512_calc_tpb_80(thr_id);

	tpb64[SHAVITE] = x11_shavite512_calc_tpb_64(thr_id);
	tpb80[SHAVITE] = x11_shavite512_calc_tpb_80(thr_id);

	tpb64[SIMD] = x11_simd512_calc_tpb_64(thr_id);
	tpb80[SIMD] = x16_simd512_calc_tpb_80(thr_id);

	tpb64[ECHO] = x11_echo512_calc_tpb_64(thr_id);
	tpb80[ECHO] = x16_echo512_calc_tpb_80(thr_id);

	tpb64[HAMSI] = x13_hamsi512_calc_tpb_64(thr_id);
	tpb80[HAMSI] = x13_hamsi512_calc_tpb_80(thr_id);

	tpb64[FUGUE] = x13_fugue512_calc_tpb_64(thr_id);
	tpb80[FUGUE] = x16_fugue512_calc_tpb_80(thr_id);

	tpb64[SHABAL] = x14_shabal512_calc_tpb_64(thr_id);
	tpb80[SHABAL] = x16_shabal512_calc_tpb_80(thr_id);

	tpb64[WHIRLPOOL] = x15_whirlpool512_calc_tpb_64(thr_id);
	tpb80[WHIRLPOOL] = x15_whirlpool512_calc_tpb_80(thr_id);

	tpb64[SHA512] = x17_sha512_calc_tpb_64(thr_id);
	tpb80[SHA512] = x17_sha512_calc_tpb_80(thr_id);

}
