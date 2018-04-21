// Definitions here FNPRAA

enum x16r_algos {
	BLAKE = 0,
	BMW,
	GROESTL,
	JH,
	KECCAK,
	SKEIN,
	LUFFA,
	CUBEHASH,
	SHAVITE,
	SIMD,
	ECHO,
	HAMSI,
	FUGUE,
	SHABAL,
	WHIRLPOOL,
	SHA512,
	HASH_FUNC_COUNT
};

static const char* algo_strings[] = {
	"blake",
	"bmw",
	"groestl",
	"jh",
	"keccak",
	"skein",
	"luffa",
	"cubehash",
	"shavite",
	"simd",
	"echo",
	"hamsi",
	"fugue",
	"shabal",
	"whirlpool",
	"sha512",
	NULL
};

static const uint64_t algo_hashes[] = {
	0x0000000000000000,
	0x1111111111111111,
	0x2222222222222222,
	0x3333333333333333,
	0x4444444444444444,
	0x5555555555555555,
	0x6666666666666666,
	0x7777777777777777,
	0x8888888888888888,
	0x9999999999999999,
	0xAAAAAAAAAAAAAAAA,
	0xBBBBBBBBBBBBBBBB,
	0xCCCCCCCCCCCCCCCC,
	0xDDDDDDDDDDDDDDDD,
	0xEEEEEEEEEEEEEEEE,
	0xFFFFFFFFFFFFFFFF
};

static const char* algo80_strings[] = {
	"blake_80",
	"bmw_80",
	"groestl_80",
	"jh_80",
	"keccak_80",
	"skein_80",
	"luffa_80",
	"cubehash_80",
	"shavite_80",
	"simd_80",
	"echo_80",
	"hamsi_80",
	"fugue_80",
	"shabal_80",
	"whirlpool_80",
	"sha512_80",
	NULL
};

static const uint64_t algo80_hashes[] = {
	0x0000000000000000,
	0x1000000000000000,
	0x2000000000000000,
	0x3000000000000000,
	0x4000000000000000,
	0x5000000000000000,
	0x6000000000000000,
	0x7000000000000000,
	0x8000000000000000,
	0x9000000000000000,
	0xA000000000000000,
	0xB000000000000000,
	0xC000000000000000,
	0xD000000000000000,
	0xE000000000000000
};

// Include all of the function definitions here for ease of maintenance

// ---- 64 byte kernels
extern void quark_blake512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_blake512_calc_tpb_64(int thr_id);
extern void quark_blake512_cpu_free_64(int thr_id);

extern void quark_bmw512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_bmw512_calc_tpb_64(int thr_id);
extern void quark_bmw512_cpu_free_64(int thr_id);

extern void quark_groestl512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_groestl512_calc_tpb_64(int thr_id);
extern void quark_groestl512_cpu_free_64(int thr_id);

extern void quark_jh512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_jh512_calc_tpb_64(int thr_id);
extern void quark_jh512_cpu_free_64(int thr_id);

extern void quark_keccak512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_keccak512_calc_tpb_64(int thr_id);
extern void quark_keccak512_cpu_free_64(int thr_id);

extern void quark_skein512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_skein512_calc_tpb_64(int thr_id);
extern void quark_skein512_cpu_free_64(int thr_id);

extern void qubit_luffa512_cpu_init_64(int thr_id, uint32_t threads);
extern void qubit_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t qubit_luffa512_calc_tpb_64(int thr_id);
extern void qubit_luffa512_cpu_free_64(int thr_id);

extern void x11_cubehash512_cpu_init_64(int thr_id, uint32_t threads);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x11_cubehash512_calc_tpb_64(int thr_id);
extern void x11_cubehash512_cpu_free_64(int thr_id);

extern void x11_shavite512_cpu_init_64(int thr_id, uint32_t threads);
extern void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x11_shavite512_calc_tpb_64(int thr_id);
extern void x11_shavite512_cpu_free_64(int thr_id);

extern void x11_simd512_cpu_init_64(int thr_id, uint32_t threads);
extern void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x11_simd512_calc_tpb_64(int thr_id);
extern void x11_simd512_cpu_free_64(int thr_id);

extern void x11_echo512_cpu_init_64(int thr_id, uint32_t threads);
extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x11_echo512_calc_tpb_64(int thr_id);
extern void x11_echo512_cpu_free_64(int thr_id);

extern void x13_hamsi512_cpu_init_64(int thr_id, uint32_t threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x13_hamsi512_calc_tpb_64(int thr_id);
extern void x13_hamsi512_cpu_free_64(int thr_id);

extern void x13_fugue512_cpu_init_64(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x13_fugue512_calc_tpb_64(int thr_id);
extern void x13_fugue512_cpu_free_64(int thr_id);

extern void x14_shabal512_cpu_init_64(int thr_id, uint32_t threads);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x14_shabal512_calc_tpb_64(int thr_id);
extern void x14_shabal512_cpu_free_64(int thr_id);

extern void x15_whirlpool512_cpu_init_64(int thr_id, uint32_t threads);
extern void x15_whirlpool512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x15_whirlpool512_calc_tpb_64(int thr_id);
extern void x15_whirlpool512_cpu_free_64(int thr_id);

extern void x17_sha512_cpu_init_64(int thr_id, uint32_t threads);
extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x17_sha512_calc_tpb_64(int thr_id);
extern void x17_sha512_cpu_free_64(int thr_id);

// ---- 80 byte kernels
extern void quark_blake512_cpu_init_80(int thr_id, const uint32_t threads);
extern void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *pdata);
extern void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_blake512_calc_tpb_80(int thr_id);
extern void quark_blake512_cpu_free_80(int thr_id);

extern void quark_bmw512_cpu_init_80(int thr_id, const uint32_t threads);
extern void quark_bmw512_cpu_setBlock_80(void *pdata);
extern void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_bmw512_calc_tpb_80(int thr_id);
extern void quark_bmw512_cpu_free_80(int thr_id);

extern void quark_groestl512_cpu_init_80(int thr_id, const uint32_t threads);
extern void quark_groestl512_setBlock_80(int thr_id, uint32_t *endiandata);
extern void quark_groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_groestl512_calc_tpb_80(int thr_id);
extern void quark_groestl512_cpu_free_80(int thr_id);

extern void quark_jh512_cpu_init_80(int thr_id, const uint32_t threads);
extern void quark_jh512_setBlock_80(int thr_id, uint32_t *endiandata);
extern void quark_jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_jh512_calc_tpb_80(int thr_id);
extern void quark_jh512_cpu_free_80(int thr_id);

extern void quark_keccak512_cpu_init_80(int thr_id, uint32_t threads);
extern void quark_keccak512_setBlock_80(int thr_id, void *pdata);
extern void quark_keccak512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_keccak512_calc_tpb_80(int thr_id);
extern void quark_keccak512_cpu_free_80(int thr_id);

extern void quark_skein512_cpu_init_80(int thr_id, const uint32_t threads);
extern void quark_skein512_cpu_setBlock_80(void *pdata);
extern void quark_skein512_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t quark_skein512_calc_tpb_80(int thr_id);
extern void quark_skein512_cpu_free_80(int thr_id);

extern void qubit_luffa512_cpu_init_80(int thr_id, const uint32_t threads);
extern void qubit_luffa512_cpu_setBlock_80(void *pdata);
extern void qubit_luffa512_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t qubit_luffa512_calc_tpb_80(int thr_id);
extern void qubit_luffa512_cpu_free_80(int thr_id);

extern void x11_cubehash512_cpu_init_80(int thr_id, const uint32_t threads);
extern void x11_cubehash512_setBlock_80(int thr_id, uint32_t* endiandata);
extern void x11_cubehash512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x11_cubehash512_calc_tpb_80(int thr_id);
extern void x11_cubehash512_cpu_free_80(int thr_id);

extern void x11_shavite512_cpu_init_80(int thr_id, const uint32_t threads);
extern void x11_shavite512_setBlock_80(void *pdata);
extern void x11_shavite512_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x11_shavite512_calc_tpb_80(int thr_id);
extern void x11_shavite512_cpu_free_80(int thr_id);

extern void x16_simd512_cpu_init_80(int thr_id, const uint32_t threads);
extern void x16_simd512_setBlock_80(void *pdata);
extern void x16_simd512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x16_simd512_calc_tpb_80(int thr_id);
extern void x16_simd512_cpu_free_80(int thr_id);

extern void x16_echo512_cpu_init_80(int thr_id, const uint32_t threads);
extern void x16_echo512_setBlock_80(void *pdata);
extern void x16_echo512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x16_echo512_calc_tpb_80(int thr_id);
extern void x16_echo512_cpu_free_80(int thr_id);

extern void x13_hamsi512_cpu_init_80(int thr_id, uint32_t threads);
extern void x13_hamsi512_setBlock_80(void *pdata);
extern void x13_hamsi512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x13_hamsi512_calc_tpb_80(int thr_id);
extern void x13_hamsi512_cpu_free_80(int thr_id);

extern void x16_fugue512_cpu_init_80(int thr_id, uint32_t threads);
extern void x16_fugue512_setBlock_80(void *pdata);
extern void x16_fugue512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x16_fugue512_calc_tpb_80(int thr_id);
extern void x16_fugue512_cpu_free_80(int thr_id);

extern void x16_shabal512_cpu_init_80(int thr_id, const uint32_t threads);
extern void x16_shabal512_setBlock_80(void *pdata);
extern void x16_shabal512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x16_shabal512_calc_tpb_80(int thr_id);
extern void x16_shabal512_cpu_free_80(int thr_id);

extern void x15_whirlpool512_cpu_init_80(int thr_id, uint32_t threads);
extern void x15_whirlpool512_setBlock_80(void* endiandata);
extern void x15_whirlpool512_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x15_whirlpool512_calc_tpb_80(int thr_id);
extern void x15_whirlpool512_cpu_free_80(int thr_id);

extern void x17_sha512_cpu_init_80(int thr_id, uint32_t threads);
extern void x17_sha512_setBlock_80(void *pdata);
extern void x17_sha512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, uint32_t tpb);
extern uint32_t x17_sha512_calc_tpb_80(int thr_id);
extern void x17_sha512_cpu_free_80(int thr_id);
