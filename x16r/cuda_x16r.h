// Include all of the function definitions here for ease of maintenance

// ---- 64 byte kernels
extern void quark_blake512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_blake512_cpu_free_64(int thr_id);

extern void quark_bmw512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_bmw512_cpu_free_64(int thr_id);

extern void quark_groestl512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_groestl512_cpu_free_64(int thr_id);

extern void quark_jh512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_jh512_cpu_free_64(int thr_id);

extern void quark_skein512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_skein512_cpu_hash_64(int thr_id, const uint32_t threads, uint32_t *d_hash);
extern void quark_skein512_cpu_free_64(int thr_id);

extern void quark_keccak512_cpu_init_64(int thr_id, uint32_t threads);
extern void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_keccak512_cpu_free_64(int thr_id);

extern void qubit_luffa512_cpu_init_64(int thr_id, uint32_t threads);
extern void qubit_luffa512_cpu_hash_64(int thr_id, uint32_t threads,uint32_t *d_hash);
extern void qubit_luffa512_cpu_free_64(int thr_id);

extern void x11_cubehash512_cpu_init_64(int thr_id, uint32_t threads);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_cubehash512_cpu_free_64(int thr_id);

extern void x11_shavite512_cpu_init_64(int thr_id, uint32_t threads);
extern void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_shavite512_cpu_free_64(int thr_id);

extern void x11_simd512_cpu_init_64(int thr_id, uint32_t threads);
extern void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_simd512_cpu_free_64(int thr_id);

extern void x11_echo512_cpu_init_64(int thr_id, uint32_t threads);
extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_echo512_cpu_free_64(int thr_id);

extern void x13_hamsi512_cpu_init_64(int thr_id, uint32_t threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x13_hamsi512_cpu_free_64(int thr_id);

extern void x13_fugue512_cpu_init_64(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x13_fugue512_cpu_free_64(int thr_id);

extern void x14_shabal512_cpu_init_64(int thr_id, uint32_t threads);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x14_shabal512_cpu_free_64(int thr_id);

extern void x15_whirlpool512_cpu_init_64(int thr_id, uint32_t threads);
extern void x15_whirlpool512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x15_whirlpool512_cpu_free_64(int thr_id);

extern void x17_sha512_cpu_init_64(int thr_id, uint32_t threads);
extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x17_sha512_cpu_free_64(int thr_id);

// ---- 80 byte kernels
extern void quark_blake512_cpu_init_80(int thr_id, const uint32_t threads);
extern void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *pdata);
extern void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void quark_blake512_cpu_free_80(int thr_id);

extern void quark_bmw512_cpu_init_80(int thr_id, const uint32_t threads);
extern void quark_bmw512_cpu_setBlock_80(void *pdata);
extern void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);
extern void quark_bmw512_cpu_free_80(int thr_id);

extern void quark_groestl512_cpu_init_80(int thr_id, const uint32_t threads);
extern void quark_groestl512_setBlock_80(int thr_id, uint32_t *endiandata);
extern void quark_groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void quark_groestl512_cpu_free_80(int thr_id);

extern void quark_skein512_cpu_init_80(int thr_id, const uint32_t threads);
extern void quark_skein512_cpu_setBlock_80(void *pdata);
extern void quark_skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int swap);
extern void quark_skein512_cpu_free_80(int thr_id);

extern void qubit_luffa512_cpu_init_80(int thr_id, const uint32_t threads);
extern void qubit_luffa512_cpu_setBlock_80(void *pdata);
extern void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash);
extern void qubit_luffa512_cpu_free_80(int thr_id);

extern void quark_jh512_cpu_init_80(int thr_id, const uint32_t threads);
extern void quark_jh512_setBlock_80(int thr_id, uint32_t *endiandata);
extern void quark_jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void quark_jh512_cpu_free_80(int thr_id);

extern void quark_keccak512_cpu_init_80(int thr_id, uint32_t threads);
extern void quark_keccak512_setBlock_80(int thr_id, uint32_t *endiandata);
extern void quark_keccak512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void quark_keccak512_cpu_free_80(int thr_id);

extern void x11_cubehash512_cpu_init_80(int thr_id, const uint32_t threads);
extern void x11_cubehash512_setBlock_80(int thr_id, uint32_t* endiandata);
extern void x11_cubehash512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void x11_cubehash512_cpu_free_80(int thr_id);

extern void x11_shavite512_cpu_init_80(int thr_id, const uint32_t threads);
extern void x11_shavite512_setBlock_80(void *pdata);
extern void x11_shavite512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);
extern void x11_shavite512_cpu_free_80(int thr_id);

extern void x16_simd512_cpu_init_80(int thr_id, const uint32_t threads);
extern void x16_simd512_setBlock_80(void *pdata);
extern void x16_simd512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void x16_simd512_cpu_free_80(int thr_id);

extern void x16_echo512_cpu_init_80(int thr_id, const uint32_t threads);
extern void x16_echo512_setBlock_80(void *pdata);
extern void x16_echo512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void x16_echo512_cpu_free_80(int thr_id);

extern void x13_hamsi512_cpu_init_80(int thr_id, uint32_t threads);
extern void x13_hamsi512_setBlock_80(void *pdata);
extern void x13_hamsi512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void x13_hamsi512_cpu_free_80(int thr_id);

extern void x16_fugue512_cpu_init_80(int thr_id, uint32_t threads);
extern void x16_fugue512_setBlock_80(void *pdata);
extern void x16_fugue512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void x16_fugue512_cpu_free_80(int thr_id);

extern void x16_shabal512_cpu_init_80(int thr_id, const uint32_t threads);
extern void x16_shabal512_setBlock_80(void *pdata);
extern void x16_shabal512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void x16_shabal512_cpu_free_80(int thr_id);

extern void x15_whirlpool512_cpu_init_80(int thr_id, uint32_t threads);
extern void x15_whirlpool512_setBlock_80(void* endiandata);
extern void x15_whirlpool512_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void x15_whirlpool512_cpu_free_80(int thr_id);

extern void x17_sha512_cpu_init_80(int thr_id, uint32_t threads);
extern void x17_sha512_setBlock_80(void *pdata);
extern void x17_sha512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
extern void x17_sha512_cpu_free_80(int thr_id);
