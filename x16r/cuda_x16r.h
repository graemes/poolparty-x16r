// Include all of the function definitions here for ease of maintenance

extern void quark_blake512_cpu_init(int thr_id, uint32_t threads);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64_sp(uint32_t threads, uint32_t *d_outputHash);
extern void quark_blake512_cpu_free(int thr_id);

extern void quark_bmw512_cpu_init(int thr_id, uint32_t threads);
extern void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_groestl512_cpu_free(int thr_id);

extern void quark_jh512_cpu_init(int thr_id, uint32_t threads);
extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);
extern void quark_skein512_cpu_hash_64(int thr_id, const uint32_t threads, uint32_t *d_hash);

extern void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x11_luffa512_cpu_init(int thr_id, uint32_t threads);
extern void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads,uint32_t *d_hash);

extern void x11_cubehash512_cpu_init(int thr_id, uint32_t threads);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x11_shavite512_cpu_init(int thr_id, uint32_t threads);
extern void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern int  x11_simd512_cpu_init(int thr_id, uint32_t threads);
extern void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_simd512_cpu_free(int thr_id);

extern void x11_echo512_cpu_init(int thr_id, uint32_t threads);
extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x13_hamsi512_cpu_init(int thr_id, uint32_t threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int flag);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

// ---- 80 bytes kernels
extern void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *pdata);
extern void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

void quark_bmw512_cpu_setBlock_80(void *pdata);
void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);

void groestl512_setBlock_80(int thr_id, uint32_t *endiandata);
void groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void skein512_cpu_setBlock_80(void *pdata);
void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int swap);

extern void qubit_luffa512_cpu_setBlock_80(void *pdata);
extern void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash);

void jh512_setBlock_80(int thr_id, uint32_t *endiandata);
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

extern void quark_keccak512_cpu_init_80(int thr_id, uint32_t threads);
extern void keccak512_setBlock_80(int thr_id, uint32_t *endiandata);
extern void keccak512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void cubehash512_setBlock_80(int thr_id, uint32_t* endiandata);
void cubehash512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x11_shavite512_setBlock_80(void *pdata);
void x11_shavite512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);

void x16_shabal512_setBlock_80(void *pdata);
void x16_shabal512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_simd512_setBlock_80(void *pdata);
void x16_simd512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_echo512_cuda_init(int thr_id, const uint32_t threads);
void x16_echo512_setBlock_80(void *pdata);
void x16_echo512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_hamsi512_setBlock_80(void *pdata);
void x16_hamsi512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_fugue512_cpu_init(int thr_id, uint32_t threads);
void x16_fugue512_cpu_free(int thr_id);
void x16_fugue512_setBlock_80(void *pdata);
void x16_fugue512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_whirlpool512_init(int thr_id, uint32_t threads);
void x16_whirlpool512_setBlock_80(void* endiandata);
void x16_whirlpool512_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_sha512_setBlock_80(void *pdata);
void x16_sha512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
