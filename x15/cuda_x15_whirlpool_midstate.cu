extern "C"
{
#include "sph/sph_whirlpool.h"
#include "miner.h"
}

void whirlpool_midstate(void *state, const void *input)
{
        sph_whirlpool_context ctx;

        sph_whirlpool_init(&ctx);
        sph_whirlpool(&ctx, input, 64);

        memcpy(state, ctx.state, 64);
}

void whirl_midstate(void *state, const void *input)
{
        sph_whirlpool_context ctx;

        sph_whirlpool1_init(&ctx);
        sph_whirlpool1(&ctx, input, 64);

        memcpy(state, ctx.state, 64);
}
