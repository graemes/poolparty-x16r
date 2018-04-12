# ccminer

If you find this version useful please consider a donation to: RH4KkDFJV7FuURwrZDyQZoPWAKc4hSHuDU (graemes)

Things to watch out for:
- only builds for compute_50, compute_52 & compute_61 - change Makefile.am if you want to include other nvidia architectures ;
- only tested against cuda 9.1

This version is based primarily on:
- the Nevermore miner (https://github.com/brian112358/nevermore-miner - RWoSZX6j6WU6SVTVq5hKmdgPmmrYE9be5R) ;
- the work of Penfold in stripping down the original ccminer (https://github.com/todd1251/ccminer-x16r/tree/x16r-only - RWoyvvT5exmbs937QfRavf4fxB5mvijG6R) ;
- alexis78 kernel optimisations (https://github.com/alexis78/ccminer - RYKaoWqR5uahFioNvxabQtEBjNkBmRoRdg) 
- along with changes hand-picked from other tree's ;

Standing on the shoulders of giants.

Based on Christian Buchner's &amp; Christian H.'s CUDA project, no more active on github since 2014.

BTC donation address: 1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo (tpruvot)

A part of the recent algos were originally written by [djm34](https://github.com/djm34) and [alexis78](https://github.com/alexis78)

This variant was tested and built on Linux (ubuntu server 16.04, Fedora 27)
It is also built for Windows 7 to 10 with VStudio 2013, to stay compatible with Windows 7 and Vista.

About source code dependencies
------------------------------

This project requires some libraries to be built :

- OpenSSL (prebuilt for win)
- Curl (prebuilt for win)
- pthreads (prebuilt for win)

The tree now contains recent prebuilt openssl and curl .lib for both x86 and x64 platforms (windows).

To rebuild them, you need to clone this repository and its submodules :
    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows


Compile on Linux
----------------

Please see [INSTALL](https://github.com/tpruvot/ccminer/blob/linux/INSTALL) file or [project Wiki](https://github.com/tpruvot/ccminer/wiki/Compatibility)
