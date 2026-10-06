# AB-JChess engine source

This directory contains the C++ UCI Jieqi engine and the ABJCHESSV11 NNUE
runtime. It supports UTF-8 output and network paths.

## Build

Use an MSYS2 UCRT64 shell with GNU Make and Clang installed. From this
directory or from `src/`, run:

```sh
make -j4 build ARCH=x86-64-bmi2 COMP=clang largeboards=yes
```

The executable is `src/AB-JChess.exe`. Build artifacts and NNUE packages are
kept outside this source tree.

## Runtime

The engine accepts an authenticated `ABJCHESSV11` package. Set its location
before search with:

```text
setoption name EvalFile value C:\path\to\abjchess-v11.nnue
```

The package must match the v11 feature and architecture contract implemented
under `src/abjnnue/`.

## Tests

The native regression targets can be built with:

```sh
make -j4 abjnnue-test ARCH=x86-64-bmi2 COMP=clang largeboards=yes
```
