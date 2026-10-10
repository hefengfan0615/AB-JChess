# AB-JChess engine source

This directory contains the C++ UCI Jieqi engine and the ABJCHESSV11 NNUE
runtime. It supports UTF-8 output and network paths.

## Build

Use an MSYS2 UCRT64 shell with GNU Make and Clang installed. From this
directory or from `src/`, run:

```sh
make -j4 build ARCH=x86-64-bmi2 COMP=clang
```

The executable is `src/AB-JChess.exe`. This distribution contains source only;
building creates local objects and executables. Run `make clean` after building
to restore a source-only tree. NNUE packages are supplied separately.

## Runtime

The default EvalFile is `abjchess-20261010.nnue`. The engine accepts an
authenticated `ABJCHESSV11` package. Set another location before search with:

```text
setoption name EvalFile value C:\path\to\abjchess-20261010.nnue
```

The package must match the v11 feature and architecture contract implemented
under `src/abjnnue/`.

## Tests

The native regression targets can be built with:

```sh
make -j4 abjnnue-test ARCH=x86-64-bmi2 COMP=clang
```
