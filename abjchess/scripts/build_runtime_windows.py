"""Build isolated Windows runtime binaries/tests using MinGW GCC, without MSYS.

Example: python scripts/build_runtime_windows.py --arch avxvnni --tests
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='C:/mingw64/bin/g++.exe')
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arch', choices=('bmi2', 'avxvnni'), default='bmi2')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--scalar-transform', action='store_true')
    parser.add_argument('--tests', action='store_true')
    args = parser.parse_args()
    root = args.source_root.resolve()
    output = (args.output or root / 'build' / args.arch).resolve()
    output.mkdir(parents=True, exist_ok=True)
    flags = [
        '-std=c++17', '-O3', '-funroll-loops', '-DNDEBUG', '-DABJCHESS_V11',
        '-DIS_64BIT', '-DNO_PREFETCH', '-DUSE_POPCNT', '-DUSE_PEXT',
        '-DUSE_SSE2', '-DUSE_SSSE3', '-DUSE_SSE41', '-DUSE_AVX2',
        '-DABJNNUE_RUNTIME_REFRESH_CACHE', '-m64', '-mavx2', '-mbmi', '-mbmi2',
        '-mpopcnt', '-flto', '-flto-partition=one', '-Wall', '-Wextra',
        # GCC 15 MinGW can emit an aligned 256-bit store to a 16-byte-aligned
        # stack temporary (Engine::resize_threads). Explicit NNUE intrinsics
        # retain their 256-bit width; restrict automatic vectorization only.
        '-mprefer-vector-width=128', '-I' + str(root / 'src'),
    ]
    if args.arch == 'avxvnni':
        flags += ['-mavxvnni', '-DUSE_VNNI', '-DUSE_AVXVNNI']
    if not args.scalar_transform:
        flags += ['-DABJNNUE_RUNTIME_SIMD_TRANSFORM']
    if args.stats:
        flags += ['-DABJNNUE_RUNTIME_STATS']

    def run(command, log):
        with log.open('w', encoding='utf-8') as stream:
            stream.write(subprocess.list2cmdline(command) + '\n')
            stream.flush()
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Build failed ({result.returncode}); see {log}')

    def compile_source(source):
        obj = output / (source.stem + '.o')
        run([args.compiler, *flags, '-c', str(source), '-o', str(obj)],
            output / (source.stem + '.compile.log'))
        return obj

    sources = sorted((root / 'src').rglob('*.cpp'))
    tests = [
        'abjnnue_runtime_test', 'abjnnue_v11_contract_test', 'abjnnue_sha256_test',
        'position_protocol_test', 'reveal_test', 'skyrule_jieqi_test',
    ] if args.tests else []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        objects = list(pool.map(compile_source, sources))
        test_objects = list(pool.map(compile_source,
                                    [root / 'tests' / (name + '.cpp') for name in tests]))
    engine_objects = [obj for obj in objects if obj.stem != 'main']
    binaries = [('AB-JChess-v11-' + args.arch, objects)]
    binaries += [(obj.stem, [obj, *engine_objects]) for obj in test_objects]
    for name, inputs in binaries:
        run([args.compiler, *flags, '-static', '-Wl,--stack,16777216',
             '-o', str(output / (name + '.exe')), *map(str, inputs), '-lpthread'],
            output / (name + '.link.log'))
        print(output / (name + '.exe'), flush=True)


if __name__ == '__main__':
    main()
