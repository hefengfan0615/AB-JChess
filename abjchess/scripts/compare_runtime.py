"""Compare deterministic single-thread searches, excluding model-load time."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from queue import Queue, Empty
import re
import statistics
import subprocess
import threading


class Engine:
    def __init__(self, executable, model, cpu):
        self.executable = Path(executable).resolve()
        self.process = subprocess.Popen([str(self.executable)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
        if cpu is not None:
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            if not kernel.SetProcessAffinityMask(int(self.process._handle), 1 << cpu):
                self.process.kill()
                raise ctypes.WinError(ctypes.get_last_error())
        self.lines = Queue()
        def read():
            for line in self.process.stdout:
                self.lines.put(line.strip())
            self.lines.put(None)
        threading.Thread(target=read, daemon=True).start()
        self.send('uci')
        self.until('uciok')
        self.send(f'setoption name EvalFile value {Path(model).resolve().as_posix()}')
        self.send('setoption name Threads value 1')
        self.send('setoption name Hash value 16')
        self.send('position startpos')
        self.send('go nodes 1000')
        self.until('bestmove ')

    def send(self, command):
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

    def until(self, prefix):
        lines = []
        while True:
            try:
                line = self.lines.get(timeout=120)
            except Empty as error:
                raise RuntimeError(f'{self.executable}: response timeout') from error
            if line is None:
                raise RuntimeError(f'{self.executable} exited: {lines[-8:]}')
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def search(self, position, nodes):
        self.send('ucinewgame')
        self.send('isready')
        self.until('readyok')
        self.send('position ' + position)
        self.send(f'go nodes {nodes}')
        lines = self.until('bestmove ')
        iterations = [line for line in lines if line.startswith('info depth ')]
        if not iterations:
            raise RuntimeError(f'No search iterations: {lines}')
        final = iterations[-1]
        count = int(re.search(r'\bnodes (\d+)', final)[1])
        elapsed = int(re.search(r'\btime (\d+)', final)[1])
        return dict(nodes=count, milliseconds=elapsed, nps=count * 1000 / max(1, elapsed),
                    transcript=[re.sub(r'\b(?:nps|time) \d+ ?', '', line)
                                for line in [*iterations, lines[-1]]])

    def close(self):
        if self.process.poll() is None:
            self.send('quit')
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', action='append', required=True, help='LABEL=EXE')
    parser.add_argument('--model', required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--nodes', type=int, default=100000)
    parser.add_argument('--opening-nodes', type=int, default=300000)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = (Path(__file__).resolve().parents[1] / 'src/benchmark.cpp').read_text()
    defaults = source.split('Defaults = {', 1)[1].split('};', 1)[0]
    fens = re.findall(r'"([^"]+)"', defaults)
    cases = [('opening', 'startpos', args.opening_nodes)]
    cases += [(f'bench-{i+1:02}', 'fen ' + fen, args.nodes) for i, fen in enumerate(fens)]
    engines = {}
    report = dict(model=str(Path(args.model).resolve()), cpu=args.cpu, threads=1, hash_mb=16,
                  rounds=args.rounds, engines={}, records=[], all_equal=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for specification in args.engine:
            label, path = specification.split('=', 1)
            engines[label] = Engine(path, args.model, args.cpu)
            report['engines'][label] = dict(path=str(Path(path).resolve()),
                                           sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        reference = {}
        for run in range(args.rounds):
            order = list(engines)
            if run % 2:
                order.reverse()
            for label in order:
                for name, position, nodes in cases:
                    result = engines[label].search(position, nodes)
                    reference.setdefault(name, result['transcript'])
                    equal = result['transcript'] == reference[name]
                    report['all_equal'] &= equal
                    report['records'].append(dict(engine=label, round=run + 1, case=name,
                                                  position=position, equal=equal, **result))
                    if not equal:
                        raise RuntimeError(f'Behavior mismatch: {label}, {name}, round {run + 1}')
                print(f'round {run + 1}: {label} all {len(cases)} positions equal', flush=True)
        report['summary'] = {}
        for label in engines:
            rows = [r for r in report['records'] if r['engine'] == label]
            opening = [r['nps'] for r in rows if r['case'] == 'opening']
            suite = []
            for run in range(1, args.rounds + 1):
                part = [r for r in rows if r['round'] == run and r['case'] != 'opening']
                suite.append(sum(r['nodes'] for r in part) * 1000 / sum(r['milliseconds'] for r in part))
            report['summary'][label] = dict(opening_nps_median=statistics.median(opening),
                                             suite_nps_median=statistics.median(suite))
        print(json.dumps(report['summary'], indent=2), flush=True)
    finally:
        for engine in engines.values():
            engine.close()
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
