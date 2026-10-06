"""Read-only structural inventory of every JQv4 file; never combine separate corpora."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from create_v11_jqv4_manifest import _read_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.input_root.resolve()
    paths = subprocess.check_output(['rg', '--files', '--hidden', '-g', '*.jqv4', str(root)], text=True).splitlines()
    sources, errors = [], []
    for path in sorted(paths):
        file = Path(path)
        try:
            sources.append(asdict(_read_source(file, root, str(file.parent.relative_to(root)),
                                               hash_file=False, require_range_name=False)))
        except Exception as error:
            errors.append({'path': str(file), 'error': str(error)})
    record = {'schema': 'abjchess-v11-source-inventory-v1', 'files': len(paths),
              'valid_files': len(sources), 'bytes': sum(s['bytes'] for s in sources),
              'records': sum(s['record_end'] - s['record_start'] for s in sources),
              'payload_sha256_verified': False, 'errors': errors, 'sources': sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in record.items() if k not in ('sources', 'errors')}))
    print('errors=' + str(len(errors)))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
