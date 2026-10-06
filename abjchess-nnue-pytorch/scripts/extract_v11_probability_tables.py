"""Copy authenticated probability tables from a compatible package."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    data = args.source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != args.source_sha256:
        raise ValueError("probability source SHA-256 mismatch")
    version, length = struct.unpack_from("<II", data, 16)
    valid_headers = {
        (b"ABJCHESSV11" + b"\0" * 5, 110),
        (b"ABJCHESSV82" + b"\0" * 5, 82),
    }
    if (data[:16], version) not in valid_headers:
        raise ValueError("expected a compatible probability source package")
    # The formal package contains a literal backslash-n after its JSON object.
    text = data[24:24 + length].decode("utf-8")
    metadata, end = json.JSONDecoder().raw_decode(text)
    if text[end:].strip() not in ("", "\\n"):
        raise ValueError("unexpected trailing metadata")
    wanted = {"probability_score_to_mass.i32le": 4001 * 4,
              "probability_mass_to_score.i32le": 1901 * 4}
    tables = {}
    for entry in metadata["chunks"]:
        name = entry["name"]
        start = 24 + length + entry["data_offset"]
        payload = data[start:start + entry["size"]]
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"source chunk SHA-256 mismatch: {name}")
        if name in wanted:
            if name in tables or len(payload) != wanted[name]:
                raise ValueError(f"invalid probability table: {name}")
            tables[name] = payload
    if tables.keys() != wanted.keys():
        raise ValueError("missing probability tables")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in tables.items():
        (args.output / name).write_bytes(payload)
    record = {"source": str(args.source.resolve()), "source_sha256": digest,
              "chunks": {k: hashlib.sha256(v).hexdigest() for k, v in tables.items()}}
    (args.output / "probability-tables-provenance.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
