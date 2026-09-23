#!/usr/bin/env python3
"""Build and validate deterministic edit-order manifests.

An order manifest permutes a *fixed prefix multiset* without changing the
model/training RNG seed.  This keeps request-order robustness separate from
other stochasticity in cumulative knowledge-editing experiments.

The manifest records both an order-invariant multiset digest and an ordered
digest.  Shuffle manifests are regenerated from their declared Python RNG
seed during validation, so a hand-picked permutation cannot masquerade as a
predeclared random order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = 1
MANIFEST_KIND = "fixed_prefix_edit_order"
SHUFFLE_ALGORITHM = "python_random_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _record_digest(record: Any) -> str:
    return hashlib.sha256(_canonical_bytes(record)).hexdigest()


def _sequence_digest(records: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_record_digest(record).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _multiset_digest(records: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for record_digest in sorted(_record_digest(record) for record in records):
        digest.update(record_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_record_list(path: Path) -> List[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Edit-order source must be a JSON list: {path}")
    return payload


def _permutation(sample_size: int, *, seed: int | None) -> List[int]:
    indices = list(range(sample_size))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    return indices


def build_manifest(
    data_path: Path,
    *,
    sample_size: int,
    order_id: str,
    shuffle_seed: int | None,
) -> Dict[str, Any]:
    """Return one deterministic manifest without writing it."""

    data_path = data_path.expanduser().resolve()
    records = _read_record_list(data_path)
    if sample_size <= 0 or sample_size > len(records):
        raise ValueError(
            f"sample_size must be in [1, {len(records)}], got {sample_size}"
        )
    if not order_id or any(character.isspace() for character in order_id):
        raise ValueError("order_id must be non-empty and contain no whitespace")

    selected = records[:sample_size]
    permutation = _permutation(sample_size, seed=shuffle_seed)
    ordered = [selected[index] for index in permutation]
    order_kind = "prefix" if shuffle_seed is None else "seeded_shuffle"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "order_id": order_id,
        "order_kind": order_kind,
        "shuffle_algorithm": (
            None if shuffle_seed is None else SHUFFLE_ALGORITHM
        ),
        "shuffle_seed": shuffle_seed,
        "source_path": str(data_path),
        "source_sha256": file_sha256(data_path),
        "source_record_count": len(records),
        "selection": "prefix",
        "sample_size": sample_size,
        "selected_source_indices": [0, sample_size - 1],
        "selected_multiset_sha256": _multiset_digest(selected),
        "permutation_indices": permutation,
        "ordered_records_sha256": _sequence_digest(ordered),
    }


def validate_manifest(
    manifest_path: Path,
    *,
    data_path: Path | None = None,
    sample_size: int | None = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Validate provenance and return the declared permutation plus metadata."""

    manifest_path = manifest_path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Order manifest must be a JSON object: {manifest_path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported edit-order schema: {payload.get('schema_version')!r}"
        )
    if payload.get("kind") != MANIFEST_KIND:
        raise ValueError(f"Unexpected edit-order kind: {payload.get('kind')!r}")
    if payload.get("selection") != "prefix":
        raise ValueError("Only fixed-prefix edit-order manifests are supported")
    order_id = payload.get("order_id")
    if (
        not isinstance(order_id, str)
        or not order_id
        or any(character.isspace() for character in order_id)
    ):
        raise ValueError("Edit-order manifest has an invalid order_id")

    manifest_sample_size = int(payload.get("sample_size", -1))
    if sample_size is not None and manifest_sample_size != int(sample_size):
        raise ValueError(
            "Edit-order sample size mismatch: "
            f"manifest={manifest_sample_size}, requested={sample_size}"
        )
    if manifest_sample_size <= 0:
        raise ValueError("Edit-order manifest has a non-positive sample size")

    source_path = (
        data_path.expanduser().resolve()
        if data_path is not None
        else Path(str(payload.get("source_path", ""))).expanduser().resolve()
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"Edit-order source does not exist: {source_path}")
    actual_source_sha = file_sha256(source_path)
    if payload.get("source_sha256") != actual_source_sha:
        raise ValueError(
            "Edit-order source hash mismatch: "
            f"manifest={payload.get('source_sha256')}, actual={actual_source_sha}"
        )

    records = _read_record_list(source_path)
    if int(payload.get("source_record_count", -1)) != len(records):
        raise ValueError("Edit-order source record count changed")
    if manifest_sample_size > len(records):
        raise ValueError("Edit-order sample size exceeds source record count")
    expected_source_range = [0, manifest_sample_size - 1]
    if payload.get("selected_source_indices") != expected_source_range:
        raise ValueError(
            "Edit-order manifest does not select the declared fixed prefix"
        )

    selected = records[:manifest_sample_size]
    actual_multiset_sha = _multiset_digest(selected)
    if payload.get("selected_multiset_sha256") != actual_multiset_sha:
        raise ValueError("Edit-order selected multiset hash mismatch")

    permutation = payload.get("permutation_indices")
    if not isinstance(permutation, list) or any(
        type(index) is not int for index in permutation
    ):
        raise ValueError("Edit-order permutation must be a list of integers")
    if len(permutation) != manifest_sample_size or sorted(permutation) != list(
        range(manifest_sample_size)
    ):
        raise ValueError("Edit-order permutation is not a bijection")

    order_kind = payload.get("order_kind")
    if order_kind == "prefix":
        if payload.get("shuffle_seed") is not None:
            raise ValueError("Prefix order must not declare a shuffle seed")
        if payload.get("shuffle_algorithm") is not None:
            raise ValueError("Prefix order must not declare a shuffle algorithm")
        expected_permutation = _permutation(manifest_sample_size, seed=None)
    elif order_kind == "seeded_shuffle":
        if payload.get("shuffle_algorithm") != SHUFFLE_ALGORITHM:
            raise ValueError("Unsupported seeded-shuffle algorithm")
        seed = payload.get("shuffle_seed")
        if type(seed) is not int:
            raise ValueError("Seeded shuffle must declare an integer seed")
        expected_permutation = _permutation(manifest_sample_size, seed=seed)
    else:
        raise ValueError(f"Unknown edit-order kind: {order_kind!r}")
    if permutation != expected_permutation:
        raise ValueError(
            "Edit-order permutation does not match its declared kind/seed"
        )

    ordered = [selected[index] for index in permutation]
    actual_ordered_sha = _sequence_digest(ordered)
    if payload.get("ordered_records_sha256") != actual_ordered_sha:
        raise ValueError("Edit-order ordered-record hash mismatch")

    metadata = {
        "path": str(manifest_path),
        "sha256": file_sha256(manifest_path),
        "order_id": order_id,
        "order_kind": order_kind,
        "shuffle_algorithm": payload.get("shuffle_algorithm"),
        "shuffle_seed": payload.get("shuffle_seed"),
        "source_path": str(source_path),
        "source_sha256": actual_source_sha,
        "source_record_count": len(records),
        "sample_size": manifest_sample_size,
        "selected_multiset_sha256": actual_multiset_sha,
        "ordered_records_sha256": actual_ordered_sha,
    }
    return list(permutation), metadata


def apply_manifest_order(
    records: Sequence[Any], permutation: Sequence[int]
) -> List[Any]:
    if len(records) != len(permutation):
        raise ValueError(
            f"Cannot apply {len(permutation)} indices to {len(records)} records"
        )
    if sorted(permutation) != list(range(len(records))):
        raise ValueError("Permutation is not a bijection over the records")
    return [records[index] for index in permutation]


def _write_json_idempotently(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != rendered:
            raise FileExistsError(
                f"Refusing to overwrite a different manifest: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_suite(
    data_path: Path,
    *,
    sample_size: int,
    output_dir: Path,
    shuffle_seeds: Sequence[int],
    include_prefix: bool = True,
) -> Dict[str, Any]:
    """Build an idempotent prefix-plus-seeded-shuffles manifest suite."""

    if len(set(shuffle_seeds)) != len(shuffle_seeds):
        raise ValueError("shuffle_seeds contain duplicates")
    output_dir = output_dir.expanduser().resolve()
    specifications: List[Tuple[str, int | None]] = []
    if include_prefix:
        specifications.append(("o0_prefix", None))
    specifications.extend(
        (f"o{index}_seed{seed}", int(seed))
        for index, seed in enumerate(shuffle_seeds, start=1 if include_prefix else 0)
    )
    if not specifications:
        raise ValueError("At least one prefix or shuffled order is required")

    built: List[Dict[str, Any]] = []
    seen_ordered_hashes = set()
    common_multiset_sha: str | None = None
    for order_id, seed in specifications:
        manifest = build_manifest(
            data_path,
            sample_size=sample_size,
            order_id=order_id,
            shuffle_seed=seed,
        )
        ordered_sha = str(manifest["ordered_records_sha256"])
        if ordered_sha in seen_ordered_hashes:
            raise ValueError(f"Order {order_id} duplicates an earlier permutation")
        seen_ordered_hashes.add(ordered_sha)
        multiset_sha = str(manifest["selected_multiset_sha256"])
        if common_multiset_sha is None:
            common_multiset_sha = multiset_sha
        elif common_multiset_sha != multiset_sha:
            raise AssertionError("Built orders unexpectedly changed membership")

        path = output_dir / f"{order_id}.json"
        _write_json_idempotently(path, manifest)
        _, metadata = validate_manifest(
            path, data_path=data_path, sample_size=sample_size
        )
        built.append(metadata)

    suite = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fixed_prefix_edit_order_suite",
        "source_path": str(data_path.expanduser().resolve()),
        "source_sha256": file_sha256(data_path.expanduser().resolve()),
        "sample_size": sample_size,
        "selected_multiset_sha256": common_multiset_sha,
        "orders": built,
    }
    _write_json_idempotently(output_dir / "suite_manifest.json", suite)
    return suite


def _parse_seed_list(text: str) -> List[int]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        return []
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shuffle seeds must be comma-separated integers"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build an order-manifest suite")
    build.add_argument("--data-path", type=Path, required=True)
    build.add_argument("--sample-size", type=int, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument(
        "--shuffle-seeds",
        type=_parse_seed_list,
        default=[1729, 2718],
        help="Comma-separated seeds; default: 1729,2718",
    )
    build.add_argument(
        "--include-prefix", type=int, choices=[0, 1], default=1
    )

    validate = subparsers.add_parser("validate", help="Validate one manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--data-path", type=Path, default=None)
    validate.add_argument("--sample-size", type=int, default=None)

    args = parser.parse_args()
    if args.command == "build":
        result = build_suite(
            args.data_path,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            shuffle_seeds=args.shuffle_seeds,
            include_prefix=bool(args.include_prefix),
        )
    else:
        _, result = validate_manifest(
            args.manifest,
            data_path=args.data_path,
            sample_size=args.sample_size,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
