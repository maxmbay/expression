"""Pinned access to the official Tahoe-100M Hugging Face release."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from huggingface_hub import HfApi

REPO_ID = "tahoebio/Tahoe-100M"
REPO_TYPE = "dataset"

METADATA_FILES = (
    "metadata/sample_metadata.parquet",
    "metadata/drug_metadata.parquet",
    "metadata/cell_line_metadata.parquet",
    "metadata/gene_metadata.parquet",
    "metadata/obs_metadata.parquet",
    "LICENSE.md",
)


@dataclass(frozen=True)
class SourceFile:
    path: str
    size: int
    sha256: str | None


def list_files(revision: str, prefix: str = "") -> list[SourceFile]:
    api = HfApi()
    entries = api.list_repo_tree(
        REPO_ID, path_in_repo=prefix or None, repo_type=REPO_TYPE, revision=revision, recursive=True
    )
    files = []
    for entry in entries:
        size = getattr(entry, "size", None)
        if size is None:
            continue
        lfs = getattr(entry, "lfs", None)
        files.append(SourceFile(entry.path, int(size), getattr(lfs, "sha256", None)))
    return sorted(files, key=lambda f: f.path)


def resolve_url(path: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{path}"


def download(path: str, revision: str, root: Path, expected_size: int | None = None) -> Path:
    """Resumable streaming download. The hub client stalled on large xet-backed files here."""
    target = root / path
    if expected_size is not None and target.exists() and target.stat().st_size == expected_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    with httpx.stream(
        "GET", resolve_url(path, revision), headers=headers, follow_redirects=True, timeout=60
    ) as response:
        if response.status_code == 200 and offset:
            offset = 0
        elif response.status_code not in (200, 206):
            response.raise_for_status()
        with partial.open("ab" if offset else "wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise OSError(f"{path}: got {partial.stat().st_size} bytes, expected {expected_size}")
    partial.replace(target)
    return target


def sha256sum(path: Path, chunk_bytes: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    out: Path, revision: str, local_files: dict[str, Path], listing: list[SourceFile]
) -> dict:
    remote = {f.path: f for f in listing}
    records = []
    for repo_path, local in sorted(local_files.items()):
        checksum = sha256sum(local)
        expected = remote.get(repo_path)
        records.append(
            {
                "path": repo_path,
                "bytes": local.stat().st_size,
                "sha256": checksum,
                "remote_sha256": expected.sha256 if expected else None,
                "checksum_matches_remote": (
                    None if expected is None or expected.sha256 is None
                    else expected.sha256 == checksum
                ),
            }
        )
    manifest = {
        "repo_id": REPO_ID,
        "revision": revision,
        "source_url": f"https://huggingface.co/datasets/{REPO_ID}/tree/{revision}",
        "files": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
