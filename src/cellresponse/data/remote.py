"""Byte-range access to remote Parquet shards with one hub request per shard.

The hub's `/resolve` endpoint is rate limited; the CDN URL it redirects to is not (and is
signed for hours). We resolve once, then fetch the footer and only the needed row-group
byte spans from the CDN, and let pyarrow parse them from a sparse in-memory file.
"""

from __future__ import annotations

import io
import os
import threading

import httpx
import pyarrow.parquet as pq

from cellresponse.data.source import resolve_url

_local = threading.local()


def _client() -> httpx.Client:
    if getattr(_local, "client", None) is None:
        _local.client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=30))
    return _local.client


def _auth() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


class SparseFile(io.RawIOBase):
    """Read-only file of known size where only fetched byte ranges are readable."""

    def __init__(self, size: int):
        self.size = size
        self.position = 0
        self.ranges: list[tuple[int, bytes]] = []

    def add(self, start: int, data: bytes) -> None:
        self.ranges.append((start, data))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.position + n)
        for start, data in self.ranges:
            if start <= self.position and end <= start + len(data):
                chunk = data[self.position - start : end - start]
                self.position = end
                return chunk
        raise OSError(f"bytes {self.position}-{end} were not fetched")

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


class RemoteParquet:
    """Footer plus selected row groups of one remote Parquet file."""

    def __init__(self, repo_path: str, revision: str, tail_bytes: int = 1 << 17):
        client = _client()
        response = client.get(
            resolve_url(repo_path, revision),
            headers={**_auth(), "Range": f"bytes=-{tail_bytes}"},
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise OSError(f"{repo_path}: server ignored range request")
        self.cdn_url = str(response.url)
        self.size = int(response.headers["content-range"].rsplit("/", 1)[1])
        tail = response.content
        if tail[-4:] != b"PAR1":
            raise OSError(f"{repo_path}: missing Parquet magic")
        footer_length = int.from_bytes(tail[-8:-4], "little")
        if footer_length + 8 > len(tail):
            tail = self._range(self.size - footer_length - 8, self.size)
        self.file = SparseFile(self.size)
        self.file.add(self.size - len(tail), tail)
        self.metadata = pq.read_metadata(self.file)
        self.bytes_fetched = len(tail)

    def _range(self, start: int, stop: int) -> bytes:
        response = _client().get(self.cdn_url, headers={"Range": f"bytes={start}-{stop - 1}"})
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != stop - start:
            raise OSError(f"range {start}-{stop} returned {len(response.content)} bytes")
        return response.content

    def span(self, row_group: int, columns: tuple[str, ...]) -> tuple[int, int]:
        group = self.metadata.row_group(row_group)
        starts, stops = [], []
        for i in range(group.num_columns):
            column = group.column(i)
            if column.path_in_schema.split(".")[0] not in columns:
                continue
            start = column.data_page_offset
            if column.has_dictionary_page and column.dictionary_page_offset is not None:
                start = min(start, column.dictionary_page_offset)
            starts.append(start)
            stops.append(start + column.total_compressed_size)
        return min(starts), max(stops)

    def fetch_row_groups(
        self, row_groups: list[int], columns: tuple[str, ...], merge_gap: int = 1 << 20
    ) -> None:
        spans = sorted(self.span(g, columns) for g in row_groups)
        merged: list[list[int]] = []
        for start, stop in spans:
            if merged and start - merged[-1][1] <= merge_gap:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        for start, stop in merged:
            self.file.add(start, self._range(start, stop))
            self.bytes_fetched += stop - start

    def parquet_file(self) -> pq.ParquetFile:
        self.file.seek(0)
        return pq.ParquetFile(self.file, metadata=self.metadata)
