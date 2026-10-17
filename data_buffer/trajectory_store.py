import json
import os
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np

MAGIC = 0x54524A31
HEADER_FMT = "<IQII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class TrajectoryStore:
    def __init__(self, path, *, fsync_every: int = 1, verify: bool = False):
        self.path = Path(path)
        self.fsync_every = int(fsync_every)
        self._records = []
        self._file_end = 0
        self._num_transitions = 0
        self._appends_since_sync = 0
        if self.path.exists():
            self._f = open(self.path, "r+b")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "w+b")
        self._recover(verify=verify)

    @staticmethod
    def _serialize(episode: Dict[str, np.ndarray]):
        keys = list(episode.keys())
        arrs = {k: np.ascontiguousarray(episode[k]) for k in keys}
        n = int(arrs[keys[0]].shape[0])
        header = {
            "keys": keys,
            "dtypes": {k: arrs[k].dtype.str for k in keys},
            "shapes": {k: list(arrs[k].shape) for k in keys},
            "n_transitions": n,
        }
        hjson = json.dumps(header).encode("utf-8")
        parts = [struct.pack("<I", len(hjson)), hjson]
        for k in keys:
            parts.append(arrs[k].tobytes(order="C"))
        return b"".join(parts), n

    @staticmethod
    def _deserialize(payload: bytes) -> Dict[str, np.ndarray]:
        (hlen,) = struct.unpack_from("<I", payload, 0)
        header = json.loads(payload[4:4 + hlen].decode("utf-8"))
        out = {}
        off = 4 + hlen
        for k in header["keys"]:
            dt = np.dtype(header["dtypes"][k])
            shape = tuple(header["shapes"][k])
            count = int(np.prod(shape)) if shape else 1
            arr = np.frombuffer(payload, dtype=dt, count=count, offset=off).reshape(shape)
            out[k] = arr.copy()
            off += count * dt.itemsize
        return out

    def _recover(self, verify: bool = False) -> None:
        size = os.fstat(self._f.fileno()).st_size
        self._records = []
        off = 0
        while off + HEADER_SIZE <= size:
            self._f.seek(off)
            hdr = self._f.read(HEADER_SIZE)
            magic, plen, n, crc = struct.unpack(HEADER_FMT, hdr)
            if magic != MAGIC:
                break
            poff = off + HEADER_SIZE
            if poff + plen > size:
                break
            if verify:
                self._f.seek(poff)
                if (zlib.crc32(self._f.read(plen)) & 0xFFFFFFFF) != crc:
                    break
            self._records.append((poff, plen, n, crc))
            off = poff + plen
        valid_end = off
        if self._records and not verify:
            poff, plen, n, crc = self._records[-1]
            self._f.seek(poff)
            if (zlib.crc32(self._f.read(plen)) & 0xFFFFFFFF) != crc:
                self._records.pop()
                valid_end = poff - HEADER_SIZE
        if valid_end < size:
            self._f.truncate(valid_end)
            self._f.flush()
            os.fsync(self._f.fileno())
        self._file_end = valid_end
        self._num_transitions = sum(r[2] for r in self._records)

    def append_episode(self, episode: Dict[str, np.ndarray]) -> None:
        payload, n = self._serialize(episode)
        if n <= 0:
            return
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        hdr = struct.pack(HEADER_FMT, MAGIC, len(payload), n, crc)
        self._f.seek(self._file_end)
        self._f.write(hdr)
        self._f.write(payload)
        self._f.flush()
        self._appends_since_sync += 1
        if self.fsync_every > 0 and self._appends_since_sync >= self.fsync_every:
            os.fsync(self._f.fileno())
            self._appends_since_sync = 0
        poff = self._file_end + HEADER_SIZE
        self._records.append((poff, len(payload), n, crc))
        self._file_end = poff + len(payload)
        self._num_transitions += n

    def _read_payload(self, poff: int, plen: int) -> Dict[str, np.ndarray]:
        self._f.seek(poff)
        return self._deserialize(self._f.read(plen))

    def iter_episodes(self) -> Iterator[Dict[str, np.ndarray]]:
        for (poff, plen, n, crc) in self._records:
            yield self._read_payload(poff, plen)

    def read_tail_to_fill(self, capacity: int) -> Iterator[Dict[str, np.ndarray]]:
        cap = int(capacity)
        chosen = []
        total = 0
        for rec in reversed(self._records):
            chosen.append(rec)
            total += rec[2]
            if total >= cap:
                break
        chosen.reverse()
        for (poff, plen, n, crc) in chosen:
            yield self._read_payload(poff, plen)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def num_transitions(self) -> int:
        return self._num_transitions

    def close(self) -> None:
        if self._f is not None:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()
            self._f = None


def migrate_npz_dir(old_dir, out_file) -> int:
    old_dir = Path(old_dir)
    files = sorted(old_dir.glob("*.npz"))
    store = TrajectoryStore(out_file)
    try:
        for fn in files:
            with np.load(fn) as ep:
                store.append_episode({k: ep[k] for k in ep.files})
    finally:
        store.close()
    return len(files)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Trajectory store utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate", help="consolidate a dir of .npz episodes into one .traj")
    m.add_argument("old_dir")
    m.add_argument("out_file")
    args = ap.parse_args()
    if args.cmd == "migrate":
        count = migrate_npz_dir(args.old_dir, args.out_file)
        print(f"migrated {count} episodes -> {args.out_file}")
