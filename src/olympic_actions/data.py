"""Official split parsing and Norpix SEQ/video readers (RGB, zero-based indices)."""
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from time import perf_counter

import cv2
import numpy as np

CLASSES = (
    "basketball_layup", "bowling", "clean_and_jerk", "discus_throw",
    "hammer_throw", "high_jump", "javelin_throw", "long_jump",
    "diving_platform_10m", "pole_vault", "shot_put", "snatch",
    "diving_springboard_3m", "tennis_serve", "triple_jump", "vault",
)
ALIASES = {"platform_10m": "diving_platform_10m", "springboard_3m": "diving_springboard_3m",
           "diving_platfrom_10m": "diving_platform_10m",  # Typo in Stanford's archive folder.
           "platform_diving": "diving_platform_10m", "springboard_diving": "diving_springboard_3m",
           "gymnastic_vault": "vault"}
VIDEO_SUFFIXES = {".seq", ".avi", ".mp4", ".mov", ".mkv"}


def canonical_class(name):
    label = name.strip().lower().replace(" ", "_").replace("-", "_")
    return ALIASES.get(label, label)


@dataclass(frozen=True)
class Clip:
    clip_id: str
    path: Path
    label: str
    split: str


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def official_manifest(video_root, split_root, output):
    """Use the provided per-class text lists; never infer/randomise test membership."""
    index = {}
    for path in sorted(Path(video_root).rglob("*")):
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        label = canonical_class(path.parent.name)
        key = (label, path.stem)
        if key in index:
            raise ValueError(f"Ambiguous video ID {key}: {index[key]} and {path}")
        index[key] = path.resolve()
    rows = []
    for split in ("train", "test"):
        lists = sorted((Path(split_root) / split).glob("*.txt"))
        if not lists:
            raise ValueError(f"No official {split} lists under {split_root}")
        for listing in lists:
            label = canonical_class(listing.stem)
            if label not in CLASSES:
                raise ValueError(f"Unknown sport in {listing}")
            for line in listing.read_text().splitlines():
                stem = Path(line.strip()).stem
                if not stem:
                    continue
                path = index.get((label, stem), Path(video_root).resolve() / label / (stem + ".seq"))
                rows.append(dict(clip_id=f"{label}/{stem}", path=str(path), label=label, split=split))
    write_csv(output, rows)
    return load_manifest(output)


def load_manifest(path):
    clips, ids, paths = [], set(), set()
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"clip_id", "path", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires clip_id,path,label,split")
        for row in reader:
            video = Path(row["path"])
            if not video.is_absolute():
                video = Path(path).resolve().parent / video
            video = video.resolve()
            label = canonical_class(row["label"])
            if row["split"] not in {"train", "test"} or label not in CLASSES:
                raise ValueError(f"Invalid label or partition: {row}")
            if not row["clip_id"] or row["clip_id"] in ids or video in paths:
                raise ValueError(f"Duplicate ID/path in manifest: {row}")
            ids.add(row["clip_id"])
            paths.add(video)
            clips.append(Clip(row["clip_id"], video, label, row["split"]))
    if not clips:
        raise ValueError("Manifest is empty")
    return clips


def sample_indices(frame_count, samples):
    if frame_count < 2:
        raise ValueError("At least two decoded frames are required")
    if samples < 1:
        raise ValueError("samples must be positive")
    return np.unique(np.linspace(0, frame_count - 2, min(samples, frame_count - 1), dtype=int))


class SeqReader:
    """Read 8-bit raw/JPEG/PNG SEQ, using the layout documented by Piotr's toolbox.

    Compressed frame length includes its 4-byte prefix. Timestamp padding is
    either 8 or 16 bytes. Unsupported Bayer/16-bit streams fail explicitly.
    """
    def __init__(self, path):
        self.file = Path(path).open("rb")
        try:
            header = self.file.read(1024)
            if len(header) != 1024 or struct.unpack_from("<I", header)[0] != 0xFEED:
                raise ValueError("Invalid Norpix SEQ header")
            if header[4:24].decode("utf-16-le") != "Norpix seq":
                raise ValueError("Invalid SEQ signature")
            version, header_size = struct.unpack_from("<II", header, 28)
            if header_size != 1024 or version > 4:
                raise ValueError(f"Unsupported SEQ header/version: {header_size}/{version}")
            (self.width, self.height, self.bits, real_bits, self.image_bytes,
             self.format, self.frame_count, _, self.stride) = struct.unpack_from("<9I", header, 548)
            self.fps = struct.unpack_from("<d", header, 584)[0]
            if real_bits != 8 or self.bits not in (8, 24) or min(self.width, self.height, self.frame_count) < 1:
                raise ValueError("Unsupported SEQ dimensions/bit depth/frame count")
            self.file.seek(0, 2)
            size = self.file.tell()
            self.offsets = []
            if self.format in (100, 200):
                if self.stride < self.image_bytes or size < 1024 + self.stride * (self.frame_count - 1) + self.image_bytes:
                    raise ValueError("Truncated raw SEQ")
                self.offsets = [1024 + i * self.stride for i in range(self.frame_count)]
            elif self.format in (1, 2, 102, 201):
                offset, extra = 1024, 8
                for i in range(self.frame_count):
                    self.offsets.append(offset)
                    self.file.seek(offset)
                    prefix = self.file.read(4)
                    if len(prefix) != 4:
                        raise ValueError(f"Truncated SEQ frame {i}")
                    length = struct.unpack("<I", prefix)[0]
                    if length < 5 or offset + length + 8 > size:
                        raise ValueError(f"Invalid SEQ frame length at {i}")
                    next_offset = offset + length + extra
                    if i == 0 and self.frame_count > 1:
                        self.file.seek(next_offset)
                        if self.file.read(4) == b"\x00\x00\x00\x00":
                            extra = 16
                            next_offset += 8
                    offset = next_offset
            else:
                raise ValueError(f"Unsupported SEQ image format {self.format}")
        except Exception:
            self.file.close()
            raise

    def read(self, index):
        self.file.seek(self.offsets[index])
        if self.format in (100, 200):
            raw = self.file.read(self.image_bytes)
            channels = self.bits // 8
            if len(raw) != self.width * self.height * channels:
                raise ValueError("Unsupported raw SEQ row padding or truncated frame")
            frame = np.frombuffer(raw, np.uint8).reshape(self.height, self.width, channels)
            if channels == 1:
                return np.repeat(frame, 3, axis=2)
            return frame[:, :, ::-1].copy()
        length = struct.unpack("<I", self.file.read(4))[0]
        frame = cv2.imdecode(np.frombuffer(self.file.read(length - 4), np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"Cannot decode SEQ frame {index}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self.file.close()


def read_sampled_pairs(path, samples):
    """Decode original neighbours, never pairs of distant sampled frames."""
    started = perf_counter()
    if Path(path).suffix.lower() == ".seq":
        reader = SeqReader(path)
        try:
            indices = sample_indices(reader.frame_count, samples)
            frames = {int(i): reader.read(int(i)) for i in np.unique(np.r_[indices, indices + 1])}
            info = dict(frame_count=reader.frame_count, width=reader.width,
                        height=reader.height, fps=reader.fps)
        finally:
            reader.close()
    else:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open {path}")
            # Count actual decoded frames, rather than trusting container metadata.
            count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                count += 1
            indices = sample_indices(count, samples)
            needed = set(map(int, np.r_[indices, indices + 1]))
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = {}
            for i in range(count):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Decode failure at frame {i}")
                if i in needed:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            info = dict(frame_count=count, width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
    info["sample_indices"] = indices.tolist()
    info["decode_seconds"] = perf_counter() - started
    return [(frames[int(i)], frames[int(i + 1)]) for i in indices], info


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
