"""Download the official Stanford release; retain archives for reproducibility."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

BASE = "http://vision.stanford.edu/Datasets/OlympicSports/"
CLASSES = (
    "basketball_layup", "bowling", "clean_and_jerk", "discus_throw",
    "hammer_throw", "high_jump", "javelin_throw", "long_jump",
    "diving_platform_10m", "pole_vault", "shot_put", "snatch",
    "diving_springboard_3m", "tennis_serve", "triple_jump", "vault",
)


def fetch(name, root):
    archive = root / "downloads" / (name + ".zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        print(f"Downloading {BASE}{name}.zip", flush=True)
        with urllib.request.urlopen(BASE + name + ".zip", timeout=120) as response:
            with partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        partial.replace(archive)
    destination = root / ("splits" if name == "train_test_split" else "videos")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive entry: {member.filename}")
        zipped.extractall(destination)
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    print(f"Ready: {name}", flush=True)
    return {"archive": archive.name, "url": BASE + name + ".zip",
            "bytes": archive.stat().st_size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--classes", nargs="+", choices=CLASSES, default=list(CLASSES))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    records = [fetch("train_test_split", args.root)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records.extend(pool.map(lambda name: fetch(name, args.root), args.classes))
    (args.root / "download_provenance.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
