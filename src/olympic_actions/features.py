"""Four representations, with separate cold extraction and decode timings."""
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from skimage.feature import hog

from .data import file_sha256, read_sampled_pairs, write_csv


def environment():
    return {name: version(name) for name in ("numpy", "opencv-python-headless", "scikit-image",
                                            "scikit-learn", "torch", "torchvision", "matplotlib")}


def l2(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def grayscale(frame, size):
    return cv2.cvtColor(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY)


def hog_feature(pairs, config):
    vectors = [hog(grayscale(first, config.image_size), orientations=config.hog_orientations,
                   pixels_per_cell=(config.hog_cell_size,) * 2,
                   cells_per_block=(config.hog_block_cells,) * 2,
                   block_norm="L2-Hys", feature_vector=True) for first, _ in pairs]
    return l2(np.mean(vectors, axis=0))


def flow_histogram(flow, grid=2, bins=8):
    magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    ys = np.linspace(0, flow.shape[0], grid + 1, dtype=int)
    xs = np.linspace(0, flow.shape[1], grid + 1, dtype=int)
    descriptor = []
    for row in range(grid):
        for column in range(grid):
            cell = np.s_[ys[row]:ys[row + 1], xs[column]:xs[column + 1]]
            # modulo protects the 2*pi boundary, which belongs to the first bin.
            hist, _ = np.histogram(angle[cell] % (2 * np.pi), bins=bins,
                                   range=(0, 2 * np.pi), weights=magnitude[cell])
            descriptor.extend(hist)
    return np.asarray(descriptor, dtype=np.float32)


def flow_feature(pairs, config):
    vectors = []
    for first, second in pairs:
        flow = cv2.calcOpticalFlowFarneback(
            grayscale(first, config.image_size), grayscale(second, config.image_size), None,
            config.flow_pyr_scale, config.flow_levels, config.flow_winsize,
            config.flow_iterations, config.flow_poly_n, config.flow_poly_sigma, 0)
        vectors.append(flow_histogram(flow, config.flow_grid, config.flow_bins))
    return l2(np.mean(vectors, axis=0))


class ResNetExtractor:
    def __init__(self, config, cache_root):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
        self.torch, self.config = torch, config
        os.environ.setdefault("TORCH_HOME", str(Path(cache_root).resolve() / "torch"))
        if config.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be cpu, cuda or mps")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if config.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable")
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.device = torch.device(config.device)
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.model.requires_grad_(False)
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        # Exclude model setup and one-off kernel warmup from per-clip timing.
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, config.image_size, config.image_size, device=self.device))
        self.sync()

    def sync(self):
        if self.config.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.config.device == "mps":
            self.torch.mps.synchronize()

    def __call__(self, pairs):
        torch, config = self.torch, self.config
        frames = np.stack([cv2.resize(first, (config.image_size, config.image_size),
                                     interpolation=cv2.INTER_LINEAR) for first, _ in pairs])
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255
        tensor = (tensor - self.mean) / self.std
        with torch.inference_mode():
            vectors = [self.model(batch).cpu().numpy() for batch in tensor.split(config.resnet_batch_size)]
        self.sync()
        return l2(np.concatenate(vectors).mean(axis=0))


def extract(clips, config, output, cache_root):
    output, cache_root = Path(output), Path(cache_root)
    output.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    settings = {"config": config.to_dict(), "versions": environment(), "feature_schema": 1}
    signature = json.dumps(settings, sort_keys=True)
    log, usable, arrays = [], [], {"hog": [], "resnet": [], "flow": []}
    times, seen_hashes, extractor = [], {}, None
    for number, clip in enumerate(clips, 1):
        row = dict(clip_id=clip.clip_id, label=clip.label, split=clip.split, path=str(clip.path))
        if not clip.path.exists():
            log.append({**row, "status": "missing", "error": "File not found"})
            continue
        digest = file_sha256(clip.path)
        if digest in seen_hashes:
            previous = seen_hashes[digest]
            # Exclude both members, irrespective of partition/order.
            previous["status"] = "duplicate"
            previous["error"] = f"Exact duplicate of {clip.clip_id}"
            log.append({**row, "status": "duplicate", "error": f"Exact duplicate of {previous['clip_id']}"})
            continue
        key = hashlib.sha256((digest + signature).encode()).hexdigest()
        cached = cache_root / (key + ".npz")
        try:
            if cached.exists():
                with np.load(cached, allow_pickle=False) as saved:
                    vectors = {name: saved[name].copy() for name in arrays}
                    info = json.loads(str(saved["info"]))
            else:
                pairs, info = read_sampled_pairs(clip.path, config.samples)
        except (ValueError, OSError, cv2.error) as error:
            log.append({**row, "status": "decode_failed", "error": str(error)})
            continue
        cache_hit = cached.exists()
        if not cache_hit:
            if extractor is None:
                extractor = ResNetExtractor(config, cache_root)
            vectors = {}
            for name, function in (("hog", hog_feature), ("flow", flow_feature)):
                start = perf_counter()
                vectors[name] = function(pairs, config)
                info[name + "_seconds"] = perf_counter() - start
            extractor.sync()
            start = perf_counter()
            vectors["resnet"] = extractor(pairs)
            info["resnet_seconds"] = perf_counter() - start
            np.savez_compressed(cached, **vectors, info=json.dumps(info))
        if any(not np.isfinite(vector).all() for vector in vectors.values()):
            raise ValueError(f"Non-finite descriptor: {clip.clip_id}")
        audited = {**row, **info, "sha256": digest, "cache_hit": cache_hit, "status": "ok", "error": ""}
        seen_hashes[digest] = audited
        log.append(audited)
        usable.append(clip)
        for name in arrays:
            arrays[name].append(vectors[name])
        times.append([info[name + "_seconds"] for name in ("hog", "resnet", "flow")] + [info["decode_seconds"]])
        print(f"[{number}/{len(clips)}] {clip.clip_id} ({'cached' if cache_hit else 'extracted'})", flush=True)
    # Exact duplicates found later must be removed from every configuration.
    allowed = {row["clip_id"] for row in log if row["status"] == "ok"}
    keep = [i for i, clip in enumerate(usable) if clip.clip_id in allowed]
    fields = list(dict.fromkeys(key for row in log for key in row))
    write_csv(output / "audit.csv", log, fields)
    if not keep:
        raise ValueError(f"No usable clips. See {output / 'audit.csv'}")
    selected = [usable[i] for i in keep]
    data = {name: np.stack(values)[keep] for name, values in arrays.items()}
    data.update(clip_ids=np.array([c.clip_id for c in selected]), labels=np.array([c.label for c in selected]),
                splits=np.array([c.split for c in selected]), times=np.asarray(times)[keep])
    np.savez_compressed(output / "features.npz", **data)
    (output / "extraction.json").write_text(json.dumps({
        **settings, "source_manifest": [dict(clip_id=c.clip_id, label=c.label, split=c.split,
                                            path=str(c.path)) for c in clips],
        "usable_clips": len(selected), "excluded_clips": len(clips) - len(selected),
        "timing_columns": ["hog", "resnet", "flow", "decode"],
        "timing_note": "Cold per-clip extraction, including resizing/normalisation/pooling; excludes decode, model setup and warmup. Cache hits retain original timings.",
        "decode_note": "SEQ indexes all frame records, decodes sampled neighbours. Other formats decode sequentially twice to verify actual count.",
    }, indent=2))
    return output / "features.npz"
