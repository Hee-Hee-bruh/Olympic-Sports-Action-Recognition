"""Compare K=5/10 on the original training-only split, with fresh paired timings.

Run from the repository: python scripts/run_sampling_ablation.py
Official test videos are never opened and baseline artifacts are never replaced.
"""
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from olympic_actions.config import Config
from olympic_actions.data import CLASSES, file_sha256, load_manifest, read_sampled_pairs, write_csv
from olympic_actions.experiment import decision_scores, fit_classifier, metrics, select_c, validation_indices
from olympic_actions.features import ResNetExtractor, environment, flow_feature


def main():
    output = ROOT / "runs/sampling_ablation"
    # Refuse to mix partial/repeated timings with a completed experiment.
    output.mkdir(parents=True, exist_ok=False)
    config = Config.load(ROOT / "configs/baseline.json")
    clips = [c for c in load_manifest(ROOT / "data/manifest.csv") if c.split == "train"]
    ids = np.array([c.clip_id for c in clips])
    labels = np.array([c.label for c in clips])
    fit, val, _, test = validation_indices(labels, np.array([c.split for c in clips]), config)
    assert len(clips) == 649 and len(fit) == 519 and len(val) == 130 and not len(test)
    with (ROOT / "runs/baseline/partitions.csv").open() as stream:
        original = {r["clip_id"]: r["selection_subset"] for r in csv.DictReader(stream)}
    subsets = ["validation" if i in set(val) else "fit" for i in range(len(clips))]
    assert all(original[clip_id] == subset for clip_id, subset in zip(ids, subsets))
    baseline_files = sorted(p for p in (ROOT / "runs/baseline").iterdir() if p.is_file())
    baseline_hashes = {p.name: file_sha256(p) for p in baseline_files}
    write_csv(output / "partitions.csv", [dict(clip_id=c.clip_id, label=c.label,
              official_split="train", selection_subset=s) for c, s in zip(clips, subsets)])
    cv2.setNumThreads(1)
    # Reuse frozen weights, but compute every descriptor afresh for this study.
    extractor = ResNetExtractor(config, ROOT / ".cache/features")
    vectors = {k: {"resnet": [], "flow": [], "times": []} for k in (5, 10)}
    audit = []
    for i, clip in enumerate(clips):
        digest = file_sha256(clip.path)
        # Alternate processing order to reduce systematic order/thermal bias.
        for k in ((5, 10) if i % 2 == 0 else (10, 5)):
            settings = replace(config, samples=k)
            pairs, info = read_sampled_pairs(clip.path, k)
            extractor.sync()
            started = perf_counter()
            appearance = extractor(pairs)
            appearance_seconds = perf_counter() - started
            started = perf_counter()
            motion = flow_feature(pairs, settings)
            motion_seconds = perf_counter() - started
            assert np.isfinite(appearance).all() and np.isfinite(motion).all()
            vectors[k]["resnet"].append(appearance)
            vectors[k]["flow"].append(motion)
            vectors[k]["times"].append([appearance_seconds, motion_seconds, info["decode_seconds"]])
            audit.append(dict(clip_id=clip.clip_id, subset=subsets[i], K=k, sha256=digest,
                              sampled_indices=json.dumps(info["sample_indices"]),
                              actual_pairs=len(pairs), resnet_seconds=appearance_seconds,
                              flow_seconds=motion_seconds, decode_seconds=info["decode_seconds"]))
        if (i + 1) % 25 == 0 or i + 1 == len(clips):
            print(f"Paired extraction {i + 1}/{len(clips)} training clips", flush=True)
    write_csv(output / "extraction_audit.csv", audit)
    rows, validation_results, prediction_rows = [], {}, []
    for k, values in vectors.items():
        values = {name: np.asarray(array) for name, array in values.items()}
        np.savez_compressed(output / f"features_k{k}.npz", **values, clip_ids=ids, labels=labels)
        settings = replace(config, samples=k)
        validation_results[str(k)] = {}
        for method in ("resnet", "resnet_flow"):
            features = values["resnet"] if method == "resnet" else np.concatenate(
                [values["resnet"], values["flow"]], axis=1)
            chosen_c, candidates = select_c(features, labels, fit, val, settings, CLASSES)
            model, _ = fit_classifier(features[fit], labels[fit], chosen_c, settings)
            scores = decision_scores(model, features[val], CLASSES)
            result, predicted = metrics(labels[val], scores, CLASSES)
            timing = values["times"][val, 0].copy()
            if method == "resnet_flow":
                timing += values["times"][val, 1]
            rows.append(dict(K=k, method=method, selected_C=chosen_c, validation_clips=len(val),
                             mAP=result["mAP"], accuracy=result["accuracy"], macro_f1=result["macro_f1"],
                             extraction_ms_per_clip=float(timing.mean() * 1000),
                             extraction_median_ms=float(np.median(timing) * 1000)))
            validation_results[str(k)][method] = dict(selected_C=chosen_c, candidates=candidates,
                                                     metrics=result)
            for j, index in enumerate(val):
                prediction_rows.append(dict(K=k, method=method, clip_id=ids[index],
                                            true_label=labels[index], prediction=predicted[j],
                                            **{f"score_{c}": float(scores[j, n]) for n, c in enumerate(CLASSES)}))
    # K=10 must reproduce the original validation mAP before interpreting K=5.
    baseline = json.loads((ROOT / "runs/baseline/validation.json").read_text())
    for row in rows:
        if row["K"] == 10:
            previous = baseline[row["method"]]
            expected = next(c["mAP"] for c in previous["candidates"] if c["C"] == previous["selected_C"])
            assert row["selected_C"] == previous["selected_C"]
            assert abs(row["mAP"] - expected) < 1e-10
    assert baseline_hashes == {p.name: file_sha256(p) for p in baseline_files}
    write_csv(output / "summary.csv", rows)
    write_csv(output / "validation_predictions.csv", prediction_rows)
    (output / "validation.json").write_text(json.dumps(validation_results, indent=2))
    (output / "protocol.json").write_text(json.dumps(dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(), config=config.to_dict(),
        samples_compared=[5, 10], versions=environment(), platform=platform.platform(),
        processor=platform.processor(), python=platform.python_version(), cpu_threads=1,
        fit_clips=len(fit), validation_clips=len(val), test_clips_opened=0,
        timing_note="Fresh descriptors; alternating K order per clip. Mean on 130 validation clips; "
                    "includes resize/normalisation/pooling, excludes decoding/model setup/warmup. "
                    "D sums B and flow, excluding concatenation. One pass per K, no timing intervals.",
        study_note="Supplementary analysis performed after the original test results were available. "
                   "No new test evaluation or replacement of the original K=10 results.",
        baseline_artifacts_unchanged=True, baseline_hashes=baseline_hashes), indent=2))
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
