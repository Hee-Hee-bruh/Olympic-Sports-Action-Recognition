"""Estimate test-score uncertainty from saved predictions; no model is retrained.

Run from the repository: python scripts/bootstrap_results.py
Resample clips within each true class and use identical samples for all methods.
The intervals are conditional on the fitted models and observed class counts.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from olympic_actions.data import CLASSES, file_sha256, write_csv

METHODS = ("hog", "resnet", "flow", "resnet_flow")
REPEATS, SEED = 2000, 42


def evaluate(labels, scores):
    ap = [average_precision_score(labels == c, scores[:, j]) for j, c in enumerate(CLASSES)]
    predicted = np.asarray(CLASSES)[scores.argmax(axis=1)]
    return np.array([np.mean(ap), accuracy_score(labels, predicted)])


def main():
    baseline = ROOT / "runs/baseline"
    output = ROOT / "runs/uncertainty"
    output.mkdir(parents=True, exist_ok=False)
    hashes = {p.name: file_sha256(p) for p in baseline.iterdir() if p.is_file()}
    with (baseline / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    grouped = {m: sorted([r for r in rows if r["method"] == m], key=lambda r: r["clip_id"])
               for m in METHODS}
    ids = [r["clip_id"] for r in grouped["hog"]]
    labels = np.array([r["true_label"] for r in grouped["hog"]])
    assert len(ids) == len(set(ids)) == 134 and set(labels) == set(CLASSES)
    scores = {}
    for method in METHODS:
        assert ids == [r["clip_id"] for r in grouped[method]]
        assert labels.tolist() == [r["true_label"] for r in grouped[method]]
        scores[method] = np.array([[float(r[c]) for c in CLASSES] for r in grouped[method]])
        assert np.isfinite(scores[method]).all()
    point = np.stack([evaluate(labels, scores[m]) for m in METHODS])
    for i, method in enumerate(METHODS):
        original = json.loads((baseline / f"{method}_metrics.json").read_text())
        assert np.allclose(point[i], [original["mAP"], original["accuracy"]], atol=1e-12)
    rng = np.random.default_rng(SEED)
    strata = [np.flatnonzero(labels == c) for c in CLASSES]
    samples = np.stack([np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
                        for _ in range(REPEATS)])
    results = np.empty((REPEATS, len(METHODS), 2))
    for repeat, selected in enumerate(samples):
        for i, method in enumerate(METHODS):
            results[repeat, i] = evaluate(labels[selected], scores[method][selected])
        if (repeat + 1) % 200 == 0:
            print(f"Bootstrap {repeat + 1}/{REPEATS}", flush=True)
    np.savez_compressed(output / "bootstrap_samples.npz", indices=samples, statistics=results,
                        clip_ids=np.array(ids), methods=np.array(METHODS))
    intervals = np.quantile(results, [0.025, 0.975], axis=0)
    summaries = []
    for i, method in enumerate(METHODS):
        summaries.append(dict(method=method, mAP=point[i, 0], mAP_low=intervals[0, i, 0],
                              mAP_high=intervals[1, i, 0], accuracy=point[i, 1],
                              accuracy_low=intervals[0, i, 1], accuracy_high=intervals[1, i, 1]))
    comparisons = []
    for name, left, right in [("D minus B", 3, 1), ("B minus A", 1, 0)]:
        differences = results[:, left] - results[:, right]
        low, high = np.quantile(differences, [0.025, 0.975], axis=0)
        delta = point[left] - point[right]
        comparisons.append(dict(comparison=name, delta_mAP=delta[0], mAP_low=low[0],
                                mAP_high=high[0], delta_accuracy=delta[1],
                                accuracy_low=low[1], accuracy_high=high[1]))
    assert hashes == {p.name: file_sha256(p) for p in baseline.iterdir() if p.is_file()}
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "comparisons.csv", comparisons)
    (output / "protocol.json").write_text(json.dumps(dict(
        repeats=REPEATS, seed=SEED, confidence=0.95, method="percentile",
        resampling="Within each true class, keep its observed test count; paired across methods",
        source="runs/baseline/predictions.csv", source_sha256=hashes["predictions.csv"],
        retraining=False, new_test_predictions=False, baseline_artifacts_unchanged=True,
        limitations="Clip-level intervals with fixed models and class counts. They exclude model "
                    "selection/training variability and do not account for shared video sources. "
                    "Small classes and source dependence limit interpretation."), indent=2))
    for row in summaries + comparisons:
        print(row, flush=True)


if __name__ == "__main__":
    main()
