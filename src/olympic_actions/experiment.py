"""Select C using training-only validation, then optionally evaluate official test."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from .data import CLASSES, write_csv

METHODS = ("hog", "resnet", "flow", "resnet_flow")


def representations(data):
    return {"hog": data["hog"], "resnet": data["resnet"], "flow": data["flow"],
            # Each block is already L2 normalised; preserve equal block weights.
            "resnet_flow": np.concatenate([data["resnet"], data["flow"]], axis=1)}


def validation_indices(labels, splits, config, classes=CLASSES):
    train = np.flatnonzero(splits == "train")
    test = np.flatnonzero(splits == "test")
    if set(labels[train]) != set(classes):
        raise ValueError("Usable training set must contain every class; check audit.csv")
    fit, validation = train_test_split(train, test_size=config.validation_fraction,
                                       random_state=config.seed, stratify=labels[train])
    if set(labels[fit]) != set(classes) or set(labels[validation]) != set(classes):
        raise ValueError("Every class must occur in both fit and validation subsets")
    return np.sort(fit), np.sort(validation), train, test


def fit_classifier(features, labels, c, config):
    model = LinearSVC(C=c, multi_class="ovr", dual="auto", max_iter=config.svm_max_iter,
                      random_state=config.seed)
    started = perf_counter()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, labels)
    return model, perf_counter() - started


def decision_scores(model, features, classes):
    scores = model.decision_function(features)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    columns = [list(model.classes_).index(label) for label in classes]
    return scores[:, columns]


def metrics(labels, scores, classes):
    if not len(labels) or set(labels) != set(classes):
        raise ValueError("AP comparison requires examples of every class")
    if scores.shape != (len(labels), len(classes)) or not np.isfinite(scores).all():
        raise ValueError("Invalid class decision scores")
    predicted = np.asarray(classes)[np.argmax(scores, axis=1)]
    ap = {label: float(average_precision_score(labels == label, scores[:, i]))
          for i, label in enumerate(classes)}
    counts = confusion_matrix(labels, predicted, labels=classes)
    normalised = counts / counts.sum(axis=1, keepdims=True)
    return dict(mAP=float(np.mean(list(ap.values()))), accuracy=float(accuracy_score(labels, predicted)),
                macro_f1=float(f1_score(labels, predicted, labels=classes, average="macro", zero_division=0)),
                per_class_ap=ap, confusion_counts=counts.tolist(),
                confusion_normalised=normalised.tolist()), predicted


def select_c(features, labels, fit, validation, config, classes):
    candidates, best_c, best_map = [], None, -1.0
    for c in sorted(set(config.svm_c)):
        model, seconds = fit_classifier(features[fit], labels[fit], c, config)
        scores = decision_scores(model, features[validation], classes)
        result, _ = metrics(labels[validation], scores, classes)
        candidates.append(dict(C=c, mAP=result["mAP"], accuracy=result["accuracy"],
                               macro_f1=result["macro_f1"], fit_seconds=seconds))
        if result["mAP"] > best_map + 1e-12:
            best_c, best_map = c, result["mAP"]
    return best_c, candidates


def save_confusion(path, matrix, classes, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(11, 9))
    picture = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(len(classes)), labels=classes, rotation=75, ha="right")
    axis.set_yticks(range(len(classes)), labels=classes)
    axis.set(xlabel="Predicted sport", ylabel="True sport", title=title)
    figure.colorbar(picture, ax=axis, label="Fraction within true class")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_experiment(feature_file, output, config, evaluate_test=False, classes=CLASSES):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(feature_file, allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    extraction_path = Path(feature_file).with_name("extraction.json")
    extraction = json.loads(extraction_path.read_text())
    if extraction["config"] != json.loads(json.dumps(config.to_dict())):
        raise ValueError("Config differs from cached feature settings; re-extract with the desired config")
    labels, splits, ids = data["labels"], data["splits"], data["clip_ids"]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate video IDs in feature bundle")
    if set(splits) - {"train", "test"}:
        raise ValueError("Invalid feature partition")
    if set(labels) - set(classes):
        raise ValueError("Unknown class in feature bundle")
    for name in ("hog", "resnet", "flow"):
        if data[name].ndim != 2 or len(data[name]) != len(ids) or not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid feature matrix: {name}")
    fit, validation, train, test = validation_indices(labels, splits, config, classes)
    if evaluate_test and (not len(test) or set(labels[test]) != set(classes)):
        raise ValueError("Official test set must contain every class; check audit.csv")
    write_csv(output / "partitions.csv", [dict(clip_id=ids[i], label=labels[i], official_split=splits[i],
              selection_subset="fit" if i in set(fit) else "validation" if i in set(validation) else "test")
              for i in range(len(ids))])
    matrix = representations(data)
    validation_results, summary, prediction_rows = {}, [], []
    all_metrics, all_predictions = {}, {}
    for name, features in matrix.items():
        chosen_c, candidates = select_c(features, labels, fit, validation, config, classes)
        validation_results[name] = {"selected_C": chosen_c, "candidates": candidates}
        print(f"{name}: selected C={chosen_c} using validation mAP", flush=True)
    # Finish ALL model selection before looking at ANY test score.
    (output / "validation.json").write_text(json.dumps(validation_results, indent=2))
    with Path(feature_file).open("rb") as source:
        feature_digest = hashlib.file_digest(source, "sha256").hexdigest()
    (output / "protocol.json").write_text(json.dumps({
        "config": config.to_dict(), "classes": list(classes), "extraction": extraction,
        "feature_sha256": feature_digest,
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "python": platform.python_version(), "device": config.device, "cpu_threads": 1},
        "class_counts": {split: dict(Counter(labels[splits == split])) for split in ("train", "test")},
        "selection_counts": {"fit": len(fit), "validation": len(validation)},
        "test_evaluated": evaluate_test,
    }, indent=2))
    if not evaluate_test:
        return validation_results
    for name, features in matrix.items():
        chosen_c = validation_results[name]["selected_C"]
        model, train_seconds = fit_classifier(features[train], labels[train], chosen_c, config)
        with threadpool_limits(limits=1):
            started = perf_counter()
            scores = decision_scores(model, features[test], classes)
            inference = (perf_counter() - started) / len(test)
        result, predicted = metrics(labels[test], scores, classes)
        all_metrics[name], all_predictions[name] = result, predicted
        joblib.dump({"classifier": model, "classes": classes, "method": name,
                     "config": config.to_dict()}, output / (name + ".joblib"))
        (output / (name + "_metrics.json")).write_text(json.dumps(result, indent=2))
        save_confusion(output / (name + "_confusion.png"), result["confusion_normalised"], classes, name)
        write_csv(output / (name + "_confusion.csv"),
                  [dict(true_class=label, **dict(zip(classes, result["confusion_normalised"][i])))
                   for i, label in enumerate(classes)])
        # All rows use the same set of usable test clips. Cold original timings
        # are preserved in caches; fusion reuses B/C and does not re-extract them.
        time_columns = {"hog": [0], "resnet": [1], "flow": [2], "resnet_flow": [1, 2]}[name]
        extraction_seconds = data["times"][test][:, time_columns].sum(axis=1)
        decode_seconds = data["times"][test, 3]
        summary.append(dict(method=name, C=chosen_c, test_clips=len(test), mAP=result["mAP"],
                            accuracy=result["accuracy"], macro_f1=result["macro_f1"],
                            extraction_seconds_per_clip=float(extraction_seconds.mean()),
                            extraction_median_seconds=float(np.median(extraction_seconds)),
                            decode_seconds_per_clip=float(decode_seconds.mean()),
                            classifier_fit_seconds=train_seconds,
                            cached_inference_seconds_per_clip=inference,
                            estimated_full_seconds_per_clip=float((extraction_seconds + decode_seconds).mean()) + inference))
        for j, i in enumerate(test):
            prediction_rows.append(dict(method=name, clip_id=ids[i], true_label=labels[i],
                                        predicted_label=predicted[j], **dict(zip(classes, scores[j].tolist()))))
    write_csv(output / "summary.csv", summary)
    write_csv(output / "predictions.csv", prediction_rows)
    write_csv(output / "per_class_ap.csv", [dict(label=label, test_count=int(np.sum(labels[test] == label)),
              **{name: all_metrics[name]["per_class_ap"][label] for name in METHODS}) for label in classes])
    majority = sorted(Counter(labels[train]).items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
    (output / "research_questions.json").write_text(json.dumps({
        "RQ1": {"comparison": "resnet_flow - resnet", "delta_mAP": all_metrics["resnet_flow"]["mAP"] - all_metrics["resnet"]["mAP"],
                "delta_accuracy": all_metrics["resnet_flow"]["accuracy"] - all_metrics["resnet"]["accuracy"]},
        "RQ2": {"comparison": "resnet - hog", "delta_mAP": all_metrics["resnet"]["mAP"] - all_metrics["hog"]["mAP"],
                "delta_extraction_seconds_per_clip": summary[1]["extraction_seconds_per_clip"] - summary[0]["extraction_seconds_per_clip"]},
        "majority_baseline": {"label_from_train": majority, "test_accuracy": float(np.mean(labels[test] == majority))},
    }, indent=2))
    # Prioritise errors on which B and D disagree, with class diversity.
    candidates = []
    for j, i in enumerate(test):
        b, d = all_predictions["resnet"][j], all_predictions["resnet_flow"][j]
        if b != labels[i] or d != labels[i]:
            candidates.append(dict(clip_id=ids[i], true_label=labels[i], resnet_prediction=b,
                                   fusion_prediction=d, models_disagree=bool(b != d), manual_observation=""))
    candidates.sort(key=lambda row: (not row["models_disagree"], row["true_label"], row["clip_id"]))
    chosen, seen = [], set()
    for row in candidates:
        if row["true_label"] not in seen and len(chosen) < 10:
            chosen.append(row)
            seen.add(row["true_label"])
    for row in candidates:
        if len(chosen) >= 10:
            break
        if row not in chosen:
            chosen.append(row)
    write_csv(output / "error_review.csv", chosen,
              ["clip_id", "true_label", "resnet_prediction", "fusion_prediction", "models_disagree", "manual_observation"])
    if extraction.get("source_manifest"):
        from .error_gallery import save_error_gallery
        save_error_gallery(chosen, extraction["source_manifest"], output, config.samples)
    return summary
