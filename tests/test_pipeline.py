import json
from pathlib import Path
import struct

import cv2
import numpy as np
import pytest

from olympic_actions.config import Config
from olympic_actions.data import Clip, SeqReader, load_manifest, read_sampled_pairs, sample_indices, write_csv
from olympic_actions.experiment import decision_scores, metrics, run_experiment, select_c, validation_indices
from olympic_actions.features import flow_feature, flow_histogram, hog_feature, l2


def make_seq(path, compressed=True, padding=8, count=7):
    height, width = 32, 48
    header = bytearray(1024)
    struct.pack_into("<I", header, 0, 0xFEED)
    header[4:24] = "Norpix seq".encode("utf-16-le")
    struct.pack_into("<II", header, 28, 3, 1024)
    image_bytes = height * width * 3
    stride = image_bytes + 8
    struct.pack_into("<9I", header, 548, width, height, 24, 8, image_bytes,
                     102 if compressed else 100, count, 0, stride)
    struct.pack_into("<d", header, 584, 30.0)
    expected = []
    with path.open("wb") as output:
        output.write(header)
        for i in range(count):
            frame = np.full((height, width, 3), (i * 20, 10, 200), dtype=np.uint8)
            expected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if compressed:
                ok, encoded = cv2.imencode(".jpg", frame)
                assert ok
                output.write(struct.pack("<I", len(encoded) + 4))
                output.write(encoded.tobytes())
                output.write(struct.pack("<IHH", i, 0, 0) + b"\0" * (padding - 8))
            else:
                output.write(frame.tobytes())
                output.write(struct.pack("<IHH", i, 0, 0))
    return expected


@pytest.mark.parametrize("count,samples", [(2, 10), (4, 10), (11, 10), (100, 10), (100, 5)])
def test_sampling_uses_unique_valid_original_neighbours(count, samples):
    indices = sample_indices(count, samples)
    assert len(indices) == min(samples, count - 1)
    assert len(set(indices)) == len(indices)
    assert min(indices) == 0 and max(indices) + 1 < count


@pytest.mark.parametrize("compressed,padding", [(True, 8), (True, 16), (False, 8)])
def test_seq_layout_and_adjacent_sampling(tmp_path, compressed, padding):
    path = tmp_path / "example.seq"
    expected = make_seq(path, compressed, padding)
    pairs, info = read_sampled_pairs(path, 3)
    assert info["sample_indices"] == [0, 2, 5]
    assert info["frame_count"] == 7
    for (first, second), index in zip(pairs, info["sample_indices"]):
        np.testing.assert_allclose(first, expected[index], atol=3)
        np.testing.assert_allclose(second, expected[index + 1], atol=3)


def test_seq_rejects_truncated_stream(tmp_path):
    path = tmp_path / "truncated.seq"
    make_seq(path)
    path.write_bytes(path.read_bytes()[:-30])
    with pytest.raises(ValueError, match="Invalid SEQ frame length"):
        SeqReader(path)


def test_short_clip_rejected(tmp_path):
    path = tmp_path / "one.seq"
    make_seq(path, count=1)
    with pytest.raises(ValueError, match="two"):
        read_sampled_pairs(path, 10)


def test_flow_votes_for_rightward_motion_and_weights_magnitude():
    flow = np.zeros((8, 8, 2), np.float32)
    flow[:, :, 0] = 2
    histogram = flow_histogram(flow).reshape(4, 8)
    np.testing.assert_allclose(histogram[:, 0], 32)
    assert np.count_nonzero(histogram[:, 1:]) == 0
    flow[:, :, 0] = 0
    flow[:, :, 1] = -1
    histogram = flow_histogram(flow).reshape(4, 8)
    assert np.count_nonzero(histogram) == 4
    assert np.all(histogram[:, 6] > 0)


def test_zero_motion_and_hog_are_finite():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    config = Config()
    flow = flow_feature([(frame, frame)], config)
    hog = hog_feature([(frame, frame)], config)
    assert flow.shape == (32,) and hog.shape == (6084,)
    assert np.isfinite(flow).all() and np.isfinite(hog).all()
    assert not np.any(flow)
    np.testing.assert_array_equal(l2(np.zeros(5)), np.zeros(5))


def test_manifest_blocks_duplicate_ids_paths_and_partition_overlap(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, [dict(clip_id="a", path="x.seq", label="high_jump", split="train"),
                         dict(clip_id="b", path="x.seq", label="high_jump", split="test")])
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(manifest)


def synthetic_features():
    classes = ("basketball_layup", "bowling", "clean_and_jerk")
    labels = np.repeat(classes, 15)
    splits = np.tile(["train"] * 12 + ["test"] * 3, 3)
    rng = np.random.default_rng(42)
    features = np.eye(3)[np.repeat(np.arange(3), 15)] + rng.normal(0, 0.04, (45, 3))
    features = np.stack([l2(row) for row in features])
    return classes, labels, splits, features


def test_test_labels_cannot_change_validation_split_or_selected_c():
    classes, labels, splits, features = synthetic_features()
    config = Config()
    fit, validation, _, test = validation_indices(labels, splits, config, classes)
    assert not set(fit) & set(validation)
    assert not (set(fit) | set(validation)) & set(test)
    first, _ = select_c(features, labels, fit, validation, config, classes)
    altered = labels.copy()
    altered[test] = "anything"
    fit2, validation2, _, _ = validation_indices(altered, splits, config, classes)
    second, _ = select_c(features, altered, fit2, validation2, config, classes)
    np.testing.assert_array_equal(fit, fit2)
    np.testing.assert_array_equal(validation, validation2)
    assert first == second == 0.01  # equal validation mAP => smallest C


def test_average_precision_uses_scores_and_class_column_order():
    classes = ("bowling", "high_jump", "long_jump")
    labels = np.array(["bowling", "high_jump", "long_jump"] * 2)
    scores = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    result, predicted = metrics(labels, scores, classes)
    assert result["mAP"] == result["accuracy"] == result["macro_f1"] == 1
    np.testing.assert_array_equal(predicted, labels)
    class Model:
        classes_ = np.array(classes[::-1])
        def decision_function(self, x):
            return scores[:, ::-1]
    np.testing.assert_array_equal(decision_scores(Model(), scores, classes), scores)


def test_end_to_end_classifiers_export_expected_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    classes, labels, splits, features = synthetic_features()
    config = Config()
    bundle = tmp_path / "features.npz"
    np.savez(bundle, hog=features, resnet=features, flow=features, labels=labels, splits=splits,
             clip_ids=np.array([f"clip_{i}" for i in range(len(labels))]), times=np.ones((len(labels), 4)))
    extraction = dict(config=json.loads(json.dumps(config.to_dict())))
    (tmp_path / "extraction.json").write_text(json.dumps(extraction))
    summary = run_experiment(bundle, tmp_path / "result", config, True, classes)
    assert len(summary) == 4 and all(row["mAP"] == 1 for row in summary)
    assert summary[-1]["extraction_seconds_per_clip"] == 2
    for name in ("summary.csv", "predictions.csv", "partitions.csv", "error_review.csv",
                 "research_questions.json", "per_class_ap.csv", "resnet_confusion.png", "resnet.joblib"):
        assert (tmp_path / "result" / name).exists()


def test_validation_mode_never_exports_test_predictions(tmp_path, monkeypatch):
    classes, labels, splits, features = synthetic_features()
    config = Config()
    bundle = tmp_path / "features.npz"
    np.savez(bundle, hog=features, resnet=features, flow=features, labels=labels, splits=splits,
             clip_ids=np.array([f"clip_{i}" for i in range(len(labels))]), times=np.ones((len(labels), 4)))
    (tmp_path / "extraction.json").write_text(json.dumps(dict(config=json.loads(json.dumps(config.to_dict())))))
    run_experiment(bundle, tmp_path / "result", config, False, classes)
    assert not (tmp_path / "result" / "predictions.csv").exists()
    assert not json.loads((tmp_path / "result" / "protocol.json").read_text())["test_evaluated"]
