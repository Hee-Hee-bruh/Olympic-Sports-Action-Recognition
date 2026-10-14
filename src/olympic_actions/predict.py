import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from .config import Config
from .data import read_sampled_pairs
from .experiment import decision_scores
from .features import ResNetExtractor, flow_feature, hog_feature


def predict(video, model_file, cache_root):
    # Models must have been generated locally by this pipeline: joblib uses pickle.
    saved = joblib.load(model_file)
    config = Config(**saved["config"])
    pairs, info = read_sampled_pairs(video, config.samples)
    method = saved["method"]
    extractor = ResNetExtractor(config, cache_root) if method in {"resnet", "resnet_flow"} else None
    started = perf_counter()
    if method == "hog":
        feature = hog_feature(pairs, config)
    elif method == "flow":
        feature = flow_feature(pairs, config)
    elif method == "resnet":
        feature = extractor(pairs)
    elif method == "resnet_flow":
        feature = np.r_[extractor(pairs), flow_feature(pairs, config)]
    else:
        raise ValueError(f"Unknown model method: {method}")
    scores = decision_scores(saved["classifier"], feature[None], saved["classes"])[0]
    return {"video": str(Path(video).resolve()), "method": method,
            "predicted_label": saved["classes"][int(np.argmax(scores))],
            "decision_scores": dict(zip(saved["classes"], scores.tolist())),
            "extraction_and_svm_seconds": perf_counter() - started, **info}
