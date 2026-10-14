"""Run the official four-method comparison from the repository directory."""
import argparse
from pathlib import Path

from olympic_actions.config import Config
from olympic_actions.data import official_manifest
from olympic_actions.experiment import run_experiment
from olympic_actions.features import extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download all official data first")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--features-output", type=Path, default=Path("runs/features_k10"))
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.download:
        from concurrent.futures import ThreadPoolExecutor
        import json
        from download_dataset import CLASSES, fetch
        records = [fetch("train_test_split", args.data)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            records.extend(pool.map(lambda name: fetch(name, args.data), CLASSES))
        (args.data / "download_provenance.json").write_text(json.dumps(records, indent=2))
    clips = official_manifest(args.data / "videos", args.data / "splits", args.data / "manifest.csv")
    config = Config.load(args.config)
    features = extract(clips, config, args.features_output, Path(".cache/features"))
    run_experiment(features, args.output, config, args.evaluate_test)


if __name__ == "__main__":
    main()
