import argparse
import json
from pathlib import Path

from .config import Config
from .data import load_manifest, official_manifest


def main():
    parser = argparse.ArgumentParser(description="Olympic Sports: four fixed representations + linear SVM")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build manifest using official per-class train/test lists")
    prepare.add_argument("--videos", type=Path, default=Path("data/videos"))
    prepare.add_argument("--splits", type=Path, default=Path("data/splits"))
    prepare.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    extract = commands.add_parser("extract", help="Audit clips and cache HOG, ResNet and flow descriptors")
    extract.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    extract.add_argument("--output", type=Path, default=Path("runs/features_k10"))
    extract.add_argument("--cache", type=Path, default=Path(".cache/features"))
    extract.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    experiment = commands.add_parser("experiment", help="Select C; optionally fit full train and evaluate test")
    experiment.add_argument("--features", type=Path, default=Path("runs/features_k10/features.npz"))
    experiment.add_argument("--output", type=Path, default=Path("runs/baseline"))
    experiment.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    experiment.add_argument("--evaluate-test", action="store_true",
                            help="Evaluate official test after all validation model selection has finished")
    inference = commands.add_parser("predict", help="Classify a new clip using a locally trained model")
    inference.add_argument("video", type=Path)
    inference.add_argument("--model", type=Path, default=Path("runs/baseline/resnet_flow.joblib"))
    inference.add_argument("--cache", type=Path, default=Path(".cache/features"))
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            clips = official_manifest(args.videos, args.splits, args.output)
            print(f"Saved {len(clips)} official entries to {args.output}")
        elif args.command == "extract":
            from .features import extract
            extract(load_manifest(args.manifest), Config.load(args.config), args.output, args.cache)
        elif args.command == "experiment":
            from .experiment import run_experiment
            run_experiment(args.features, args.output, Config.load(args.config), args.evaluate_test)
        else:
            from .predict import predict
            print(json.dumps(predict(args.video, args.model, args.cache), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
