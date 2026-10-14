# Olympic Sports Action Recognition

This implementation follows the Introduction and methodology in the latest
project report, `output/olympic_sports_class_report.tex`, located in the parent
project directory. The Python pipeline is independent of the original notebook.

The project addresses two research questions:

- **RQ1:** Does adding motion features to ResNet improve classification? Compare D with B.
- **RQ2:** Do pretrained ResNet features outperform HOG, and how do their feature extraction costs compare? Compare B with A.

| ID | Video representation | Default dimensions | Classifier |
|---|---|---:|---|
| A | Mean HOG descriptors, L2-normalised | 6,084 | Linear SVM, one-vs-rest |
| B | Mean frozen ImageNet ResNet-18 features, L2-normalised | 512 | Linear SVM, one-vs-rest |
| C | Mean Farneback flow histograms with magnitude-weighted orientation votes, L2-normalised | 32 | Linear SVM, one-vs-rest |
| D | Concatenated B and C features, with each block normalised separately | 544 | Linear SVM, one-vs-rest |

## Running the project

**Python 3.11 or later** is required. On the current development machine, the
`.venv` environment is already installed and ready to use. On a new machine,
create an environment with a compatible Python version:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
```

Run the following commands from the repository directory:

```bash
source .venv/bin/activate

# Download the Stanford release and official train/test split (~5.45 GB of archives).
# Archives and extracted .seq files are stored in data/ and excluded from Git.
python scripts/download_dataset.py --workers 4

# Build the clip manifest from the train/*.txt and test/*.txt lists.
olympic prepare

# Audit files, decode frames, and cache all three feature types.
olympic extract

# Select C on validation data. This command does not produce test results.
MPLCONFIGDIR=.cache/matplotlib olympic experiment

# After fixing preprocessing and sampling, fit on all training clips and evaluate the test set.
MPLCONFIGDIR=.cache/matplotlib olympic experiment --evaluate-test

# Classify a new clip using a saved model.
olympic predict /path/to/clip.seq --model runs/baseline/resnet_flow.joblib
```

Official ResNet-18 weights are downloaded to `.cache/features/torch/` on first
use. Internet access is required to download the dataset and weights; subsequent
runs can operate offline. SVM outputs are **decision scores**, not probabilities.

The entire pipeline can also be run with one command. Include `--evaluate-test`
after finalising the configuration:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/run_pipeline.py --download --evaluate-test
```

## Data preparation and preprocessing

Source: [Stanford Olympic Sports, release 2010.09.07](http://vision.stanford.edu/Datasets/OlympicSports/).
The dataset page was accessible over HTTP during the baseline run. The downloader
uses the official archive URLs published on that page and records their SHA-256
checksums and URLs in `data/download_provenance.json`.
The verified split contains **649 training clips and 134 test clips** across
16 classes. The pipeline preserves this split rather than creating a new random
test partition.

The reader directly supports 8-bit raw/JPEG/PNG Norpix `.seq` files with 8-byte
or 16-byte frame padding, following the
[reader specification in Piotr's Toolbox](https://github.com/pdollar/toolbox/blob/master/videos/seqReaderPlugin.m).
Conversion to AVI is unnecessary. Converted `.mp4`, `.avi`, `.mov`, and `.mkv`
files are also supported through OpenCV.

The manifest contains `clip_id,path,label,split`. Relative paths are resolved
from the manifest's directory. `prepare` maps class names from the official split
lists, including `platform 10m` to `diving_platform_10m` and `springboard 3m` to
`diving_springboard_3m`. The misspelled folder name `diving_platfrom_10m` in the
Stanford archive is also mapped to the correct label without renaming the source
data.

Repeated clip IDs or file paths in the manifest are rejected. Matching SHA-256
checksums identify exact duplicates, and both clips are excluded from every
configuration. Missing files, decoding failures, and clips with fewer than two
frames are recorded in the audit. Evaluation stops with a clear error if any
class is absent from the training, validation, or test subset.

For a clip with T frames, up to K=10 unique indices are sampled uniformly from
`[0, T-2]`. Appearance features use the frame at each sampled index. Optical flow
uses that frame and its **immediately following original frame**, rather than two
distant frames from the sampled list.

All frames are resized directly to 224x224 pixels. HOG and flow use grayscale
images; ResNet uses RGB images and ImageNet mean/std normalisation. The
[torchvision weights](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18)
are fixed to `IMAGENET1K_V1`. Direct resizing follows the proposal, rather than
torchvision's default resize-to-256 and centre-crop transform. The model remains
in `eval()` mode with frozen weights and batch-normalisation statistics. No data
augmentation is applied.

## Experimental protocol and outputs

`configs/baseline.json` stores all settings. A stratified 20% validation subset
is drawn **only from the training partition**, using seed 42. Each configuration
selects C from `{0.01, 0.1, 1, 10}` using validation mAP; ties are resolved by
choosing the smaller C. Model selection finishes for **all four configurations**
before classifiers are refitted on the full official training partition and
metrics are computed on the official test partition.

AP is computed one-vs-rest from decision scores. mAP is the mean AP across all
16 classes. Accuracy uses the class with the highest score; macro-F1 and confusion
matrices follow the same class order. All configurations use the same common set
of usable clips.

| File | Contents |
|---|---|
| `runs/features_k10/audit.csv` | Metadata, sampled indices, SHA-256 checksums, decode status, exclusions, and cache hits |
| `runs/features_k10/features.npz` | HOG, ResNet, and flow features; clip IDs, labels, official splits, and per-clip timings |
| `runs/features_k10/extraction.json` | Configuration, library versions, manifest, and timing definitions |
| `runs/baseline/partitions.csv` | Exact clip IDs for the fitting, validation, and test subsets |
| `runs/baseline/validation.json` | C-grid results and selected C values |
| `runs/baseline/protocol.json` | Settings, hardware, library versions, class counts, and feature hash |
| `runs/baseline/summary.csv` | mAP, accuracy, macro-F1, and extraction/training/inference timings for A-D |
| `runs/baseline/per_class_ap.csv` | AP and test sample counts for each class |
| `runs/baseline/predictions.csv` | Labels and decision scores for every test clip and configuration |
| `runs/baseline/*_confusion.csv/png` | Row-normalised confusion matrices |
| `runs/baseline/*_metrics.json` | Metrics and original confusion counts |
| `runs/baseline/*.joblib` | Classifiers, methods, class order, and preprocessing configuration |
| `runs/baseline/research_questions.json` | D-B and B-A comparisons, plus majority-class baseline accuracy |
| `runs/baseline/error_review.csv` | Up to 10 errors, prioritising class diversity, with manual observations |
| `runs/baseline/error_frames.png` | Actual sampled frames for inspecting B/D errors, with clip IDs and Stanford/YouTube attribution |

Load only `.joblib` files created by you or this pipeline. Feature extraction
timing includes resizing, normalisation, descriptor computation, and pooling.
Decoding and ResNet initialisation/warmup are timed separately. Cached features
retain the original cold-run timings so cache access is not mistaken for feature
extraction speed.

D reuses B and C features. Its extraction cost is the sum of B and C, excluding
the vector concatenation copy. `estimated_full_seconds_per_clip` adds decoding,
extraction, and cached-feature SVM inference; it is an estimate rather than a
separate end-to-end latency measurement. Inference timing is measured over the
test batch.

The SEQ reader checks frame records throughout each clip and decodes sampled
frames; it does not decode every frame. Other video formats are read twice to
count actual frames and retrieve the samples, so decoding timings can differ
between formats.

CPU is the default for consistent benchmarking. Set `device` to `mps` or `cuda`
on compatible hardware if desired. Results from different devices should be
reported as separate benchmarks.

## Optional K=5 validation ablation

Copy the configuration, set `samples` to 5, and use separate output directories:

```bash
olympic extract --config configs/k5.json --output runs/features_k5
olympic experiment --config configs/k5.json --features runs/features_k5/features.npz --output runs/validation_k5
```

Compare B and D against K=10 on validation data to select K before evaluating
the test set. Do not use test results to select K. Changing the configuration
or library versions creates a different cache key, preventing reuse of vectors
computed with different preprocessing settings.

## Tests

```bash
MPLCONFIGDIR=.cache/matplotlib python -m pytest -q
```

Tests cover raw/JPEG SEQ files, padding, truncation, short clips, adjacent-frame
sampling, orientation/magnitude histograms, zero-motion normalisation, duplicate
manifest entries, training-only model selection, score ordering, and exported
metrics, models, and artifacts.

Synthetic tests verify the implementation. Research results must use the actual
Stanford dataset. Explanations involving camera motion, background cues, or poses
require inspection of the clips; confusion matrices alone do not establish the
causes of errors.

## Baseline results from 14 October 2026

All **783/783 clips** were downloaded and processed, with no missing files,
decoding failures, or duplicate SHA-256 checksums. The official **649 train /
134 test** split was preserved. Validation used 130 training clips, leaving
519 clips for fitting candidate classifiers. The experiment used CPU execution
with one thread, seed 42, K=10, and no ResNet fine-tuning.

| Configuration | C selected on validation | Test mAP | Test accuracy | Macro-F1 | Extraction per clip |
|---|---:|---:|---:|---:|---:|
| A: HOG | 10 | 47.42% | 41.79% | 39.42% | 22.35 ms |
| B: ResNet-18 | 10 | 69.24% | 62.69% | 58.85% | 93.34 ms |
| C: Flow | 1 | 37.11% | 35.82% | 32.62% | 33.22 ms |
| D: ResNet + flow | 10 | 68.49% | 62.69% | 59.69% | 126.56 ms |

**RQ1:** Adding flow reduced mAP by 0.75 percentage points. Accuracy was unchanged
(84/134 correct predictions for both B and D), while macro-F1 increased by
0.84 percentage points. D corrected eight errors made by B but also introduced
eight errors on clips B had classified correctly. This run therefore does not
support the hypothesis that motion improves overall mAP; its effect varies
across classes.

AP increased for javelin throw (+55.43 percentage points), high jump (+18.85),
and platform diving (+8.06), but decreased for triple jump (-24.25), snatch
(-21.22), and pole vault (-12.10). Javelin throw and triple jump each have only
four test clips, so these AP changes should be interpreted alongside the sample
counts.

**RQ2:** ResNet increased mAP by 21.82 percentage points and accuracy by
20.90 percentage points compared with HOG. Its average feature extraction cost
was approximately 4.18 times higher on this CPU. The majority-class predictor
selected from the training labels achieved 7.46% test accuracy.

B's confusion matrix shows frequent confusion between hammer throw and discus
throw, clean and jerk and snatch, and triple jump and long jump. The inspected
error frames show small athletes, blurred images, and similar sports venues or
diving areas. These are observations from illustrative frames; they do not
establish the causes of errors or the influence of camera motion.
`runs/baseline/error_review.csv` records individual observations for further
inspection of the complete clips.

Model D was reloaded and used to predict directly from a `.seq` clip. Its decision
scores matched those saved during evaluation. **All 21 tests passed.** Detailed
results are stored in `runs/baseline/`. Dataset files, caches, and large result
artifacts are kept locally and excluded from Git.
