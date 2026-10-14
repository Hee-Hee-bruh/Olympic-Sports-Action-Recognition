"""Render actual sampled frames for manual inspection; do not infer causes."""
from pathlib import Path
import math

from .data import read_sampled_pairs


def save_error_gallery(rows, manifest, output, samples):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = {row["clip_id"]: row["path"] for row in manifest}
    columns = 2
    figure, axes = plt.subplots(math.ceil(len(rows) / columns), columns,
                                figsize=(13, 4 * math.ceil(len(rows) / columns)), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, rows):
        pairs, info = read_sampled_pairs(paths[row["clip_id"]], samples)
        middle = len(pairs) // 2
        axis.imshow(pairs[middle][0])
        clip = row["clip_id"].split("/")[-1]
        axis.set_title(f"True: {row['true_label']} | frame {info['sample_indices'][middle]}\n"
                       f"B: {row['resnet_prediction']} | D: {row['fusion_prediction']}\n"
                       f"Stanford Olympic Sports / YouTube: {clip}", fontsize=9)
    figure.tight_layout()
    figure.savefig(Path(output) / "error_frames.png", dpi=150)
    plt.close(figure)
