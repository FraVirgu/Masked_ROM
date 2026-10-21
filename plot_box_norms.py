"""Box plot of the per-box norms of u_star, err_raw and err_robin.

Same boxes and fields as pod_analysis_dataset.py, but only the norms: one
read of every box file, no decomposition. The norm does not depend on the dof
order, so the boxes are not re-sorted onto the reference grid.

Whiskers span the 1st-99th percentile (percentiles, unlike 1.5 IQR, mean the
same thing on a log axis); boxes beyond them are drawn as points.

Run:

    python plot_box_norms.py -data dataset -exclude case_00054 case_00045
"""

import argparse
import os

import numpy as np

from pod_analysis_dataset import COLORS, FIELDS, LABELS, list_boxes


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Box plot of the per-box field norms over the dataset.")
    parser.add_argument("-data", type=str, default="dataset")
    parser.add_argument("-out", type=str, default="pod_norm")
    parser.add_argument("-exclude", nargs="+", default=[],
                        help="case folder names to leave out, e.g. case_00042")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases, boxes = list_boxes(args.data)
    skip = set(args.exclude)
    boxes = [b for b in boxes
             if os.path.basename(os.path.dirname(os.path.dirname(b)))
             not in skip]
    fields = list(FIELDS)

    norms = {f: np.empty(len(boxes)) for f in fields}
    for m, path in enumerate(boxes):
        with np.load(path) as b:
            for f in fields:
                norms[f][m] = np.linalg.norm(FIELDS[f](b))
        if (m + 1) % 500 == 0 or m + 1 == len(boxes):
            print(f"  {m + 1}/{len(boxes)} boxes")

    np.savez_compressed(os.path.join(args.out, "box_norms.npz"),
                        boxes=np.array(boxes, dtype=str), **norms)
    for f in fields:
        q = np.percentile(norms[f], [1, 25, 50, 75, 99])
        print(f"[{f}] p1 {q[0]:.3e}  q1 {q[1]:.3e}  median {q[2]:.3e}  "
              f"q3 {q[3]:.3e}  p99 {q[4]:.3e}")

    plot(norms, fields, len(boxes), args.out)


def plot(norms, fields, n_boxes, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [norms[f][norms[f] > 0] for f in fields]
    bp = ax.boxplot(data, whis=(1, 99), widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    flierprops=dict(marker="o", ms=2, alpha=0.3,
                                    markeredgewidth=0))
    for patch, flier, f in zip(bp["boxes"], bp["fliers"], fields):
        patch.set(facecolor=COLORS[f], alpha=0.35, edgecolor=COLORS[f], lw=1.5)
        flier.set(markerfacecolor=COLORS[f])
    for f, d, x in zip(fields, data, range(1, len(fields) + 1)):
        med = np.median(d)
        ax.text(x + 0.3, med, f"{med:.3g}", va="center", fontsize=9,
                color="0.25")

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(fields) + 1),
                  [LABELS[f] for f in fields], fontsize=9)
    ax.set_ylabel(r"$\|\cdot\|_2$ over the box dofs")
    ax.set_title(f"Per-box norms, {n_boxes} boxes\n"
                 "(box: quartiles, whiskers: 1st-99th percentile)")
    ax.grid(True, axis="y", which="major", color="0.88", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "box_norms.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
