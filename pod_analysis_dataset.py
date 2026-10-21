"""POD analysis of the local (subdomain) fields over the whole dataset.

Every box has the same size and resolution (LENGTH_SUB, n_sub cells per
side), so all saved boxes of all cases live on one reference grid and can be
stacked as snapshots of a single local space. No train/validation/test split:
every finished case and every saved box goes in.

Fields analysed (one POD basis each):

    u_star       global solution restricted to the box        (the target)
    err_raw      u_star - u_raw      error BEFORE the Robin flux
    err_robin    u_star - u_robin    error AFTER  the Robin flux

For each field:

    1. singular values of the snapshot matrix S = [s_1 ... s_M]
       (from the eigenvalues of the n x n correlation S S^T, accumulated box by
       box, so S itself is never held in memory)
    2. projection error vs rank r, averaged over ALL boxes of ALL cases:
           e_m(r) = ||s_m - V_r V_r^T s_m|| / ||s_m||
       reported as the mean, the median and the 90th percentile over m, plus
       the global ratio sqrt(sum_m ||s_m - V_r V_r^T s_m||^2 / sum_m ||s_m||^2)
       = sqrt(sum_{k>r} sigma_k^2 / sum_k sigma_k^2)
    3. exponential rate of both curves (least squares slope of log10 vs r)

Precision: going through S S^T, sigma_k / sigma_0 and the relative errors
bottom out at ~1e-8 (sqrt of machine eps). Anything below FLOOR is round-off.

Pass 1 accumulates S S^T; pass 2 projects every snapshot on the basis.

Run:

    python pod_analysis_dataset.py -data dataset
    python pod_analysis_dataset.py -data dataset -fields u_star err_robin -rmax 300
"""

import argparse
import glob
import json
import os
import time

import numpy as np

FIELDS = {
    "u_star": lambda b: b["u_star"],
    "err_raw": lambda b: b["u_star"] - b["u_raw"],
    "err_robin": lambda b: b["u_star"] - b["u_robin"],
}
LABELS = {
    "u_star": r"$u^\star$",
    "err_raw": r"$u^\star - u_{raw}$  (before Robin)",
    "err_robin": r"$u^\star - u_{robin}$  (after Robin)",
}
COLORS = {"u_star": "#2a6fdb", "err_raw": "#d9822b", "err_robin": "#1f9e74"}

# A snapshot whose norm is below this is skipped in the relative errors (it
# still enters the correlation matrix, where it contributes nothing anyway).
ZERO_NORM = 1e-14

# Going through S S^T squares the condition number: sigma_k / sigma_0 and the
# relative projection errors are only meaningful down to ~sqrt(eps) = 1e-8.
# Below this they are round-off, not decay.
FLOOR = 1e-7


# =============================================================================
# dataset traversal
# =============================================================================

def list_boxes(data_dir):
    """All box files of all finished cases, sorted for reproducibility."""
    cases = sorted(d for d in glob.glob(os.path.join(data_dir, "case_*"))
                   if os.path.isfile(os.path.join(d, "done")))
    boxes = []
    for c in cases:
        boxes += sorted(glob.glob(os.path.join(c, "subdomains", "box_*.npz")))
    return cases, boxes


def reference_order(coords, length_sub, n_sub):
    """Permutation taking a box's dofs to lexicographic (x, y, z) grid order.

    The boxes are separate meshes; sorting by the integer grid index makes the
    dof order identical in every box regardless of how dolfin numbered it.
    """
    h = length_sub / n_sub
    idx = np.rint(coords / h).astype(int)
    perm = np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0]))
    if len(np.unique(idx, axis=0)) != len(idx):
        raise ValueError("box dofs do not map one-to-one onto the grid")
    return perm


def iter_snapshots(boxes, fields, length_sub, n_sub, batch, normalize):
    """Yield ({field: (n_dofs, <=batch) array}, {field: raw norms}) blocks.

    With `normalize`, every nonzero snapshot is scaled to unit norm, so each
    box weighs the same in the POD regardless of its magnitude.
    """
    def flush(buf):
        blk = {f: np.stack(v, axis=1) for f, v in buf.items()}
        nrm = {f: np.linalg.norm(S, axis=0) for f, S in blk.items()}
        if normalize:
            for f, S in blk.items():
                ok = nrm[f] > ZERO_NORM
                S[:, ok] /= nrm[f][ok]
        return blk, nrm

    buf = {f: [] for f in fields}
    for path in boxes:
        with np.load(path) as b:
            perm = reference_order(b["coords"], length_sub, n_sub)
            for f in fields:
                buf[f].append(np.asarray(FIELDS[f](b), dtype=float)[perm])
        if len(buf[fields[0]]) == batch:
            yield flush(buf)
            buf = {f: [] for f in fields}
    if buf[fields[0]]:
        yield flush(buf)


def report_norms(boxes, norms, top):
    """Spread of the per-box norms, and the cases holding the largest ones.

    A few boxes many orders of magnitude above the rest own the whole POD:
    sigma collapses after ~their count and every other box sits under the
    precision floor. This is the first thing to look at.
    """
    case_of = np.array([os.path.basename(os.path.dirname(os.path.dirname(p)))
                        for p in boxes])
    for f, nrm in norms.items():
        nz = nrm[nrm > ZERO_NORM]
        print(f"\n[{f}] box norms: min {nz.min():.3e}  median "
              f"{np.median(nz):.3e}  p99 {np.percentile(nz, 99):.3e}  "
              f"max {nz.max():.3e}   ({len(nrm) - len(nz)} zero boxes)")
        energy = nrm ** 2 / np.sum(nrm ** 2)
        print(f"  {'case':<12} {'boxes':>6} {'max norm':>11} "
              f"{'median norm':>12} {'energy share':>13}")
        per_case = []
        for c in np.unique(case_of):
            m = case_of == c
            per_case.append((c, m.sum(), nrm[m].max(), np.median(nrm[m]),
                             energy[m].sum()))
        for c, k, mx, md, e in sorted(per_case, key=lambda r: -r[4])[:top]:
            print(f"  {c:<12} {k:6d} {mx:11.3e} {md:12.3e} {e:13.2%}")


# =============================================================================
# analysis
# =============================================================================

def decay_rate(y, r_lo, r_hi):
    """Slope of log10(y) vs r on [r_lo, r_hi): y ~ 10^(slope r).

    Values at the precision floor (see FLOOR) are dropped, or the flat
    floor drags the slope toward zero.
    """
    r = np.arange(len(y))
    m = (r >= r_lo) & (r < r_hi) & (y > FLOOR)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(r[m], np.log10(y[m]), 1)[0])


def first_rank_below(err, tol):
    """Smallest r with err[r] <= tol (err[r] = error with r modes), or None."""
    hit = np.nonzero(err <= tol)[0]
    return int(hit[0]) if hit.size else None


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Singular value decay and POD projection error of the "
                    "subdomain fields over the whole dataset.")
    parser.add_argument("-data", type=str, default="dataset")
    parser.add_argument("-out", type=str, default="pod_analysis")
    parser.add_argument("-fields", nargs="+", choices=tuple(FIELDS),
                        default=list(FIELDS))
    parser.add_argument("-rmax", type=int, default=500,
                        help="largest rank for the projection error curve")
    parser.add_argument("-batch", type=int, default=256,
                        help="boxes per block in the S S^T accumulation")
    parser.add_argument("-tols", nargs="+", type=float,
                        default=[1e-1, 1e-2, 1e-3, 1e-4])
    parser.add_argument("-normalize", action="store_true",
                        help="POD of s/||s||: every box weighs the same")
    parser.add_argument("-exclude", nargs="+", default=[],
                        help="case folder names to leave out, e.g. case_00042")
    parser.add_argument("-top", type=int, default=10,
                        help="cases listed in the norm report")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases, boxes = list_boxes(args.data)
    skip = set(args.exclude)
    cases = [c for c in cases if os.path.basename(c) not in skip]
    boxes = [b for b in boxes
             if os.path.basename(os.path.dirname(os.path.dirname(b)))
             not in skip]
    if not boxes:
        raise SystemExit(f"no finished case with boxes under {args.data}")
    with open(os.path.join(cases[0], "params.json")) as f:
        p0 = json.load(f)
    length_sub, n_sub = float(p0["length_sub"]), int(p0["n_sub"])
    n = (n_sub + 1) ** 3
    print(f"{len(cases)} cases, {len(boxes)} boxes, {n} dofs per box")

    # ---- pass 1: correlation matrices --------------------------------------
    t0 = time.time()
    C = {f: np.zeros((n, n)) for f in args.fields}
    norms = {f: [] for f in args.fields}
    done = 0
    for blk, nrm in iter_snapshots(boxes, args.fields, length_sub, n_sub,
                                   args.batch, args.normalize):
        for f in args.fields:
            C[f] += blk[f] @ blk[f].T
            norms[f].append(nrm[f])
        done += blk[args.fields[0]].shape[1]
        print(f"  pass 1: {done}/{len(boxes)} boxes  ({time.time() - t0:.0f}s)")
    norms = {f: np.concatenate(v) for f, v in norms.items()}
    report_norms(boxes, norms, args.top)

    rmax = min(args.rmax, n, len(boxes))
    sigma, basis = {}, {}
    for f in args.fields:
        lam, V = np.linalg.eigh(C.pop(f))
        lam, V = lam[::-1], V[:, ::-1]
        sigma[f] = np.sqrt(np.clip(lam, 0.0, None))
        basis[f] = np.ascontiguousarray(V[:, :rmax])
        print(f"  eigh {f}: done ({time.time() - t0:.0f}s)")

    # ---- pass 2: projection error of every snapshot ------------------------
    # ||s - V_r V_r^T s||^2 = ||s||^2 - sum_{k<r} (v_k^T s)^2, so one product
    # with the rmax-mode basis gives the error at every rank at once.
    rel = {f: [] for f in args.fields}       # (n_snapshots, rmax + 1)
    sq_err = {f: np.zeros(rmax + 1) for f in args.fields}
    sq_norm = {f: 0.0 for f in args.fields}
    done = 0
    for blk, _ in iter_snapshots(boxes, args.fields, length_sub, n_sub,
                                 args.batch, args.normalize):
        for f in args.fields:
            S = blk[f]
            nrm2 = np.einsum("ij,ij->j", S, S)
            coef2 = (basis[f].T @ S) ** 2
            cum = np.vstack([np.zeros((1, S.shape[1])),
                             np.cumsum(coef2, axis=0)])
            err2 = np.clip(nrm2[None, :] - cum, 0.0, None)
            sq_err[f] += err2.sum(axis=1)
            sq_norm[f] += nrm2.sum()
            ok = nrm2 > ZERO_NORM ** 2
            rel[f].append((np.sqrt(err2[:, ok] / nrm2[ok])).T)
        done += blk[args.fields[0]].shape[1]
        print(f"  pass 2: {done}/{len(boxes)} boxes  ({time.time() - t0:.0f}s)")

    # ---- summary ------------------------------------------------------------
    out = {"n_cases": len(cases), "n_boxes": len(boxes), "n_dofs": n,
           "rmax": rmax, "normalize": args.normalize,
           "excluded": np.array(sorted(skip), dtype=str)}
    summary = {}
    for f in args.fields:
        R = np.vstack(rel[f])
        s = sigma[f]
        energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        g = np.sqrt(sq_err[f] / sq_norm[f])
        out.update({
            f"{f}_sigma": s,
            f"{f}_energy": energy,
            f"{f}_proj_mean": R.mean(axis=0),
            f"{f}_proj_median": np.median(R, axis=0),
            f"{f}_proj_p90": np.percentile(R, 90, axis=0),
            f"{f}_proj_global": g,
            f"{f}_n_zero": len(boxes) - R.shape[0],
        })
        r_hi = rmax
        summary[f] = {
            "sigma_rate_log10_per_mode": decay_rate(s / s[0], 1, r_hi),
            "proj_mean_rate_log10_per_mode":
                decay_rate(out[f"{f}_proj_mean"], 1, r_hi),
            "rank_mean_proj_below": {
                str(t): first_rank_below(out[f"{f}_proj_mean"], t)
                for t in args.tols},
            "rank_global_proj_below": {
                str(t): first_rank_below(g, t) for t in args.tols},
            "boxes_with_zero_field": int(len(boxes) - R.shape[0]),
        }
    np.savez_compressed(os.path.join(args.out, "pod_analysis.npz"), **out)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(dict(n_cases=len(cases), n_boxes=len(boxes), n_dofs=n,
                       rmax=rmax, normalize=args.normalize,
                       excluded=sorted(skip), fields=summary), fh, indent=2)

    print(f"\n{len(cases)} cases, {len(boxes)} boxes, n = {n}, rmax = {rmax}")
    for f in args.fields:
        sm = summary[f]
        print(f"\n[{f}]")
        print(f"  sigma_k/sigma_0 rate   : 10^({sm['sigma_rate_log10_per_mode']:.4f} r)")
        print(f"  mean proj. error rate  : 10^({sm['proj_mean_rate_log10_per_mode']:.4f} r)")
        for t in args.tols:
            print(f"  rank for mean err <= {t:.0e}: "
                  f"{sm['rank_mean_proj_below'][str(t)]}   "
                  f"global: {sm['rank_global_proj_below'][str(t)]}")
        if sm["boxes_with_zero_field"]:
            print(f"  ({sm['boxes_with_zero_field']} boxes with a zero field "
                  f"left out of the relative errors)")

    plot(out, args.fields, rmax, args.out)


def plot(out, fields, rmax, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for f in fields:
        s = out[f"{f}_sigma"][:rmax]
        a1.semilogy(s / s[0], color=COLORS[f], lw=2, label=LABELS[f])
        r = np.arange(rmax + 1)
        a2.semilogy(r, out[f"{f}_proj_mean"], color=COLORS[f], lw=2,
                    label=LABELS[f] + " mean")
        a2.fill_between(r, out[f"{f}_proj_median"], out[f"{f}_proj_p90"],
                        color=COLORS[f], alpha=0.15, lw=0)
    a1.set(xlabel="mode k", ylabel=r"$\sigma_k / \sigma_0$",
           title="Singular value decay")
    a2.set(xlabel="rank r",
           ylabel=r"$\|s - V_rV_r^Ts\| / \|s\|$",
           title="Projection error, mean over all boxes\n(band: median - 90th pct)")
    for a in (a1, a2):
        a.grid(True, which="major", color="0.88", lw=0.8)
        a.legend(frameon=False, fontsize=9)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle(f"{out['n_cases']} cases, {out['n_boxes']} boxes, "
                 f"{out['n_dofs']} dofs per box"
                 + (", normalised snapshots" if out["normalize"] else ""))
    fig.tight_layout()
    path = os.path.join(out_dir, "pod_analysis.png")
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
