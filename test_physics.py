"""
Physical sanity checks for the 3D-1D solution (exterior='dirichlet').

Verification (are the equations solved right?) is already covered by
test_exterior.py. This script asks the different question: does the solution
obey the physics the model claims to describe?

  1. BOUNDS / maximum principle
        0 <= u3d <= max(u1d) <= p_in.
        The tissue is fed only by the vessels, so it cannot exceed the source,
        and nothing generates negative concentration. A violation is a real
        defect, not a tolerance issue.

  2. MONOTONICITY along the tree
        u1d must decrease from the inlet outwards — concentration cannot rise
        as you move away from the only source.

  3. MASS CONSERVATION  (the strongest single check)
        Steady state: what the vessels deliver = what the tissue absorbs.
            delivery   = 1' G (u1d - C u3d)     [the coupling term, discretely]
            absorption = int_Omega sigma3d * u3d dx
        These are computed from the SAME operators the solver used, so a
        mismatch points at C, G, or the sign of the coupling — not at
        quadrature.

  4. RESOLUTION
        The 3D cell size must be small enough to resolve the vessels. If
        hmax(V) >> R, the line source is smeared over a whole cell and the
        perivascular gradient — the entire point of a 3D-1D model — cannot
        form.

Run:
    python test_physics.py -name Prova_14_07 -n 40
    python test_physics.py -name Prova_14_07 -n 40 -beta 50   # nitsche sweep
"""
import argparse
import os
import numpy as np
from dolfin import assemble, Constant, dx, Function, vertex_to_dof_map

from CCO_Domain import CCOVascularMesh, boundary_from_obj
from Solver import Solver3D1D


P_IN = 1.0   # inlet value hardcoded in Solver._assemble_system


def check_bounds(s, u3d, u1d, tol):
    print("\n" + "-" * 70)
    print("1. BOUNDS / maximum principle")
    print("-" * 70)

    # exterior dofs are eliminated to exactly 0 and are not part of the domain
    if s.exterior == "dirichlet":
        u3d_dom = u3d[s.int_dofs]
    else:
        u3d_dom = u3d

    lo3, hi3 = u3d_dom.min(), u3d_dom.max()
    lo1, hi1 = u1d.min(), u1d.max()

    print(f"  u1d  range : [{lo1:.6f}, {hi1:.6f}]")
    print(f"  u3d  range : [{lo3:.6f}, {hi3:.6f}]   (interior dofs only)")
    print(f"  p_in       : {P_IN}")

    # These are DISCRETIZATION quantities, not algebraic identities: the line
    # source is smeared over cells of size h >> R, and Nitsche imposes the inlet
    # only approximately. A small overshoot is expected and must SHRINK under
    # refinement (raise -n) and stronger enforcement (raise -beta). What would
    # be a real defect is an overshoot that is O(1), or one that does not
    # vanish as h -> 0.
    over1 = (hi1 - P_IN) / P_IN
    over3 = (hi3 - hi1) / max(abs(hi1), 1e-30)
    print(f"  u1d overshoot over p_in      : {over1:+.3e}  (rel)")
    print(f"  u3d overshoot over max(u1d)  : {over3:+.3e}  (rel)")

    ok = True
    if over1 > tol:
        print(f"  FAIL  max(u1d) exceeds p_in by {over1:.2%}")
        print( "        The Nitsche inlet BC is only weakly enforced — raise -beta.")
        ok = False
    if over3 > tol:
        print(f"  FAIL  max(u3d) exceeds max(u1d) by {over3:.2%}")
        print( "        Expected from smearing when h >> R; must vanish as -n grows.")
        print( "        If it does NOT shrink under refinement, suspect the sign of")
        print( "        the coupling blocks.")
        ok = False
    if lo3 < -tol:
        print(f"  FAIL  min(u3d) = {lo3:.3e} < 0")
        ok = False
    if ok:
        print(f"  PASS  0 <= u3d <= max(u1d) <= p_in  (to {tol:g})")
    return ok


def monotonicity(s, u1d):
    """
    Walk the vessel tree outward from the inlet and measure whether u1d ever
    RISES. There is a single source, so concentration cannot increase as you
    move away from it. Returns the raw numbers; the printing/verdict is done by
    the callers, so the report and stdout cannot disagree.

    Note this is a LOCAL check and is genuinely independent of mass balance:
    a solution can conserve mass globally while still rising along a branch,
    because a surplus in one place can cancel a deficit in another.
    """
    v2d  = vertex_to_dof_map(s.W[1])
    vals = u1d[v2d]                      # value at each mesh vertex

    mesh   = s.meshQ
    nv     = mesh.num_vertices()
    adj    = [[] for _ in range(nv)]
    for a, b in mesh.cells():
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    inlet = [v for v in range(nv) if s.Q_markers[v] == s.inlet_tag]
    if not inlet:
        return None
    root = inlet[0]

    # BFS gives each vertex its distance from the inlet, so "outward" is well
    # defined even at bifurcations.
    depth, order, queue = {root: 0}, [root], [root]
    while queue:
        v = queue.pop(0)
        for w in adj[v]:
            if w not in depth:
                depth[w] = depth[v] + 1
                order.append(w)
                queue.append(w)

    # Judge each rise RELATIVE to the span of u1d: a rise of 1e-4 on a field
    # spanning 0.33 is solver noise, not backflow.
    span  = max(vals.max() - vals.min(), 1e-30)
    worst, n_out, rises = 0.0, 0, []
    for v in order:
        for w in adj[v]:
            if depth.get(w, -1) == depth[v] + 1:       # w is further out
                n_out += 1
                rise   = (vals[w] - vals[v]) / span
                rises.append(rise)
                worst = max(worst, rise)

    return dict(
        root        = root,
        inlet_val   = float(vals[root]),
        span        = float(span),
        n_out       = n_out,
        worst_rise  = float(worst),
        rises       = rises,
    )


def check_monotone(s, u1d, tol):
    """u1d must not increase as you move away from the inlet."""
    print("\n" + "-" * 70)
    print("2. MONOTONICITY of u1d along the tree  (inlet -> outlets)")
    print("-" * 70)

    m = monotonicity(s, u1d)
    if m is None:
        print("  SKIP  no inlet vertex found")
        return True

    bad = sum(1 for r in m["rises"] if r > tol)
    print(f"  inlet vertex   : {m['root']}  (u1d = {m['inlet_val']:.6f})")
    print(f"  outward edges  : {m['n_out']}   u1d span = {m['span']:.6f}")
    print(f"  worst rise     : {m['worst_rise']:+.3e}  (relative to span)")
    if bad:
        print(f"  FAIL  u1d increases on {bad} outward edges by more than {tol:g}")
        return False
    print(f"  PASS  u1d decreases from the inlet to the outlets (to {tol:g})")
    return True


def check_mass(s, u3d, u1d, tol):
    """
    Steady state:  vessel delivery == tissue absorption.

        3D eq:  -div(k3 grad u) + k3*u = gamma*(u1d - u3d)*delta_Lambda

    Test with v = 1. The diffusion term integrates to a boundary flux, and the
    3D outer boundary carries no BC (natural / zero flux), so it vanishes.
    What is left is:

        int k3*u3d dx  =  sum gamma*(u1d - C u3d)

    Computed from the assembled C and G — the very operators the solver used.
    """
    print("\n" + "-" * 70)
    print("3. MASS CONSERVATION  (delivery == absorption)")
    print("-" * 70)

    # --- delivery: 1' G (u1d - C u3d) ---
    avg      = s.C @ u3d                 # u3d averaged onto the centerline
    delivery = float(np.ones(len(u1d)) @ (s.G @ (u1d - avg)))

    # --- absorption: int k3 * u3d dx, over the interior only ---
    f = Function(s.W[0])
    f.vector()[:] = u3d
    absorption = assemble(Constant(s.sigma3d) * f * s.d_omega(222))

    denom = max(abs(delivery), abs(absorption), 1e-30)
    rel   = abs(delivery - absorption) / denom

    print(f"  vessel delivery    int gamma*(u1d - u3d) = {delivery:.8e}")
    print(f"  tissue absorption  int sigma3d * u3d dx  = {absorption:.8e}")
    print(f"  relative imbalance                       = {rel:.3e}")

    # This is a DISCRETE balance, not an algebraic identity: the Dirac line
    # source is smeared onto a mesh with h >> R, so a sub-percent mismatch is
    # ordinary discretization error and must shrink under refinement. A SIGN
    # error in the coupling would not show up as 0.1% — it would put the two
    # sides off by a factor of -1 or 2.
    if rel < tol:
        print(f"  PASS  delivery == absorption to {rel:.2e} — the coupling")
        print( "        operators (C, G) and their signs are consistent.")
        return True
    print(f"  FAIL  imbalance {rel:.2e} exceeds tol {tol:g}.")
    print( "        If this SHRINKS as -n grows it is discretization error, not a")
    print( "        bug. If it is O(1) or refinement-independent, the coupling")
    print( "        operator or its sign is wrong.")
    return False


def check_resolution(s):
    print("\n" + "-" * 70)
    print("4. RESOLUTION — can the 3D mesh see the vessels?")
    print("-" * 70)

    h      = s.W[0].mesh().hmax()
    r      = np.array(s.Q_radii.array())
    rmin, rmax = r.min(), r.max()

    print(f"  hmax(V)        = {h:.5f}")
    print(f"  vessel radius  = [{rmin:.5f}, {rmax:.5f}]")
    print(f"  h / R          = [{h/rmax:.1f}, {h/rmin:.1f}]")

    if h > rmin:
        print(f"  WARN  cells are up to {h/rmin:.0f}x LARGER than the thinnest vessel.")
        print( "        The line source is smeared over a cell, so the perivascular")
        print( "        gradient cannot form. u3d will look like a smooth blob that")
        print( "        barely knows the tree is there. Refine (-n) until h < R, or")
        print( "        accept that the near-vessel field is unresolved.")
        return False
    print("  PASS  h < R: the mesh resolves the vessels")
    return True


def verdict(beta_rows, n_rows, single=None, tol=1e-2):
    """
    Turn the results into a diagnosis. Kept separate from the printing so the
    report file and stdout can never disagree.
    """
    v = []

    if single:
        wr = single["worst_rise"]
        if wr is not None:
            if wr > tol:
                v.append(
                    f"MONOTONICITY — FAILING. Walking outward from the inlet, u1d "
                    f"RISES by up to {wr:+.3e} (relative to its span) on some edge. "
                    f"With a single source, concentration cannot increase as you "
                    f"move away from it."
                )
            else:
                v.append(
                    f"MONOTONICITY — OK. u1d decreases from the inlet to the "
                    f"outlets on every edge (worst rise {wr:+.3e}, within noise). "
                    f"Pressure falls along the tree, as it must with one source."
                )

        if single["imbal"] < 1e-5:
            v.append(
                f"MASS CONSERVATION — OK. Delivery {single['delivery']:.6e} vs "
                f"absorption {single['absorption']:.6e}, relative imbalance "
                f"{single['imbal']:.2e}. These are computed by completely "
                f"different routes (the discrete coupling operators C and G vs a "
                f"volume integral), so agreement to this level is strong evidence "
                f"the coupling is correct."
            )
        else:
            v.append(
                f"MASS CONSERVATION — imbalance {single['imbal']:.2e}. Run "
                f"-sweep n: if it shrinks under refinement it is discretization "
                f"error; if it grows, the coupling is broken."
            )

        if single["pou"] < 1e-8:
            v.append(
                f"COUPLING OPERATOR — OK. max|C@1 - 1| = {single['pou']:.2e}. The "
                f"circle-average reproduces constants exactly, so it is "
                f"mass-preserving by construction."
            )
        else:
            v.append(
                f"COUPLING OPERATOR — BROKEN. max|C@1 - 1| = {single['pou']:.2e}. "
                f"Averaging the constant 1 over the circle must return 1 on every "
                f"row, for any mesh and any radius. It does not, so C is not "
                f"mass-preserving and no coupling built from it can conserve mass."
            )

        if single["h_over_R"] > 1.0:
            v.append(
                f"RESOLUTION — the 3D cells are up to {single['h_over_R']:.0f}x "
                f"larger than the thinnest vessel (h={single['h']:.4f}). The line "
                f"source is smeared over a cell, so the perivascular gradient is "
                f"unresolved and u3d overshoots max(u1d) by "
                f"{single['over_3d']:+.2e}. This is a genuine modelling "
                f"limitation, not a bug: resolving it needs local refinement "
                f"around the centerlines."
            )

    if beta_rows:
        o0 = beta_rows[0][1]["over_1d"]
        o1 = beta_rows[-1][1]["over_1d"]
        b1 = beta_rows[-1][0]
        if o1 < o0 / 10:
            v.append(
                f"NITSCHE INLET — RESOLVED. The u1d overshoot above p_in falls "
                f"{o0:.2e} -> {o1:.2e} as beta goes {beta_rows[0][0]:g} -> {b1:g}. "
                f"It was only a weakly enforced Nitsche BC. Use beta >= 500."
            )
        else:
            v.append(
                f"NITSCHE INLET — UNRESOLVED. The u1d overshoot does not respond "
                f"to beta ({o0:.2e} -> {o1:.2e}). The inlet BC is not the cause; "
                f"look at the 1D operator."
            )

    if n_rows:
        i0, i1 = n_rows[0][1]["imbal"], n_rows[-1][1]["imbal"]
        p_max  = max(m["pou"]       for _, m in n_rows)
        d_max  = max(m["c_dropped"] for _, m in n_rows)

        # Below ~1e-5 the imbalance is at the level of the linear solver's own
        # tolerance: it is noise in the last digits, not a trend. Comparing
        # 1e-8 to 1e-7 as if it were a convergence rate is meaningless.
        i_max = max(m["imbal"] for _, m in n_rows)
        if i_max < 1e-5:
            v.append(
                f"MASS CONSERVATION — OK at every refinement level (worst "
                f"{i_max:.2e}). This is at the level of the linear solver's own "
                f"tolerance, i.e. conservation holds exactly; the variation "
                f"across rows is solver noise, not a trend."
            )
        elif i1 > i0:
            v.append(
                f"MASS CONSERVATION — FAILING, AND DIVERGING. The imbalance GROWS "
                f"under refinement ({i0:.2e} -> {i1:.2e}, {i1/max(i0,1e-30):.0f}x). "
                f"A discretization error converges as h -> 0; this does the "
                f"opposite. So the cause is NOT the coarse mesh / smearing — it is "
                f"a defect in the coupling itself."
            )
        else:
            v.append(
                f"MASS CONSERVATION — consistent. The imbalance shrinks under "
                f"refinement ({i0:.2e} -> {i1:.2e}), i.e. ordinary discretization "
                f"error."
            )

        if p_max > 1e-8:
            v.append(
                f"ROOT CAUSE (suspected) — PARTITION OF UNITY VIOLATED. "
                f"max|C@1 - 1| = {p_max:.3e}. Averaging the constant function 1 "
                f"over the circle must return exactly 1 on every row, for any mesh "
                f"and any radius; this is a property of the operator, not the "
                f"solution. In average_matrix_diff_radii, quadrature points that "
                f"miss the mesh are skipped ('if c >= limit: continue') while "
                f"curve_measure still divides by the FULL weight sum, so any row "
                f"whose circle pokes outside the mesh comes out short. C loses "
                f"mass, and no coupling built from it can conserve mass."
            )
        else:
            v.append(
                f"PARTITION OF UNITY — OK. max|C@1 - 1| = {p_max:.3e}, so the "
                f"averaging operator is mass-preserving and the imbalance "
                f"originates elsewhere."
            )

        if d_max > 1e-12:
            v.append(
                f"CONTRIBUTING — the dirichlet elimination is discarding real "
                f"coupling weight (max dropped = {d_max:.3e}). Averaging circles "
                f"near the boundary reach into cells classified exterior, and "
                f"zeroing those columns of C removes genuine coupling. This gets "
                f"worse as the mesh refines and the interface moves."
            )
        else:
            v.append(
                "The dirichlet elimination discards no coupling weight "
                "(max dropped = 0), so it is not contributing to the imbalance."
            )

    return v


def write_report(path, args, single, beta_rows, n_rows):
    """Write the full diagnostic record: raw metrics as JSON + a readable report."""
    import json
    from datetime import datetime

    payload = {
        "generated"  : datetime.now().isoformat(timespec="seconds"),
        "case"       : args.name,
        "exterior"   : "dirichlet",
        "p_in"       : P_IN,
        "single"     : single,
        "beta_sweep" : [m for _, m in beta_rows],
        "mesh_sweep" : [m for _, m in n_rows],
        "verdict"    : verdict(beta_rows, n_rows, single, args.tol),
    }
    with open(path + ".json", "w") as f:
        json.dump(payload, f, indent=2)

    L = []
    L.append("=" * 78)
    L.append(f"PHYSICS DIAGNOSTIC — {args.name}")
    L.append(f"generated {payload['generated']}")
    L.append("=" * 78)
    L.append("")
    L.append("Verification (are the equations solved correctly?) is covered by")
    L.append("test_exterior.py: dirichlet == restrict to ~1e-11. This file asks the")
    L.append("different question — does the solution obey the physics?")
    L.append("")

    if single:
        L.append("-" * 78)
        L.append("SINGLE CONFIGURATION")
        L.append("-" * 78)
        L.append(f"  n = {single['n']}   beta = {single['beta']}   "
                 f"sigma3d = {single['sigma3d']}   kappa = {single['kappa']}")
        L.append(f"  3D dofs {single['ndofs']} ({single['n_int_dofs']} interior)  "
                 f"h = {single['h']:.5f}   h/Rmin = {single['h_over_R']:.1f}")
        L.append("")
        L.append(f"  u1d range            : [{single['u1d_min']:.6f}, "
                 f"{single['u1d_max']:.6f}]")
        L.append(f"  u3d range (interior) : [{single['u3d_min']:.6f}, "
                 f"{single['u3d_max']:.6f}]")
        L.append(f"  u1d over p_in        : {single['over_1d']:+.3e}")
        L.append(f"  u3d over max(u1d)    : {single['over_3d']:+.3e}")
        L.append("")
        L.append("  MONOTONICITY (does u1d fall from the inlet to the outlets?)")
        L.append(f"    outward edges      : {single['n_out']}")
        L.append(f"    worst rise         : {single['worst_rise']:+.3e}  "
                 f"(relative to the u1d span; >0 means u1d went UP)")
        L.append("")
        L.append("  MASS BALANCE (delivery == absorption)")
        L.append(f"    vessel delivery    : {single['delivery']:.8e}")
        L.append(f"    tissue absorption  : {single['absorption']:.8e}")
        L.append(f"    mass imbalance     : {single['imbal']:.3e}")
        L.append("")
        L.append("  COUPLING OPERATOR")
        L.append(f"    max|C@1 - 1|       : {single['pou']:.3e}  "
                 f"(must be ~0: averaging 1 over the circle must give 1)")
        L.append(f"    coupling wt dropped: {single['c_dropped']:.3e}")
        L.append("")

    if beta_rows:
        L.append("-" * 78)
        L.append("BETA SWEEP — is the u1d overshoot just weak Nitsche enforcement?")
        L.append("-" * 78)
        L.append(f"{'beta':>8} | {'u1d over p_in':>14} | {'u3d over u1d':>13} | "
                 f"{'mass imbal':>11}")
        for b, m in beta_rows:
            L.append(f"{b:>8.0f} | {m['over_1d']:>+14.3e} | {m['over_3d']:>+13.3e} "
                     f"| {m['imbal']:>11.3e}")
        L.append("")

    if n_rows:
        L.append("-" * 78)
        L.append("MESH SWEEP — do the errors vanish under refinement?")
        L.append("(A discretization error converges as h -> 0. A formulation bug")
        L.append(" does not. This is the decisive test.)")
        L.append("-" * 78)
        L.append("('worst rise' > 0 means u1d went UP somewhere on the way out.)")
        L.append("-" * 78)
        L.append(f"{'n':>4} | {'3D dofs':>8} | {'h/Rmin':>7} | {'u3d over u1d':>13} "
                 f"| {'worst rise':>11} | {'mass imbal':>11} | {'|C@1 - 1|':>11}")
        for n, m in n_rows:
            wr = m["worst_rise"]
            L.append(f"{n:>4} | {m['ndofs']:>8} | {m['h_over_R']:>7.1f} | "
                     f"{m['over_3d']:>+13.3e} | "
                     f"{(f'{wr:+.3e}' if wr is not None else 'n/a'):>11} | "
                     f"{m['imbal']:>11.3e} | {m['pou']:>11.3e}")
        L.append("")

    L.append("=" * 78)
    L.append("VERDICT")
    L.append("=" * 78)
    for i, line in enumerate(payload["verdict"], 1):
        L.append("")
        # wrap at 76 cols
        words, cur = line.split(), ""
        out = []
        for w in words:
            if len(cur) + len(w) + 1 > 74:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        out.append(cur)
        L.append(f"{i}. {out[0]}")
        for o in out[1:]:
            L.append(f"   {o}")
    L.append("")
    L.append("=" * 78)

    text = "\n".join(L)
    with open(path + ".txt", "w") as f:
        f.write(text + "\n")

    print("\n" + text)
    print(f"\nReport written:\n  {path}.txt\n  {path}.json")


def run(boundary, name, n, beta):
    """Build + solve one configuration."""
    return Solver3D1D(
        path_to_1D_mesh = f"./nets/{name}/{name}_",
        boundary        = boundary,
        n               = n,
        sigma3d         = 1e-3,
        sigma1d         = 1.0,
        kappa           = 1.0,
        beta_nitsche    = beta,
        exterior        = "dirichlet",
    ).build().solve()


def metrics(s):
    """
    The same quantities the checks above report, but returned quietly so the
    sweeps can tabulate them. These are all DISCRETIZATION errors — the point of
    a sweep is to watch them shrink.
    """
    u3d = s.u3d.vector().get_local()
    u1d = s.u1d.vector().get_local()
    dom = u3d[s.int_dofs] if s.exterior == "dirichlet" else u3d

    hi1, hi3 = u1d.max(), dom.max()

    avg        = s.C @ u3d
    delivery   = float(np.ones(len(u1d)) @ (s.G @ (u1d - avg)))
    f          = Function(s.W[0])
    f.vector()[:] = u3d
    absorption = assemble(Constant(s.sigma3d) * f * s.d_omega(222))
    imbalance  = abs(delivery - absorption) / max(
        abs(delivery), abs(absorption), 1e-30
    )

    # PARTITION OF UNITY: averaging the constant function 1 over the circle must
    # give exactly 1, on every row, for any mesh and any radius. This is a
    # property of the OPERATOR, independent of the solution. If C@1 != 1, C is
    # losing mass — and any coupling built from it cannot conserve mass either.
    # In average_matrix_diff_radii, quadrature points that miss the mesh are
    # skipped (`if c >= limit: continue`) while curve_measure still divides by
    # the FULL weight sum, so such rows come out short.
    row_sum = np.asarray(s.C @ np.ones(s.C.shape[1])).ravel()
    pou_err = np.abs(row_sum - 1.0).max()

    mono = monotonicity(s, u1d)

    r = np.array(s.Q_radii.array())
    return dict(
        n          = s.n,
        beta       = s.beta_nitsche,
        h          = s.W[0].mesh().hmax(),
        h_over_R   = s.W[0].mesh().hmax() / r.min(),
        ndofs      = s.W[0].dim(),
        n_int_dofs = int(len(s.int_dofs)) if s.int_dofs is not None else s.W[0].dim(),
        u1d_min    = float(u1d.min()),
        u1d_max    = float(hi1),
        u3d_min    = float(dom.min()),
        u3d_max    = float(hi3),
        over_1d    = (hi1 - P_IN) / P_IN,                 # u1d above the inlet
        over_3d    = (hi3 - hi1) / max(abs(hi1), 1e-30),  # u3d above max(u1d)
        delivery   = delivery,
        absorption = float(absorption),
        imbal      = imbalance,
        pou        = float(pou_err),
        c_dropped  = float(s.C_dropped),
        worst_rise = mono["worst_rise"] if mono else None,
        n_out      = mono["n_out"]      if mono else 0,
        niters     = s.niters,
        solve_time = s.solve_time,
        sigma3d    = s.sigma3d,
        sigma1d    = s.sigma1d_ref,
        kappa      = s.kappa,
    )


def sweep_beta(boundary, args, betas=(5.0, 50.0, 500.0, 5000.0)):
    """
    Nitsche only imposes the inlet BC approximately; the error decreases as beta
    grows. If max(u1d) -> p_in as beta grows, the overshoot was weak enforcement
    and nothing more.
    """
    print("\n" + "=" * 70)
    print(f"BETA SWEEP  (n={args.n}) — does the u1d overshoot come from Nitsche?")
    print("=" * 70)
    print(f"{'beta':>8} | {'u1d over p_in':>14} | {'u3d over u1d':>13} | "
          f"{'mass imbal':>11}")
    print("-" * 70)

    rows = []
    for b in betas:
        m = metrics(run(boundary, args.name, args.n, b))
        rows.append((b, m))
        print(f"{b:>8.0f} | {m['over_1d']:>+14.3e} | {m['over_3d']:>+13.3e} | "
              f"{m['imbal']:>11.3e}")

    print("-" * 70)
    first, last = rows[0][1]["over_1d"], rows[-1][1]["over_1d"]
    if last < first / 10:
        print("  The u1d overshoot COLLAPSES as beta grows => it was simply a")
        print("  weakly-enforced Nitsche inlet BC. Not a defect; raise beta.")
    elif last < first:
        print("  The u1d overshoot shrinks with beta, but slowly. Push beta higher.")
    else:
        print("  The u1d overshoot does NOT respond to beta => it is not the inlet")
        print("  BC. Look at the 1D operator itself.")
    return rows


def sweep_n(boundary, args, ns=(20, 40, 60, 80)):
    """
    THE decisive test. Every failure so far is a discretization error, and a
    discretization error must vanish as h -> 0. A formulation bug does not.
    """
    print("\n" + "=" * 70)
    print(f"MESH SWEEP  (beta={args.beta}) — do the errors vanish under refinement?")
    print("=" * 70)
    print(f"{'n':>4} | {'h/Rmin':>7} | {'u3d over u1d':>13} | {'worst rise':>11} | "
          f"{'mass imbal':>11} | {'|C@1 - 1|':>11}")
    print("-" * 78)

    rows = []
    for n in ns:
        m = metrics(run(boundary, args.name, n, args.beta))
        rows.append((n, m))
        wr = m["worst_rise"]
        print(f"{n:>4} | {m['h_over_R']:>7.1f} | {m['over_3d']:>+13.3e} | "
              f"{(f'{wr:+.3e}' if wr is not None else 'n/a'):>11} | "
              f"{m['imbal']:>11.3e} | {m['pou']:>11.3e}")

    print("-" * 78)
    i0, i1 = rows[0][1]["imbal"], rows[-1][1]["imbal"]
    p_max  = max(m["pou"] for _, m in rows)

    if i1 > i0:
        print(f"  The mass imbalance GROWS under refinement ({i0:.1e} -> {i1:.1e}).")
        print( "  A discretization error converges; this diverges. So this is NOT")
        print( "  smearing — it is a defect in the coupling operator itself.")
    else:
        print("  The mass imbalance shrinks under refinement => discretization error.")

    if p_max > 1e-8:
        print(f"\n  |C@1 - 1| = {p_max:.3e}: the averaging operator VIOLATES the")
        print( "  partition of unity. Averaging the constant 1 over the circle must")
        print( "  return exactly 1 on every row, for any mesh. It does not, so C is")
        print( "  losing mass — and no coupling built from it can conserve mass.")
        print( "  In average_matrix_diff_radii, quadrature points that miss the mesh")
        print( "  are skipped, but curve_measure still divides by the FULL weight")
        print( "  sum. Rows whose circle pokes outside the mesh come out short.")
    else:
        print(f"\n  |C@1 - 1| = {p_max:.3e}: the averaging operator is a valid")
        print( "  partition of unity, so the imbalance lies elsewhere.")
    print("\n  NOTE: fully resolving Rmin needs h < Rmin, i.e. n > ~350 on [-1,1]^3")
    print("        (~4e7 cells) — out of reach by uniform refinement. Real 3D-1D")
    print("        codes refine locally around the centerlines instead.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-name",  type=str,   default="Prova_14_07")
    ap.add_argument("-graph", type=str,   default="graph/liver_toy")
    ap.add_argument("-obj",   type=str,   default="graph/liver_toy/domain.obj")
    ap.add_argument("-n",     type=int,   default=40)
    ap.add_argument("-beta",  type=float, default=5.0,
                    help="Nitsche penalty on the 1D inlet BC")
    ap.add_argument("-tol",   type=float, default=1e-2,
                    help="tolerance on the DISCRETIZATION errors (bounds, mass). "
                         "These are not algebraic identities: they shrink as -n "
                         "grows. 1e-2 is honest for a mesh with h >> R.")
    ap.add_argument("-sweep", type=str, default=None,
                    choices=["beta", "n", "both"],
                    help="beta: is the u1d overshoot just weak Nitsche? "
                         "n: do the errors vanish under refinement (the decisive "
                         "test)? both: run each in turn.")
    ap.add_argument("-out",   type=str, default=None,
                    help="path prefix for the result files (.txt + .json). "
                         "Default: ./test_solution/<name>_physics")
    args = ap.parse_args()

    out = args.out or f"./test_solution/{args.name}_physics"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    cco = CCOVascularMesh(graph_folder=args.graph, obj_path=args.obj,
                          name=args.name)
    cco.load().build().export_xdmf()
    boundary = boundary_from_obj(obj_path=args.obj, scale=cco.scale,
                                 center=cco.center)

    if args.sweep:
        beta_rows = sweep_beta(boundary, args) if args.sweep in ("beta", "both") else []
        n_rows    = sweep_n(boundary, args)    if args.sweep in ("n", "both")    else []
        write_report(out, args, None, beta_rows, n_rows)
        return 0

    s = run(boundary, args.name, args.n, args.beta)

    u3d = s.u3d.vector().get_local()
    u1d = s.u1d.vector().get_local()

    print("\n" + "=" * 70)
    print(f"PHYSICAL CHECKS  (n={args.n}, beta={args.beta}, "
          f"sigma3d={s.sigma3d}, kappa={s.kappa})")
    print("=" * 70)

    r = [
        ("bounds",         check_bounds(s, u3d, u1d, args.tol)),
        ("monotonicity",   check_monotone(s, u1d, args.tol)),
        ("mass balance",   check_mass(s, u3d, u1d, args.tol)),
        ("resolution",     check_resolution(s)),
    ]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in r:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    hard = [n for n, ok in r[:3] if not ok]   # 4 is a warning, not a failure
    if hard:
        print(f"\n  Failing: {', '.join(hard)}.")
        print( "  Whether these are discretization error or a real defect is decided")
        print( "  by refinement, not by one run. Run:  -sweep both")
    else:
        print("\n  The solution obeys bounds, monotonicity and mass conservation.")

    write_report(out, args, metrics(s), [], [])
    return 0 if not hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
