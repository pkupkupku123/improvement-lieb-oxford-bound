#!/usr/bin/env python3
"""Reproduce the three-panel figure from the bundled exact certificates.

Requires Python >= 3.9, NumPy and Matplotlib. No TeX installation is needed.
Run from the package root: python reproduce.py figure
All defaults are relative to this script, independent of the working directory.
The certificate's `nu` is a storage placeholder equal to mu: the signed
single-channel auxiliary measure must instead be reconstructed as mu + sigma_1.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

HERE = Path(__file__).resolve().parent
COLORS = {1: "#0000FF", 2: "#FF0000", 3: "#008000", 5: "#000000"}
STYLES = {1: "-", 2: "--", 3: "-.", 5: ":"}
MARKERS = {1: "s", 2: "o", 3: "^", 5: "D"}


def read_certificate(path, channels, expected_hash):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"Certificate checksum mismatch: {path}")
    d = json.loads(raw)
    if d["format"] != "covariance-lo-linear-reciprocal-v1":
        raise ValueError("Expected a covariance certificate")
    if len(d["covariance_channels"]) != channels or d["nu"] != d["mu"]:
        raise ValueError("Unexpected channel representation")
    if sum(d["mu"]) != d["mass_scale"] or min(d["mu"]) < 0:
        raise ValueError("Invalid mu normalization")
    if any(sum(z) != 0 for z in d["covariance_channels"]):
        raise ValueError("A channel has nonzero total charge")
    r = np.asarray(d["radii"], dtype=float) / d["coordinate_scale"]
    p = np.asarray(d["mu"], dtype=float) / d["mass_scale"]
    z = np.asarray(d["covariance_channels"], dtype=float) / d["mass_scale"]
    a = np.asarray(d["edges"], dtype=float) / d["coordinate_scale"]
    h = np.asarray(d["allocation"], dtype=float) / d["allocation_scale"]
    if not (np.all(np.diff(r) > 0) and np.all(np.diff(a) > 0)):
        raise ValueError("Radii and interpolation nodes must be increasing")
    if len(a) != len(h) or np.any(h <= 0):
        raise ValueError("Invalid control function nodes")
    # Appendix C: W_ij = p_i p_j - sum_l z_li z_lj.
    wpos = np.maximum(np.outer(p, p) - z.T @ z, 0.0)
    return dict(raw=d, r=r, p=p, a=a, h=h,
                origin_P=wpos.sum(), tail_H=27 / 128 * (wpos @ r**3).sum())


def control_function(a, profile):
    """f(a)=a^3 h(a)/2; h is linear in 1/a, with Appendix C extensions.

    Conversion to floating point is for visualization only. This function is
    not a replacement for the exact-arithmetic feasibility verifier.
    """
    a = np.asarray(a, dtype=float)
    if np.any(a < 0):
        raise ValueError("a must be nonnegative")
    nodes, values = profile["a"], profile["h"]
    h = np.empty_like(a)
    low, high = a < nodes[0], a > nodes[-1]
    middle = ~(low | high)
    h[low] = profile["origin_P"] * a[low]**3
    h[high] = profile["tail_H"]
    h[middle] = np.interp(1 / a[middle], 1 / nodes[::-1], values[::-1])
    return a**3 * h / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--output-dir", type=Path, default=HERE.parent / "results" / "figures")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3, 5], choices=[1, 2, 3, 5])
    ap.add_argument("--a-min", type=float, default=0.0)
    ap.add_argument("--a-max", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--nu-zoom-x", type=float, nargs=2, default=[0.04, 0.36],
                    metavar=("MIN", "MAX"), help="Linear inset radius limits")
    ap.add_argument("--nu-zoom-y", type=float, nargs=2, default=[-0.35, 0.10],
                    metavar=("MIN", "MAX"), help="Linear inset signed-weight limits")
    args = ap.parse_args()
    if not 0 <= args.a_min < args.a_max:
        ap.error("Require 0 <= a-min < a-max")
    if not (0 <= args.nu_zoom_x[0] < args.nu_zoom_x[1]
            and args.nu_zoom_y[0] < args.nu_zoom_y[1]):
        ap.error("Invalid inset limits")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((args.data_dir / "provenance.json").read_text())
    profiles = {m: read_certificate(args.data_dir / f"m{m}" / "certificate.json", m,
                provenance[str(m)]["sha256"]) for m in sorted(set(args.channels) | {1})}

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        # Computer Modern matches the manuscript's LaTeX mathematics.
        # Ordinary text retains the reference figure's sans-serif styling.
        "mathtext.fontset": "cm",
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 1.1, "lines.linewidth": 1.35,
        "legend.fontsize": 8, "legend.frameon": True,
        "legend.fancybox": False, "legend.edgecolor": "0.4",
        "legend.facecolor": "white", "legend.framealpha": 1,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), layout="constrained")
    ax_mu, ax_f, ax_nu = axes
    report = {}
    for m in args.channels:
        d = profiles[m]
        nz = d["p"] > 0
        # Exact shell masses: connecting segments are guides to the eye only.
        ax_mu.plot(d["r"][nz], d["p"][nz], color=COLORS[m], ls=STYLES[m],
                   marker=MARKERS[m], ms=2.6, lw=0.85, mew=0,
                   mfc=COLORS[m], label=fr"$m={m}$")
        sample = np.unique(np.r_[np.linspace(args.a_min, args.a_max, 2000),
                    d["a"][(d["a"] >= args.a_min) & (d["a"] <= args.a_max)]])
        f = control_function(sample, d)
        ax_f.plot(sample, f, color=COLORS[m], ls=STYLES[m], label=fr"$m={m}$")
        np.savetxt(args.output_dir / f"mu_m{m}.csv", np.c_[d["r"][nz], d["p"][nz]],
                   delimiter=",", header="radius,shell_weight", comments="", fmt="%.15g")
        np.savetxt(args.output_dir / f"f_m{m}.csv", np.c_[sample, f], delimiter=",",
                   header="a,f", comments="", fmt="%.15g")
        # Verify the plotted reconstruction against every stored node.
        np.testing.assert_allclose(control_function(d["a"], d), d["a"]**3 * d["h"] / 2,
                                   rtol=2e-14, atol=0)
        report[str(m)] = dict(mu_shells=int(nz.sum()), mu_mass=float(d["p"].sum()),
                             claimed_constant=d["raw"]["claimed_constant"])
    ax_mu.set(xlabel=r"Shell radius $r$", ylabel=r"Shell weight $p_i$",
              title=r"(a) $\mu$: shell weights", xlim=(0, 0.52), ylim=(0, None))
    ax_mu.legend(loc="upper left", handlelength=1.8)
    ax_mu.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_f.set(xlabel=r"$a$", ylabel=r"$f(a)$", title=r"(b) Control function $f$",
             xlim=(args.a_min, args.a_max), ylim=(0, None))
    ax_f.legend(loc="upper left", handlelength=2.2)
    ax_f.yaxis.set_major_locator(MaxNLocator(5))

    d = profiles[1]
    # Integer addition preserves the exact support and sign before conversion.
    q_int = [p + z for p, z in zip(d["raw"]["mu"], d["raw"]["covariance_channels"][0])]
    if sum(q_int) != d["raw"]["mass_scale"]:
        raise ValueError("Signed nu is not normalized")
    q = np.asarray(q_int, dtype=float) / d["raw"]["mass_scale"]
    for mask, color, marker, label in [(q > 0, COLORS[1], "s", r"$q_i>0$"),
                                       (q < 0, COLORS[2], "o", r"$q_i<0$")]:
        ax_nu.vlines(d["r"][mask], 0, q[mask], color=color, lw=0.85)
        ax_nu.scatter(d["r"][mask], q[mask], color=color, marker=marker, s=9,
                      linewidths=0, label=label, zorder=3)
    ax_nu.axhline(0, color="black", lw=1.0, ls=(0, (5, 4)))
    ax_nu.set(xlabel=r"Shell radius $r$", ylabel=r"Signed shell weight $q_i$",
              title=r"(c) $\nu=\mu+\sigma_1$: $m=1$", xlim=(0, 1.05))
    ax_nu.legend(loc="upper right", handletextpad=0.2)
    ax_zoom = ax_nu.inset_axes([0.36, 0.44, 0.60, 0.34])
    for mask, color, marker in [(q > 0, COLORS[1], "s"), (q < 0, COLORS[2], "o")]:
        ax_zoom.vlines(d["r"][mask], 0, q[mask], color=color, lw=0.65)
        ax_zoom.scatter(d["r"][mask], q[mask], color=color, marker=marker,
                        s=5, linewidths=0, zorder=3)
    ax_zoom.axhline(0, color="black", lw=0.65, ls=(0, (5, 4)))
    ax_zoom.set(xlim=args.nu_zoom_x, ylim=args.nu_zoom_y)
    ax_zoom.set_xlabel(r"$r$", fontsize=8, labelpad=0)
    ax_zoom.set_ylabel(r"$q_i$", fontsize=8, labelpad=0)
    ax_zoom.tick_params(which="major", labelsize=6.5, length=2.5, width=0.7, pad=1)
    ax_zoom.tick_params(which="minor", length=1.5, width=0.5)
    ax_zoom.xaxis.set_major_locator(MaxNLocator(3))
    ax_zoom.yaxis.set_major_locator(MaxNLocator(3))
    for spine in ax_zoom.spines.values():
        spine.set_linewidth(0.7)
    ax_nu.indicate_inset_zoom(ax_zoom, edgecolor="0.45", linewidth=0.65, alpha=0.8)
    np.savetxt(args.output_dir / "nu_m1.csv", np.c_[d["r"][q != 0], q[q != 0]],
               delimiter=",", header="radius,signed_shell_weight", comments="", fmt="%.15g")
    report["signed_nu"] = dict(shells=int(np.count_nonzero(q)), negative_shells=int((q < 0).sum()),
                               mass=float(q.sum()), total_variation=float(abs(q).sum()))
    report["nu_inset"] = dict(xlim=args.nu_zoom_x, ylim=args.nu_zoom_y, scale="linear")
    for ax in [*axes, ax_zoom]:
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="both", top=False, right=False)
        ax.grid(False)
        ax.set_axisbelow(True)
    for suffix in ("pdf", "svg", "png"):
        path = args.output_dir / f"optimized_channel_profiles.{suffix}"
        fig.savefig(path, dpi=args.dpi)
        print(path)
    plt.close(fig)
    (args.output_dir / "plot_checks.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
