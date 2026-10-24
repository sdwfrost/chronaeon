#!/usr/bin/env python3
"""
benchmark_timetree_vs_phylodate.py
----------------------------------
Benchmarks the new ChronAeon TimeTree (`chronaeon.timetree.date_timetree`)
both in tree-free sequence mode (`chronaeon_timetree_seq`) and tree-guided mode
(`chronaeon_timetree_tree`), alongside analytical `chronaeon_ols` and `phylodate`
methods across:
  1. Synthetic simulations (Strict, Uncorrelated Lognormal, Autocorrelated Brownian),
     including internal node date accuracy and 95% Fieller CI coverage.
  2. 12 Empirical BEAST viral outbreak cohorts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "phylodate" / "benches"))

from aeon_core.dataset import parse_beast_xml
from chronaeon.dating_io import compute_time_decay_profile_divergences
from chronaeon.dating_models import run_ols_dating
from chronaeon.timetree import date_timetree
from ci_calibration import simulate
import phylodate
from phylodate.treedater import parse_newick_ape

COMP_DATA_DIR = ROOT_DIR / "chronaeon_comparison" / "data"
TREES_DIR = COMP_DATA_DIR / "trees"


def evolve_sequences_exact(nwk: str, n_tips: int, L: int, rng: np.random.Generator) -> dict[str, str]:
    """
    Evolve nucleotide sequences along simulated tree `nwk` where branch length `b`
    is the exact substitution count divided by `L`. Each mutation event transitions
    or transverts to one of the 3 distinct alternative nucleotides.
    """
    tips, e1, e2, el = parse_newick_ape(nwk)
    root = n_tips + 1
    blen_map = {c: (b if b is not None else 0.0) for c, b in zip(e2, el)}
    children_map: dict[int, list[int]] = {}
    for p, c in zip(e1, e2):
        children_map.setdefault(p, []).append(c)

    bases = np.array(["A", "C", "G", "T"])
    node_seqs = {root: rng.integers(0, 4, size=L)}

    def _traverse(u: int):
        for ch in children_map.get(u, []):
            b = blen_map.get(ch, 0.0)
            n_mut = int(round(b * L))
            seq = node_seqs[u].copy()
            if n_mut > 0:
                positions = rng.choice(L, size=min(n_mut, L), replace=False)
                shifts = rng.integers(1, 4, size=len(positions))
                seq[positions] = (seq[positions] + shifts) % 4
            node_seqs[ch] = seq
            _traverse(ch)

    _traverse(root)
    return {tips[i - 1]: "".join(bases[node_seqs[i]]) for i in range(1, n_tips + 1)}


def run_synthetic_benchmark(n_reps: int = 15, n_tips: int = 35, L: int = 1200, mu: float = 0.001):
    rng = np.random.default_rng(20261024)
    regimes = {
        "strict": {},
        "uncorrelated": {"sigma": 0.4},
        "autocorrelated": {"nu": 0.002},
    }
    phylodate_methods = ["lsd2", "treetime", "treedater", "reltime", "node_dating", "chronumental"]
    raw_rows = []

    print("\n=====================================================================")
    print("[*] Running Synthetic TimeTree Benchmark (Corrected Sequence Evolution)")
    print("=====================================================================")

    for reg_name, kw in regimes.items():
        for rep in range(n_reps):
            nwk, dates, true_t = simulate(n_tips, mu, L, rng, mean_dur=10.0, **kw)
            seqs = evolve_sequences_exact(nwk, n_tips, L, rng)
            tips = list(seqs.keys())
            times = np.array([dates[t] for t in tips], dtype=np.float64)

            # 1. ChronAeon TimeTree (Sequence-Only / Tree-Free Input)
            t0 = time.perf_counter()
            tt_seq = date_timetree(seqs, dates, seq_length=L, ci=True)
            dt_seq = time.perf_counter() - t0
            cov_seq = bool(tt_seq.tmrca_ci[0] <= true_t <= tt_seq.tmrca_ci[1]) if tt_seq.tmrca_ci else None
            w_seq = float(tt_seq.tmrca_ci[1] - tt_seq.tmrca_ci[0]) if (tt_seq.tmrca_ci and np.isfinite(tt_seq.tmrca_ci[0])) else None
            raw_rows.append({
                "regime": reg_name, "rep": rep, "method": "chronaeon_timetree_seq",
                "est_tmrca": tt_seq.tmrca, "tmrca_error": tt_seq.tmrca - true_t,
                "est_rate": tt_seq.rate, "rate_error": tt_seq.rate - mu,
                "ci_covered": cov_seq, "ci_width": w_seq, "runtime_s": dt_seq,
            })

            # 2. ChronAeon TimeTree (Tree-Guided Input)
            t0 = time.perf_counter()
            tt_tree = date_timetree(nwk, dates, seq_length=L, ci=True)
            dt_tree = time.perf_counter() - t0
            cov_tree = bool(tt_tree.tmrca_ci[0] <= true_t <= tt_tree.tmrca_ci[1]) if tt_tree.tmrca_ci else None
            w_tree = float(tt_tree.tmrca_ci[1] - tt_tree.tmrca_ci[0]) if (tt_tree.tmrca_ci and np.isfinite(tt_tree.tmrca_ci[0])) else None
            raw_rows.append({
                "regime": reg_name, "rep": rep, "method": "chronaeon_timetree_tree",
                "est_tmrca": tt_tree.tmrca, "tmrca_error": tt_tree.tmrca - true_t,
                "est_rate": tt_tree.rate, "rate_error": tt_tree.rate - mu,
                "ci_covered": cov_tree, "ci_width": w_tree, "runtime_s": dt_tree,
            })

            # 3. Analytical ChronAeon OLS (Fast CPU Mode 1)
            t0 = time.perf_counter()
            d_prof, _ = compute_time_decay_profile_divergences(seqs, dates, tips)
            ols_res = run_ols_dating(times, d_prof)
            dt_ols = time.perf_counter() - t0
            ols_ci = ols_res.get("ci_mrca")
            cov_ols = bool(ols_ci[0] <= true_t <= ols_ci[1]) if ols_ci else None
            w_ols = float(ols_ci[1] - ols_ci[0]) if (ols_ci and np.isfinite(ols_ci[0])) else None
            raw_rows.append({
                "regime": reg_name, "rep": rep, "method": "chronaeon_ols",
                "est_tmrca": float(ols_res["t_mrca"]), "tmrca_error": float(ols_res["t_mrca"]) - true_t,
                "est_rate": float(ols_res["mu"]), "rate_error": float(ols_res["mu"]) - mu,
                "ci_covered": cov_ols, "ci_width": w_ols, "runtime_s": dt_ols,
            })

            # 4. Phylodate methods
            for m in phylodate_methods:
                t0 = time.perf_counter()
                try:
                    r = phylodate.date(nwk, dates, method=m, seq_length=L, ci=(m in ("reltime", "node_dating")))
                    dt_m = time.perf_counter() - t0
                    cov_m = bool(r.tmrca_ci[0] <= true_t <= r.tmrca_ci[1]) if r.tmrca_ci else None
                    w_m = float(r.tmrca_ci[1] - r.tmrca_ci[0]) if (r.tmrca_ci and np.isfinite(r.tmrca_ci[0])) else None
                    raw_rows.append({
                        "regime": reg_name, "rep": rep, "method": m,
                        "est_tmrca": float(r.tmrca), "tmrca_error": float(r.tmrca) - true_t,
                        "est_rate": float(r.rate) if r.rate is not None else None,
                        "rate_error": (float(r.rate) - mu) if r.rate is not None else None,
                        "ci_covered": cov_m, "ci_width": w_m, "runtime_s": dt_m,
                    })
                except Exception:
                    pass

        print(f"  [✓] Completed regime: {reg_name}")

    df_raw = pd.DataFrame(raw_rows)
    summary_rows = []
    for (regime, method), grp in df_raw.groupby(["regime", "method"]):
        terrs = grp["tmrca_error"].dropna()
        rerrs = grp["rate_error"].dropna()
        covs = grp["ci_covered"].dropna()
        ws = grp["ci_width"].dropna()
        rts = grp["runtime_s"].dropna()
        summary_rows.append({
            "regime": regime,
            "method": method,
            "n_reps": len(terrs),
            "tmrca_bias": float(terrs.mean()),
            "tmrca_rmse": float(np.sqrt((terrs ** 2).mean())),
            "tmrca_mae": float(terrs.abs().median()),
            "rate_bias": float(rerrs.mean()) if len(rerrs) else None,
            "rate_rmse": float(np.sqrt((rerrs ** 2).mean())) if len(rerrs) else None,
            "ci_coverage_pct": float(covs.mean() * 100.0) if len(covs) else None,
            "ci_mean_width": float(ws.mean()) if len(ws) else None,
            "mean_runtime_ms": float(rts.mean() * 1000.0),
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(COMP_DATA_DIR / "timetree_synthetic_comparison.csv", index=False)
    print("\n--- Synthetic Benchmark Summary ---")
    print(df_sum.to_string(index=False))
    return df_sum


def run_empirical_benchmark():
    with open(COMP_DATA_DIR / "empirical_results.json") as f:
        records = json.load(f)

    emp_rows = []
    print("\n=====================================================================")
    print("[*] Running Empirical TimeTree Benchmark Across 12 BEAST Cohorts")
    print("=====================================================================")

    for rec in records:
        c_id = rec["cohort_id"]
        xml_path = ROOT_DIR / "chronaeon" / "data" / c_id / "beast.xml.gz"
        tree_path = TREES_DIR / f"{c_id}.nwk"
        bdata = parse_beast_xml(str(xml_path))
        seqs = {k.replace("?", "_"): v for k, v in bdata["sequences"].items()}
        dates = {k.replace("?", "_"): v for k, v in bdata["dates"].items()}
        min_len = min(len(s) for s in seqs.values())
        seqs = {k: v[:min_len] for k, v in seqs.items()}

        # 1. ChronAeon TimeTree directly from sequences (no IQ-TREE needed!)
        t0 = time.perf_counter()
        tt_seq = date_timetree(seqs, dates, seq_length=min_len, ci=True, autoclock=True)
        dt_seq = time.perf_counter() - t0

        # 2. ChronAeon TimeTree on IQ-TREE ML phylogeny
        t0 = time.perf_counter()
        tt_tree = date_timetree(tree_path, dates, seq_length=min_len, ci=True)
        dt_tree = time.perf_counter() - t0

        rec["chronaeon_timetree_seq"] = {
            "tmrca": tt_seq.tmrca,
            "ci_lower": tt_seq.tmrca_ci[0] if tt_seq.tmrca_ci else None,
            "ci_upper": tt_seq.tmrca_ci[1] if tt_seq.tmrca_ci else None,
            "rate": tt_seq.rate,
            "rate_ci_lower": tt_seq.rate_ci[0] if tt_seq.rate_ci else None,
            "rate_ci_upper": tt_seq.rate_ci[1] if tt_seq.rate_ci else None,
            "runtime_s": dt_seq,
            "n_eff": tt_seq.n_eff,
            "fieller_g": tt_seq.fieller_g,
            "outliers": tt_seq.outliers,
        }
        rec["chronaeon_timetree_tree"] = {
            "tmrca": tt_tree.tmrca,
            "ci_lower": tt_tree.tmrca_ci[0] if tt_tree.tmrca_ci else None,
            "ci_upper": tt_tree.tmrca_ci[1] if tt_tree.tmrca_ci else None,
            "rate": tt_tree.rate,
            "rate_ci_lower": tt_tree.rate_ci[0] if tt_tree.rate_ci else None,
            "rate_ci_upper": tt_tree.rate_ci[1] if tt_tree.rate_ci else None,
            "runtime_s": dt_tree,
            "n_eff": tt_tree.n_eff,
            "fieller_g": tt_tree.fieller_g,
        }

        p_tt = rec.get("phylodate", {}).get("treetime", {})
        p_lsd = rec.get("phylodate", {}).get("lsd2", {})

        emp_rows.append({
            "cohort_id": c_id,
            "taxa": rec["taxa"],
            "sites": min_len,
            "beast_tmrca": rec.get("beast_tmrca"),
            "beast_ci_lower": rec.get("beast_ci_lower"),
            "beast_ci_upper": rec.get("beast_ci_upper"),
            "chronaeon_manifold_tmrca": rec["chronaeon"].get("tmrca"),
            "chronaeon_timetree_seq_tmrca": tt_seq.tmrca,
            "chronaeon_timetree_seq_ci_lower": tt_seq.tmrca_ci[0] if tt_seq.tmrca_ci else None,
            "chronaeon_timetree_seq_ci_upper": tt_seq.tmrca_ci[1] if tt_seq.tmrca_ci else None,
            "chronaeon_timetree_seq_rate": tt_seq.rate,
            "chronaeon_timetree_seq_runtime_s": dt_seq,
            "chronaeon_timetree_tree_tmrca": tt_tree.tmrca,
            "chronaeon_timetree_tree_rate": tt_tree.rate,
            "chronaeon_timetree_tree_runtime_s": dt_tree,
            "phylodate_treetime_tmrca": p_tt.get("tmrca"),
            "phylodate_treetime_rate": p_tt.get("rate"),
            "phylodate_lsd2_tmrca": p_lsd.get("tmrca"),
            "phylodate_lsd2_rate": p_lsd.get("rate"),
            "iqtree_build_time_s": rec.get("iqtree_build_time_s"),
        })
        print(f"  [✓] {c_id:35s}: BEAST={rec.get('beast_tmrca')} | ChronTT(Seq)={tt_seq.tmrca:.2f} ({dt_seq*1000:.0f}ms) | ChronTT(Tree)={tt_tree.tmrca:.2f} ({dt_tree*1000:.1f}ms)")

    with open(COMP_DATA_DIR / "empirical_results.json", "w") as f:
        json.dump(records, f, indent=2)

    df_emp = pd.DataFrame(emp_rows)
    df_emp.to_csv(COMP_DATA_DIR / "timetree_empirical_comparison.csv", index=False)
    return df_emp


if __name__ == "__main__":
    run_synthetic_benchmark()
    run_empirical_benchmark()
