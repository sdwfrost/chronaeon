"""
evaluate_skyline_comparison.py
------------------------------
Quantifies and visualizes how coalescent skyline plots (N_e(t)) and
Lineages-Through-Time (LTT) curves differ between:
  - ChronAeon TimeTree from raw sequences (`chronaeon_timetree_seq`)
  - ChronAeon TimeTree on ML topology (`chronaeon_timetree_tree`)
  - Phylodate methods (`lsd2`, `treetime`, `treedater`, `reltime`,
    `node_dating`, `wlogdate`, `chronumental`, `chronos`)

Evaluates:
  1. Synthetic heterochronous coalescent simulations with known ground-truth
     effective population size trajectories N_e(t):
       (a) Constant population size: N_e(t) = 15.0
       (b) Exponential epidemic expansion: N_e(t) = N_0 * exp(r * (t - t_0))
       (c) Epidemic boom-bust wave (Gaussian peak in log N_e(t))
  2. Empirical viral outbreak datasets from the BEAST benchmark:
       - Rabies Skygrid (Gill et al. 2020)
       - Ebola Sierra Leone (Gire et al. 2014)
       - Chikungunya Brazil (Naveca et al. 2019)
       - Dengue Polyepoch (Suchard et al. 2020)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "chronaeon" / "src"))
sys.path.insert(0, str(ROOT / "phylodate" / "python"))

from aeon_core.dataset import parse_beast_xml
from chronaeon.timetree import (
    _all_nodes_to_tips_distances,
    _distance_space_soft_profile_divergences,
    _find_optimal_chronaeon_root,
    _parse_newick_adj,
    date_timetree,
)
import phylodate


def root_substitution_newick(nwk: str, tip_dates: Dict[str, float]) -> str:
    """Roots an unrooted ML substitution Newick using ChronAeon's continuous edge rooter."""
    tips, adj, _ = _parse_newick_adj(nwk)
    n_tips = len(tips)
    times = np.array([tip_dates[t] for t in tips], dtype=np.float64)
    dist_all = _all_nodes_to_tips_distances(adj, n_tips)
    D = dist_all[:n_tips, :n_tips].copy()
    d_prof, _ = _distance_space_soft_profile_divergences(D, times)
    root_id, children, blen, _, _ = _find_optimal_chronaeon_root(
        adj, n_tips, times, d_prof, respect_existing_root=False
    )

    def _emit(u: int) -> str:
        if u < n_tips:
            return tips[u]
        return "(" + ",".join(f"{_emit(v)}:{blen.get(v, 0.0):.6f}" for v in children.get(u, [])) + ")"

    return _emit(root_id) + ";"


# =========================================================================
# 1. Dated Newick Parser -> Exact Tip & Internal Node Dates
# =========================================================================

def extract_tree_node_times(
    newick: str,
    tip_dates: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses a dated rooted Newick tree (where branch lengths are in time units)
    and returns:
      - coal_times: sorted array of internal node (coalescent) calendar times (len N-1)
      - sample_times: sorted array of tip sampling calendar times (len N)
      - internal_branch_taus: array of all branch durations tau_e >= 0
    Anchors the absolute calendar time scale so that the mean tip time matches
    mean(tip_dates).
    """
    s = re.sub(r"\[&[^\]]*\]", "", newick).strip()
    if s.endswith(";"):
        s = s[:-1].strip()

    tokens = re.findall(r"\(|\)|,|:[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[^(),:;]+", s)

    stack: List[int] = []
    children: Dict[int, List[Tuple[int, float]]] = {}
    leaf_names: Dict[int, str] = {}
    next_id = 0
    curr_node = None

    i = 0
    n_tok = len(tokens)
    while i < n_tok:
        tok = tokens[i]
        if tok == "(":
            nid = next_id
            next_id += 1
            children[nid] = []
            stack.append(nid)
            i += 1
        elif tok == ",":
            i += 1
        elif tok == ")":
            curr_node = stack.pop()
            i += 1
            if i < n_tok and not tokens[i].startswith(":") and tokens[i] not in ("(", ")", ","):
                i += 1
            bl = 0.0
            if i < n_tok and tokens[i].startswith(":"):
                bl = max(0.0, float(tokens[i][1:]))
                i += 1
            if stack:
                children[stack[-1]].append((curr_node, bl))
        else:
            name = tok.strip().strip("'\"")
            nid = next_id
            next_id += 1
            leaf_names[nid] = name
            i += 1
            bl = 0.0
            if i < n_tok and tokens[i].startswith(":"):
                bl = max(0.0, float(tokens[i][1:]))
                i += 1
            if stack:
                children[stack[-1]].append((nid, bl))
            else:
                curr_node = nid

    root_id = curr_node if curr_node is not None else 0

    # Preorder traversal from root: compute relative depth from root
    rel_time: Dict[int, float] = {root_id: 0.0}
    taus: List[float] = []
    visit_stack = [root_id]
    while visit_stack:
        u = visit_stack.pop()
        for v, bl in children.get(u, []):
            rel_time[v] = rel_time[u] + bl
            taus.append(bl)
            if v in children:
                visit_stack.append(v)

    # Align relative depth to calendar dates using matched tips
    offsets = []
    for nid, name in leaf_names.items():
        if name in tip_dates:
            offsets.append(float(tip_dates[name]) - rel_time[nid])
    t_root = float(np.median(offsets)) if offsets else 0.0

    coal_times = []
    for u, ch_list in children.items():
        mult = max(1, len(ch_list) - 1)
        for _ in range(mult):
            coal_times.append(t_root + rel_time[u])
    coal_times.sort()

    sample_times = sorted(
        float(tip_dates.get(name, t_root + rel_time[nid]))
        for nid, name in leaf_names.items()
    )
    return np.array(coal_times, dtype=np.float64), np.array(sample_times, dtype=np.float64), np.array(taus, dtype=np.float64)


def extract_method_coal_and_sample_times(
    method: str,
    res,
    tip_dates: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extracts sorted internal coalescent calendar times (c_1 <= ... <= c_{N-1}),
    sorted tip sampling calendar times (s_1 <= ... <= s_N), and the fraction of
    collapsed/zero-duration coalescent waiting intervals (dc <= 1e-4 yr) across
    all ChronAeon and Phylodate methods.
    """
    sample_times = np.sort(np.array(list(tip_dates.values()), dtype=np.float64))

    if method in ("chronaeon_timetree_seq", "chronaeon_timetree_tree"):
        coals = [
            float(v)
            for k, v in res.node_dates.items()
            if k == "ROOT" or str(k).startswith("internal_")
        ]
        coals_sorted = np.sort(coals)
        zero_frac = float(np.mean(np.diff(coals_sorted) <= 1e-4)) if len(coals_sorted) > 1 else 0.0
    elif method == "lsd2":
        coals_arr, _, _ = extract_tree_node_times(res.raw.nexus, tip_dates)
        coals = list(coals_arr)
        coals_sorted = np.sort(coals)
        zero_frac = float(np.mean(np.diff(coals_sorted) <= 1e-4)) if len(coals_sorted) > 1 else 0.0
    elif method == "treetime":
        coals = [float(d) for k, d in res.raw["node_dates"].items() if k not in tip_dates]
        coals_sorted = np.sort(coals)
        zero_frac = float(np.mean(np.diff(coals_sorted) <= 1e-4)) if len(coals_sorted) > 1 else 0.0
    elif method == "treedater":
        coals_arr, _, taus = extract_tree_node_times(res.newick, tip_dates)
        coals = list(coals_arr)
        zero_frac = float(np.mean(taus <= 1e-4)) if len(taus) > 0 else 0.0
    elif method == "reltime":
        tips_set = set(res.raw["T"]["tips"])
        coals = [float(d) for nid, d in res.raw["times"].items() if nid not in tips_set]
        coals_sorted = np.sort(coals)
        zero_frac = float(np.mean(np.diff(coals_sorted) <= 1e-4)) if len(coals_sorted) > 1 else 0.0
    elif method == "node_dating":
        n_t = len(tip_dates)
        coals = [float(d) for d in res.raw[n_t:]]
        coals_sorted = np.sort(coals)
        zero_frac = float(np.mean(np.diff(coals_sorted) <= 1e-4)) if len(coals_sorted) > 1 else 0.0
    elif method == "wlogdate":
        coals = [float(x) for x in re.findall(r"t=([\d.eE+\-]+)", res.newick)]
        if not coals:
            coals_arr, _, _ = extract_tree_node_times(res.newick, tip_dates)
            coals = list(coals_arr)
        coals_sorted = np.sort(coals)
        zero_frac = float(np.mean(np.diff(coals_sorted) <= 1e-4)) if len(coals_sorted) > 1 else 0.0
    elif method == "chronumental":
        coals = [float(d) for k, d in res.raw["node_dates"].items() if k not in tip_dates]
        coals_sorted = np.sort(coals)
        zero_frac = float(np.mean(np.diff(coals_sorted) <= 1e-4)) if len(coals_sorted) > 1 else 0.0
    elif method == "chronos":
        max_t = max(tip_dates.values())
        n_t = int(res.raw["n"])
        coals = [float(max_t - a) for a in res.raw["ages"][n_t + 1:] if not math.isnan(a)]
        coals_sorted = np.sort(coals)
        zero_frac = float(np.mean(np.diff(coals_sorted) <= 1e-4)) if len(coals_sorted) > 1 else 0.0
    else:
        coals_arr, _, taus = extract_tree_node_times(res.newick, tip_dates)
        coals = list(coals_arr)
        zero_frac = float(np.mean(taus <= 1e-4)) if len(taus) > 0 else 0.0

    coal_arr = np.sort(np.array(coals, dtype=np.float64))
    return coal_arr, sample_times, zero_frac


# =========================================================================
# 2. Heterochronous LTT & Coalescent Skyline (Classic, Generalized, Grid)
# =========================================================================

def compute_ltt_and_skyline(
    coal_times: np.ndarray,
    sample_times: np.ndarray,
    epsilon: float = 0.25,
    grid_times: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes from sorted coalescent times (c_1 <= ... <= c_{N-1}) and tip sample
    times (s_1 <= ... <= s_N):
      1. Lineages-Through-Time (LTT) step function (ltt_t, ltt_k)
      2. Generalized heterochronous coalescent skyline (Strimmer & Pybus 2001 /
         Rodrigo & Felsenstein 1999) pooling coalescent intervals shorter than
         `epsilon`:
             N_e_hat(I_m) = (1 / r_m) * sum_{j in I_m} C(k_j, 2) * dt_j
      3. Piecewise-constant / kernel-smoothed skyline evaluated on `grid_times`.
    """
    # Events moving BACKWARD in time from most recent tip to root:
    #   - Tip sample at t: adds +1 lineage going backward in time
    #   - Coalescent event at t: removes -1 lineage going backward in time
    # Equivalently, moving FORWARD in time from root to tips:
    #   - Root starts at 1 lineage just before first split, then each coalescent
    #     adds +1 lineage, and each tip sample removes -1 lineage.
    events: List[Tuple[float, int]] = []
    for t in sample_times:
        events.append((float(t), +1))  # backward: sampling adds a lineage
    for t in coal_times:
        events.append((float(t), -1))  # backward: coalescence merges 2 -> 1

    # Sort backward in time (from max(t) down to min(t)); at identical timestamps,
    # process tips (+1) before coalescences (-1) so k >= 2 at coalescence.
    events.sort(key=lambda x: (-x[0], -x[1]))

    k = 0
    t_prev = events[0][0]

    # Build elementary intervals going backward in time
    elem_t_start = []
    elem_t_end = []
    elem_k = []
    elem_is_coal = []

    ltt_t = [t_prev]
    ltt_k = [0]

    for t_curr, delta in events:
        dt = max(0.0, t_prev - t_curr)
        is_coal = (delta == -1)
        if dt > 0.0 or is_coal:
            elem_t_start.append(t_prev)
            elem_t_end.append(t_curr)
            elem_k.append(max(k, 2 if is_coal else k))
            elem_is_coal.append(is_coal)
        k += delta
        t_prev = t_curr
        ltt_t.append(t_curr)
        ltt_k.append(max(0, k))

    # Group elementary intervals into generalized skyline windows with span >= epsilon
    # and at least 1 coalescent event per window
    win_mid_t = []
    win_t_hi = []
    win_t_lo = []
    win_ne = []

    acc_binom_dt = 0.0
    acc_dt = 0.0
    acc_coals = 0
    cur_t_hi = elem_t_start[0] if elem_t_start else 0.0

    for t_hi, t_lo, k_val, is_c in zip(elem_t_start, elem_t_end, elem_k, elem_is_coal):
        dt = max(0.0, t_hi - t_lo)
        binom = 0.5 * k_val * (k_val - 1) if k_val >= 2 else 0.0
        acc_binom_dt += binom * dt
        acc_dt += dt
        if is_c:
            acc_coals += 1

        if acc_coals >= 1 and acc_dt >= epsilon:
            ne_val = max(1e-3, acc_binom_dt / acc_coals)
            win_t_hi.append(cur_t_hi)
            win_t_lo.append(t_lo)
            win_mid_t.append(0.5 * (cur_t_hi + t_lo))
            win_ne.append(ne_val)
            acc_binom_dt = 0.0
            acc_dt = 0.0
            acc_coals = 0
            cur_t_hi = t_lo

    # Flush remainder if any coalescences remain
    if acc_coals >= 1:
        t_lo_last = elem_t_end[-1] if elem_t_end else cur_t_hi
        ne_val = max(1e-3, acc_binom_dt / max(acc_coals, 1))
        if ne_val <= 1e-3 and win_ne:
            ne_val = win_ne[-1]
        win_t_hi.append(cur_t_hi)
        win_t_lo.append(t_lo_last)
        win_mid_t.append(0.5 * (cur_t_hi + t_lo_last))
        win_ne.append(ne_val)

    win_mid_t_arr = np.array(win_mid_t[::-1], dtype=np.float64)
    win_ne_arr = np.array(win_ne[::-1], dtype=np.float64)

    # Evaluate on common grid_times if requested
    grid_ne = None
    grid_ltt = None
    if grid_times is not None:
        grid_ne = np.full(len(grid_times), np.nan, dtype=np.float64)
        if len(win_mid_t_arr) >= 2:
            # Interpolate in log-Ne space within [min_coal_t, max_sample_t]
            t_min_tree = float(np.min(coal_times))
            t_max_tree = float(np.max(sample_times))
            mask = (grid_times >= t_min_tree) & (grid_times <= t_max_tree)
            if np.any(mask):
                log_ne = np.interp(
                    grid_times[mask],
                    win_mid_t_arr,
                    np.log(np.maximum(win_ne_arr, 1e-3)),
                )
                grid_ne[mask] = np.exp(log_ne)
        elif len(win_mid_t_arr) == 1:
            t_min_tree = float(np.min(coal_times))
            t_max_tree = float(np.max(sample_times))
            mask = (grid_times >= t_min_tree) & (grid_times <= t_max_tree)
            grid_ne[mask] = win_ne_arr[0]

        # Evaluate LTT on grid_times: k(t) = (# coalescences <= t) + 1 - (# samples <= t)
        n_c = np.searchsorted(coal_times, grid_times, side="right")
        n_s = np.searchsorted(sample_times, grid_times, side="right")
        grid_ltt = np.where(grid_times < coal_times[0], 0.0, np.maximum(0, 1 + n_c - n_s)).astype(np.float64)

    return {
        "ltt_t": np.array(ltt_t[::-1], dtype=np.float64),
        "ltt_k": np.array(ltt_k[::-1], dtype=np.float64),
        "sky_t": win_mid_t_arr,
        "sky_ne": win_ne_arr,
        "grid_ne": grid_ne,
        "grid_ltt": grid_ltt,
    }


# =========================================================================
# 3. Heterochronous Coalescent Tree + Sequence Simulator
# =========================================================================

def simulate_heterochronous_coalescent(
    n_tips: int,
    seq_len: int,
    mu_true: float,
    ne_func,
    rng: np.random.Generator,
    t_min_sample: float = 2000.0,
    t_max_sample: float = 2020.0,
) -> Tuple[str, Dict[str, float], Dict[str, str], np.ndarray, np.ndarray]:
    """
    Simulates a serial-sample (heterochronous) Kingman coalescent tree backward in time
    under an arbitrary demographic trajectory `ne_func(t)` via thinning/time-rescaling,
    then simulates JC69 nucleotide sequences of length `seq_len` at rate `mu_true`.
    Returns:
      (sub_newick, tip_dates, seq_dict, true_coal_times, true_sample_times)
    """
    sample_times = np.sort(rng.uniform(t_min_sample, t_max_sample, size=n_tips))
    tip_names = [f"T{i:03d}" for i in range(n_tips)]
    tip_dates = {tip_names[i]: float(sample_times[i]) for i in range(n_tips)}

    # Backward simulation from t_max_sample
    pending_samples = [(float(sample_times[i]), i) for i in range(n_tips)]
    pending_samples.sort(key=lambda x: x[0])  # ascending, pop from end

    t_curr = pending_samples.pop()[0]
    active_nodes = [n_tips - 1]
    node_times: Dict[int, float] = {i: float(sample_times[i]) for i in range(n_tips)}
    children: Dict[int, List[int]] = {}
    next_node = n_tips
    coal_times = []

    dt_step = 0.02
    while len(active_nodes) > 1 or pending_samples:
        next_sample_t = pending_samples[-1][0] if pending_samples else -1e9
        k = len(active_nodes)
        if k <= 1:
            # Jump directly to next sample time
            t_next, s_id = pending_samples.pop()
            t_curr = t_next
            active_nodes.append(s_id)
            continue

        # Propose small step backward in time of size min(dt_step, t_curr - next_sample_t)
        step = min(dt_step, t_curr - next_sample_t) if t_curr > next_sample_t else dt_step
        if step <= 1e-9 and pending_samples:
            t_next, s_id = pending_samples.pop()
            t_curr = t_next
            active_nodes.append(s_id)
            continue

        ne_now = max(0.2, float(ne_func(t_curr - 0.5 * step)))
        rate_coal = 0.5 * k * (k - 1) / ne_now
        wait = float(rng.exponential(1.0 / rate_coal))
        if wait < step:
            t_curr -= wait
            # Coalesce two random active lineages
            chosen = rng.choice(len(active_nodes), size=2, replace=False)
            i1, i2 = sorted(chosen, reverse=True)
            u = active_nodes.pop(i1)
            v = active_nodes.pop(i2)
            w = next_node
            next_node += 1
            node_times[w] = t_curr
            children[w] = [u, v]
            active_nodes.append(w)
            coal_times.append(t_curr)
        else:
            t_curr -= step
            if pending_samples and abs(t_curr - pending_samples[-1][0]) < 1e-9:
                _, s_id = pending_samples.pop()
                active_nodes.append(s_id)

    root_id = active_nodes[0]

    # Simulate sequences along tree with Poisson mutations -> substitution branch lengths
    bases = np.array(["A", "C", "G", "T"])
    seqs_int: Dict[int, np.ndarray] = {root_id: rng.integers(0, 4, size=seq_len)}
    sub_blen: Dict[int, float] = {}

    stack = [root_id]
    while stack:
        u = stack.pop()
        for v in children.get(u, []):
            tau = max(0.0, node_times[v] - node_times[u])
            exp_subs = mu_true * tau
            p_mut = 0.75 * (1.0 - math.exp(-4.0 * exp_subs / 3.0))
            mut_mask = rng.random(seq_len) < p_mut
            n_mut = int(np.sum(mut_mask))
            child_seq = seqs_int[u].copy()
            if n_mut > 0:
                shifts = rng.integers(1, 4, size=n_mut)
                child_seq[mut_mask] = (child_seq[mut_mask] + shifts) % 4
            seqs_int[v] = child_seq
            p_hat = min(0.70, n_mut / float(seq_len))
            sub_blen[v] = max(1e-7, -0.75 * math.log(1.0 - 4.0 * p_hat / 3.0))
            if v in children:
                stack.append(v)

    def _to_nwk(u: int) -> str:
        if u < n_tips:
            return tip_names[u]
        parts = [f"{_to_nwk(ch)}:{sub_blen[ch]:.6f}" for ch in children[u]]
        return "(" + ",".join(parts) + ")"

    sub_newick = _to_nwk(root_id) + ";"
    seq_dict = {tip_names[i]: "".join(bases[seqs_int[i]]) for i in range(n_tips)}
    return (
        sub_newick,
        tip_dates,
        seq_dict,
        np.sort(np.array(coal_times, dtype=np.float64)),
        np.sort(sample_times),
    )


# =========================================================================
# 4. Run Synthetic & Empirical Skyline Comparison
# =========================================================================

def run_all():
    out_dir = ROOT / "chronaeon_comparison" / "data"
    fig_dir = ROOT / "chronaeon_comparison" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    methods_to_compare = [
        "chronaeon_timetree_seq",
        "chronaeon_timetree_tree",
        "lsd2",
        "treetime",
        "treedater",
        "reltime",
        "node_dating",
        "wlogdate",
        "chronumental",
    ]

    # Define 3 synthetic demographic scenarios
    scenarios = [
        (
            "constant",
            "Constant Demography ($N_e(t) = 15$)",
            lambda t: 15.0,
        ),
        (
            "exponential",
            "Exponential Epidemic Growth",
            lambda t: 2.0 * math.exp(0.14 * max(0.0, t - 1995.0)),
        ),
        (
            "boombust",
            "Epidemic Boom-Bust Wave",
            lambda t: 2.5 + 22.0 * math.exp(-0.5 * ((t - 2011.0) / 3.2) ** 2),
        ),
    ]

    n_reps = 8
    n_tips = 40
    seq_len = 1500
    mu_true = 1.5e-3

    syn_rows = []
    # Keep representative curves for plotting
    rep_curves = {}

    for scen_key, scen_title, ne_fn in scenarios:
        grid_t = np.linspace(1992.0, 2020.0, 140)
        true_ne_grid = np.array([ne_fn(t) for t in grid_t], dtype=np.float64)

        agg_ne = {m: [] for m in methods_to_compare}
        agg_ltt = {m: [] for m in methods_to_compare}
        true_ltt_list = []

        for rep in range(n_reps):
            rng = np.random.default_rng(7000 + 100 * len(scen_key) + rep)
            sub_nwk, tip_dates, seq_dict, true_coals, true_samples = simulate_heterochronous_coalescent(
                n_tips=n_tips,
                seq_len=seq_len,
                mu_true=mu_true,
                ne_func=ne_fn,
                rng=rng,
            )

            true_sky = compute_ltt_and_skyline(true_coals, true_samples, epsilon=0.6, grid_times=grid_t)
            true_ltt_list.append(true_sky["grid_ltt"])

            for m in methods_to_compare:
                try:
                    if m == "chronaeon_timetree_seq":
                        res = date_timetree(seq_dict, tip_dates, seq_length=seq_len, ci=False)
                    elif m == "chronaeon_timetree_tree":
                        res = date_timetree(sub_nwk, tip_dates, seq_length=seq_len, ci=False)
                    elif m == "wlogdate":
                        res = phylodate.date(sub_nwk, tip_dates, method=m, seq_length=seq_len, nrep=2)
                    else:
                        res = phylodate.date(sub_nwk, tip_dates, method=m, seq_length=seq_len)

                    c_times, s_times, zero_branch_frac = extract_method_coal_and_sample_times(m, res, tip_dates)
                    sky = compute_ltt_and_skyline(c_times, s_times, epsilon=0.6, grid_times=grid_t)

                    # Compare on overlapping support where both true tree and estimated tree exist
                    valid = (~np.isnan(sky["grid_ne"])) & (grid_t >= max(true_coals[0], c_times[0])) & (grid_t <= 2019.5)
                    if np.sum(valid) >= 5:
                        log_err = np.log10(np.maximum(sky["grid_ne"][valid], 1e-2)) - np.log10(true_ne_grid[valid])
                        log10_mae = float(np.mean(np.abs(log_err)))
                        log10_rmse = float(np.sqrt(np.mean(log_err ** 2)))
                        log10_bias = float(np.mean(log_err))
                    else:
                        log10_mae, log10_rmse, log10_bias = np.nan, np.nan, np.nan

                    # LTT L1 integrated error
                    ltt_mae = float(np.mean(np.abs(sky["grid_ltt"] - true_sky["grid_ltt"])))

                    # Compare quantile distribution of coalescent times vs truth
                    q_grid = np.linspace(0.0, 1.0, len(true_coals))
                    c_quant = np.quantile(c_times, q_grid)
                    coal_mae = float(np.mean(np.abs(c_quant - true_coals)))
                    tmrca_err = float(c_times[0] - true_coals[0])

                    syn_rows.append({
                        "scenario": scen_key,
                        "rep": rep,
                        "method": m,
                        "skyline_log10_mae": log10_mae,
                        "skyline_log10_rmse": log10_rmse,
                        "skyline_log10_bias": log10_bias,
                        "ltt_mae": ltt_mae,
                        "coal_times_mae": coal_mae,
                        "tmrca_err": tmrca_err,
                        "zero_branch_frac": zero_branch_frac,
                    })

                    agg_ne[m].append(sky["grid_ne"])
                    agg_ltt[m].append(sky["grid_ltt"])
                except Exception as exc:
                    print(f"[WARN] {scen_key} rep={rep} method={m}: {exc}")

        rep_curves[scen_key] = {
            "title": scen_title,
            "grid_t": grid_t,
            "true_ne": true_ne_grid,
            "true_ltt": np.nanmedian(np.vstack(true_ltt_list), axis=0),
            "agg_ne": {m: np.nanmedian(np.vstack(v), axis=0) for m, v in agg_ne.items() if v},
            "agg_ltt": {m: np.nanmedian(np.vstack(v), axis=0) for m, v in agg_ltt.items() if v},
        }

    df_syn = pd.DataFrame(syn_rows)
    df_syn.to_csv(out_dir / "skyline_synthetic_comparison.csv", index=False)

    summary_syn = (
        df_syn.groupby(["scenario", "method"])
        .agg(
            skyline_log10_mae=("skyline_log10_mae", "mean"),
            skyline_log10_rmse=("skyline_log10_rmse", "mean"),
            skyline_log10_bias=("skyline_log10_bias", "mean"),
            ltt_mae=("ltt_mae", "mean"),
            coal_times_mae=("coal_times_mae", "mean"),
            tmrca_mae=("tmrca_err", lambda s: float(np.mean(np.abs(s)))),
            zero_branch_frac=("zero_branch_frac", "mean"),
        )
        .reset_index()
    )
    print("\n=== SYNTHETIC SKYLINE & LTT BENCHMARK SUMMARY ===")
    print(summary_syn.to_string(index=False))

    # =====================================================================
    # Empirical Skyline Comparison on 4 Representative Viral Studies
    # =====================================================================
    emp_studies = [
        ("16_rabies_northamerica_biek2007", "Rabies Virus North America (Biek et al. 2007)", 1.5),
        ("00_ebola_sierraleone_gire2014", "Ebola Sierra Leone (Gire et al. 2014)", 0.04),
        ("03_chikungunya_brazil_naveca2019", "Chikungunya Brazil (Naveca et al. 2019)", 0.15),
        ("11_dengue_polyepoch_suchard2020", "Dengue Virus (Suchard et al. 2020)", 1.2),
    ]

    emp_methods = [
        "chronaeon_timetree_seq",
        "chronaeon_timetree_tree",
        "lsd2",
        "treetime",
        "treedater",
        "reltime",
        "wlogdate",
    ]

    emp_curves = {}
    emp_rows = []

    for study_id, study_label, eps_val in emp_studies:
        xml_path = ROOT / "chronaeon" / "data" / study_id / "beast.xml.gz"
        nwk_path = ROOT / "chronaeon_comparison" / "data" / "trees" / f"{study_id}.nwk"
        if not xml_path.exists() or not nwk_path.exists():
            continue
        parsed = parse_beast_xml(str(xml_path))
        raw_seqs = {k.replace("?", "_"): v for k, v in parsed["sequences"].items()}
        min_l = min(len(v) for v in raw_seqs.values())
        seq_dict = {k: v[:min_l] for k, v in raw_seqs.items()}
        tip_dates = {k.replace("?", "_"): float(v) for k, v in parsed["dates"].items() if v is not None}
        seq_len_emp = min_l
        iqtree_nwk = nwk_path.read_text().strip()
        rooted_iqtree_nwk = root_substitution_newick(iqtree_nwk, tip_dates)

        t_max = max(tip_dates.values())
        emp_curves[study_id] = {"label": study_label, "methods": {}}

        for m in emp_methods:
            try:
                if m == "chronaeon_timetree_seq":
                    res = date_timetree(seq_dict, tip_dates, seq_length=seq_len_emp, ci=False)
                elif m == "chronaeon_timetree_tree":
                    res = date_timetree(iqtree_nwk, tip_dates, seq_length=seq_len_emp, ci=False)
                elif m == "wlogdate":
                    res = phylodate.date(rooted_iqtree_nwk, tip_dates, method=m, seq_length=seq_len_emp, nrep=2)
                else:
                    res = phylodate.date(rooted_iqtree_nwk, tip_dates, method=m, seq_length=seq_len_emp)

                c_times, s_times, zero_branch_frac = extract_method_coal_and_sample_times(m, res, tip_dates)
                grid_emp = np.linspace(float(c_times[0]), float(t_max), 120)
                sky = compute_ltt_and_skyline(c_times, s_times, epsilon=eps_val, grid_times=grid_emp)
                emp_curves[study_id]["methods"][m] = {
                    "grid_t": grid_emp,
                    "grid_ne": sky["grid_ne"],
                    "ltt_t": sky["ltt_t"],
                    "ltt_k": sky["ltt_k"],
                    "tmrca": float(c_times[0]),
                    "median_ne": float(np.nanmedian(sky["grid_ne"])),
                    "zero_branch_frac": zero_branch_frac,
                }
                emp_rows.append({
                    "study_id": study_id,
                    "method": m,
                    "tmrca": float(c_times[0]),
                    "median_ne": float(np.nanmedian(sky["grid_ne"])),
                    "iqr_log10_ne": float(
                        np.nanpercentile(np.log10(np.maximum(sky["grid_ne"], 1e-3)), 75)
                        - np.nanpercentile(np.log10(np.maximum(sky["grid_ne"], 1e-3)), 25)
                    ),
                    "zero_branch_frac": zero_branch_frac,
                })
            except Exception as exc:
                print(f"[WARN] empirical {study_id} {m}: {exc}")

    df_emp = pd.DataFrame(emp_rows)
    df_emp.to_csv(out_dir / "skyline_empirical_comparison.csv", index=False)
    print("\n=== EMPIRICAL SKYLINE SUMMARY ===")
    print(df_emp.to_string(index=False))

    # =====================================================================
    # Generate White Paper Figures
    # =====================================================================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    palette = {
        "chronaeon_timetree_seq": "#D55E00",   # Vermilion (bold)
        "chronaeon_timetree_tree": "#0072B2",  # Deep Blue (bold)
        "lsd2": "#009E73",                     # Bluish green
        "treetime": "#CC79A7",                 # Reddish purple
        "treedater": "#E69F00",                # Orange
        "reltime": "#56B4E9",                  # Sky blue
        "wlogdate": "#7F7F7F",                 # Grey
        "node_dating": "#999933",              # Olive
    }
    labels_pretty = {
        "chronaeon_timetree_seq": "ChronAeon TimeTree (Seq, Tree-Free)",
        "chronaeon_timetree_tree": "ChronAeon TimeTree (ML Tree)",
        "lsd2": "Phylodate: LSD2",
        "treetime": "Phylodate: TreeTime",
        "treedater": "Phylodate: treedater",
        "reltime": "Phylodate: RelTime",
        "wlogdate": "Phylodate: wLogDate",
        "node_dating": "Phylodate: node.dating",
    }

    # Figure 1: 2x3 Synthetic Skyline (top row) & LTT (bottom row) across 3 regimes
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharex="col")
    plot_methods = ["chronaeon_timetree_seq", "chronaeon_timetree_tree", "lsd2", "treetime", "treedater", "reltime"]

    for col_i, (scen_key, scen_title, _) in enumerate(scenarios):
        rc = rep_curves[scen_key]
        gt = rc["grid_t"]
        ax_ne = axes[0, col_i]
        ax_ltt = axes[1, col_i]

        ax_ne.plot(gt, rc["true_ne"], color="black", lw=2.6, ls="--", label="Ground Truth $N_e(t)$", zorder=10)
        ax_ltt.plot(gt, rc["true_ltt"], color="black", lw=2.6, ls="--", label="Ground Truth LTT", zorder=10)

        for m in plot_methods:
            if m in rc["agg_ne"]:
                lw = 2.2 if m.startswith("chronaeon") else 1.4
                alpha = 0.95 if m.startswith("chronaeon") else 0.75
                ax_ne.plot(gt, rc["agg_ne"][m], color=palette[m], lw=lw, alpha=alpha, label=labels_pretty[m])
                ax_ltt.plot(gt, rc["agg_ltt"][m], color=palette[m], lw=lw, alpha=alpha, label=labels_pretty[m])

        ax_ne.set_yscale("log")
        ax_ne.set_title(scen_title, fontweight="bold")
        ax_ne.grid(True, alpha=0.25)
        ax_ltt.grid(True, alpha=0.25)
        ax_ltt.set_xlabel("Calendar Year")
        if col_i == 0:
            ax_ne.set_ylabel("Effective Pop. Size $\\hat{N}_e(t)$ (log scale)")
            ax_ltt.set_ylabel("Lineages Through Time $L(t)$")

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_dir / "fig_whitepaper_synthetic_skyline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: 2x2 Empirical Viral Outbreak Skylines & LTT Insets
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    axes_flat = axes.flatten()
    for idx, (study_id, study_label, _) in enumerate(emp_studies):
        ax = axes_flat[idx]
        if study_id not in emp_curves:
            continue
        m_dict = emp_curves[study_id]["methods"]
        for m in ["chronaeon_timetree_seq", "chronaeon_timetree_tree", "lsd2", "treetime", "treedater", "reltime"]:
            if m not in m_dict:
                continue
            d = m_dict[m]
            lw = 2.3 if m.startswith("chronaeon") else 1.4
            ls = "-" if m != "chronaeon_timetree_seq" else "-."
            ax.plot(d["grid_t"], d["grid_ne"], color=palette[m], lw=lw, ls=ls, label=labels_pretty[m])

        ax.set_yscale("log")
        ax.set_title(study_label, fontweight="bold")
        ax.set_xlabel("Calendar Year")
        ax.set_ylabel("Skyline $\\hat{N}_e(t) \\cdot g$ (yr)")
        ax.grid(True, alpha=0.25)

    handles, leg_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_dir / "fig_whitepaper_empirical_skyline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: 1x3 Diagnostic Panel (Skyline log10-MAE, Zero-Wait Polytomy Fraction, Runtime vs Error)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # Panel A: Grouped bar chart of Synthetic Skyline log10-MAE across scenarios
    ax0 = axes[0]
    diag_methods = ["chronaeon_timetree_seq", "chronaeon_timetree_tree", "lsd2", "treetime", "treedater", "reltime", "wlogdate"]
    short_names = ["CA-Seq", "CA-Tree", "LSD2", "TreeTime", "treedater", "RelTime", "wLogDate"]
    x_pos = np.arange(len(diag_methods))
    width = 0.25
    scen_colors = {"constant": "#4E79A7", "exponential": "#F28E2B", "boombust": "#E15759"}
    scen_labels = {"constant": "Constant", "exponential": "Exponential", "boombust": "Boom-Bust"}

    for s_i, (s_key, _, _) in enumerate(scenarios):
        sub = summary_syn[summary_syn["scenario"] == s_key].set_index("method")
        vals = [float(sub.loc[m, "skyline_log10_mae"]) if m in sub.index else 0.0 for m in diag_methods]
        ax0.bar(x_pos + (s_i - 1) * width, vals, width=width, color=scen_colors[s_key], label=scen_labels[s_key], alpha=0.9)

    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(short_names, rotation=28, ha="right")
    ax0.set_ylabel("Skyline $\\log_{10} \\hat{N}_e(t)$ MAE")
    ax0.set_title("A. Demographic Skyline Accuracy", fontweight="bold")
    ax0.legend(frameon=True, fontsize=8.5)
    ax0.grid(True, axis="y", alpha=0.25)

    # Panel B: Polytomy / Zero-Wait Interval Collapse Fraction on Empirical Datasets
    ax1 = axes[1]
    emp_short = {
        "16_rabies_northamerica_biek2007": "Rabies",
        "00_ebola_sierraleone_gire2014": "Ebola",
        "03_chikungunya_brazil_naveca2019": "CHIKV",
        "11_dengue_polyepoch_suchard2020": "Dengue",
    }
    emp_m_list = ["chronaeon_timetree_seq", "chronaeon_timetree_tree", "lsd2", "treetime", "treedater", "wlogdate"]
    emp_m_names = ["CA-Seq", "CA-Tree", "LSD2", "TreeTime", "treedater", "wLogDate"]
    x_e = np.arange(len(emp_m_list))
    w_e = 0.18
    e_cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    for e_i, (st_id, st_short) in enumerate(emp_short.items()):
        sub_e = df_emp[df_emp["study_id"] == st_id].set_index("method")
        vals_e = [100.0 * float(sub_e.loc[m, "zero_branch_frac"]) if m in sub_e.index else 0.0 for m in emp_m_list]
        ax1.bar(x_e + (e_i - 1.5) * w_e, vals_e, width=w_e, color=e_cols[e_i], label=st_short, alpha=0.9)

    ax1.set_xticks(x_e)
    ax1.set_xticklabels(emp_m_names, rotation=28, ha="right")
    ax1.set_ylabel("Collapsed Zero-Wait Intervals (%)")
    ax1.set_title("B. Zero-Wait Discretization ($\\Delta t_j \\leq (\\mu L)^{-1}$)", fontweight="bold")
    ax1.legend(frameon=True, fontsize=8.5, ncol=2)
    ax1.grid(True, axis="y", alpha=0.25)

    # Panel C: Coalescent Waiting-Time Quantile MAE vs Skyline log10-MAE
    ax2 = axes[2]
    mean_by_m = summary_syn.groupby("method").agg(
        sky_mae=("skyline_log10_mae", "mean"),
        coal_mae=("coal_times_mae", "mean"),
        ltt_mae=("ltt_mae", "mean"),
    )
    offsets = {
        "chronaeon_timetree_seq": (-98, -11),
        "chronaeon_timetree_tree": (-88, 6),
        "lsd2": (-35, -14),
        "treetime": (-55, -12),
        "treedater": (6, 4),
        "reltime": (-48, 7),
        "wlogdate": (7, 3),
    }
    for m in diag_methods:
        if m not in mean_by_m.index:
            continue
        row = mean_by_m.loc[m]
        c_col = palette.get(m, "#333333")
        marker = "D" if m.startswith("chronaeon") else "o"
        sz = 110 if m.startswith("chronaeon") else 75
        ax2.scatter(row["coal_mae"], row["sky_mae"], color=c_col, marker=marker, s=sz, zorder=5, edgecolors="black", linewidths=0.8)
        ax2.annotate(
            labels_pretty.get(m, m).replace("Phylodate: ", "").replace("ChronAeon TimeTree ", "CA-"),
            (row["coal_mae"], row["sky_mae"]),
            xytext=offsets.get(m, (5, 4)),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax2.set_xlim(0.345, 0.422)
    ax2.set_ylim(0.237, 0.269)

    ax2.set_xlabel("Internal Coalescent Quantile MAE (years)")
    ax2.set_ylabel("Mean Skyline $\\log_{10} \\hat{N}_e(t)$ MAE")
    ax2.set_title("C. Node-Time Fidelity vs. Skyline Error", fontweight="bold")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_whitepaper_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved skyline figures to {fig_dir}")


if __name__ == "__main__":
    run_all()
