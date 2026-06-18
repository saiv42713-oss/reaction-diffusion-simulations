#!/usr/bin/env python3
"""
circuit_classifier_runner.py
────────────────────────────────────────────────────────────────────────────────
Loads valid_circuits.json, pre-filters with a fast LSA gate, then runs
classify_regime() on each passing circuit in parallel with incremental
checkpointing.

Usage
─────
    python circuit_classifier_runner.py               # fresh run
    python circuit_classifier_runner.py --resume      # continue from checkpoint
    python circuit_classifier_runner.py --dry-run 20  # test on first 20 circuits

Output
──────
    checkpoint_results.csv   — written incrementally every SAVE_EVERY circuits
    classification_results.csv — final sorted output written when all done
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ── Patch SIM_PARAMS BEFORE importing classify_regime functions ───────────────
# These override the hardcoded values inside classify_regime.py so we don't
# have to touch that file.  Must happen before any worker imports the module.
import classify_regime as _cr
_cr.SIM_PARAMS.update({
    "Ny":                 100,
    "Nx":                 100,
    "steps":              50000,
    "stopping_threshold": 1e-4,
    "min_steps":          5000,
    "save_every":         500,
})

from classify_regime import classify_regime, BASE_PARAMS          # noqa: E402
from finding_steady_states import fast_stable_steady_state        # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT        = Path(__file__).parent
CIRCUITS_FILE    = REPO_ROOT / "valid_circuits.json"
CHECKPOINT_FILE  = REPO_ROOT / "checkpoint_results.csv"
OUTPUT_FILE      = REPO_ROOT / "classification_results.csv"

# ── Run config ────────────────────────────────────────────────────────────────
N_JOBS     = 8     # increased from 4 — use more cores for parallel workers
SAVE_EVERY = 500   # flush checkpoint every this many circuits


# ══════════════════════════════════════════════════════════════════════════════
#  Parameter mapping — Option A: fixed reference point for all circuits
# ══════════════════════════════════════════════════════════════════════════════
def extract_params(circuit: dict) -> tuple[float, float]:
    """
    Return (act_prod_rate, inh_prod_rate) for a given circuit entry.

    OPTION A — fixed canonical reference point for all circuits.

    act_prod_rate=5.0, inh_prod_rate=5.0 is the Fig 1C Turing reference point
    (na=ni=3, ba=5, bi=5, Di=10).  Every circuit in valid_circuits.json is
    classified at this single parameter pair.

    ⚠️  NOTE: classify_regime() does not currently use circuit topology — it
    runs the JAPI simulation with fixed BASE_PARAMS regardless of the circuit's
    nodes/edges.  This means all 1,849 circuits will receive identical inputs
    and produce the same regime label.  This run is therefore a pipeline
    validation: it proves checkpointing, parallelism, and CSV output are
    correct before a topology-aware classify_regime() is built.

    To upgrade to Option B (per-circuit parameter sweep), replace this
    function with a sweep over a (ba, bi) grid and update process_circuit()
    to iterate over it.
    """
    _ = circuit   # topology metadata not used at this stage
    return 5.0, 5.0


# ══════════════════════════════════════════════════════════════════════════════
#  LSA pre-filter
# ══════════════════════════════════════════════════════════════════════════════
def lsa_can_turing(act_prod_rate: float, inh_prod_rate: float) -> bool:
    """
    Fast heuristic gate — returns False if a Turing instability is impossible,
    so we can label those circuits OFF without running the full simulation.

    NOTE: with Option A (all circuits at ba=5.0, bi=5.0), this gate evaluates
    the same params every time.  (5.0, 5.0) is the Turing reference — it will
    always pass, making the LSA gate a no-op for this run.  It activates
    meaningfully once Option B (per-circuit ba/bi sweep) is implemented.

    Returns True  → send to full simulation
    Returns False → label OFF immediately (saved in 'source' column as 'lsa_filter')
    """
    try:
        params = {
            **BASE_PARAMS,
            "act_prod_rate": act_prod_rate,
            "inh_prod_rate": inh_prod_rate,
        }
        a_ss, i_ss, _ = fast_stable_steady_state(params, activator_type="juxtacrine")

        # Heuristic 1: if steady-state activator is zero AND inhibitor
        # overwhelmingly dominates → likely pure OFF, not worth simulating.
        if a_ss == 0.0 and inh_prod_rate > 4.0 * act_prod_rate:
            return False

        # Heuristic 2: if inhibitor-to-activator ratio is extreme → OFF.
        ratio = inh_prod_rate / (act_prod_rate + 1e-10)
        if ratio > 20.0:
            return False

        return True

    except Exception:
        # If LSA itself crashes, let the full simulation decide.
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  Per-circuit worker (runs in child process via joblib)
# ══════════════════════════════════════════════════════════════════════════════
def process_circuit(circuit: dict) -> dict:
    """
    Run one circuit through LSA gate → classify_regime() → return result dict.
    All exceptions are caught and recorded; worker never crashes the whole run.
    """
    # ── Re-apply SIM_PARAMS patch inside the worker ───────────────────────────
    # joblib loky spawns fresh processes on macOS — they re-import classify_regime
    # from disk and get the original values.  We must patch again here so the
    # actual simulation sees the optimized settings, not the hardcoded defaults.
    import classify_regime as _cr_local
    _cr_local.SIM_PARAMS.update({
        "Ny":                 100,
        "Nx":                 100,
        "steps":              50000,
        "stopping_threshold": 1e-4,
        "min_steps":          5000,
        "save_every":         500,
    })

    idx = circuit["index"]
    t0  = time.time()

    try:
        act_prod_rate, inh_prod_rate = extract_params(circuit)
    except NotImplementedError as e:
        raise
    except Exception as e:
        return _error_row(idx, 0.0, 0.0, f"extract_params failed: {e}", t0)

    # ── LSA gate ──────────────────────────────────────────────────────────────
    if not lsa_can_turing(act_prod_rate, inh_prod_rate):
        return {
            "index":          idx,
            "act_prod_rate":  act_prod_rate,
            "inh_prod_rate":  inh_prod_rate,
            "regime":         "OFF",
            "source":         "lsa_filter",
            "elapsed_s":      round(time.time() - t0, 2),
            "error":          "",
        }

    # ── Full simulation ───────────────────────────────────────────────────────
    try:
        regime = classify_regime(act_prod_rate, inh_prod_rate)
        return {
            "index":          idx,
            "act_prod_rate":  act_prod_rate,
            "inh_prod_rate":  inh_prod_rate,
            "regime":         regime,
            "source":         "simulation",
            "elapsed_s":      round(time.time() - t0, 2),
            "error":          "",
        }
    except Exception as e:
        return _error_row(idx, act_prod_rate, inh_prod_rate,
                          traceback.format_exc(limit=3), t0)


def _error_row(idx, ba, bi, msg, t0):
    return {
        "index":          idx,
        "act_prod_rate":  ba,
        "inh_prod_rate":  bi,
        "regime":         "ERROR",
        "source":         "simulation",
        "elapsed_s":      round(time.time() - t0, 2),
        "error":          msg.replace("\n", " | ")[:300],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ══════════════════════════════════════════════════════════════════════════════
def load_checkpoint() -> set[int]:
    """Return set of circuit indices already written to checkpoint."""
    if not CHECKPOINT_FILE.exists():
        return set()
    df = pd.read_csv(CHECKPOINT_FILE, usecols=["index"])
    return set(df["index"].tolist())


def append_checkpoint(rows: list[dict]) -> None:
    """Append a batch to the checkpoint CSV; write header only on first call."""
    df = pd.DataFrame(rows)
    write_header = not CHECKPOINT_FILE.exists()
    df.to_csv(CHECKPOINT_FILE, mode="a", header=write_header, index=False)
    print(f"    ✓ Checkpoint flushed ({len(rows)} rows → {CHECKPOINT_FILE.name})")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main(resume: bool = False, dry_run: int = 0):
    print("=" * 72)
    print("  JAPI Circuit Classifier Runner")
    print("=" * 72)
    print(f"  Grid:              {_cr.SIM_PARAMS['Ny']}×{_cr.SIM_PARAMS['Nx']}")
    print(f"  Steps:             {_cr.SIM_PARAMS['steps']:,}")
    print(f"  stopping_threshold:{_cr.SIM_PARAMS['stopping_threshold']}")
    print(f"  n_jobs:            {N_JOBS}")
    print(f"  save_every:        {SAVE_EVERY}")
    print(f"  resume:            {resume}")
    print()

    # ── Load circuits ─────────────────────────────────────────────────────────
    print(f"Loading {CIRCUITS_FILE} ...")
    with open(CIRCUITS_FILE) as f:
        all_circuits: list[dict] = json.load(f)
    print(f"  {len(all_circuits):,} circuits in file.")

    if dry_run > 0:
        all_circuits = all_circuits[:dry_run]
        print(f"  DRY RUN: trimmed to {len(all_circuits)} circuits.")

    # ── Resume logic: back up checkpoint on fresh run to prevent duplicates ───
    if resume:
        done_indices = load_checkpoint()
        print(f"  Resuming — {len(done_indices):,} circuits already done, skipping.")
    else:
        if CHECKPOINT_FILE.exists():
            backup = CHECKPOINT_FILE.with_suffix(".bak.csv")
            CHECKPOINT_FILE.rename(backup)
            print(f"  Fresh run — old checkpoint backed up to {backup.name}")
        done_indices = set()

    circuits = [c for c in all_circuits if c["index"] not in done_indices]
    print(f"  {len(circuits):,} circuits to classify.\n")

    if not circuits:
        print("Nothing to do — all circuits already checkpointed.")
        _finalize()
        return

    # ── Batch loop ────────────────────────────────────────────────────────────
    total       = len(circuits)
    n_batches   = (total + SAVE_EVERY - 1) // SAVE_EVERY
    wall0       = time.time()
    done_so_far = len(done_indices)

    for batch_num in range(n_batches):
        batch_start = batch_num * SAVE_EVERY
        batch       = circuits[batch_start : batch_start + SAVE_EVERY]

        print(f"── Batch {batch_num + 1}/{n_batches}  "
              f"(circuits {done_so_far + batch_start + 1}–"
              f"{done_so_far + batch_start + len(batch)} of "
              f"{done_so_far + total}) ──────────────")

        batch_t0 = time.time()
        results: list[dict] = Parallel(
            n_jobs=N_JOBS,
            backend="loky",
            verbose=5,
        )(delayed(process_circuit)(c) for c in batch)

        append_checkpoint(results)

        # ── Batch summary ─────────────────────────────────────────────────────
        counts: dict[str, int] = {}
        errors = []
        for r in results:
            regime = r["regime"]
            counts[regime] = counts.get(regime, 0) + 1
            if regime == "ERROR":
                errors.append(f"    index {r['index']}: {r['error'][:100]}")

        elapsed_batch = time.time() - batch_t0
        avg_s         = elapsed_batch / max(len(batch), 1)
        remaining     = (total - batch_start - len(batch)) * avg_s
        print(f"    Regimes this batch: {counts}")
        print(f"    Batch time: {elapsed_batch:.1f}s  |  "
              f"avg {avg_s:.1f}s/circuit  |  "
              f"est. remaining: {remaining/60:.1f} min")
        if errors:
            print(f"    ⚠️  {len(errors)} errors:")
            for e in errors[:5]:
                print(e)
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more (see checkpoint CSV)")
        print()

    total_wall = time.time() - wall0
    print(f"All batches complete in {total_wall/60:.1f} min.")
    _finalize()


def _finalize():
    """Merge checkpoint into sorted, deduplicated final CSV and print summary."""
    if not CHECKPOINT_FILE.exists():
        print("No checkpoint file found — nothing to finalize.")
        return

    df = pd.read_csv(CHECKPOINT_FILE)
    # Deduplicate in case of any repeat runs — keep the last result per index
    df = (df.sort_values("index")
            .drop_duplicates(subset=["index"], keep="last")
            .reset_index(drop=True))
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n══════════════════════════════════════════════════")
    print("  CLASSIFICATION SUMMARY")
    print("══════════════════════════════════════════════════")
    regime_counts = df["regime"].value_counts()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(df)
        bar = "█" * int(pct / 2)
        print(f"  {regime:<12} {count:>5}  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':<12} {len(df):>5}")

    lsa_skipped = (df["source"] == "lsa_filter").sum()
    if lsa_skipped:
        print(f"\n  {lsa_skipped} circuits labeled OFF by LSA pre-filter (no simulation run).")

    errors = (df["regime"] == "ERROR").sum()
    if errors:
        print(f"\n  ⚠️  {errors} circuits errored — inspect 'error' column in CSV.")

    print(f"\n  → {OUTPUT_FILE}")
    print("══════════════════════════════════════════════════\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify all valid JAPI circuits via simulate + checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip circuits already written to checkpoint_results.csv"
    )
    parser.add_argument(
        "--dry-run", type=int, default=0, metavar="N",
        help="Run only the first N circuits (for testing)"
    )
    args = parser.parse_args()
    main(resume=args.resume, dry_run=args.dry_run)