"""Chronology audit: for every rolling fold, print the date ranges of history and labels and assert no overlap.

    python -m src.audit data/hm
"""
import argparse

import pandas as pd

from src.run_hm import load_phases


def day(ms):
    return pd.to_datetime(int(ms), unit="ms").strftime("%Y-%m-%d")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--folds", type=int, default=4)
    args = p.parse_args()

    print("| fold | phase | history ends | labels span | eligible customers | customers with a label | share |")
    print("|---|---|---|---|---|---|---|")
    for shift in range(args.folds):
        split, val, test = load_phases(args.data_dir, kcore="train", shift_weeks=shift)
        for phase in (val, test):
            h_end, l_start, l_end = phase.hist["timestamp"].max(), phase.labels["timestamp"].min(), phase.labels["timestamp"].max()
            assert h_end < l_start, f"shift {shift} {phase.name}: history overlaps labels"
            n_label = phase.labels["u"].nunique()
            print(f"| shift {shift} | {phase.name} | {day(h_end)} | {day(l_start)} to {day(l_end)} | "
                  f"{split.n_users:,} | {n_label:,} | {n_label / split.n_users:.1%} |", flush=True)
        assert val.labels["timestamp"].max() < test.hist["timestamp"].max() + 1, "validation labels must precede the test cutoff"
        assert test.hist["timestamp"].max() < test.labels["timestamp"].min()
    print("\nall chronology assertions passed")


if __name__ == "__main__":
    main()
