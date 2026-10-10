#!/usr/bin/env python3
"""Reproduce the primary-sample annual water-minus-air peak timing comparison."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=Path("data/samples/analysis/annual_panel.parquet"))
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=Path("data/samples/analysis/peak_timing.csv"))
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    panel = panel[(panel.sample_scope == "primary") &
                  panel.water_mwmt7.notna() & panel.air_mwmt7.notna() &
                  panel.water_peak_doy.notna() & panel.air_peak_doy.notna()].copy()
    panel["offset_days"] = ((panel.water_peak_doy - panel.air_peak_doy + 183) % 365) - 183

    rows = []
    for site, group in panel.groupby("site"):
        if len(group) < 8:
            continue
        rows.append({"site": site, "tier": group.tier.iloc[0],
                     "offset_days": float(group.offset_days.median()),
                     "n_years": int(len(group))})
    per_site = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    per_site.to_csv(args.output, index=False)

    rng = np.random.default_rng(args.seed)
    for tier, group in per_site.groupby("tier", sort=True):
        x = group.offset_days.to_numpy()
        boot = np.array([rng.choice(x, len(x), replace=True).mean()
                         for _ in range(args.replicates)])
        print(f"{tier}: n={len(x)}, mean={x.mean():.3f}, median={np.median(x):.1f}, "
              f"positive_share={(x > 0).mean():.3f}, "
              f"mean_bootstrap_95ci={np.quantile(boot, [.025, .975]).round(3).tolist()}")

    tail = per_site.loc[per_site.tier == "tailwater", "offset_days"].to_numpy()
    ref = per_site.loc[per_site.tier == "reference", "offset_days"].to_numpy()
    boot_diff = np.array([rng.choice(tail, len(tail), replace=True).mean() -
                          rng.choice(ref, len(ref), replace=True).mean()
                          for _ in range(args.replicates)])
    print(f"tailwater-reference mean difference={tail.mean() - ref.mean():.3f}; "
          f"bootstrap_95ci={np.quantile(boot_diff, [.025, .975]).round(3).tolist()}")
    print(f"Welch two-sided p={stats.ttest_ind(tail, ref, equal_var=False).pvalue:.6f}; "
          f"Mann-Whitney two-sided p={stats.mannwhitneyu(tail, ref, alternative='two-sided').pvalue:.6f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
