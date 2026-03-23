"""
Benchmark suite using diagnosis summaries.
Computes descriptive stats and bootstrap confidence intervals.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(values, metric="mean", n=1000, alpha=0.05):
    values = np.array(values, dtype=float)
    if values.size == 0:
        return None

    stats = []
    rng = np.random.default_rng(42)
    for _ in range(n):
        sample = rng.choice(values, size=values.size, replace=True)
        if metric == "mean":
            stats.append(sample.mean())
        elif metric == "median":
            stats.append(np.median(sample))
        else:
            stats.append(sample.mean())

    lo = np.percentile(stats, 100 * (alpha / 2))
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def load_summaries(summary_dir):
    rows = []
    for summary_file in Path(summary_dir).glob("*_summary.json"):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            detections = data.get("detections", [])
            if len(detections) != 1:
                continue
            det = detections[0]
            rows.append(
                {
                    "file": summary_file.name,
                    "angle": det.get("cobb_angle_primary"),
                    "severity": det.get("severity"),
                    "confidence": det.get("confidence"),
                }
            )
        except Exception:
            continue
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run benchmark stats")
    parser.add_argument("--summaries", default="outputs/diagnosis")
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()

    rows = load_summaries(args.summaries)
    if not rows:
        print("No summary files found.")
        return 1

    df = pd.DataFrame(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    angles = df["angle"].dropna().astype(float).to_list()
    mean_ci = bootstrap_ci(angles, metric="mean")
    median_ci = bootstrap_ci(angles, metric="median")

    report_lines = []
    report_lines.append("BENCHMARK SUMMARY")
    report_lines.append(f"Samples: {len(df)}")
    report_lines.append(f"Angle mean: {np.mean(angles):.2f}")
    report_lines.append(f"Angle median: {np.median(angles):.2f}")
    if mean_ci:
        report_lines.append(f"Mean 95% CI: [{mean_ci[0]:.2f}, {mean_ci[1]:.2f}]")
    if median_ci:
        report_lines.append(f"Median 95% CI: [{median_ci[0]:.2f}, {median_ci[1]:.2f}]")
    report_lines.append("")
    report_lines.append("Severity counts:")
    for sev, count in df["severity"].value_counts().items():
        report_lines.append(f"  {sev}: {count}")

    report_path = out_dir / "benchmark_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    df.to_csv(out_dir / "benchmark_summary.csv", index=False)

    print("\n".join(report_lines))
    print(f"\nSaved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
