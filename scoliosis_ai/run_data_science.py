"""
Run data science analysis for Scoliosis AI.
Collects dataset stats, training curves, and Cobb angle summaries.
"""

from pathlib import Path
import json
import sys

from src.data_analysis import ScoliosisDataAnalyzer


def find_latest_results_csv():
    results = list(Path("models/detection").glob("**/results.csv"))
    return max(results, key=lambda p: p.stat().st_mtime) if results else None


def load_cobb_angles_from_summaries(summary_dir):
    angles = {}
    summary_files = list(Path(summary_dir).glob("*_summary.json"))
    for summary_file in summary_files:
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            detections = data.get("detections", [])
            if len(detections) != 1:
                continue
            angle = detections[0].get("cobb_angle_primary")
            if angle is None:
                continue
            angles[summary_file.stem] = float(angle)
        except Exception:
            continue
    return angles


def find_dataset_yaml():
    candidates = [Path("data.yaml"), Path("data") / "data.yaml"]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    analyzer = ScoliosisDataAnalyzer(output_dir="outputs/analysis")
    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = []

    # Dataset stats
    dataset_yaml = find_dataset_yaml()
    if dataset_yaml:
        stats = analyzer.analyze_dataset(str(dataset_yaml))
        if stats:
            report.append("DATASET STATS")
            report.append(f"Classes: {stats['num_classes']}")
            report.append(f"Train images: {stats['train_images']}")
            report.append(f"Val images: {stats['val_images']}")
            report.append(f"Total images: {stats['total_images']}")
            report.append("")

    # Training history plot
    results_csv = find_latest_results_csv()
    if results_csv:
        ok = analyzer.plot_training_history(str(results_csv))
        if ok:
            report.append("TRAINING HISTORY")
            report.append(f"Results CSV: {results_csv}")
            report.append("Plot: outputs/analysis/training_history.png")
            report.append("")

    # Cobb angle statistics from diagnosis summaries
    angles = load_cobb_angles_from_summaries("outputs/diagnosis")
    if angles:
        stats = analyzer.cobb_angle_statistics(angles)
        report.append("COBB ANGLE STATS")
        report.append(f"Count: {stats['count']}")
        report.append(f"Mean: {stats['mean']:.2f}")
        report.append(f"Median: {stats['median']:.2f}")
        report.append(f"Std: {stats['std']:.2f}")
        report.append("Plot: outputs/analysis/cobb_angle_stats.png")
        report.append("")

    report_text = "\n".join(report) if report else "No data available."

    report_file = output_dir / "data_science_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved report: {report_file}")


if __name__ == "__main__":
    sys.exit(main() or 0)
