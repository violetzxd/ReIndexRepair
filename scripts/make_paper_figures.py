from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-analysis", type=Path)
    parser.add_argument("--boundary-audit", type=Path)
    parser.add_argument("--human-statistics", type=Path)
    parser.add_argument("--locator-sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.human_statistics:
        if args.main_analysis or args.boundary_audit:
            parser.error("--human-statistics cannot be combined with auxiliary audit inputs")
        human = json.loads(args.human_statistics.read_text(encoding="utf-8"))
        if human.get("paper_evidence") is not True:
            raise ValueError("refusing to plot human results without paper_evidence=true")
        main_summary = human["primary_hypotheses"]["substitution"]
        boundary_summary = human["primary_hypotheses"]["boundary"]
        main_partial = main_summary["partial_recurrence"]
        main_complete = main_summary["complete_lineage_recurrence"]
        boundary_partial = boundary_summary["partial_recurrence"]
        boundary_complete = boundary_summary["complete_lineage_recurrence"]
    else:
        if not args.main_analysis or not args.boundary_audit:
            parser.error("provide --human-statistics or both --main-analysis and --boundary-audit")
        main = json.loads(args.main_analysis.read_text(encoding="utf-8"))["paired_primary"]
        boundary = json.loads(args.boundary_audit.read_text(encoding="utf-8"))["summary"]
        main_partial = main["partial_visible_deletion"]
        main_complete = main["complete_lineage_deletion"]
        boundary_partial = boundary["repair_mirage"]
        boundary_complete = boundary["complete_lineage_recurrence"]
    locator = json.loads(args.locator_sweep.read_text(encoding="utf-8"))["summary"]["conditions"]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.15), constrained_layout=True)

    groups = [f"Substitution\n($n={main_partial['eligible']}$)", f"Boundary\n($n={boundary_partial['eligible']}$)"]
    partial = [100 * main_partial["rate"], 100 * boundary_partial["rate"]]
    complete = [100 * main_complete["rate"], 100 * boundary_complete["rate"]]
    partial_counts = [f"{main_partial['events']}/{main_partial['eligible']}", f"{boundary_partial['events']}/{boundary_partial['eligible']}"]
    complete_counts = [f"{main_complete['events']}/{main_complete['eligible']}", f"{boundary_complete['events']}/{boundary_complete['eligible']}"]
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, partial, width, label="Partial / trace-derived", color="#C44E52")
    axes[0].bar(x + width / 2, complete, width, label="Complete lineage", color="#4C72B0")
    for positions, values, counts in ((x - width / 2, partial, partial_counts), (x + width / 2, complete, complete_counts)):
        for position, value, count in zip(positions, values, counts):
            axes[0].text(position, value + 1.3, count, ha="center", va="bottom", fontsize=6.5)
    axes[0].set_ylabel("Target recurrence among $F_0$-safe (%)")
    axes[0].set_xticks(x, groups)
    axes[0].set_ylim(0, max(45, 1.25 * max(partial + complete, default=0)))
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")
    axes[0].set_title("(a) Full-pipeline repair mirage")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.5)

    fpr0 = sorted((row for row in locator if row["false_positive_documents"] == 0), key=lambda row: row["locator_recall"])
    recalls = [100 * row["locator_recall"] for row in fpr0]
    residual = [100 * row["attack"]["initially_risky_incidents"]["rate"] for row in fpr0]
    axes[1].plot(recalls, residual, marker="o", linewidth=1.6, color="#C44E52", label="Residual poison@5")
    axes[1].set_xlabel("Injected source-locator recall (%)")
    axes[1].set_ylabel("Initially risky incidents remaining (%)")
    axes[1].set_xticks(recalls)
    axes[1].set_ylim(-4, 104)
    axes[1].grid(color="#DDDDDD", linewidth=0.5)
    axes[1].set_title("(b) Sensitivity to missed sources")
    axes[1].legend(frameon=False, fontsize=6.3, loc="lower left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
