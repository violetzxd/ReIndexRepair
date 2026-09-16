from __future__ import annotations

import argparse
import json
from pathlib import Path


def latex_command(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def count_rate(value: dict) -> str:
    return f"{value['events']}/{value['eligible']}"


def format_p(value: float) -> str:
    text = f"{value:.4f}"
    return text[1:] if text.startswith("0.") else text


def format_ci(value: dict) -> str:
    low, high = (100 * number for number in value["two_sided_95_exact_interval"])
    low_text = f"{low:.2f}" if 0 < low < 1 else f"{low:.1f}"
    high_text = f"{high:.2f}" if 0 < high < 1 else f"{high:.1f}"
    return f"{low_text.rstrip('0').rstrip('.')}--{high_text.rstrip('0').rstrip('.')}"


def agreement_text(agreement: dict) -> str:
    names = (("target adoption", "target_adopted"), ("correct adoption", "correct_adopted"), ("completeness", "complete"))
    parts = []
    for label, key in names:
        raw = 100 * agreement[key]["raw_agreement"]
        kappa = agreement[key]["cohen_kappa_three_class"]
        kappa_text = r"\mathrm{n/a}" if kappa is None else f"{kappa:.2f}"
        parts.append(f"{label} {raw:.1f}\\%/$\\kappa={kappa_text}$")
    return ", ".join(parts)


def sensitivity_text(stats: dict) -> str:
    least = stats["incomplete_output_sensitivity"]["least_favorable"]
    most = stats["incomplete_output_sensitivity"]["most_favorable"]
    fragments = []
    for label, key in (("substitution", "substitution"), ("boundary", "boundary")):
        low = 100 * least[key]["risk_difference_partial_minus_complete"]
        high = 100 * most[key]["risk_difference_partial_minus_complete"]
        fragments.append(f"{label} {low:.1f}--{high:.1f} points")
    least_p = [least[key]["holm_adjusted_p"] for key in ("substitution", "boundary")]
    most_p = [most[key]["holm_adjusted_p"] for key in ("substitution", "boundary")]
    if all(value < 0.05 for value in least_p + most_p):
        conclusion = "Holm significance remains below .05 throughout"
    elif all(value >= 0.05 for value in least_p) and all(value < 0.05 for value in most_p):
        least_text = "/".join(format_p(value) for value in least_p)
        conclusion = rf"neither least-favorable Holm test is significant ($p_{{\mathrm{{Holm}}}}={least_text}$)"
    else:
        conclusion = "Holm significance is not invariant to these endpoint assignments"
    return "Directional assignment of incomplete endpoints gives risk-difference ranges of " + " and ".join(fragments) + f"; {conclusion}."


def build_commands(stats: dict) -> list[str]:
    if stats.get("paper_evidence") is not True or stats.get("status") != "final_human_adjudicated_semantic_analysis":
        raise ValueError("refusing to render paper inputs without final human paper evidence")
    primary = stats["primary_hypotheses"]
    substitution = primary["substitution"]
    boundary = primary["boundary"]
    annotation = stats["human_annotation"]
    mechanism = stats["boundary_mechanism"]
    repair = stats["repair_lattice"]
    sub_flow = f"94\\!\\to\\!{substitution['initial_target_adopted']}\\!\\to\\!{substitution['eligible']}"
    boundary_flow = f"50\\!\\to\\!{boundary['initial_target_adopted']}\\!\\to\\!{boundary['eligible']}"
    action_labels = {
        "target_span": "span",
        "visible_sentence": "sentence",
        "all_attack_sentences": "all-attack",
        "document_quarantine": "quarantine",
    }
    action_counts = repair["selected_action_counts"]
    action_phrase = ", ".join(
        f"{count} {action_labels.get(action, action.replace('_', '-'))}"
        for action, count in sorted(action_counts.items(), key=lambda item: (action_labels.get(item[0], item[0]), item[0]))
    ) or "none"
    commands = [
        latex_command("EvidenceBanner", ""),
        latex_command("SemanticFigureLead", "Human-adjudicated semantic evaluation"),
        latex_command("SubFlow", sub_flow),
        latex_command("SubPartialCount", count_rate(substitution["partial_recurrence"])),
        latex_command("SubCompleteCount", count_rate(substitution["complete_lineage_recurrence"])),
        latex_command("SubUnconditional", f"{count_rate(substitution['unconditional_partial_recurrence'])}={100 * substitution['unconditional_partial_recurrence']['rate']:.1f}\\%"),
        latex_command("SubRiskDifference", f"{100 * substitution['risk_difference_partial_minus_complete']:.1f}"),
        latex_command("SubRiskDifferenceCI", "--".join(f"{100 * value:.1f}" for value in substitution["paired_bootstrap_95_interval"])),
        latex_command("SubPartialCI", format_ci(substitution["partial_recurrence"])),
        latex_command("SubCompleteCI", format_ci(substitution["complete_lineage_recurrence"])),
        latex_command("SubCompleteCIUpper", format_ci(substitution["complete_lineage_recurrence"]).split("--")[1]),
        latex_command("SubHolm", format_p(substitution["holm_adjusted_p"])),
        latex_command("BoundaryFlow", boundary_flow),
        latex_command("BoundaryPartialCount", count_rate(boundary["partial_recurrence"])),
        latex_command("BoundaryCompleteCount", count_rate(boundary["complete_lineage_recurrence"])),
        latex_command("BoundaryPartialCI", format_ci(boundary["partial_recurrence"])),
        latex_command("BoundaryCompleteCI", format_ci(boundary["complete_lineage_recurrence"])),
        latex_command("BoundaryUnconditional", f"{count_rate(boundary['unconditional_partial_recurrence'])}={100 * boundary['unconditional_partial_recurrence']['rate']:.1f}\\%"),
        latex_command("BoundaryHolm", format_p(boundary["holm_adjusted_p"])),
        latex_command(
            "BoundaryMechanismSentence",
            "Independently of answer labels, migration occurs in "
            f"{count_rate(mechanism['migration_all'])} frozen constructions and "
            f"{count_rate(mechanism['migration_among_human_eligible'])} human-eligible cases; "
            + (
                f"it accompanies all {mechanism['partial_recurrence_total']} partial-arm recurrences."
                if mechanism["all_partial_recurrences_have_migration"]
                else f"it accompanies {mechanism['partial_recurrence_with_migration']}/"
                     f"{mechanism['partial_recurrence_total']} partial-arm recurrences."
            ),
        ),
        latex_command("AnnotationStatusParagraph", f"Two independent humans labeled all {annotation['outputs']} condition-blinded canonical outputs; {annotation['independent_unclear_field_decisions']} independent field decisions were \\texttt{{Unclear}}, and a third person adjudicated {annotation['adjudicated_items']} items. Raw agreement and three-class Cohen's $\\kappa$ were {agreement_text(annotation['agreement'])}. Final labels contain {annotation['final_incomplete_outputs']} incomplete outputs, all retained in the reported flow."),
        latex_command("PrimaryInferenceSentence", sensitivity_text(stats)),
        latex_command("RepairFeasibleCount", f"{repair['reindexrepair_feasible']}/{repair['incidents']}"),
        latex_command("RepairCoverage", f"{100 * repair['reindexrepair_feasible_coverage']:.1f}"),
        latex_command("RepairAbstainCount", f"{repair['incidents'] - repair['reindexrepair_feasible']}/{repair['incidents']}"),
        latex_command("RepairActionSentence", f"For feasible cases, the least-token choices are {action_phrase}, with median {repair['edit_tokens_median']:.1f} and mean {repair['edit_tokens_mean']:.1f} edited tokens."),
        latex_command("RepairFinalistCount", f"{repair['certified_exact_view_oracle_recall_count']}/{repair['reindexrepair_feasible']}"),
        latex_command("RepairSelectedCorrect", str(repair["reindexrepair_correct_adopted"])),
        latex_command("RepairDocumentCorrect", f"{repair['document_quarantine_valid_correct']}/{repair['incidents']}"),
    ]
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stats = json.loads(args.statistics.read_text(encoding="utf-8"))
    commands = build_commands(stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("% Generated from locked, human-adjudicated statistics.\n" + "\n".join(commands) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "commands": len(commands)}, indent=2))


if __name__ == "__main__":
    main()
