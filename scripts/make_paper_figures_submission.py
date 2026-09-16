"""Submission typography only; preserve frozen experiment scripts and data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-statistics', type=Path, required=True)
    parser.add_argument('--locator-sweep', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stats = json.loads(args.human_statistics.read_text(encoding='utf-8'))
    if stats.get('paper_evidence') is not True:
        raise ValueError('Human evidence gate must pass')
    hypotheses = stats['primary_hypotheses']
    pairs = [(hypotheses[k]['partial_recurrence'], hypotheses[k]['complete_lineage_recurrence'])
             for k in ('substitution', 'boundary')]
    locator = json.loads(args.locator_sweep.read_text(encoding='utf-8'))['summary']['conditions']
    rows = sorted((r for r in locator if r['false_positive_documents'] == 0), key=lambda r:r['locator_recall'])
    plt.rcParams.update({'font.family':'sans-serif', 'font.size':10.5,
                         'text.color':'#243247', 'axes.labelcolor':'#243247',
                         'xtick.color':'#243247', 'ytick.color':'#243247',
                         'axes.linewidth':.7, 'pdf.fonttype':42, 'ps.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3), constrained_layout=True)
    x = np.arange(2)
    for j, (label, color) in enumerate((('Trace-derived', '#C76459'), ('Complete lineage', '#387C8D'))):
        values = [100*pair[j]['rate'] for pair in pairs]
        positions = x + (j-.5)*.34
        axes[0].bar(positions, values, .30, label=label, color=color,
                    edgecolor='white', linewidth=.6, hatch='///' if j else None)
        for position, value, pair in zip(positions, values, pairs):
            axes[0].text(position, value+1.3, f"{pair[j]['events']}/{pair[j]['eligible']}",
                         ha='center', va='bottom', fontsize=10.5, color='#243247')
    axes[0].set_xticks(x, [f"Substitution\n(n={pairs[0][0]['eligible']})", f"Boundary\n(n={pairs[1][0]['eligible']})"])
    axes[0].set_ylabel('Target recurrence (%)')
    axes[0].set_ylim(0, 50)
    axes[0].set_yticks([0,10,20,30,40,50])
    axes[0].legend(frameon=False, fontsize=10, ncol=2, loc='lower left',
                   bbox_to_anchor=(-.06, 1.01), handlelength=1.2, columnspacing=.8)
    axes[0].set_title('(a) Answer recurrence', fontsize=11, fontweight='bold', pad=29)
    recalls = [100*r['locator_recall'] for r in rows]
    residual = [100*r['attack']['initially_risky_incidents']['rate'] for r in rows]
    axes[1].plot(recalls, residual, marker='o', markersize=4.8,
                 markeredgecolor='white', markeredgewidth=.7, linewidth=1.6,
                 color='#387C8D')
    for index in (4,5):
        if len(rows) > index:
            count = rows[index]['attack']['initially_risky_incidents']
            axes[1].annotate(f"{count['events']}/{count['eligible']}",
                            (recalls[index], residual[index]),
                            xytext=(-7, 9) if index == 4 else (-14, 9),
                            textcoords='offset points', ha='center', fontsize=10)
    axes[1].set_xlabel('Injected locator recall (%)')
    axes[1].set_ylabel('Residual exposure (%)')
    axes[1].set_xticks([0,20,40,60,80,100])
    axes[1].set_ylim(-3,112)
    axes[1].set_yticks([0,25,50,75,100])
    axes[1].set_title('(b) Missed-source sensitivity', fontsize=11, fontweight='bold', pad=29)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#A5AFBC')
        ax.spines['bottom'].set_color('#A5AFBC')
        ax.tick_params(length=3, width=.6)
        ax.grid(axis='y', color='#E1E6EB', linewidth=.6)
        ax.set_axisbelow(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.png'), dpi=240, bbox_inches='tight')

if __name__ == '__main__':
    main()
