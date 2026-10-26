"""Read results/summary.json and draw reproducible comparison plots.

Only plot measured values; no synthetic experimental results are generated.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', default='results/full/summary.json')
    p.add_argument('--out', default='results/full/plots')
    a = p.parse_args()
    import matplotlib.pyplot as plt
    with open(a.summary, encoding='utf-8') as f:
        records = json.load(f)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = sorted(set(row['method'] for row in records))
    signals = [('gate_coverage', 'Coverage allowed by gate', 'gate_coverage_vs_noise.png'),
               ('gate_answerable_recall', 'Answerable-context acceptance rate', 'gate_recall_vs_noise.png'),
               ('lexical_accuracy_on_answerable', 'Lexical answer match on answerable cases', 'lexical_match_vs_noise.png')]
    for metric, title, filename in signals:
        plotted = False
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in methods:
            sub = [r for r in records if r['method'] == method and r['noise'] != 'all'
                   and r.get(metric) is not None]
            if not sub:
                continue
            sub.sort(key=lambda x: float(x['noise']))
            ax.plot([float(r['noise']) for r in sub], [r[metric] for r in sub],
                    marker='o', label=method)
            plotted = True
        if plotted:
            ax.set_xlabel('Requested noise fraction')
            ax.set_ylabel(metric)
            ax.set_ylim(-.05, 1.05)
            ax.set_title(title)
            ax.grid(True, alpha=.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / filename, dpi=160)
            print('Wrote', out / filename)
        plt.close(fig)


if __name__ == '__main__':
    main()
