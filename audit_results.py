#!/usr/bin/env python3
"""Audit benchmark checkpoint for errors, paired deltas and numeric-score disagreements.

Usage: python audit_results.py results_smoke
No OpenAI client or network is required.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

VARIANTS = ("raw_multiturn", "history_selection", "one_line_rewrite")


def latest_rows(path: Path) -> dict:
    """Load per-call records, explaining how to recover a missing checkpoint."""
    if not path.is_file():
        directory = path.parent
        location_status = (
            "Directory exists" if directory.is_dir()
            else "Directory does NOT exist"
        )
        present = (", ".join(sorted(x.name for x in directory.iterdir()))
                   if directory.is_dir() else "(none)")
        nested = sorted(directory.rglob("rows.jsonl")) if directory.is_dir() else []
        nearby = ("\nNested candidate results:\n" + "\n".join(f"  {x}" for x in nested[:10])) if nested else ""
        raise SystemExit(
            f"Missing checkpoint: {path}\n"
            f"{location_status}. Files here: {present}{nearby}\n"
            "The summary.md only has aggregate statistics; it cannot recover individual "
            "predictions or detailed error records.\n"
            "First locate the original checkpoint with:\n"
            "  find ~ -type f -name rows.jsonl -print\n"
            "Then run: python3 audit_results.py /path/to/results_directory\n"
            "If no checkpoint exists, back up the result directory and re-run the "
            "benchmark with exactly the same original arguments to regenerate it "
            "(this will incur API calls)."
        )
    rows = {}
    with path.open(encoding="utf-8") as fd:
        for line in fd:
            if line.strip():
                r = json.loads(line)
                rows[(r["suite"], r["id"], r["variant"])] = r
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path, nargs="?", default=Path("results_smoke"))
    args = p.parse_args()
    source = args.directory if args.directory.is_file() else args.directory / "rows.jsonl"
    rows = latest_rows(source)
    if not rows:
        raise SystemExit(f"Empty checkpoint: {source}; there are no rows to audit.")
    print(f"Latest unique suite/sample/variant records: {len(rows)}")
    errors = [r for r in rows.values() if r.get("error")]
    print(f"Errors: {len(errors)}")
    for r in errors:
        print(f"  {r['suite']} {r['id']} {r['variant']}: {r['error'][:300]}")
    for suite in sorted({k[0] for k in rows}):
        print(f"\n=== {suite} ===")
        grouped = defaultdict(dict)
        for (s, i, v), row in rows.items():
            if s == suite:
                grouped[i][v] = row
        for v in VARIANTS:
            rr = [x[v] for x in grouped.values() if v in x]
            ok = [x for x in rr if not x.get('error')]
            all_acc = sum(float(x['correct']) for x in rr) / len(rr) if rr else 0
            ok_acc = sum(float(x['correct']) for x in ok) / len(ok) if ok else 0
            print(f"{v:18s} all={all_acc:.3f} ({len(rr)}) completed={ok_acc:.3f} ({len(ok)}) failures={len(rr)-len(ok)}")
        for left, right in [("history_selection", "raw_multiturn"),
                            ("one_line_rewrite", "raw_multiturn")]:
            valid = [(i, x[left], x[right]) for i, x in grouped.items()
                     if left in x and right in x]
            successes = [(i, a, b) for i, a, b in valid
                         if not a.get('error') and not b.get('error')]
            def wl(pairs):
                return (sum(a['correct'] > b['correct'] for _,a,b in pairs),
                        sum(a['correct'] == b['correct'] for _,a,b in pairs),
                        sum(a['correct'] < b['correct'] for _,a,b in pairs))
            print(f"{left} vs {right}: operational W/T/L={wl(valid)}, "
                  f"completed-only W/T/L={wl(successes)}")
        print("Disagreements / failed preps:")
        for i, variant_rows in grouped.items():
            if not all(v in variant_rows for v in VARIANTS):
                continue
            if len({variant_rows[v]['correct'] for v in VARIANTS}) == 1 and not any(
                variant_rows[v].get('error') for v in VARIANTS):
                continue
            row = variant_rows['raw_multiturn']
            print(f"  {i}: q={row['question']!r}; gold={row['gold']!r}; gold_exec={row.get('gold_exec')!r}")
            for v in VARIANTS:
                r = variant_rows[v]
                print(f"    {v}: correct={r['correct']} pred={r.get('prediction')!r} error={r.get('error')}")
        if suite == 'hard_convfinqa':
            print("Numeric-correct but text F1 < 0.5 (inspect unit semantics):")
            for i, variant_rows in grouped.items():
                for v, r in variant_rows.items():
                    if not r.get('error') and r['correct'] > 0.5 and r.get('f1',1) < 0.5:
                        print(f"  {i} {v}: gold={r['gold']!r}; executed={r.get('gold_exec')!r}; "
                              f"pred={r.get('prediction')!r}; token_f1={r.get('f1')}")

if __name__ == '__main__':
    main()
