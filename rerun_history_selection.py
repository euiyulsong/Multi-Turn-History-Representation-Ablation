#!/usr/bin/env python3
"""Rerun ONLY history_selection with a recall-oriented selector prompt.

Loads the EXACT existing samples.jsonl, checkpoints new outputs separately, and
atomically replaces just history_selection entries in prep.jsonl / rows.jsonl.
Raw/rewrite rows are preserved. A backup is made before the merge.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import benchmark as bm

SELECTOR_INSTRUCTIONS = """You are a conversation history selector.

Choose past question/answer turns that may be needed to answer the CURRENT QUESTION.
The goal is to preserve answer-relevant context, not to minimize the number of turns.

Rules:
1. Keep past turns that mention relevant entities, facts, constraints, time periods,
   comparisons, numerical values, or intermediate calculation results.
2. Resolve implicit references such as "it", "they", "that", "the same period",
   "the previous result", "that total", or a follow-up to a previous question.
3. Include ALL transitive dependencies: if a selected answer depends on another
   earlier turn to understand what it refers to, keep that earlier turn too.
4. Prefer RECALL over compression. If a turn might be needed and you are unsure,
   INCLUDE it. When references are ambiguous, keep enough turns to resolve them.
5. Pay special attention to nearby turns but do not omit older dependencies.
6. Select an empty list ONLY if the current question is clearly standalone.
7. Judge dependencies using the conversation given. The SOURCE DOCUMENT will be
   given to the downstream answer model, but is not shown to you here.
8. Do not solve, rewrite, or answer the current question.

Return ONLY JSON with the key "indices" and an array of zero-based, unique
history turn indexes in chronological order. Example: {"indices": [0, 2]}.
"""


def selector_prompt(sample: bm.Sample) -> str:
    return (
        SELECTOR_INSTRUCTIONS
        + "\nHISTORY:\n" + bm.history_text(sample.history)
        + "\n\nCURRENT QUESTION:\n" + sample.question
    )


def selected_answer(sample: bm.Sample, opts: dict) -> tuple[dict, dict | None]:
    """Perform new selection and answer; return compatible prep and row records."""
    try:
        sel = bm.llm(
            opts["model"], [{"role": "user", "content": selector_prompt(sample)}],
            opts["prep_effort"], max_output_tokens=opts["prep_max_output_tokens"],
            retries=opts["prep_retries"], json_schema=bm.SELECTOR_SCHEMA,
            validator=lambda txt: bm.parse_selector_output(txt, len(sample.history)),
        )
        idx = sel["parsed"]
        prep = {"id": sample.id, "variant": "history_selection", "indices": idx,
                "rewrite": None, "usage": sel,
                "selector_version": "recall_v1"}
        ans = bm.llm(
            opts["model"],
            bm.answer_messages(sample, "history_selection", idx, None),
            opts["answer_effort"], max_output_tokens=opts["max_output_tokens"],
            retries=opts["answer_retries"],
        )
        metrics = bm.score(sample, ans["text"])
        rec = {
            "id": sample.id, "suite": sample.suite, "variant": "history_selection",
            "history_n": len(sample.history), "kept_n": len(idx),
            "question": sample.question, "gold": sample.gold,
            "gold_exec": sample.gold_exec, "prediction": ans["text"],
            "rewrite": None, "selected_indices": idx,
            "answer_input_tokens": ans["input_tokens"],
            "answer_output_tokens": ans["output_tokens"],
            "prep_input_tokens": sel["input_tokens"],
            "prep_output_tokens": sel["output_tokens"],
            "reasoning_tokens": ans["reasoning_tokens"] + sel["reasoning_tokens"],
            "answer_api_attempts": ans["api_attempts"],
            "prep_api_attempts": sel["api_attempts"],
            "answer_seconds": ans["seconds"],
            "prep_seconds": sel["seconds"],
            "total_seconds": round(ans["seconds"] + sel["seconds"], 3),
            "correct": metrics["correct"], "em": metrics["em"], "f1": metrics["f1"],
            "error": None, "selector_version": "recall_v1",
        }
        return rec, prep
    except Exception as exc:
        return {
            "id": sample.id, "suite": sample.suite, "variant": "history_selection",
            "history_n": len(sample.history), "question": sample.question,
            "gold": sample.gold, "gold_exec": sample.gold_exec,
            "prediction": "", "correct": 0.0, "em": 0.0, "f1": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "selector_version": "recall_v1",
        }, None


def unique_latest(path: Path) -> dict[tuple[str, str], dict]:
    return {(r["id"], r["variant"]): r for r in bm.read_jsonl(path)}


def write_atomic(path: Path, records: list[dict]) -> None:
    tmp = path.with_name(path.name + ".recall-tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def require_files(out: Path) -> None:
    required = ("samples.jsonl", "rows.jsonl", "prep.jsonl", "config.json")
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise SystemExit(f"Missing {missing} in {out}. Do not regenerate new samples; use the original output directory.")


def commit(out: Path, samples: list[bm.Sample], staged: dict[tuple[str, str], dict],
           staged_prep: dict[tuple[str, str], dict]) -> Path:
    originals = unique_latest(out / "rows.jsonl")
    original_prep = unique_latest(out / "prep.jsonl")
    ids = {s.id for s in samples}
    required = {(s.id, "history_selection") for s in samples}
    if not required.issubset(staged) or not required.issubset(staged_prep):
        raise RuntimeError("Staged results incomplete; originals were not changed")
    if any(staged[k].get("error") or not str(staged[k].get("prediction") or "").strip()
           for k in required):
        raise RuntimeError("Staged results contain failures; originals were not changed")
    if any(k not in originals for s in samples for k in
           ((s.id, "raw_multiturn"), (s.id, "one_line_rewrite"))):
        raise RuntimeError("Some original raw/rewrite rows are missing; refusing to overwrite")
    old_baselines = {k: v for k, v in originals.items() if k[1] != "history_selection"}
    # Merge with prior rows; only the history_selection keys for these exact IDs change.
    new_rows = dict(originals)
    new_rows.update({k: staged[k] for k in required})
    new_preps = dict(original_prep)
    new_preps.update({k: staged_prep[k] for k in required})
    assert {k: v for k, v in new_rows.items() if k[1] != "history_selection"} == old_baselines

    from collections import Counter
    sizes = Counter(s.suite for s in samples)
    if sizes.get("easy_coqa") != sizes.get("hard_convfinqa"):
        raise RuntimeError(f"Unequal suite sizes {sizes}; summary format expects the same n")

    backup_root = out / "backups"
    backup_root.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = backup_root / f"before_recall_{stamp}"
    if backup.exists():
        raise RuntimeError(f"Backup destination exists: {backup}")
    backup.mkdir()
    for name in ("rows.jsonl", "prep.jsonl", "summary.md"):
        src = out / name
        if src.exists():
            shutil.copy2(src, backup / name)

    # Build new report BEFORE modifying any existing results.
    summary_tmp = out / "summary.recall-tmp.md"
    bm.summarize(list(new_rows.values()), summary_tmp, sizes["easy_coqa"])
    content = summary_tmp.read_text(encoding="utf-8")
    content = content.replace(
        "Every sample receives the same source document.",
        "History Selection has been rerun using the recall-oriented selector prompt (recall_v1). "
        "Raw Multi-turn and One-line Rewrite results are unchanged.\n\n"
        "Every sample receives the same source document.", 1,
    )
    summary_tmp.write_text(content, encoding="utf-8")
    write_atomic(out / "rows.jsonl", list(new_rows.values()))
    write_atomic(out / "prep.jsonl", list(new_preps.values()))
    os.replace(summary_tmp, out / "summary.md")
    return backup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results_smoke"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prep-effort", choices=("none", "low", "medium", "high"),
                    help="Optional override; default uses original config")
    ap.add_argument("--reset-stage", action="store_true",
                    help="Discard only prior recall-v1 temporary checkpoints")
    args = ap.parse_args()
    out = args.out
    require_files(out)
    if not os.environ.get("OPENAI_API_KEY"):
        ap.error("Set OPENAI_API_KEY")
    if args.workers < 1:
        ap.error("--workers must be >= 1")

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    opts = {"model": config.get("model", "gpt-6-luna"),
            "prep_effort": args.prep_effort or config.get("prep_effort", "low"),
            "answer_effort": config.get("answer_effort", "medium"),
            "prep_max_output_tokens": config.get("prep_max_output_tokens", 1800),
            "prep_retries": config.get("prep_retries", 2),
            "max_output_tokens": config.get("max_output_tokens", 1400),
            "answer_retries": config.get("answer_retries", 2)}
    allowed = {f.name for f in fields(bm.Sample)}
    samples = [bm.Sample(**{k: v for k, v in x.items() if k in allowed})
               for x in bm.read_jsonl(out / "samples.jsonl")]
    if not samples or len(samples) != len({s.id for s in samples}):
        raise SystemExit("samples.jsonl is empty or contains duplicate IDs")
    originals = unique_latest(out / "rows.jsonl")
    if any((s.id, "history_selection") not in originals for s in samples):
        raise SystemExit("Original selection rows missing; refusing to mix datasets")

    # Checkpoints separated from existing results until every new result succeeds.
    stage = out / "history_selection_recall_v1"
    if args.reset_stage and stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(exist_ok=True)
    signature = hashlib.sha256(json.dumps({
        "selector": SELECTOR_INSTRUCTIONS, "opts": opts,
        "ids": [s.id for s in samples],
        "sample_hash": hashlib.sha256((out / "samples.jsonl").read_bytes()).hexdigest(),
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    sig_path = stage / "signature.txt"
    if sig_path.exists() and sig_path.read_text().strip() != signature:
        raise SystemExit("Old recall stage uses different settings or samples; rerun with --reset-stage")
    sig_path.write_text(signature + "\n")
    stage_rows = stage / "rows.jsonl"
    stage_prep = stage / "prep.jsonl"
    prior_rows = unique_latest(stage_rows)
    prior_prep = unique_latest(stage_prep)
    valid = {(s.id, "history_selection") for s in samples
             if (s.id, "history_selection") in prior_rows
             and (s.id, "history_selection") in prior_prep
             and not prior_rows[(s.id, "history_selection")].get("error")
             and str(prior_rows[(s.id, "history_selection")].get("prediction") or "").strip()}
    pending = [s for s in samples if (s.id, "history_selection") not in valid]
    print(f"Model: {opts['model']} | Selection: recall_v1 | Input samples: {len(samples)}")
    print(f"Checkpointed: {len(valid)} | Will rerun: {len(pending)} ONLY history_selection")
    print("Original raw_multiturn and one_line_rewrite API calls: ZERO", flush=True)
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            tasks = {executor.submit(selected_answer, s, opts): s for s in pending}
            for i, future in enumerate(concurrent.futures.as_completed(tasks), 1):
                row, prep = future.result()
                bm.append_jsonl(stage_rows, row)
                if prep is not None:
                    bm.append_jsonl(stage_prep, prep)
                status = ("ERROR " + row["error"][:160]) if row.get("error") else f"correct={row['correct']} kept={row['kept_n']}"
                print(f"[{len(valid)+i}/{len(samples)}] {row['suite']} {row['id']} {status}", flush=True)

    new_rows = unique_latest(stage_rows)
    new_prep = unique_latest(stage_prep)
    succeeded = sum((s.id, "history_selection") in new_rows
                    and (s.id, "history_selection") in new_prep
                    and not new_rows[(s.id, "history_selection")].get("error")
                    and bool(str(new_rows[(s.id, "history_selection")].get("prediction") or "").strip())
                    for s in samples)
    if succeeded != len(samples):
        print(f"\nOnly {succeeded}/{len(samples)} successes. Existing results_smoke untouched.")
        print(f"Fix the errors then run this SAME command to resume. Check {stage_rows}")
        raise SystemExit(1)
    existing_rows = unique_latest(out / "rows.jsonl")
    existing_prep = unique_latest(out / "prep.jsonl")
    keys = {(s.id, "history_selection") for s in samples}
    if all(existing_rows.get(k) == new_rows.get(k) and
           existing_prep.get(k) == new_prep.get(k) for k in keys):
        print("\nRecall-v1 selection results already applied; no changes or new backup needed.")
        return
    backup = commit(out, samples, new_rows, new_prep)
    print(f"\nUpdated ONLY history_selection in {out}/rows.jsonl and prep.jsonl")
    print(f"Regenerated: {out / 'summary.md'}")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
