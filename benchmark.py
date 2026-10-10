#!/usr/bin/env python3
"""CoQA vs ConvFinQA: raw multi-turn / LLM-selected history / one-line rewrite.

No gold *current* answer is supplied to the model. Gold prior answers are used in
all conditions to isolate the effect of history representation (teacher forcing).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import string
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

VARIANTS = ("raw_multiturn", "history_selection", "one_line_rewrite")
_client_tls = threading.local()
_write_lock = threading.Lock()


@dataclass
class Sample:
    id: str
    suite: str
    document: str
    history: list[dict[str, str]]
    question: str
    gold: str
    gold_exec: Any = None


def clean_spaces(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def table_to_text(t: Any) -> str:
    if isinstance(t, dict):
        lines = []
        for col, rows in t.items():
            if isinstance(rows, dict):
                lines.append(f"Column: {col}")
                lines.extend(f"  {row}: {value}" for row, value in rows.items())
            else:
                lines.append(f"{col}: {rows}")
        return "\n".join(lines)
    if isinstance(t, list):
        return "\n".join(" | ".join(map(str, row)) if isinstance(row, list) else str(row) for row in t)
    return str(t or "")


def to_list(x: Any) -> list:
    """HF CoQA may expose arrays or old serialized JSON strings."""
    if isinstance(x, str):
        x = json.loads(x)
    return list(x)


def field_text(row: Any) -> str:
    return str(row.get("input_text", "") if isinstance(row, dict) else row)


def choose_target(qs: list[str], ans: list[str], rng: random.Random,
                  min_hist: int, max_hist: int) -> int | None:
    eligible = [i for i in range(min_hist, min(len(qs), len(ans), max_hist + 1))
                if clean_spaces(qs[i]) and clean_spaces(ans[i])]
    return rng.choice(eligible) if eligible else None


def load_coqa(seed: int, n: int, min_hist: int, max_hist: int,
              max_doc_chars: int) -> list[Sample]:
    # HF hosts an auto-parquet conversion of the Stanford validation split.
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/coqa", split="validation")
    rng = random.Random(seed)
    candidates = []
    for row_idx, rec in enumerate(ds):
        story = str(rec.get("story") or "")
        if not (100 <= len(story) <= max_doc_chars):
            continue
        qs = [field_text(q) for q in to_list(rec["questions"])]
        ans_obj = rec["answers"]
        if isinstance(ans_obj, str):
            ans_obj = json.loads(ans_obj)
        if isinstance(ans_obj, dict):  # dictionary-of-arrays format
            ans = [str(a) for a in ans_obj["input_text"]]
        else:
            ans = [field_text(a) for a in ans_obj]
        idx = choose_target(qs, ans, rng, min_hist, max_hist)
        if idx is None:
            continue
        history = [{"q": qs[j], "a": ans[j]} for j in range(idx)]
        candidates.append(Sample(
            id=f"coqa:{rec.get('id', row_idx)}:{idx}", suite="easy_coqa",
            document=story, history=history, question=qs[idx], gold=ans[idx]))
    rng.shuffle(candidates)
    if len(candidates) < n:
        raise ValueError(f"CoQA: only {len(candidates)} eligible conversations, requested {n}")
    return candidates[:n]


def load_convfinqa(seed: int, n: int, min_hist: int, max_hist: int,
                    max_doc_chars: int) -> list[Sample]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="sharick008/convfinqa",
                           filename="convfinqa_dataset.json", repo_type="dataset")
    all_data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = all_data["dev"]
    rng = random.Random(seed + 113)
    candidates = []
    for row_idx, rec in enumerate(records):
        doc = rec["doc"]
        pre = doc.get("pre_text") or []
        post = doc.get("post_text") or []
        pre_text = "\n".join(map(str, pre)) if isinstance(pre, list) else str(pre)
        post_text = "\n".join(map(str, post)) if isinstance(post, list) else str(post)
        document = f"PRE-TABLE TEXT:\n{pre_text}\n\nTABLE:\n{table_to_text(doc.get('table'))}\n\nPOST-TABLE TEXT:\n{post_text}"
        if len(document) > max_doc_chars:
            continue
        d = rec["dialogue"]
        qs = [str(x) for x in d["conv_questions"]]
        ans = [str(x) for x in d["conv_answers"]]
        gold_exe = d.get("executed_answers", [])
        if len(qs) != len(ans) or len(gold_exe) != len(qs):
            continue
        idx = choose_target(qs, ans, rng, min_hist, max_hist)
        if idx is None or gold_exe[idx] is None:
            continue
        history = [{"q": qs[j], "a": ans[j]} for j in range(idx)]
        candidates.append(Sample(
            id=f"convfinqa:{rec.get('id', row_idx)}:{idx}", suite="hard_convfinqa",
            document=document, history=history, question=qs[idx],
            gold=ans[idx], gold_exec=gold_exe[idx]))
    rng.shuffle(candidates)
    if len(candidates) < n:
        raise ValueError(f"ConvFinQA: only {len(candidates)} eligible conversations, requested {n}. "
                         "Try --min-history 2 or --max-doc-chars 30000")
    return candidates[:n]


def client() -> OpenAI:
    if not hasattr(_client_tls, "client"):
        _client_tls.client = OpenAI(timeout=120.0, max_retries=2)
    return _client_tls.client


def llm(model: str, messages: list[dict[str, str]], effort: str,
        max_output_tokens: int = 1800, retries: int = 2,
        json_schema: dict | None = None, validator=None) -> dict:
    """Retry incomplete/empty/malformed responses; account for *all* attempts.

    Output-token limits include invisible reasoning tokens. An empty output_text is
    therefore often an exhausted output budget, not an empty valid prediction.
    """
    spent = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
             "seconds": 0.0, "api_attempts": 0}
    diagnostics = []
    use_schema = json_schema is not None
    max_attempts = retries + 1
    for attempt in range(max_attempts):
        budget = max_output_tokens * (2 ** attempt)
        kwargs = dict(model=model, input=messages, reasoning={"effort": effort},
                      max_output_tokens=budget, store=False)
        if use_schema:
            kwargs["text"] = {"format": json_schema}
        t0 = time.perf_counter()
        try:
            resp = client().responses.create(**kwargs)
        except Exception as exc:
            # Model-specific incompatibility with Structured Outputs: use the
            # identical prompt without schema on this *same* attempt.
            error_str = str(exc).lower()
            format_unsupported = (getattr(exc, "status_code", None) == 400
                                  and any(x in error_str for x in
                                          ("json_schema", "text.format", "response_format", "structured output")))
            if not (use_schema and format_unsupported):
                raise
            use_schema = False
            kwargs.pop("text", None)
            diagnostics.append(f"structured format unsupported: {str(exc)[:100]}")
            resp = client().responses.create(**kwargs)
        spent["seconds"] += time.perf_counter() - t0
        spent["api_attempts"] += 1
        usage = getattr(resp, "usage", None)
        output_details = getattr(usage, "output_tokens_details", None)
        spent["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        spent["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        reasoning = getattr(output_details, "reasoning_tokens", 0) or 0
        spent["reasoning_tokens"] += reasoning
        status = getattr(resp, "status", None)
        incomplete = getattr(resp, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None)
        message = (getattr(resp, "output_text", None) or "").strip()
        diagnostics.append(f"attempt={attempt+1} status={status} reason={reason} "
                           f"budget={budget} reasoning_tokens={reasoning} "
                           f"text_len={len(message)}")
        if status not in (None, "completed") or not message:
            # Refusals and content-filtered responses should not be silently
            # treated as valid; retries here leave an informative error.
            continue
        try:
            parsed = validator(message) if validator is not None else None
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            diagnostics.append(f"validation={type(exc).__name__}: {str(exc)[:140]}")
            continue
        return {"text": message, "parsed": parsed, "seconds": round(spent["seconds"], 3),
                "input_tokens": spent["input_tokens"],
                "output_tokens": spent["output_tokens"],
                "reasoning_tokens": spent["reasoning_tokens"],
                "api_attempts": spent["api_attempts"]}
    raise ValueError("No usable LLM output after retries: " + "; ".join(diagnostics))


SELECTOR_SCHEMA = {
    "type": "json_schema", "name": "history_turn_selection", "strict": True,
    "schema": {
        "type": "object", "properties": {
            "indices": {"type": "array", "items": {"type": "integer"}}
        }, "required": ["indices"], "additionalProperties": False,
    },
}


def parse_selector_output(output: str, n: int) -> list[int]:
    """Accept a structured object or legacy raw array, validating bounds."""
    data = json.loads(output)
    if isinstance(data, dict):
        data = data.get("indices")
    if not isinstance(data, list):
        raise ValueError(f"Selector output lacks an indices array: {output[:150]!r}")
    return indices_from_output(json.dumps(data), n)


def history_text(history: list[dict[str, str]]) -> str:
    return "\n".join(f"[{i}] User: {x['q']}\n    Assistant: {x['a']}"
                     for i, x in enumerate(history))


def indices_from_output(output: str, n: int) -> list[int]:
    # Strictly avoid treating free-form explanations as indices.
    match = re.fullmatch(r"\s*\[(?:\s*\d+\s*(?:,\s*\d+\s*)*)?\]\s*", output)
    if match is None:
        raise ValueError(f"Selector output is not a JSON integer array: {output[:250]!r}")
    indexes = json.loads(output)
    if (not isinstance(indexes, list)
            or not all(type(i) is int and 0 <= i < n for i in indexes)
            or len(set(indexes)) != len(indexes)):
        raise ValueError(f"Invalid history indices: {indexes}")
    return sorted(indexes)


def select_history(s: Sample, model: str, effort: str,
                   max_output_tokens: int = 1800, retries: int = 2) -> tuple[list[int], dict]:
    prompt = (
        "Select the minimal past turns required to answer the CURRENT question, "
        "including transitive dependencies and earlier numerical results. "
        "Include any past Q/A that resolves a reference, entity, comparison, or intermediate calculation. "
        "Ignore unrelated turns. The source document will be available later. "
        "Reply ONLY with JSON containing an 'indices' integer array of zero-based "
        "turn indices in chronological order; e.g. {\"indices\": [0,2]}. "
        "An empty array is allowed. Do not answer the current question.\n\n"
        f"HISTORY:\n{history_text(s.history)}\n\nCURRENT QUESTION: {s.question}"
    )
    r = llm(model, [{"role": "user", "content": prompt}], effort,
            max_output_tokens=max_output_tokens, retries=retries,
            json_schema=SELECTOR_SCHEMA,
            validator=lambda output: parse_selector_output(output, len(s.history)))
    return r["parsed"], r


def rewrite_question(s: Sample, model: str, effort: str,
                     max_output_tokens: int = 1800, retries: int = 2) -> tuple[str, dict]:
    prompt = (
        "Rewrite the CURRENT conversational question into EXACTLY ONE self-contained "
        "English question on a single line. Preserve intent, constraints, named entities, "
        "and all numerical values needed from preceding user/assistant turns; explicitly "
        "include prior computed answers whenever the current step depends on them. "
        "Do not solve the current question, do not add unsupported facts, do not cite the history, "
        "and do not create more than one question. "
        "Output ONLY that single-line question, no prefix or explanation.\n\n"
        f"HISTORY:\n{history_text(s.history)}\n\nCURRENT QUESTION: {s.question}"
    )
    def validate_one_line(output: str) -> str:
        answer = clean_spaces(output)
        if not answer or len(answer) > 1500 or not answer.endswith("?"):
            raise ValueError(f"Not a valid single-line question: {answer[:180]!r}")
        return answer
    r = llm(model, [{"role": "user", "content": prompt}], effort,
            max_output_tokens=max_output_tokens, retries=retries,
            validator=validate_one_line)
    return r["parsed"], r


def answer_messages(s: Sample, variant: str, indices: list[int] | None,
                    rewrite: str | None) -> list[dict[str, str]]:
    prompt = (
        "Answer the last user question using the supplied SOURCE DOCUMENT and conversation "
        "information. Be faithful to the document. Previous assistant answers (if any) "
        "are historical context. Give only the short final answer; for numeric questions "
        "return a number with units or %, if appropriate. Do not show calculations. "
        "If the information is missing, say 'unknown'.\n\nSOURCE DOCUMENT:\n" + s.document
    )
    messages = [{"role": "system", "content": prompt}]
    if variant == "raw_multiturn":
        prior = s.history
        current = s.question
    elif variant == "history_selection":
        assert indices is not None
        prior = [s.history[i] for i in indices]
        current = s.question
    else:
        assert rewrite is not None
        prior = []
        current = rewrite
    for turn in prior:
        messages.append({"role": "user", "content": turn["q"]})
        messages.append({"role": "assistant", "content": turn["a"]})
    messages.append({"role": "user", "content": current})
    return messages


def normalized(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return clean_spaces(s)


def text_metrics(pred: str, gold: str) -> tuple[float, float]:
    a, b = normalized(pred).split(), normalized(gold).split()
    em = float(a == b)
    if not a or not b:
        return em, float(a == b)
    shared = sum((Counter(a) & Counter(b)).values())
    f1 = 2 * shared / (len(a) + len(b)) if shared else 0.0
    return em, f1


def parse_number(x: Any) -> tuple[float | None, bool]:
    if isinstance(x, bool):
        return None, False
    if isinstance(x, (int, float)):
        return float(x), False
    s = clean_spaces(x).replace(",", "").replace("−", "-")
    # The answer itself can contain a number or unit, e.g. '$1,200 million'.
    match = re.search(r"(?<![\w.])([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", s)
    if not match:
        return None, False
    num = float(match.group(1))
    pct = "%" in s or bool(re.search(r"\bpercent\b", s, flags=re.I))
    return num, pct


def finance_correct(pred: str, gold: Any) -> float:
    exp, _ = parse_number(gold)
    if exp is None:
        return float(normalized(pred) == normalized(str(gold)))
    got, is_percent = parse_number(pred)
    if got is None:
        return 0.0
    # A percent expression may correspond to a ratio execution value (0.12=12%).
    # Also permit percentage points if the execution value is already 12.
    candidates = [got, got / 100.0] if is_percent else [got]
    return float(any(math.isclose(val, exp, rel_tol=0.01, abs_tol=1e-4)
                     for val in candidates))


def score(s: Sample, pred: str) -> dict[str, float]:
    em, f1 = text_metrics(pred, s.gold)
    if s.suite == "easy_coqa":
        return {"correct": em, "em": em, "f1": f1}
    return {"correct": finance_correct(pred, s.gold_exec), "em": em, "f1": f1}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fp:
        return [json.loads(x) for x in fp if x.strip()]


def append_jsonl(path: Path, rec: dict) -> None:
    with _write_lock:
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fp.flush()


def process_sample(s: Sample, variant: str, args: argparse.Namespace,
                   prep_path: Path, results_path: Path,
                   cached_prep: dict, cached_results: dict) -> dict:
    key = (s.id, variant)
    # Old versions permanently cached errors as zero accuracy. Retry those.
    if key in cached_results and not cached_results[key].get("error") \
            and str(cached_results[key].get("prediction") or "").strip():
        return cached_results[key]
    preproc = cached_prep.get(key)
    try:
        if preproc is None:
            if variant == "history_selection":
                indexes, r = select_history(s, args.model, args.prep_effort, args.prep_max_output_tokens, args.prep_retries)
                preproc = {"id": s.id, "variant": variant, "indices": indexes,
                           "rewrite": None, "usage": r}
            elif variant == "one_line_rewrite":
                rewrite, r = rewrite_question(s, args.model, args.prep_effort, args.prep_max_output_tokens, args.prep_retries)
                preproc = {"id": s.id, "variant": variant, "indices": None,
                           "rewrite": rewrite, "usage": r}
            else:
                preproc = {"id": s.id, "variant": variant, "indices": None,
                           "rewrite": None, "usage": {}}
            append_jsonl(prep_path, preproc)
        messages = answer_messages(s, variant, preproc.get("indices"), preproc.get("rewrite"))
        r = llm(args.model, messages, args.answer_effort, args.max_output_tokens, args.answer_retries)
        metrics = score(s, r["text"])
        p_usage = preproc["usage"]
        rec = {
            "id": s.id, "suite": s.suite, "variant": variant,
            "history_n": len(s.history), "kept_n": len(preproc.get("indices") or []) if variant == "history_selection" else None,
            "question": s.question, "gold": s.gold, "gold_exec": s.gold_exec,
            "prediction": r["text"], "rewrite": preproc.get("rewrite"),
            "selected_indices": preproc.get("indices"),
            "answer_input_tokens": r["input_tokens"],
            "answer_output_tokens": r["output_tokens"],
            "prep_input_tokens": p_usage.get("input_tokens", 0),
            "prep_output_tokens": p_usage.get("output_tokens", 0),
            "reasoning_tokens": r["reasoning_tokens"] + p_usage.get("reasoning_tokens", 0),
            "answer_api_attempts": r["api_attempts"],
            "prep_api_attempts": p_usage.get("api_attempts", 0),
            "answer_seconds": r["seconds"],
            "prep_seconds": p_usage.get("seconds", 0),
            "total_seconds": round(r["seconds"] + p_usage.get("seconds", 0), 3),
            "correct": metrics["correct"], "em": metrics["em"], "f1": metrics["f1"],
            "error": None,
        }
    except Exception as ex:
        rec = {"id": s.id, "suite": s.suite, "variant": variant,
               "history_n": len(s.history), "question": s.question,
               "gold": s.gold, "gold_exec": s.gold_exec,
               "prediction": "", "correct": 0.0, "em": 0.0, "f1": 0.0,
               "error": f"{type(ex).__name__}: {ex}"}
    append_jsonl(results_path, rec)
    return rec


def compact_jsonl(path: Path) -> None:
    """Keep the latest row for each (id, variant) after retrying older errors."""
    if not path.exists():
        return
    latest = {(rec["id"], rec["variant"]): rec for rec in read_jsonl(path)}
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for rec in latest.values():
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def mean(records: list[dict], key: str) -> float:
    return sum(float(x.get(key, 0) or 0) for x in records) / len(records) if records else 0.0


def paired_ci(pairs: list[tuple[dict, dict]], seed: int = 42,
              iterations: int = 2000) -> tuple[float, float, float]:
    """Paired bootstrap CI of accuracy difference (A-B), resampling questions."""
    if not pairs:
        return 0.0, 0.0, 0.0
    deltas = [float(a["correct"]) - float(b["correct"]) for a, b in pairs]
    point = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
                  for _ in range(iterations))
    return point, boot[int(0.025 * iterations)], boot[int(0.975 * iterations)]


def summarize(results: list[dict], out: Path, n: int) -> None:
    lines = ["# Multi-turn history representation ablation", "",
             "Every sample receives the same source document. History uses gold previous answers.",
             "Scores include failures as incorrect. Token/latency metrics only include successful calls.",
             "", "| Suite | Variant | n | Accuracy/EM | Token F1 | Mean answer input tokens | "
             "Mean TOTAL tokens | Mean answer sec | Mean TOTAL sec | Mean kept turns | Errors | Completed accuracy |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for suite in ("easy_coqa", "hard_convfinqa"):
        for variant in VARIANTS:
            subset = [x for x in results if x.get("suite") == suite and x.get("variant") == variant]
            successes = [x for x in subset if not x.get("error")]
            t = (mean(successes, "answer_input_tokens") + mean(successes, "answer_output_tokens")
                 + mean(successes, "prep_input_tokens") + mean(successes, "prep_output_tokens"))
            # Raw keeps all previous Q/A turns. A zero here was a reporting bug.
            kept = (mean(successes, "history_n") if variant == "raw_multiturn"
                    else mean(successes, "kept_n") if variant == "history_selection"
                    else 0.0)
            completed_acc = f"{mean(successes, 'correct'):.3f}" if successes else "N/A"
            lines.append(f"| {suite} | {variant} | {len(subset)}/{n} | {mean(subset,'correct'):.3f} | "
                         f"{mean(subset,'f1'):.3f} | {mean(successes,'answer_input_tokens'):.1f} | "
                         f"{t:.1f} | {mean(successes,'answer_seconds'):.2f} | "
                         f"{mean(successes,'total_seconds'):.2f} | "
                         f"{kept:.2f} | {len(subset)-len(successes)} | {completed_acc} |")
    lines.extend(["", "*Easy accuracy = normalized Exact Match; hard accuracy = executed-answer "
                  "numeric match (relative tolerance 1%, or absolute 1e-4).",
                  "*Total seconds are per-sample sequential API wall times (prep + answer), not wall time "
                  "with concurrent workers. Total tokens include both LLM stages.",
                  "*The primary Accuracy column treats pipeline errors as wrong (end-to-end operational score). "
                  "Completed accuracy excludes API/preprocessing failures and is not directly comparable when failure rates differ.",
                  "*Mean kept turns counts Q/A pairs seen by the answer model: all for Raw, chosen for Selection, "
                  "and zero for Rewrite.", ""])
    # Paired A>B / A<B counts over matching IDs. Useful even with n=100.
    grouped = {(x["suite"], x["id"], x["variant"]): x for x in results}
    lines.extend(["## Paired wins / losses (same samples)", "",
                  "| Suite | Comparison | Win | Tie | Loss | Δ accuracy | 95% paired bootstrap CI |", "|---|---|---:|---:|---:|---:|---:|"])
    for suite in ("easy_coqa", "hard_convfinqa"):
        for left, right in (("history_selection", "raw_multiturn"),
                            ("one_line_rewrite", "raw_multiturn"),
                            ("history_selection", "one_line_rewrite")):
            ids = {x["id"] for x in results if x.get("suite") == suite}
            pairs = [(grouped[(suite, i, left)], grouped[(suite, i, right)])
                     for i in ids if (suite, i, left) in grouped and (suite, i, right) in grouped]
            win = sum(a["correct"] > b["correct"] for a, b in pairs)
            tie = sum(a["correct"] == b["correct"] for a, b in pairs)
            loss = sum(a["correct"] < b["correct"] for a, b in pairs)
            delta, ci_lo, ci_hi = paired_ci(pairs)
            lines.append(f"| {suite} | {left} vs {right} | {win} | {tie} | {loss} | "
                         f"{delta:+.3f} | [{ci_lo:+.3f}, {ci_hi:+.3f}] |")
    # Separately report differences conditional on BOTH strategies completing.
    # This distinguishes actual answer-quality effects from selector/API failure.
    lines.extend(["", "## Paired wins / losses — both completed (answer quality only)", "",
                  "| Suite | Comparison | n paired | Win | Tie | Loss | Δ accuracy |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for suite in ("easy_coqa", "hard_convfinqa"):
        for left, right in (("history_selection", "raw_multiturn"),
                            ("one_line_rewrite", "raw_multiturn"),
                            ("history_selection", "one_line_rewrite")):
            ids = {x["id"] for x in results if x.get("suite") == suite}
            paired = [(grouped[(suite, i, left)], grouped[(suite, i, right)])
                      for i in ids if (suite, i, left) in grouped and (suite, i, right) in grouped
                      and not grouped[(suite, i, left)].get("error")
                      and not grouped[(suite, i, right)].get("error")]
            wins = sum(a["correct"] > b["correct"] for a, b in paired)
            ties = sum(a["correct"] == b["correct"] for a, b in paired)
            losses = sum(a["correct"] < b["correct"] for a, b in paired)
            delta = sum(a["correct"] - b["correct"] for a, b in paired) / len(paired) if paired else 0.0
            lines.append(f"| {suite} | {left} vs {right} | {len(paired)} | "
                         f"{wins} | {ties} | {losses} | {delta:+.3f} |")
    lines.extend(["", "Bootstrap intervals resample matching question pairs (2,000 resamples). "
                  f"This report contains up to {n} examples per dataset; check paired n and error counts.",
                  "For small smoke tests, confidence intervals are descriptive only.",
                  "", "## Selected disagreements", ""])
    for suite in ("easy_coqa", "hard_convfinqa"):
        ids = sorted({x["id"] for x in results if x.get("suite") == suite})
        count = 0
        for i in ids:
            rowset = [grouped.get((suite, i, v)) for v in VARIANTS]
            if any(r is None for r in rowset):
                continue
            if len({r["correct"] for r in rowset}) == 1:
                continue
            orig = rowset[0]
            lines.append(f"- **{suite}** / `{i}` — Question: {orig['question'][:120]} "
                         f"/ Gold: `{str(orig['gold'])[:80]}` "
                         f"/ raw={rowset[0]['correct']:.0f}, "
                         f"selection={rowset[1]['correct']:.0f}, "
                         f"rewrite={rowset[2]['correct']:.0f}")
            count += 1
            if count >= 8:
                break
        if count == 0:
            lines.append(f"- **{suite}**: no fully matched disagreements yet.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="samples PER dataset")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--prep-effort", default="low", choices=["none", "low", "medium", "high"])
    parser.add_argument("--answer-effort", default="medium", choices=["none", "low", "medium", "high"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=6)
    parser.add_argument("--max-doc-chars", type=int, default=20000)
    parser.add_argument("--max-output-tokens", type=int, default=1400)
    parser.add_argument("--prep-max-output-tokens", type=int, default=1800,
                        help="initial preprocessing budget INCLUDING reasoning; retries double it")
    parser.add_argument("--prep-retries", type=int, default=2)
    parser.add_argument("--answer-retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Please set OPENAI_API_KEY")
    if (args.n < 1 or args.workers < 1 or args.min_history < 1
            or args.max_history < args.min_history or args.prep_max_output_tokens < 256
            or args.max_output_tokens < 256 or args.prep_retries < 0 or args.answer_retries < 0):
        parser.error("Invalid --n, --workers, or history bounds")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "config.json"
    settings = vars(args).copy()
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        # The four newly introduced knobs did not exist in v1. Allow resuming
        # from v1 checkpoints while still rejecting changed data or model.
        if any(settings.get(k) != v for k, v in previous.items()):
            parser.error("Output directory belongs to different config. Use another --out directory.")
    manifest.write_text(json.dumps(settings, indent=2))

    samples = (load_coqa(args.seed, args.n, args.min_history, args.max_history, args.max_doc_chars)
               + load_convfinqa(args.seed, args.n, args.min_history, args.max_history, args.max_doc_chars))
    with (out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False, default=str) + "\n")
    prep_path, results_path = out / "prep.jsonl", out / "rows.jsonl"
    cached_prep = {(x["id"], x["variant"]): x for x in read_jsonl(prep_path)}
    cached_results = {(x["id"], x["variant"]): x for x in read_jsonl(results_path)}
    jobs = [(s, v) for s in samples for v in VARIANTS]
    rng = random.Random(args.seed + 19)
    rng.shuffle(jobs)  # prevent systematic variant-order / server-load bias
    final_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_sample, s, v, args, prep_path, results_path,
                               cached_prep, cached_results): (s.id, v) for s, v in jobs}
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            row = fut.result()
            final_rows.append(row)
            print(f"[{idx}/{len(jobs)}] {row['suite']} {row['variant']} "
                  f"correct={row['correct']} {('ERROR '+row['error'][:120]) if row.get('error') else ''}",
                  flush=True)
            if idx % 30 == 0:
                summarize(final_rows, out / "summary.md", args.n)
    summarize(final_rows, out / "summary.md", args.n)
    compact_jsonl(results_path)
    compact_jsonl(prep_path)
    print(f"\nFinished. See {out / 'summary.md'} and {results_path}")


if __name__ == "__main__":
    main()
