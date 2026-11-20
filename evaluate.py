#!/usr/bin/env python3
"""
ChartTrap evaluator: run image+question pairs through a multimodal model, grade against the
deterministic ground truth, and mine the failures for patterns.

  python evaluate.py --data data --provider mock                      # no API key, tests the plumbing
  python evaluate.py --data data --provider anthropic --model claude-sonnet-5-5
  python evaluate.py --data data --provider openai --model gpt-4o

Outputs (in --out, default results/<provider>_<model>/):
  results.jsonl   one row per question (raw response, parsed answer, verdict, failure tag)
  failures.jsonl  only the wrong ones, with the image path for manual review
  report.md       accuracy by question type / trap feature, ranked failure patterns
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROMPT = """You are analysing a chart image. Answer the question using only what is visible in the image.

Question: {question}
{format_hint}

Briefly explain how you read the axes and marks, then finish with a last line in exactly this form:
FINAL ANSWER: <answer>"""

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}


# =========================================================================== model clients
class MockClient:
    """Simulated model for testing the pipeline offline. It is NOT a benchmark of anything."""

    def __init__(self, seed: int = 0):
        self.name = "mock"
        self.seed = seed

    def ask(self, image_path: Path, prompt: str, item: dict) -> str:
        rng = random.Random(f"{self.seed}:{item['id']}")
        ft = item["features"]
        p_err = 0.10 + 0.25 * (ft["scale"] == "log") + 0.15 * ft["truncated_axis"] + 0.15 * ft["dual_axis"] \
            + 0.10 * ft["clutter"] + {"easy": 0, "medium": 0.05, "hard": 0.15}[item["tier"]]
        wrong = rng.random() < p_err
        ans, d = item["answer"], item.get("distractors", {})
        if item["answer_type"] == "numeric":
            if wrong:
                options = [d.get("linear_reading"), d.get("other_axis_reading"), d.get("visual_ratio"),
                           d.get("other_series_same_category"),
                           (rng.choice(list(d["neighbors"].values())) if d.get("neighbors") else None),
                           ans * rng.uniform(1.15, 1.4)]
                options = [o for o in options if o is not None]
                ans = rng.choice(options)
            else:
                ans = rng.uniform(item["accept_lo"], item["accept_hi"])
            return f"Reading the marks against the axis.\nFINAL ANSWER: {ans:.2f}"
        if item["answer_type"] == "integer":
            if wrong:
                ans = max(0, ans + rng.choice([-1, 1, 2]))
            return f"Counting bars above the line.\nFINAL ANSWER: {ans}"
        if wrong:
            pool = [c for c in item["choices"] if c != ans]
            ans = d.get("runner_up") if (d.get("runner_up") and rng.random() < 0.6) else rng.choice(pool)
        return f"Comparing heights.\nFINAL ANSWER: {ans}"


class AnthropicClient:
    def __init__(self, model: str):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def ask(self, image_path: Path, prompt: str, item: dict) -> str:
        data = base64.b64encode(image_path.read_bytes()).decode()
        msg = self.client.messages.create(
            model=self.model, max_tokens=800,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                {"type": "text", "text": prompt}]}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAIClient:
    def __init__(self, model: str):
        from openai import OpenAI  # pip install openai
        self.client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def ask(self, image_path: Path, prompt: str, item: dict) -> str:
        data = base64.b64encode(image_path.read_bytes()).decode()
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=800,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                {"type": "text", "text": prompt}]}])
        return r.choices[0].message.content or ""


def make_client(provider: str, model: str):
    if provider == "mock":
        return MockClient()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set; set it before using --provider anthropic.")
        return AnthropicClient(model or "claude-sonnet-5-5")
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set; set it before using --provider openai.")
        return OpenAIClient(model or "gpt-4o")
    raise SystemExit(f"unknown provider {provider}")


def is_retryable_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code in (408, 409, 429) or status_code >= 500
    return type(error).__name__ in {
        "APIConnectionError", "APITimeoutError", "ConnectTimeout", "ReadTimeout",
        "ConnectError", "NetworkError", "TimeoutException",
    }


def ask_with_retry(client, image_path, prompt, item, attempts=4):
    for k in range(attempts):
        try:
            return client.ask(image_path, prompt, item), None
        except Exception as e:
            if k == attempts - 1 or not is_retryable_error(e):
                return "", f"{type(e).__name__}: {e}"
            time.sleep(2 ** k)


# =========================================================================== parsing
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def final_answer_text(resp: str) -> str:
    hits = re.findall(r"final answer\s*[:\-]\s*(.+)", resp, flags=re.I)
    if hits:
        return hits[-1].strip().strip("*` ")
    lines = [l for l in resp.strip().splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def parse_number(text: str):
    t = text.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if m:
        return float(m.group())
    for w, v in NUMBER_WORDS.items():
        if re.search(rf"\b{w}\b", text.lower()):
            return float(v)
    return None


def parse_category(text: str, choices: list):
    nt = norm(text)
    for c in choices:
        if norm(c) == nt:
            return c
    hits = [c for c in choices if re.search(rf"\b{re.escape(norm(c))}\b", nt)]
    if len(hits) > 1:  # prefer the longest label if one is nested in another ("India" vs "Indian Ocean")
        longest = max(len(norm(h)) for h in hits)
        hits = [h for h in hits if len(norm(h)) == longest]
    return hits[0] if len(hits) == 1 else None


def parse_answer(resp: str, item: dict):
    tail = final_answer_text(resp)
    at = item["answer_type"]
    if at == "category":
        return parse_category(tail, item["choices"])
    num = parse_number(tail)
    if num is None:
        return None
    if at == "integer":
        return int(round(num)) if abs(num - round(num)) < 1e-6 else None
    return num


# =========================================================================== grading + failure taxonomy
def grade(item: dict, parsed) -> bool:
    if parsed is None:
        return False
    at = item["answer_type"]
    if at == "numeric":
        return item["accept_lo"] <= parsed <= item["accept_hi"]
    return parsed == item["answer"]


def close(x, v, rel=0.06) -> bool:
    return v is not None and abs(x - v) <= rel * max(abs(v), 1e-9)


def classify_failure(item: dict, parsed, api_error=None) -> str:
    """Heuristic tags. Each one names a concrete visual mistake rather than just 'wrong'."""
    if api_error:
        return "api_error"
    if parsed is None:
        return "unparseable_or_refused"
    d, t = item.get("distractors", {}), item["qtype"]
    if item["answer_type"] == "numeric":
        truth = item["answer"]
        if t == "ratio":
            if close(parsed, d.get("visual_ratio"), 0.10):
                return "truncated_axis_visual_ratio"        # compared bar heights, not values
            if close(parsed, d.get("inverse_ratio"), 0.10):
                return "ratio_inverted"
            return "ratio_miscalculation"
        if close(parsed, d.get("linear_reading"), 0.10):
            return "log_scale_read_as_linear"
        if close(parsed, d.get("other_axis_reading"), 0.08):
            return "dual_axis_confusion"
        if close(parsed, d.get("other_series_same_category"), 0.04):
            return "series_confusion"
        if any(close(parsed, v, 0.03) for v in d.get("neighbors", {}).values()):
            return "neighbor_category_misread"
        if truth and parsed and (close(parsed / 10, truth, 0.1) or close(parsed * 10, truth, 0.1)):
            return "magnitude_error"
        half = max(truth - item["accept_lo"], item["accept_hi"] - truth)
        return "imprecise_estimate" if abs(parsed - truth) <= 2.5 * half else "gross_misread"
    if item["answer_type"] == "integer":
        return "count_off_by_one" if abs(parsed - item["answer"]) == 1 else "count_off_by_many"
    if parsed == d.get("runner_up"):
        return "close_call_confusion"
    if t == "largest_gap" and parsed == d.get("tallest_second_series"):
        return "absolute_height_vs_difference"
    ch = item["choices"]
    if abs(ch.index(parsed) - ch.index(item["answer"])) == 1:
        return "adjacent_category"
    return "wrong_category"


# =========================================================================== analysis
def wilson(k: int, n: int, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def bucket(item: dict):
    f = item["features"]
    nb = "<=6" if f["n_categories"] <= 6 else ("7-9" if f["n_categories"] <= 9 else "10+")
    return {
        "tier": item["tier"], "scale": f["scale"], "truncated_axis": f["truncated_axis"],
        "dual_axis": f["dual_axis"], "clutter": f["clutter"], "gridlines": f["gridlines"],
        "axis_color_hint": f["axis_color_hint"], "n_categories": nb,
    }


def acc_table(rows, keyfn):
    g = defaultdict(lambda: [0, 0])
    for r in rows:
        for k in keyfn(r):
            g[k][1] += 1
            g[k][0] += r["correct"]
    return g


def build_report(rows, model_name) -> str:
    n, k = len(rows), sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)
    L = [f"# ChartTrap report: {model_name}", "",
         f"**Overall accuracy: {k}/{n} = {k / max(n, 1):.1%}** (95% CI {lo:.1%} to {hi:.1%})", ""]

    def table(title, g, order=None):
        L.extend([f"## {title}", "", "| group | n | accuracy | 95% CI |", "|---|---|---|---|"])
        for key in (order or sorted(g, key=str)):
            if key not in g:
                continue
            c, m = g[key]
            a, b = wilson(c, m)
            L.append(f"| {key} | {m} | {c / m:.1%} | {a:.0%} to {b:.0%} |")
        L.append("")

    table("By question type", acc_table(rows, lambda r: [r["qtype"]]))
    table("By difficulty tier", acc_table(rows, lambda r: [r["tier"]]), ["easy", "medium", "hard"])

    feat = acc_table(rows, lambda r: [f"{k}={v}" for k, v in bucket(r).items() if k != "tier"])
    L.extend(["## By visual trap (each row is a separate slice, slices overlap)", "",
              "| feature | n | accuracy | 95% CI |", "|---|---|---|---|"])
    for key in sorted(feat):
        c, m = feat[key]
        a, b = wilson(c, m)
        L.append(f"| {key} | {m} | {c / m:.1%} | {a:.0%} to {b:.0%} |")
    L.append("")

    fails = [r for r in rows if not r["correct"]]
    tags = Counter(r["failure_tag"] for r in fails)
    L.extend(["## Failure patterns (ranked)", "", "| pattern | count | share of failures | example ids |", "|---|---|---|---|"])
    for tag, c in tags.most_common():
        ex = ", ".join([r["id"] for r in fails if r["failure_tag"] == tag][:3])
        L.append(f"| {tag} | {c} | {c / max(len(fails), 1):.0%} | {ex} |")
    L.append("")

    worst = [(key, c / m, m) for key, (c, m) in feat.items() if m >= 5]
    worst.sort(key=lambda x: x[1])
    L.extend(["## Hardest conditions (n >= 5)", ""])
    L.extend(f"- `{key}`: {a:.0%} accurate over {m} questions" for key, a, m in worst[:5])
    L.append("")
    return "\n".join(L)


# =========================================================================== main
def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--provider", choices=["mock", "anthropic", "openai"], default="mock")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tier", default="", help="comma list, e.g. hard")
    ap.add_argument("--qtypes", default="", help="comma list, e.g. argmax,read_value")
    a = ap.parse_args()

    items = load_jsonl(a.data / "dataset.jsonl")
    if a.tier:
        items = [i for i in items if i["tier"] in a.tier.split(",")]
    if a.qtypes:
        items = [i for i in items if i["qtype"] in a.qtypes.split(",")]
    if a.limit:
        items = items[:a.limit]
    if not items:
        ap.error("No questions matched the selected filters.")
    if a.workers < 1:
        ap.error("--workers must be at least 1.")

    client = make_client(a.provider, a.model)
    out = a.out or Path("results") / re.sub(r"[^\w.-]+", "_", client.name)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {r["id"] for r in load_jsonl(res_path)} if res_path.exists() else set()
    todo = [i for i in items if i["id"] not in done]
    print(f"{client.name}: {len(todo)} to run, {len(done)} already done")

    lock = threading.Lock()

    def work(item):
        if item["answer_type"] == "category":
            fmt = "The final answer must be exactly one of: " + ", ".join(item["choices"])
        elif item["answer_type"] == "integer":
            fmt = "The final answer must be a single whole number."
        else:
            fmt = "The final answer must be a single number with no units."
        prompt = PROMPT.format(question=item["question"], format_hint=fmt)
        resp, err = ask_with_retry(client, a.data / item["image"], prompt, item)
        parsed = parse_answer(resp, item) if not err else None
        ok = grade(item, parsed)
        row = {"id": item["id"], "qtype": item["qtype"], "tier": item["tier"], "features": item["features"],
               "image": item["image"], "question": item["question"], "answer_type": item["answer_type"],
               "truth": item["answer"], "parsed": parsed, "correct": ok,
               "failure_tag": None if ok else classify_failure(item, parsed, err),
               "response": resp, "error": err}
        with lock, open(res_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for n, fu in enumerate(as_completed(futs), 1):
            fu.result()  # re-raise any worker exception instead of silently dropping rows
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")

    rows = [r for r in load_jsonl(res_path) if r["id"] in {i["id"] for i in items}]
    with open(out / "failures.jsonl", "w") as f:
        for r in rows:
            if not r["correct"]:
                f.write(json.dumps(r) + "\n")
    report = build_report(rows, client.name)
    (out / "report.md").write_text(report)
    print(report)
    print(f"\nWrote {out}/results.jsonl, failures.jsonl, report.md")


if __name__ == "__main__":
    main()