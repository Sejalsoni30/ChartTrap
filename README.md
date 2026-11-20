# ChartTrap

ChartTrap is a synthetic benchmark for a practical business-analytics problem: people and
vision models can misread chart values, axes, and visual encodings. In dashboards, mistaking
a logarithmic scale for a linear one, overlooking a truncated baseline, confusing dual axes,
or comparing bar heights instead of the labeled values can lead to incorrect forecasts,
budget decisions, and performance reviews.

## What it does

- **`generate.py`** samples business-themed data and renders bar, grouped-bar, and dual-axis
  charts. It creates questions about values, categories, counts, gaps, and ratios. It then
  scans the rendered PNG pixels independently of the source data, checks mark positions
  against the expected values, and re-derives each question's answer from those pixel
  measurements. Questions that fail the pixel audit are discarded.
- **`evaluate.py`** sends each image and question to a selected provider, parses the final
  answer, and grades it against the reference answer (with an allowed interval for numeric
  estimates). It writes per-question results and incorrect examples, and groups failures
  into visual-reading patterns such as log-scale, dual-axis, truncated-axis, category, and
  magnitude errors. The `mock` provider is only an offline plumbing test, not a model score.

Generated datasets and evaluation outputs are local artifacts and are ignored by Git.

## Setup (PowerShell)

From `D:\charttrap`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.10 or newer is required.

## Generate, revalidate, evaluate

Generate 30 charts with a fixed seed:

```powershell
python generate.py --out data --n 30 --seed 7
```

Re-measure the saved PNGs and validate all stored answers against the pixels:

```powershell
python generate.py --revalidate data
```

Run the offline pipeline test first:

```powershell
python evaluate.py --data data --provider mock
```

Results are written under `results\mock` (or the selected provider/model directory) as
`results.jsonl`, `failures.jsonl`, and `report.md`. Existing results are resumed by question
ID; remove the selected results directory to start that evaluation from scratch.

## Real vision-model providers

Install dependencies above, then provide an API key through the current PowerShell session.
Do not paste keys into source files or commit them.

Anthropic:

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
python evaluate.py --data data --provider anthropic --model claude-sonnet-5-5
```

OpenAI:

```powershell
$env:OPENAI_API_KEY = "your-key"
python evaluate.py --data data --provider openai --model gpt-4o
```

The evaluator checks that the provider key is set and records final request errors as
`api_error` failures. Transient connection, timeout, rate-limit, and server errors are
retried; permanent request errors are not.

## Useful options

```powershell
python evaluate.py --data data --provider mock --tier hard --limit 10
python evaluate.py --data data --provider mock --qtypes argmax,read_value
python evaluate.py --data data --provider mock --workers 2 --out results\mock-small
```

Use `python generate.py --help` and `python evaluate.py --help` for all options.
