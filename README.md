# Amazon Support Agent — Hiver take-home

An AI triage layer for `AmazonHelp`'s public Twitter support: classify the
customer's intent, decide auto-handle vs. escalate-to-human (with a stated
reason), and draft a reply grounded in how this brand has actually resolved
similar issues before.

The full write-up — problem framing, baselines, failure analysis, "what's
misleading about my headline number", decision log — is in **[REPORT.md](REPORT.md)**
and **[DECISIONS.md](DECISIONS.md)**. This file is reproduction only.

## Reproducing in under 15 minutes

Two paths. Pick one.

### A. Offline replay (recommended, ~2 minutes, no API key)

Every LLM call made while building this is committed at
`artifacts/cache/llm_cache.jsonl`. This replays them — no network, no cost,
byte-identical to the numbers in the report.

```bash
make setup                 # venv + deps, ~1 min
make test                  # every module's self-check, no network
make reproduce             # replays the cache -> predictions + metrics + agreement
cat artifacts/results/tables.md
```

`make reproduce` sets `LLM_OFFLINE=1`. A cache miss raises immediately instead
of silently calling out, so you'll know if something needed a call the cache
doesn't have (it shouldn't, for the committed golden set).

### B. Full rebuild from raw data (~12–15 minutes, needs an LLM)

```bash
make setup
make data                  # downloads the 516MB TWCS csv (~2 min), verifies size+header
make corpus                # brand corpus + temporal split, ~90s
make golden                # samples the golden slices (already committed; reruns identically)
export ANTHROPIC_API_KEY=sk-...   # or: export LLM_BACKEND=cli  (uses `claude -p`)
make eval                  # runs B0/B1/B2/AGENT + judge over all 220 golden cases, ~8-10 min
make metrics
make agreement
```

Set `LLM_BACKEND=api` (default when `ANTHROPIC_API_KEY` is set) or
`LLM_BACKEND=cli` to use a local Claude Code subscription instead of a key.
Either way, every call is cached in `artifacts/cache/llm_cache.jsonl` on the way
past, so a second run costs nothing.

## What's in the repo

```
src/
  textnorm.py     text cleaning + PII scrubbing + language gate (audited, see notes/lang_audit.md)
  taxonomy.py     12 intents + 7 escalation rules + cost model — fixed before any labelling
  retrieval.py    TF-IDF nearest-neighbour over the brand's historical cases
  agent.py        classify + retrieve + draft + route, one LLM call + deterministic guardrails
  baselines.py    B0 majority, B1 keyword-rules, B2 retrieval-copy
  judge.py        LLM-as-judge reply-quality rubric (grounded/helpful/tone/safe + hallucination flag)
  llm.py          cached client, two backends (api / cli), used by everything above

scripts/
  fetch_data.py         downloads + verifies the raw dataset
  build_corpus.py       brand selection + cleaning + temporal train/eval split
  sample_golden.py      draws the golden random + enriched slices from the held-out pool
  prelabel_golden.py    pass A: pre-annotates the golden set (sees the brand's real reply)
  prelabel_pass_b.py    pass B: independent second pre-annotation (does not see the reply)
  adjudicate.py         interactive CLI — the only script that writes gold labels
  rate_replies.py       interactive CLI — blind human rating of reply quality
  run_eval.py           runs all 4 systems + judge over the golden set
  metrics.py            accuracy / macro-F1 / routing errors / cost, per slice
  agreement.py          judge-vs-human agreement, pass A/B agreement, gold-label provenance
  make_tables.py        renders REPORT.md's tables from the metrics/agreement JSON

data/
  raw/                  (gitignored — 516MB, fetched by fetch_data.py)
  processed/             corpus.jsonl (retrieval KB) + eval_pool.jsonl (held-out) + split.json
  golden/                unlabelled_{random,enriched}.jsonl, preannotated.jsonl, labelled.jsonl

artifacts/
  cache/llm_cache.jsonl  every LLM call made in this project — the reproduction artifact
  results/               predictions.jsonl, metrics.json, agreement.json, tables.md

notes/lang_audit.md      hand audit of the language-filter heuristic
DECISIONS.md             10-15+ non-obvious decisions and why
REPORT.md                the graded report
```

## Data provenance

Primary dataset per the assignment: Kaggle
[`thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(the "TWCS" corpus, ~2.8M tweets). `scripts/fetch_data.py` pulls the identical
file from a public Hugging Face mirror
([`SunidhiSriram/twcs`](https://huggingface.co/datasets/SunidhiSriram/twcs))
rather than Kaggle directly, so reproduction needs no Kaggle credentials; the
fetcher verifies byte count and header before use.

## Brand: AmazonHelp

Chosen by measurement, not popularity — see [DECISIONS.md](DECISIONS.md#data-and-brand)
item 1 and [REPORT.md](REPORT.md) for the full comparison. In short: the
biggest accounts (`AppleSupport`, `TMobileHelp`) mostly deflect customers to
DM/phone rather than resolving in-thread, which leaves nothing to ground a
reply-drafting agent in. AmazonHelp resolves in public at a measured 5.3%
deflection rate with 168k usable historical reply pairs.

## What "make reproduce" actually checks

`make test` runs every module's `demo()`/`__main__` self-check — the guardrail
logic (`agent.py`), the text/PII scrubbing (`textnorm.py`), the metrics math
(`agreement.py`) — against fixed inputs, no network, in a few seconds. This is
the fast signal that the *logic* still behaves; `make reproduce` is the slow
signal that the *numbers* still match.
