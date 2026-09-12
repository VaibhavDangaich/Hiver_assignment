# Reproduce the headline numbers. See README "Reproducing in under 15 minutes".
PY := .venv/bin/python
BRAND ?= AmazonHelp

.PHONY: help setup test data corpus golden eval metrics agreement reproduce clean

help:
	@grep -E '^[a-z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## create venv and install deps
	python3 -m venv .venv && $(PY) -m pip -q install --upgrade pip && \
	$(PY) -m pip -q install -r requirements.txt

test: ## run every module self-check (no network, no API key)
	$(PY) src/textnorm.py && $(PY) src/taxonomy.py && $(PY) src/retrieval.py && \
	$(PY) src/baselines.py && $(PY) src/agent.py && $(PY) src/judge.py && \
	$(PY) scripts/agreement.py --demo
	@echo "all self-checks passed"

data: ## download the raw TWCS csv (516MB, ~2 min)
	$(PY) scripts/fetch_data.py

corpus: ## build the brand corpus + temporal split (~90s)
	$(PY) scripts/build_corpus.py --brand $(BRAND)

golden: ## sample the golden slices from the held-out pool
	$(PY) scripts/sample_golden.py

eval: ## run all systems + judge over the golden set
	$(PY) scripts/run_eval.py

metrics: ## compute headline metrics
	$(PY) scripts/metrics.py

agreement: ## judge-vs-human agreement + gold provenance
	$(PY) scripts/agreement.py

# The graded path: replays the committed LLM cache, so it needs no API key and
# no network. Fails loudly rather than silently calling out if the cache misses.
reproduce: ## headline numbers from the committed cache (offline, ~60s)
	LLM_OFFLINE=1 $(PY) scripts/run_eval.py
	LLM_OFFLINE=1 $(PY) scripts/metrics.py
	LLM_OFFLINE=1 $(PY) scripts/agreement.py

clean:
	rm -rf artifacts/results/*.jsonl artifacts/results/*.json
