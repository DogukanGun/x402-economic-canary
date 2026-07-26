PY := .venv/bin/python
RESULTS := results

.DEFAULT_GOAL := help

.PHONY: help setup reproduce paper-spec ablations figures report validate test serve mock-asp all clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the pinned venv and install everything
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) -e . --no-deps

reproduce:  ## Reproduce the published numbers (must be bit-for-bit)
	$(PY) -m casper_pay_guard.cli reproduce

paper-spec:  ## Run Section 5 as written, both framings
	$(PY) -m casper_pay_guard.experiment --mode both

ablations:  ## Compute Table 1 and Table 3
	$(PY) -m casper_pay_guard.ablation

figures:  ## Regenerate Figures 3 and 4
	$(PY) -m casper_pay_guard.figures --mode forecast
	$(PY) -m casper_pay_guard.figures --mode oracle

report:  ## Write results/paper_delta.md (runs every configuration)
	$(PY) -m casper_pay_guard.report

validate:  ## Run the 14 validation criteria on every configuration
	$(PY) -m casper_pay_guard.validate

test:  ## Run the full test suite
	$(PY) -m pytest tests/ -q

serve:  ## Run the priced MCP stdio server
	$(PY) -m casper_pay_guard.mcp_server

mock-asp:  ## Run the misbehaving mock ASP on :8402
	$(PY) -m casper_pay_guard.mock_asp

all: reproduce paper-spec ablations figures report validate test  ## Everything

clean:  ## Remove generated artifacts
	rm -rf $(RESULTS) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
