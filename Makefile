# Convenience targets for macOS/Linux. On Windows run the commands directly (see README).
.PHONY: dev test test-fast lint format smoke

MODEL ?= runs/safedecide-base/best

dev:       ; uv sync
test:      ; uv run pytest -q
test-fast: ; uv run pytest -q -m "not e2e"
lint:      ; uv run ruff check src tests && uv run ruff format --check src tests
format:    ; uv run ruff format src tests && uv run ruff check --fix src tests
smoke:     ; uv run safedecide train --config configs/smoke.yaml
