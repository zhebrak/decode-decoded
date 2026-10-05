# Everyday commands, all on a laptop. Sessions on a GPU box go through scripts/box.sh (README.md).
#   make check                    tests and lint
#   make dry SESSION=S2           every command of a session, no GPU
#   make rehearse                 the L2 path end to end on this CPU (llama.cpp + AIPerf), no GPU
#   make results SESSIONS="<S2 dir> <S3 dir>" CALIBRATE="<S1 dir>"   cells (S1: ruler, noise, sentinels), patterns.json
#   make page figures             the page (site/) and the post's figures (site/figures/)
UV ?= uv
PY := $(UV) run python
GRID ?= configs/grid/v1.yaml
SESSION ?= S2
SESSIONS ?=
CALIBRATE ?=
CELLS := results/cells.parquet

.PHONY: check test lint dry rehearse results patterns page figures preview fixture-cells sample

check: test lint
test:
	$(UV) run pytest -q
lint:
	$(UV) run ruff check .

dry:
	$(PY) -m anatomy.grid $(GRID) --session $(SESSION) --dry-run

RUNS ?= $(HOME)/decode-anatomy-runs
REHEARSAL := rehearsal-$(shell git rev-parse --short HEAD)   # a session per commit: never an older code's finished unit
rehearse:
	$(PY) -m anatomy.grid configs/grid/rehearsal.yaml --block rehearsal --run --no-gpu --session-id $(REHEARSAL)
	@# a CPU has no ruler: a nominal bandwidth stands in
	$(PY) -m anatomy.tidy $(RUNS)/$(REHEARSAL) --bandwidth 5e10 --out /tmp/decode-anatomy-rehearsal
	@# the smoke gate's items, read on the rehearsal: a laptop has no ruler, no provisioning and no traces, so those
	@# items fail or stay untested here by design; what must pass is that the gate runs and reads the unit
	-$(PY) -m anatomy.gate smoke --session $(RUNS)/$(REHEARSAL) --grid configs/grid/rehearsal.yaml --bandwidth 5e10

results:
	@test -n "$(SESSIONS)" || (echo 'set SESSIONS="<S2 S3 dirs>" CALIBRATE="<S1 dir>"' && exit 2)
	$(PY) -m anatomy.tidy $(SESSIONS) --calibrate $(CALIBRATE) --out results
	$(PY) -m anatomy.report.patterns $(CELLS) --json results/patterns.json

patterns:
	$(PY) -m anatomy.report.patterns $(CELLS) --json results/patterns.json

page:
	@if [ -f $(CELLS) ]; then $(PY) -m anatomy.report.page --cells $(CELLS) --out site; \
	else $(PY) -m anatomy.report.page --out site; fi

figures:
	@if [ -f $(CELLS) ]; then $(PY) -m anatomy.report.static --cells $(CELLS) --out site/figures; \
	else $(PY) -m anatomy.report.static --out site/figures; fi

preview: page
	@echo "open site/index.html"

# the page tests' real cell table (tests/test_report_real.py), rebuilt by the current tidy from the NVL validation
# session's raw data: after any change to tidy's columns
fixture-cells:
	$(PY) -m anatomy.tidy $(RUNS)/nvl-n2-20261001 --calibrate $(RUNS)/nvl-ruler-d-20261001 --out /tmp/da-fixture-cells
	cp /tmp/da-fixture-cells/cells.parquet /tmp/da-fixture-cells/cells.csv /tmp/da-fixture-cells/cell-strips.json \
	   /tmp/da-fixture-cells/strips.json tests/fixtures/cells-nvl/

sample:
	scripts/sample.sh $(ENGINE) $(MODEL)
