# TopoFuse — common tasks. Override vars on the command line, e.g.
#   make train DATA=./SYN_dataset DATASET=syn SEED=0 CONFIG=configs/syn.yaml
PYTHON  ?= python3
DATA    ?= ./SYN_dataset
DATASET ?= syn
SEED    ?= 0
CONFIG  ?= configs/syn.yaml
OUT     ?= runs/topofuse

.PHONY: help install test syn train eval paper figures results clean

help:
	@echo "targets: install  test  syn  train  eval  paper  figures  results  clean"

install:
	$(PYTHON) -m pip install -e '.[sam]'

test:
	$(PYTHON) -m pytest -q

syn:
	bash scripts/generate_syn.sh $(DATA)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG) --data-root $(DATA) \
	  --dataset $(DATASET) --seed $(SEED) --out $(OUT)

eval:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --data-root $(DATA) \
	  --dataset $(DATASET) --ckpt $(OUT)/seed$(SEED)/best.pt \
	  --out $(OUT)/seed$(SEED)/eval

paper:
	bash scripts/run_paper.sh

results:
	$(PYTHON) scripts/collect_results.py --runs runs/ --out RESULTS.md

figures:
	$(PYTHON) scripts/make_figures.py --runs runs/ --out figures/

clean:
	rm -rf runs figures RESULTS.md **/__pycache__ .pytest_cache
