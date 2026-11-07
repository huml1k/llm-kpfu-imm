PYTHON ?= .venv/bin/python

.PHONY: check calibrate eval-v1 eval-v2 compare

check:
	$(PYTHON) -m py_compile client_rag.py eval/*.py

calibrate:
	$(PYTHON) eval/calibrate.py

eval-v1:
	$(PYTHON) eval/run.py --version v1

eval-v2:
	$(PYTHON) eval/run.py --version v2

compare:
	$(PYTHON) eval/compare.py
