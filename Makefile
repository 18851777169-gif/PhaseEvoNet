.PHONY: smoke test clean

smoke:
	python -m phase_evonet.cli smoke --output-dir .tmp/smoke --seed 42
	python -m pytest -q -p no:cacheprovider tests/test_smoke.py

test:
	python -m pytest -q

clean:
	python -c "import shutil; from pathlib import Path; targets = [Path('.pytest_cache'), Path('.tmp'), Path('build'), Path('dist'), *Path('.').glob('*.egg-info'), *Path('src').glob('*.egg-info')]; [shutil.rmtree(path, ignore_errors=True) for path in targets]"
