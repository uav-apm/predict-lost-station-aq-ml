# Testing Guide

This project uses `pytest`.

## Setup

Create and activate the virtual environment, then install the project in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Run All Tests

```powershell
python -m pytest
```

Or explicitly through the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Run Targeted Test Files

```powershell
python -m pytest tests/test_data.py
python -m pytest tests/test_cli.py
python -m pytest tests/test_sample_data_smoke.py
```

## Run One Test

```powershell
python -m pytest tests/test_cli.py::test_period_helpers_support_year_month_and_day_ranges
```

## Useful Test Groups

- `tests/test_data.py`: loading, normalization, imputation, and column statistics
- `tests/test_cli.py`: config and CLI helper behavior
- `tests/test_pipeline_config_output.py`: artifact creation
- `tests/test_sample_data_smoke.py`: end-to-end train/eval/predict behavior
- `tests/test_backends.py`: model backend resolution
- `tests/test_metrics.py`: metric calculations
- `tests/test_predict_scalar.py`: scalar prediction path

## If Temp Directory Permissions Fail

In this workspace, pytest may hit Windows permission issues in the default temp/cache location. If that happens, run with a writable local temp directory:

```powershell
$env:TEMP="$PWD\tests\_tmp\python-temp"
$env:TMP="$PWD\tests\_tmp\python-temp"
.\.venv\Scripts\python.exe -m pytest
```

If you want to limit the run while debugging:

```powershell
$env:TEMP="$PWD\tests\_tmp\python-temp"
$env:TMP="$PWD\tests\_tmp\python-temp"
.\.venv\Scripts\python.exe -m pytest tests/test_data.py
```

## Notes

- The smoke tests copy the example config into a temporary folder before running, so they do not depend on editing the checked-in example profile directly.
- After changing package paths or imports, it is useful to run:

```powershell
.\.venv\Scripts\python.exe -m compileall src tests
```

before running the full test suite.
