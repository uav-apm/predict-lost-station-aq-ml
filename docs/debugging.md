# Debugging and Development

## Useful Entry Points

- [`../src/cli.py`](../src/cli.py)
- [`../src/data.py`](../src/data.py)
- [`../src/forecasting.py`](../src/forecasting.py)
- [`../src/pipeline.py`](../src/pipeline.py)

## Debugging Training

Common breakpoints:

- `train_cmd()`
- `_prepare_training_data()`
- `_build_supervised_frame()`
- `build_supervised_dataset()`
- `StationForecastPipeline.fit()`

Example launch pattern:

```text
Program: .venv\Scripts\python.exe
Arguments: -c "import sys; from src.cli import train_cmd; sys.argv=['aq-spatial-reconstruction-train','--config','configs/delhi/delhi.yaml']; train_cmd()"
Working directory: d:\source\research\aq-spatial-reconstruction
```

## Debugging in the Terminal

```powershell
python -m pdb -c continue -c "b src.cli.train_cmd" -c "b src.cli._prepare_training_data" -c "b src.forecasting.build_supervised_dataset" -c "b src.pipeline.StationForecastPipeline.fit" -c "run" -c "import sys; sys.argv=['aq-spatial-reconstruction-train','--config','configs/delhi/delhi.yaml']; from src.cli import train_cmd; train_cmd()"
```

## Dumping Pre-Fit Inputs

Set this in the config:

```yaml
artifacts:
  dump_training_data: true
```

That writes `training_dump/` with:

- supervised rows
- raw feature matrix
- model-input feature matrix
- target values
- station matrix
- summary metadata

## Tests

Run:

```powershell
python -m pytest
```

In this workspace, pytest temp/cache cleanup may need a writable local temp directory if the default temp location is permission-restricted.

Prefer keeping test temp files under `tests/_tmp`:

```powershell
$env:TEMP="$PWD\tests\_tmp\python-temp"
$env:TMP="$PWD\tests\_tmp\python-temp"
python -m pytest
```
