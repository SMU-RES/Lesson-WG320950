# Financial RNN Assignment

This project downloads financial time-series data, saves the dataset to CSV, trains a simple recurrent neural network, prints the model architecture, evaluates predictions, and saves a comparison plot.

## Environment setup

```powershell
uv venv
uv pip install --python .\.venv\Scripts\python.exe -r .\requirements.txt
```

## Run the project

```powershell
.\.venv\Scripts\python .\src\train_rnn.py
```

Optional arguments:

```powershell
.\.venv\Scripts\python .\src\train_rnn.py --ticker MSFT --period 5y --sequence-length 20 --epochs 60
```

## Outputs

- `data/financial_data.csv`: downloaded financial dataset
- `outputs/result_plot.png`: actual vs predicted close prices
- `outputs/screenshots/`: auto-generated screenshot materials for submission

The script also prints:

- dataset summary
- RNN model architecture
- train loss during training
- test MSE and MAE

## Generate screenshot materials

```powershell
.\.venv\Scripts\python .\src\generate_screenshots.py
```

This creates:

- `outputs/screenshots/dataset_screenshot.png`
- `outputs/screenshots/code_screenshot.png`
- `outputs/screenshots/network_structure_screenshot.png`
- `outputs/screenshots/submission_collage.png`

The required personal picture from the digital platform still needs to be added manually by you.
