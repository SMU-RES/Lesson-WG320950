from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
CSV_PATH = DATA_DIR / "financial_data.csv"
PLOT_PATH = OUTPUT_DIR / "result_plot.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download financial data and train a simple RNN model."
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to download.")
    parser.add_argument("--period", default="5y", help="Download period for yfinance.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
        help="Number of time steps used to predict the next value.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--hidden-size", type=int, default=32, help="RNN hidden size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_financial_data(ticker: str, period: str) -> pd.DataFrame:
    downloaded = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if downloaded is None or downloaded.empty:
        raise RuntimeError(f"No data downloaded for ticker {ticker!r}.")

    df = cast(pd.DataFrame, downloaded)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise RuntimeError(f"Downloaded dataset is missing columns: {missing_columns}")

    selected_df = df[required_columns].dropna().reset_index(drop=True)
    return cast(pd.DataFrame, selected_df)


def min_max_scale(values: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
    denominator = data_max - data_min
    if denominator == 0:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - data_min) / denominator).astype(np.float32)


def inverse_min_max_scale(values: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
    return values * (data_max - data_min) + data_min


def create_sequences(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[float] = []

    for start_index in range(len(values) - sequence_length):
        end_index = start_index + sequence_length
        features.append(values[start_index:end_index])
        targets.append(values[end_index])

    if not features:
        raise RuntimeError(
            "Not enough data points to create sequences. Increase the download period or reduce sequence length."
        )

    x_array = np.asarray(features, dtype=np.float32).reshape(-1, sequence_length, 1)
    y_array = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    return x_array, y_array


def prepare_datasets(
    close_values: np.ndarray, sequence_length: int
) -> tuple[TensorDataset, TensorDataset, np.ndarray, np.ndarray]:
    split_index = int(len(close_values) * 0.8)
    if split_index <= sequence_length or len(close_values) - split_index <= 1:
        raise RuntimeError("Dataset is too small for the requested train/test split.")

    train_series = close_values[:split_index]
    train_min = float(train_series.min())
    train_max = float(train_series.max())

    scaled_all = min_max_scale(close_values, train_min, train_max)
    train_scaled = scaled_all[:split_index]
    test_scaled = scaled_all[split_index - sequence_length :]

    x_train, y_train = create_sequences(train_scaled, sequence_length)
    x_test, y_test = create_sequences(test_scaled, sequence_length)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    return train_dataset, test_dataset, y_test, np.array([train_min, train_max], dtype=np.float32)


class SimpleRNNRegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        initial_hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=inputs.device,
        )
        sequence_output, _ = self.rnn(inputs, initial_hidden)
        return self.output(sequence_output[:, -1, :])


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, ...]],
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_examples = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_inputs.size(0)
            seen_examples += batch_inputs.size(0)

        epoch_loss = running_loss / seen_examples
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:>3}/{epochs}: train_loss={epoch_loss:.6f}")


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader[tuple[torch.Tensor, ...]],
    device: torch.device,
    scale_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_predictions = model(batch_inputs).cpu().numpy()
            predictions.append(batch_predictions)
            targets.append(batch_targets.numpy())

    scaled_predictions = np.vstack(predictions).squeeze(-1)
    scaled_targets = np.vstack(targets).squeeze(-1)
    train_min, train_max = float(scale_bounds[0]), float(scale_bounds[1])

    actual_values = inverse_min_max_scale(scaled_targets, train_min, train_max)
    predicted_values = inverse_min_max_scale(scaled_predictions, train_min, train_max)

    mse = float(np.mean((predicted_values - actual_values) ** 2))
    mae = float(np.mean(np.abs(predicted_values - actual_values)))
    return actual_values, predicted_values, mse, mae


def save_plot(dates: pd.Series, actual_values: np.ndarray, predicted_values: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual_values, label="Actual Close Price", linewidth=2)
    plt.plot(dates, predicted_values, label="Predicted Close Price", linewidth=2)
    plt.title("Simple RNN Prediction on Financial Data")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data for ticker={args.ticker}, period={args.period}...")
    data_frame = download_financial_data(args.ticker, args.period)
    data_frame.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved dataset to: {CSV_PATH}")
    print(data_frame.head())
    print(f"Dataset rows: {len(data_frame)}")

    close_values = data_frame["Close"].to_numpy(dtype=np.float32)
    train_dataset, test_dataset, _, scale_bounds = prepare_datasets(
        close_values, args.sequence_length
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleRNNRegressor(hidden_size=args.hidden_size).to(device)

    print("\nModel architecture:")
    print(model)

    train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    actual_values, predicted_values, mse, mae = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        scale_bounds=scale_bounds,
    )

    test_dates = data_frame["Date"].iloc[len(data_frame) - len(actual_values) :]
    save_plot(test_dates, actual_values, predicted_values)

    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Saved result plot to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
