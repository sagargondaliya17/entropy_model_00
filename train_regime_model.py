import torch
from torch.utils.data import DataLoader
import pandas as pd

from dataset import EntropyDataset
from entropy_model import RegimeClassifier  # your NN model

# -----------------------
# Load enriched data
# -----------------------
df = pd.read_csv("data/entropy_outputs.csv")

feature_cols = [
    "repo_rate",
    "cpi",
    "gdp",
    "liquidity",
    "volatility",
    "entropy",
    "range_prob",
]

target_col = "market_regime"

# -----------------------
# Dataset & Loader
# -----------------------
dataset = EntropyDataset(
    data=df,
    feature_cols=feature_cols,
    target_col=target_col,
    window_size=5,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

# -----------------------
# Model
# -----------------------
model = RegimeClassifier(
    input_dim=len(feature_cols),
    window_size=5,
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# Training loop
# -----------------------
for epoch in range(20):
    total_loss = 0.0

    for X, y in loader:
        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y.long())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

print("Phase-2 regime training completed.")
