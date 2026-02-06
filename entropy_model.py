import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

STATE_NAMES = ["RANGE", "TREND", "BREAKOUT", "PANIC"]
NUM_STATES = len(STATE_NAMES)
RANGE_IDX = 0

def shannon_entropy(p, eps=1e-8):
    p = torch.clamp(p, eps, 1.0)
    return -torch.sum(p * torch.log(p))

def kl_divergence(p, q, eps=1e-8):
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * torch.log(p / q))

class EntropyStateModel:
    def __init__(self):
        self.state_prob = torch.ones(NUM_STATES) / NUM_STATES
        self.prev_state_prob = self.state_prob.clone()

    def update(self, signal_logits):
        logits = torch.log(self.state_prob + 1e-8) + signal_logits
        self.prev_state_prob = self.state_prob.clone()
        self.state_prob = F.softmax(logits, dim=0)

    def inject_entropy(self, intensity):
        if intensity <= 0:
            return
        uniform = torch.ones(NUM_STATES) / NUM_STATES
        self.prev_state_prob = self.state_prob.clone()
        self.state_prob = (1 - intensity) * self.state_prob + intensity * uniform

    def entropy(self):
        return shannon_entropy(self.state_prob)

    def instability(self):
        return kl_divergence(self.state_prob, self.prev_state_prob)

class MarketEncoder(nn.Module):
    def __init__(self, in_dim=None, input_dim=None):
        super().__init__()
        if in_dim is None and input_dim is None:
            raise ValueError("Provide either in_dim or input_dim.")
        if in_dim is None:
            in_dim = input_dim
        self.linear = nn.Linear(in_dim, NUM_STATES)

    def forward(self, x):
        return self.linear(x)

class OptionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, NUM_STATES)

    def forward(self, x):
        state_signal = self.linear(x)
        implied_entropy = 0.6 * x[0] + 0.4 * torch.abs(x[1])
        return state_signal, implied_entropy

class MacroCalendar:
    def __init__(self, df):
        self.events = []

        cols = set(df.columns)
        if {"date", "pre_window", "post_window", "entropy_weight", "transition_shock"}.issubset(cols):
            for _, r in df.iterrows():
                self.events.append({
                    "date": datetime.fromisoformat(str(r["date"])),
                    "pre": int(r["pre_window"]),
                    "post": int(r["post_window"]),
                    "entropy": float(r["entropy_weight"]),
                    "transition": float(r["transition_shock"])
                })
            return

        if "DATE" in cols:
            entropy_cols = {
                "RBI_POLICY": 0.25,
                "FOMC": 0.20,
                "CPI_RELEASE": 0.15,
                "GDP_RELEASE": 0.15,
                "BUDGET_DAY": 0.30,
            }
            for _, r in df.iterrows():
                entropy = 0.0
                for c, w in entropy_cols.items():
                    if c in df.columns:
                        entropy += float(r[c]) * w
                if entropy <= 0:
                    continue
                self.events.append({
                    "date": datetime.fromisoformat(str(r["DATE"])),
                    "pre": 2,
                    "post": 2,
                    "entropy": float(entropy),
                    "transition": float(entropy),
                })
            return

        raise ValueError(
            "Unsupported macro schema. Expected either "
            "['date','pre_window','post_window','entropy_weight','transition_shock'] "
            "or a DATE + event flag columns schema."
        )

    def entropy_effect(self, date):
        total = 0.0
        for e in self.events:
            d = (e["date"] - date).days
            if -e["post"] <= d <= e["pre"]:
                decay = 1 - abs(d) / max(e["pre"], 1)
                total += decay * e["entropy"]
        return total


class RegimeClassifier(nn.Module):
    """
    Supervised regime classifier.
    Consumes windowed features (including entropy).
    """

    def __init__(self, input_dim, window_size, hidden_dim=64, num_classes=3):
        super().__init__()

        self.window_size = window_size
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim * window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, window_size, input_dim)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.net(x)
