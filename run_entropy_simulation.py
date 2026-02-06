import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from entropy_model import (
    EntropyStateModel,
    MacroCalendar,
    MarketEncoder,
    OptionEncoder,
    RANGE_IDX,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run phase-1 entropy simulation. Supports large local options file that is not "
            "tracked in git."
        )
    )
    parser.add_argument("--price-path", default="data/nifty_daily.csv")
    parser.add_argument("--macro-path", default="data/macro_events.csv")
    parser.add_argument(
        "--options-path",
        default="data/options_daily.csv",
        help=(
            "Path to local options dataset. Keep this file local (gitignored) if it is too "
            "large for GitHub. If not found, simulation runs in market+macro mode."
        ),
    )
    parser.add_argument("--output-path", default="data/entropy_outputs.csv")
    return parser.parse_args()


def _clean_price_frame(price: pd.DataFrame) -> pd.DataFrame:
    price.columns = [c.replace("\ufeff", "") for c in price.columns]
    price["Date"] = pd.to_datetime(price["Date"], dayfirst=True, errors="coerce")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in price.columns:
            price[c] = pd.to_numeric(price[c], errors="coerce")

    price = price.dropna(subset=["Date", "Close", "High", "Low"]).copy()
    price = price.sort_values("Date").reset_index(drop=True)

    if "Volume" not in price.columns or price["Volume"].isna().all():
        price["Volume"] = 0.0
    else:
        price["Volume"] = price["Volume"].ffill().fillna(0.0)

    return price


def _load_options_frame(path: str) -> pd.DataFrame:
    opt = pd.read_csv(path)
    opt.columns = [c.replace("\ufeff", "") for c in opt.columns]

    if "Date" not in opt.columns:
        raise ValueError(f"Options file at {path} must contain a Date column.")

    required = ["iv_atm", "iv_change", "iv_skew", "pcr_oi", "oi_concentration"]
    missing = [c for c in required if c not in opt.columns]
    if missing:
        raise ValueError(
            f"Options file at {path} is missing required columns: {missing}."
        )

    opt["Date"] = pd.to_datetime(opt["Date"], dayfirst=True, errors="coerce")
    for c in required:
        opt[c] = pd.to_numeric(opt[c], errors="coerce")

    opt = opt.dropna(subset=["Date"])
    opt = opt.sort_values("Date")
    return opt[["Date", *required]]


def main():
    args = parse_args()

    price = _clean_price_frame(pd.read_csv(args.price_path))
    macro = pd.read_csv(args.macro_path)
    macro["DATE"] = pd.to_datetime(macro["DATE"], errors="coerce")
    calendar = MacroCalendar(macro)

    # Optional local options file for richer entropy signals.
    options_path = Path(args.options_path)
    use_options = options_path.exists()

    if use_options:
        opt = _load_options_frame(str(options_path))
        df = price.merge(opt, on="Date", how="left")
    else:
        df = price.copy()

    # Market features
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["range"] = (df["High"] - df["Low"]) / df["Close"]
    vol_std = df["Volume"].std()
    if vol_std and not np.isnan(vol_std) and vol_std > 0:
        df["vol_z"] = (df["Volume"] - df["Volume"].mean()) / vol_std
    else:
        df["vol_z"] = 0.0

    # Macro features for phase-2 compatibility.
    df = df.merge(macro, left_on="Date", right_on="DATE", how="left")
    for c in ["RBI_POLICY", "FOMC", "CPI_RELEASE", "GDP_RELEASE", "BUDGET_DAY"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["repo_rate"] = df["RBI_POLICY"].astype(float)
    df["cpi"] = df["CPI_RELEASE"].astype(float)
    df["gdp"] = df["GDP_RELEASE"].astype(float)
    df["liquidity"] = df["FOMC"].astype(float)
    df["volatility"] = df["BUDGET_DAY"].astype(float)

    df = df.dropna(subset=["log_ret", "range", "vol_z"]).copy()

    market_features = torch.tensor(
        df[["log_ret", "range", "vol_z"]].values,
        dtype=torch.float32,
    )

    if use_options:
        option_cols = ["iv_atm", "iv_change", "iv_skew", "pcr_oi", "oi_concentration"]
        for c in option_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        option_features = torch.tensor(df[option_cols].values, dtype=torch.float32)
        option_encoder = OptionEncoder()
    else:
        option_features = None
        option_encoder = None

    market_enc = MarketEncoder(input_dim=3)
    state_model = EntropyStateModel()

    entropy_series = []
    range_prob_series = []

    dates = df["Date"].tolist()
    for t in range(len(df)):
        sig_mkt = market_enc(market_features[t])
        state_model.update(sig_mkt)

        if option_encoder is not None and option_features is not None:
            sig_opt, opt_entropy = option_encoder(option_features[t])
            state_model.update(sig_opt)
            state_model.inject_entropy(float(torch.clamp(opt_entropy, 0.0, 0.5)))

        macro_entropy = calendar.entropy_effect(dates[t].to_pydatetime())
        state_model.inject_entropy(float(np.clip(macro_entropy, 0.0, 0.5)))

        entropy_series.append(state_model.entropy().item())
        range_prob_series.append(state_model.state_prob[RANGE_IDX].item())

    df["entropy"] = entropy_series
    df["range_prob"] = range_prob_series

    # Weak labels so phase-2 can train end-to-end.
    df["market_regime"] = np.select(
        [
            df["log_ret"] > 0.01,
            df["log_ret"] < -0.01,
        ],
        [1, 2],
        default=0,
    ).astype(int)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)

    mode = "market+macro+options" if use_options else "market+macro"
    print(f"Phase-1 entropy simulation completed ({mode}). Output: {args.output_path}")
    if not use_options:
        print(
            "Note: options file not found. Place local file at "
            f"{args.options_path} or pass --options-path /your/path/options_daily.csv"
        )


if __name__ == "__main__":
    main()
