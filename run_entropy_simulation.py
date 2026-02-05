from entropy_model import *

# -----------------------
# Load data
# -----------------------
price = pd.read_csv("data/nifty_daily.csv", parse_dates=["Date"])
opt = pd.read_csv("data/option_daily.csv", parse_dates=["Date"])
macro = pd.read_csv("data/macro_events.csv")

df = price.merge(opt, on="Date")
calendar = MacroCalendar(macro)

# -----------------------
# Feature engineering
# -----------------------
df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
df["range"] = (df["High"] - df["Low"]) / df["Close"]
df["vol_z"] = (df["Volume"] - df["Volume"].mean()) / df["Volume"].std()
df.dropna(inplace=True)

market_features = torch.tensor(
    df[["log_ret", "range", "vol_z"]].values,
    dtype=torch.float32
)

option_features = torch.tensor(
    df[["iv_atm", "iv_change", "iv_skew", "pcr_oi", "oi_concentration"]].values,
    dtype=torch.float32
)

dates = df["Date"].tolist()

# -----------------------
# Models (stateful)
# -----------------------
market_enc = MarketEncoder(input_dim=3)
option_enc = OptionEncoder()
state_model = EntropyStateModel()

# -----------------------
# Online simulation
# -----------------------
entropy_series = []
range_prob_series = []

for t in range(len(df)):
    # Market signal
    sig_mkt = market_enc(market_features[t])
    state_model.update(sig_mkt)

    # Option signal + entropy
    sig_opt, opt_entropy = option_enc(option_features[t])
    state_model.update(sig_opt)

    state_model.inject_entropy(
        float(torch.clamp(opt_entropy, 0.0, 0.5))
    )

    # Macro entropy
    macro_entropy = calendar.entropy_effect(dates[t])
    state_model.inject_entropy(macro_entropy)

    entropy_series.append(state_model.entropy().item())
    range_prob_series.append(state_model.state_prob[RANGE_IDX].item())

# -----------------------
# Persist outputs
# -----------------------
df["entropy"] = entropy_series
df["range_prob"] = range_prob_series

df.to_csv("data/entropy_outputs.csv", index=False)

print("Phase-1 entropy simulation completed.")
