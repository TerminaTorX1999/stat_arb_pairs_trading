# pairs_trading_stat_arb.py

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint

# ---------------------------
# 1. Data Extraction
# ---------------------------
symbols = ["MSFT", "AAPL"]
start_date = "2019-01-01"
end_date = "2021-12-31"

data = yf.download(symbols, start=start_date, end=end_date)["Adj Close"]
data.dropna(inplace=True)

# ---------------------------
# 2. Cointegration Test
# ---------------------------
score, pvalue, _ = coint(data[symbols[0]], data[symbols[1]])
print("Cointegration p-value:", pvalue)

# ---------------------------
# 3. Spread Construction
# ---------------------------
hedge_ratio = np.polyfit(data[symbols[1]], data[symbols[0]], 1)[0]
data["Spread"] = data[symbols[0]] - hedge_ratio * data[symbols[1]]

# ---------------------------
# 4. Z-score Calculation
# ---------------------------
window = 20
data["Spread_Mean"] = data["Spread"].rolling(window).mean()
data["Spread_Std"] = data["Spread"].rolling(window).std()
data["Zscore"] = (data["Spread"] - data["Spread_Mean"]) / data["Spread_Std"]

data.dropna(inplace=True)

# ---------------------------
# 5. Signal Generation
# ---------------------------
entry_threshold = 2.0
exit_threshold = 0.5

data["Long_Spread"] = data["Zscore"] < -entry_threshold
data["Short_Spread"] = data["Zscore"] > entry_threshold
data["Exit"] = abs(data["Zscore"]) < exit_threshold

# ---------------------------
# 6. Position Management
# ---------------------------
position = 0
positions = []

for _, row in data.iterrows():
    if position == 0:
        if row["Long_Spread"]:
            position = 1
        elif row["Short_Spread"]:
            position = -1
    elif row["Exit"]:
        position = 0
    positions.append(position)

data["Position"] = positions

# ---------------------------
# 7. Backtest
# ---------------------------
data["Return_1"] = data[symbols[0]].pct_change()
data["Return_2"] = data[symbols[1]].pct_change()

data["Strategy_Return"] = data["Position"].shift(1) * (
    data["Return_1"] - hedge_ratio * data["Return_2"]
)

data["Cumulative_Strategy"] = (1 + data["Strategy_Return"]).cumprod()

# ---------------------------
# 8. Risk Controls
# ---------------------------
max_drawdown = (
    data["Cumulative_Strategy"] / data["Cumulative_Strategy"].cummax() - 1
).min()

print("Max Drawdown:", max_drawdown)

# ---------------------------
# 9. Visualization
# ---------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(data["Cumulative_Strategy"], label="Pairs Trading Strategy")
plt.legend()
plt.title("Statistical Arbitrage / Pairs Trading")
plt.show()
