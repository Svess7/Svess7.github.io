import sys
import pandas as pd
import yfinance as yf


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with Close, 20-day SMA, and 14-day RSI."""
    # Coerce Close to a Series to handle edge cases (e.g., single-row downloads)
    close = pd.Series(prices["Close"])  # ensures index exists even if scalar
    out = pd.DataFrame({"Close": close})

    # 20-day Simple Moving Average
    out["SMA20"] = close.rolling(window=20, min_periods=20).mean()

    # 14-day RSI (Wilder-style using simple rolling averages for brevity)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    out["RSI14"] = 100 - (100 / (1 + rs))

    return out


def main() -> None:
    """Download data and print last 5 rows of indicators."""
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        print(f"No data downloaded for {ticker}.")
        return

    indicators = compute_indicators(df)[["Close", "SMA20", "RSI14"]]
    print(indicators.tail(5))


if __name__ == "__main__":
    main()


