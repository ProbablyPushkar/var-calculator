import yfinance as yf
from curl_cffi import requests
import pandas as pd
from datetime import datetime, timedelta

session = requests.Session(impersonate="chrome")

def get_combined_stock_data(tickers, period_days=120, interval='1d'):
    from datetime import datetime, timedelta

    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_days)

    price_data = []
    failed_tickers = []

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
                session=session,
                auto_adjust=False,
                progress=False,
            )
            if df.empty:
                failed_tickers.append(ticker)
                continue

            df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
            price_data.append(df)

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            failed_tickers.append(ticker)

    if not price_data:
        raise ValueError("No valid stock data could be fetched.")

    combined_df = pd.concat(price_data, axis=1)
    combined_df.columns.name = None
    combined_df.columns = [col if isinstance(col, str) else col[-1] for col in combined_df.columns]
    combined_df.reset_index(inplace=True)
    combined_df.dropna(inplace=True)

    return combined_df, failed_tickers



if __name__ == "__main__":
    tickers = ['AAPL', 'TSLA', 'GOOGL']
    df = get_combined_stock_data(tickers)
    print(df.head())

    # Save to CSV (optional)
    df.to_csv("combined_stock_data.csv", index=False)