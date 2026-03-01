import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_watchlist(file_path='watchlist.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fetch_data(symbol, start_date, end_date):
    print(f"\n正在抓取 {symbol} 的數據...")
    df = yf.download(symbol, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def apply_composite_strategy(df):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 複合訊號
    df['Signal'] = 0.0
    condition = (df['SMA20'] > df['SMA50']) & (df['MACD'] > df['Signal_Line']) & (df['RSI'] < 75)
    df.loc[condition, 'Signal'] = 1.0
    
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).fillna(1).cumprod()
    df['Cum_Market_Return'] = (1 + df['Market_Return']).fillna(1).cumprod()
    
    return df

def calculate_metrics(df):
    """計算專業績效指標"""
    if df.empty: return {}
    
    # 1. 總投報率
    total_return = df['Cum_Strategy_Return'].iloc[-1] - 1
    
    # 2. 年化報酬率 (CAGR)
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    cagr = (df['Cum_Strategy_Return'].iloc[-1])**(1/years) - 1
    
    # 3. 最大回撤 (Max Drawdown)
    rolling_max = df['Cum_Strategy_Return'].cummax()
    drawdown = (df['Cum_Strategy_Return'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # 4. 夏普值 (Sharpe Ratio) - 假設無風險利率 2%
    strategy_std = df['Strategy_Return'].std() * np.sqrt(252)
    sharpe = (cagr - 0.02) / strategy_std if strategy_std != 0 else 0
    
    return {
        "total_return": total_return * 100,
        "cagr": cagr * 100,
        "max_drawdown": max_drawdown * 100,
        "sharpe": sharpe
    }

def main():
    watchlist = load_watchlist()
    start = "2020-01-01"
    end = "2025-01-01"
    summary = []
    
    for item in watchlist:
        symbol = item['symbol']
        name = item['name']
        df = fetch_data(symbol, start, end)
        if df.empty or len(df) < 50: continue
            
        df = apply_composite_strategy(df)
        metrics = calculate_metrics(df)
        
        print(f"--- {name} ({symbol}) ---")
        print(f"年化報酬率 (CAGR): {metrics['cagr']:.2f}%")
        print(f"最大跌幅 (MDD): {metrics['max_drawdown']:.2f}%")
        print(f"夏普值 (Sharpe): {metrics['sharpe']:.2f}")
        
        summary.append({"symbol": f"{name} ({symbol})", **metrics})
    
    print("\n" + "="*60)
    print(f"{'股票代號':<25} | {'年化(%)':>8} | {'MDD(%)':>8} | {'夏普值':>8}")
    print("-"*60)
    for s in sorted(summary, key=lambda x: x['cagr'], reverse=True):
        print(f"{s['symbol']:<25} | {s['cagr']:>8.2f}% | {s['max_drawdown']:>8.2f}% | {s['sharpe']:>8.2f}")
    print("="*60)
    print("* 提示: 若年化報酬率 > 10% 且 MDD 較小，則此策略優於一般 ETF。")

if __name__ == "__main__":
    main()
