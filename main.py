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

def apply_pro_strategy(df):
    """
    專業複合策略：SMA + MACD + RSI + Volume
    1. 趨勢：SMA 20 > 50
    2. 動能：MACD 快線 > 慢線
    3. 安全：RSI < 80 (避免極度過熱)
    4. 實力：今日成交量 > 5日平均成交量 (量增確認)
    """
    df = df.copy()
    
    # 指標計算
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
    
    # 成交量均線
    df['Vol_SMA5'] = df['Volume'].rolling(window=5).mean()
    
    # --- 專業訊號邏輯 ---
    df['Signal'] = 0.0
    condition = (df['SMA20'] > df['SMA50']) & \
                (df['MACD'] > df['Signal_Line']) & \
                (df['RSI'] < 80) & \
                (df['Volume'] > df['Vol_SMA5']) # 量增確認
    
    df.loc[condition, 'Signal'] = 1.0
    
    # 計算報酬
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).fillna(1).cumprod()
    df['Cum_Market_Return'] = (1 + df['Market_Return']).fillna(1).cumprod()
    
    return df

def calculate_metrics(df):
    if df.empty or len(df) < 2: return {"cagr": 0, "max_drawdown": 0, "sharpe": 0}
    
    # 年化報酬率 (CAGR)
    days = (df.index[-1] - df.index[0]).days
    years = max(days / 365.25, 0.1)
    cagr = (df['Cum_Strategy_Return'].iloc[-1])**(1/years) - 1
    
    # 最大回撤 (MDD)
    rolling_max = df['Cum_Strategy_Return'].cummax()
    drawdown = (df['Cum_Strategy_Return'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # 夏普值 (Sharpe)
    returns = df['Strategy_Return'].dropna()
    if len(returns) > 0 and returns.std() != 0:
        sharpe = (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252))
    else:
        sharpe = 0
    
    return {
        "cagr": cagr * 100,
        "max_drawdown": max_drawdown * 100,
        "sharpe": sharpe
    }

def main():
    watchlist = load_watchlist()
    start = "2020-01-01"
    end = "2025-01-01"
    summary = []
    
    # 基準參考：0050 Buy & Hold
    print("\n--- 正在計算 0050 直接買入持有作為 Benchmark ---")
    df_0050 = fetch_data("0050.TW", start, end)
    b_h_0050_return = (df_0050['Close'].iloc[-1] / df_0050['Close'].iloc[0]) - 1
    b_h_0050_cagr = (1 + b_h_0050_return)**(1 / ((df_0050.index[-1] - df_0050.index[0]).days / 365.25)) - 1
    
    for item in watchlist:
        symbol = item['symbol']
        name = item['name']
        df = fetch_data(symbol, start, end)
        if df.empty or len(df) < 50: continue
            
        df = apply_pro_strategy(df)
        metrics = calculate_metrics(df)
        summary.append({"symbol": f"{name} ({symbol})", **metrics})
    
    print("\n" + "="*70)
    print(f"{'股票代號':<25} | {'年化(%)':>8} | {'MDD(%)':>8} | {'夏普值':>8}")
    print("-"*70)
    for s in sorted(summary, key=lambda x: x['cagr'], reverse=True):
        win_text = "⭐" if s['cagr'] > b_h_0050_cagr * 100 else ""
        print(f"{s['symbol']:<25} | {s['cagr']:>8.2f}% | {s['max_drawdown']:>8.2f}% | {s['sharpe']:>8.2f} {win_text}")
    print("="*70)
    print(f"* 基準線 (0050 直接買入持有) 年化報酬率: {b_h_0050_cagr*100:.2f}%")
    print("* 標記 ⭐ 代表該策略成功贏過 0050 直接買入持有。")

if __name__ == "__main__":
    main()
