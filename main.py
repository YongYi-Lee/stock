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

def apply_breakout_strategy(df):
    """
    起漲股策略：帶量突破
    1. 價格突破：今日收盤價 > 過去 20 日的最高價 (Donchian Channel)
    2. 成交量爆發：今日成交量 > 過去 20 日平均成交量的 2 倍
    3. 趨勢過濾：價格 > 60 日均線 (季線)
    4. 停損/停利：股價跌破 20 日均線即出場
    """
    df = df.copy()
    
    # 計算指標
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    df['High20'] = df['High'].shift(1).rolling(window=20).max() # 過去20日最高價
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    
    # --- 起漲訊號邏輯 ---
    df['Signal'] = 0.0
    
    # 買進條件：突破高點 + 倍量 + 季線之上
    buy_condition = (df['Close'] > df['High20']) & \
                    (df['Volume'] > df['Vol_SMA20'] * 2.0) & \
                    (df['Close'] > df['SMA60'])
    
    # 賣出條件 (出場)：跌破 20 日線
    sell_condition = (df['Close'] < df['SMA20'])
    
    # 狀態機器：處理持倉邏輯
    in_position = False
    for i in range(len(df)):
        if not in_position:
            if buy_condition.iloc[i]:
                df.iloc[i, df.columns.get_loc('Signal')] = 1.0
                in_position = True
        else:
            if sell_condition.iloc[i]:
                df.iloc[i, df.columns.get_loc('Signal')] = 0.0
                in_position = False
            else:
                df.iloc[i, df.columns.get_loc('Signal')] = 1.0
                
    # 計算報酬
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).fillna(1).cumprod()
    df['Cum_Market_Return'] = (1 + df['Market_Return']).fillna(1).cumprod()
    
    return df

def calculate_metrics(df):
    if df.empty or len(df) < 2: return {"cagr": 0, "max_drawdown": 0}
    days = (df.index[-1] - df.index[0]).days
    years = max(days / 365.25, 0.1)
    cagr = (df['Cum_Strategy_Return'].iloc[-1])**(1/years) - 1
    rolling_max = df['Cum_Strategy_Return'].cummax()
    drawdown = (df['Cum_Strategy_Return'] - rolling_max) / rolling_max
    return {"cagr": cagr * 100, "max_drawdown": drawdown.min() * 100}

def main():
    watchlist = load_watchlist()
    start = "2020-01-01"
    end = "2025-01-01"
    summary = []
    
    # Benchmark 0050 B&H
    df_0050 = fetch_data("0050.TW", start, end)
    b_h_0050_cagr = ((df_0050['Close'].iloc[-1] / df_0050['Close'].iloc[0])**(1/((df_0050.index[-1] - df_0050.index[0]).days / 365.25)) - 1) * 100
    
    for item in watchlist:
        symbol = item['symbol']
        name = item['name']
        df = fetch_data(symbol, start, end)
        if df.empty or len(df) < 60: continue
            
        df = apply_breakout_strategy(df)
        metrics = calculate_metrics(df)
        summary.append({"symbol": f"{name} ({symbol})", **metrics})
    
    print("\n" + "="*65)
    print(f"{'股票代號':<25} | {'年化報酬(%)':>12} | {'最大回撤(%)':>12}")
    print("-"*65)
    for s in sorted(summary, key=lambda x: x['cagr'], reverse=True):
        win_text = "🚀" if s['cagr'] > b_h_0050_cagr else ""
        print(f"{s['symbol']:<25} | {s['cagr']:>12.2f}% | {s['max_drawdown']:>12.2f}% {win_text}")
    print("="*65)
    print(f"* 基準線 (0050 直接買入持有) 年化報酬率: {b_h_0050_cagr:.2f}%")
    print("* 標記 🚀 代表該策略成功抓到『起漲噴發』並贏過大盤。")

if __name__ == "__main__":
    main()
