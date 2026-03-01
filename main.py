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

def apply_strategy(df):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    df['Signal'] = 0.0
    df.loc[df.index[20:], 'Signal'] = np.where(df['SMA20'][20:] > df['SMA50'][20:], 1.0, 0.0)
    
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)
    
    df['Cum_Market_Return'] = (1 + df['Market_Return']).cumprod()
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    return df

def backtest(df, symbol_name, initial_capital=1000000):
    final_return = df['Cum_Strategy_Return'].iloc[-1]
    final_value = initial_capital * final_return
    roi = (final_return - 1) * 100
    
    print(f"--- {symbol_name} 回測結果 ---")
    print(f"最終資產: ${final_value:,.0f} | 總投報率 (ROI): {roi:.2f}%")
    return {"symbol": symbol_name, "roi": roi, "final_value": final_value}

def plot_results(df, symbol, name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(df['Close'], label='Price', color='black', alpha=0.3)
    ax1.plot(df['SMA20'], label='SMA 20', color='blue')
    ax1.plot(df['SMA50'], label='SMA 50', color='red')
    
    buy_signals = df[(df['Signal'] == 1) & (df['Signal'].shift(1) == 0)]
    sell_signals = df[(df['Signal'] == 0) & (df['Signal'].shift(1) == 1)]
    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy')
    ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell')
    
    ax1.set_title(f'{name} ({symbol}) Price and SMA Strategy')
    ax1.legend()
    
    ax2.plot(df['Cum_Strategy_Return'], label='Strategy Return', color='orange')
    ax2.plot(df['Cum_Market_Return'], label='Market Return', color='gray', linestyle='--')
    ax2.set_title(f'{name} Cumulative Returns')
    ax2.legend()
    
    plt.tight_layout()
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = f"{output_dir}/{symbol}_backtest.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"圖表已儲存至: {plot_path}")

def main():
    watchlist = load_watchlist()
    start = "2020-01-01"
    end = "2025-01-01"
    
    summary = []
    
    for item in watchlist:
        symbol = item['symbol']
        name = item['name']
        
        df = fetch_data(symbol, start, end)
        if df.empty:
            continue
            
        df = apply_strategy(df)
        result = backtest(df, f"{name} ({symbol})")
        plot_results(df, symbol, name)
        summary.append(result)
    
    print("\n" + "="*30)
    print("      所有自選股回測總結")
    print("="*30)
    for s in sorted(summary, key=lambda x: x['roi'], reverse=True):
        print(f"{s['symbol']:<20}: {s['roi']:>8.2f}%")

if __name__ == "__main__":
    main()
