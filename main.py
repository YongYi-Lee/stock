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
    """
    複合策略：SMA + MACD + RSI
    1. SMA: 20 > 50 (趨勢向上)
    2. MACD: DIF > DEA (動能轉強)
    3. RSI: < 70 (避免過度追高)
    """
    df = df.copy()
    
    # 1. 計算 SMA
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # 2. 計算 MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. 計算 RSI (14日)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- 複合訊號邏輯 ---
    # 買進條件：SMA黃金交叉 AND MACD金叉 AND RSI沒超買
    df['Signal'] = 0.0
    condition = (df['SMA20'] > df['SMA50']) & \
                (df['MACD'] > df['Signal_Line']) & \
                (df['RSI'] < 70)
    
    df.loc[condition, 'Signal'] = 1.0
    
    # 計算報酬
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)
    df['Cum_Market_Return'] = (1 + df['Market_Return']).cumprod()
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    return df

def backtest(df, symbol_name, initial_capital=1000000):
    if df.empty or 'Cum_Strategy_Return' not in df:
        return None
    final_return = df['Cum_Strategy_Return'].iloc[-1]
    final_value = initial_capital * final_return
    roi = (final_return - 1) * 100
    
    print(f"--- {symbol_name} 複合策略回測結果 ---")
    print(f"最終資產: ${final_value:,.0f} | 總投報率 (ROI): {roi:.2f}%")
    return {"symbol": symbol_name, "roi": roi, "final_value": final_value}

def plot_advanced_results(df, symbol, name):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 圖 1: 價格與買賣點
    ax1.plot(df['Close'], label='Price', color='black', alpha=0.3)
    ax1.plot(df['SMA20'], label='SMA 20', color='blue', linestyle='--')
    ax1.plot(df['SMA50'], label='SMA 50', color='red', linestyle='--')
    
    buy_signals = df[(df['Signal'] == 1) & (df['Signal'].shift(1) == 0)]
    sell_signals = df[(df['Signal'] == 0) & (df['Signal'].shift(1) == 1)]
    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy')
    ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell')
    ax1.set_title(f'{name} ({symbol}) Composite Strategy (SMA+MACD+RSI)')
    ax1.legend()

    # 圖 2: RSI
    ax2.plot(df['RSI'], label='RSI (14)', color='purple')
    ax2.axhline(70, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(30, color='green', linestyle=':', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.legend(loc='upper left')

    # 圖 3: 累積報酬率
    ax3.plot(df['Cum_Strategy_Return'], label='Strategy Return', color='orange', linewidth=2)
    ax3.plot(df['Cum_Market_Return'], label='Market Return', color='gray', linestyle='--', alpha=0.7)
    ax3.set_title('Cumulative Returns Comparison')
    ax3.legend()

    plt.tight_layout()
    output_dir = "results_composite"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = f"{output_dir}/{symbol}_composite.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"詳細圖表已儲存至: {plot_path}")

def main():
    watchlist = load_watchlist()
    start = "2020-01-01"
    end = "2025-01-01"
    summary = []
    
    for item in watchlist:
        symbol = item['symbol']
        name = item['name']
        df = fetch_data(symbol, start, end)
        if df.empty or len(df) < 50:
            continue
            
        df = apply_composite_strategy(df)
        result = backtest(df, f"{name} ({symbol})")
        if result:
            plot_advanced_results(df, symbol, name)
            summary.append(result)
    
    print("\n" + "="*40)
    print("      複合策略 (SMA+MACD+RSI) 總結")
    print("="*40)
    for s in sorted(summary, key=lambda x: x['roi'], reverse=True):
        print(f"{s['symbol']:<25}: {s['roi']:>8.2f}%")

if __name__ == "__main__":
    main()
