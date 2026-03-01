import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fetch_data(symbol, start_date, end_date):
    print(f"正在抓取 {symbol} 的數據...")
    df = yf.download(symbol, start=start_date, end=end_date)
    # 處理 MultiIndex columns (yfinance 新版特性)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def apply_strategy(df):
    """
    簡單均線策略：SMA 20 突破 SMA 50 買進，跌破賣出。
    """
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # 產生訊號：1 為持有，0 為空倉
    df['Signal'] = 0.0
    df.loc[df.index[20:], 'Signal'] = np.where(df['SMA20'][20:] > df['SMA50'][20:], 1.0, 0.0)
    
    # 計算每日報酬與策略報酬
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)
    
    # 計算累積報酬
    df['Cum_Market_Return'] = (1 + df['Market_Return']).cumprod()
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    return df

def backtest(df, initial_capital=1000000):
    """
    計算最終資產價值
    """
    final_return = df['Cum_Strategy_Return'].iloc[-1]
    final_value = initial_capital * final_return
    roi = (final_return - 1) * 100
    
    print(f"\n--- 回測結果 ---")
    print(f"初始資金: ${initial_capital:,.0f}")
    print(f"最終資產: ${final_value:,.0f}")
    print(f"總投報率 (ROI): {roi:.2f}%")
    return final_value

def plot_results(df, symbol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 上圖：股價與均線
    ax1.plot(df['Close'], label='Price', color='black', alpha=0.3)
    ax1.plot(df['SMA20'], label='SMA 20', color='blue')
    ax1.plot(df['SMA50'], label='SMA 50', color='red')
    
    # 標記買賣點
    buy_signals = df[(df['Signal'] == 1) & (df['Signal'].shift(1) == 0)]
    sell_signals = df[(df['Signal'] == 0) & (df['Signal'].shift(1) == 1)]
    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy')
    ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell')
    
    ax1.set_title(f'{symbol} Price and SMA Strategy')
    ax1.legend()
    
    # 下圖：累積報酬率
    ax2.plot(df['Cum_Strategy_Return'], label='Strategy Return', color='orange')
    ax2.plot(df['Cum_Market_Return'], label='Market Return', color='gray', linestyle='--')
    ax2.set_title('Cumulative Returns')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = "backtest_result.png"
    plt.savefig(plot_path)
    print(f"\n圖表已儲存至: {plot_path}")
    # plt.show() # 在本地端執行時可取消註解

def main():
    symbol = "2330.TW"
    start = "2020-01-01" # 稍微拉長時間看效果
    end = "2025-01-01"
    
    df = fetch_data(symbol, start, end)
    
    if df.empty:
        print("無法取得數據。")
        return
        
    df = apply_strategy(df)
    backtest(df)
    plot_results(df, symbol)

if __name__ == "__main__":
    main()
