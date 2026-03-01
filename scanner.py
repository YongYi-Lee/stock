import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 擴大掃描清單：包含上市(.TW)與上櫃(.TWO)熱門股
SCAN_LIST = [
    # 上市熱門
    "2330.TW", "2317.TW", "2454.TW", "2303.TW", "2382.TW", "3231.TW", "2357.TW", "2603.TW", "2609.TW", "2615.TW",
    "2308.TW", "2376.TW", "2408.TW", "2353.TW", "2324.TW", "2881.TW", "2882.TW", "2886.TW", "2891.TW", "5871.TW",
    # 上櫃熱門 (.TWO)
    "6488.TWO", "8069.TWO", "3293.TWO", "3105.TWO", "5483.TWO", "6147.TWO", "1815.TWO", "3529.TWO", "6510.TWO", "8299.TWO",
    # 飆股潛力 (近期波動大)
    "3661.TW", "3035.TW", "3443.TW", "6669.TW", "1513.TW", "1519.TW", "1503.TW", "2371.TW"
]

def scan_stock(symbol):
    try:
        # 抓取最近 3 個月的數據確保指標正確
        df = yf.download(symbol, period="3mo", progress=False)
        if df.empty or len(df) < 60:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 計算指標
        current_price = df['Close'].iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        
        sma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        sma60 = df['Close'].rolling(window=60).mean().iloc[-1]
        high20 = df['High'].shift(1).rolling(window=20).max().iloc[-1]
        vol_avg20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        
        # 判斷邏輯
        is_breakout = current_price > high20
        is_vol_burst = current_vol > (vol_avg20 * 1.8) # 放寬到 1.8 倍增加機會
        is_above_trend = current_price > sma60
        
        # 額外判斷：今天是否剛突破 (如果昨天已經突破就不是「起點」)
        prev_price = df['Close'].iloc[-2]
        prev_high20 = df['High'].shift(1).rolling(window=20).max().iloc[-2]
        was_not_breakout = prev_price <= prev_high20
        
        if is_breakout and is_vol_burst and is_above_trend:
            return {
                "symbol": symbol,
                "price": current_price,
                "vol_ratio": current_vol / vol_avg20,
                "is_fresh": was_not_breakout # 是否為新鮮的第一天突破
            }
    except Exception as e:
        pass
    return None

def main():
    print(f"--- 啟動起漲股掃描器 (掃描目標: {len(SCAN_LIST)} 檔) ---")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("正在大海撈針，請稍候...
")
    
    hits = []
    for symbol in SCAN_LIST:
        result = scan_stock(symbol)
        if result:
            hits.append(result)
            print(f"找到潛在起漲股: {symbol}!")

    print("
" + "="*50)
    print(f"{'股票代號':<15} | {'目前價格':>10} | {'量能放大':>10} | {'狀態'}")
    print("-" * 50)
    
    if not hits:
        print("今日市場平靜，未掃描到符合『帶量突破』條件的起漲股。")
    else:
        for h in hits:
            status = "🔥 新突破" if h['is_fresh'] else "📈 持續轉強"
            print(f"{h['symbol']:<15} | {h['price']:>10.2f} | {h['vol_ratio']:>9.1f}x | {status}")
            
    print("="*50)
    print("提示: 建議搭配籌碼面(外資/投信買超)進一步確認。")

if __name__ == "__main__":
    main()
