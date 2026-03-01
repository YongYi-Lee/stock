import yfinance as yf
import pandas as pd

def fetch_data(symbol, start_date, end_date):
    """
    獲取指定股票在一段時間內的歷史數據。
    """
    print(f"正在抓取 {symbol} 從 {start_date} 到 {end_date} 的數據...")
    data = yf.download(symbol, start = start_date, end = end_date)
    return data

def main():
    # 範例：抓取台積電 (2330.TW) 或 S&P 500 (SPY) 的數據
    symbol = "2330.TW"
    start = "2024-01-01"
    end = "2025-01-01"
    
    df = fetch_data(symbol, start, end)
    
    if not df.empty:
        print("
數據獲取成功！前 5 筆資料如下：")
        print(df.head())
        # 將資料存為 CSV 以供後續回測使用
        # filename = f"{symbol}_history.csv"
        # df.to_csv(filename)
        # print(f"
已將數據存檔至 {filename}")
    else:
        print("
無法獲取數據，請檢查代號或網路連線。")

if __name__ == "__main__":
    main()
