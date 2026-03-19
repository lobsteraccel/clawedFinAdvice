import yfinance as yf
import pandas as pd
import pandas_ta as ta
import sys
import warnings
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

MIN_MARKET_CAP = 2e9
MIN_DOLLAR_VOLUME = 20e6


def fetch_stock_data(ticker):

    try:
        ticker = ticker.strip().upper()
        search_ticker = f"{ticker}-USD" if ticker in ["BTC", "ETH", "SOL"] else ticker
        stock = yf.Ticker(search_ticker)

        df = stock.history(period='2y', interval='1d', auto_adjust=True)
        info = stock.info

        if not info or len(info) < 5:
            return ticker, {}, None

        return ticker, info, df

    except:
        return ticker, {}, None


def get_macro_data():

    try:
        spy = yf.download("SPY", period='1y', progress=False, auto_adjust=True)
        vix = yf.download("^VIX", period='6mo', progress=False, auto_adjust=True)

        spy_last = float(spy['Close'].iloc[-1])
        spy_ma200 = float(spy['Close'].rolling(200).mean().iloc[-1])
        spy_trend = spy_last > spy_ma200

        spy_perf = (spy_last / float(spy['Close'].iloc[-63])) - 1
        vix_level = float(vix['Close'].iloc[-1])

        return spy_perf, spy_trend, vix_level

    except Exception as e:
        print("Macro data error:", e)
        return 0, True, 20


def scout_stocks(ticker_list):

    alerts = []

    print(f"DecisionEngine V5.0 [Institutional Hybrid]: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    spy_perf, spy_trend, vix_level = get_macro_data()

    print("Market Macro Pulse:")
    print(f"  VIX Level: {vix_level:.1f}")
    print(f"  SPY Trend: {'UPTREND' if spy_trend else 'DOWNTREND'}")

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_stock_data, ticker_list))

    for idx, (ticker, info, df) in enumerate(results):

        if df is None or df.empty or len(df) < 252:
            continue

        try:

            price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            open_price = float(df['Open'].iloc[-1])

            curr_vol = float(df['Volume'].iloc[-1])
            avg_vol = float(df['Volume'].tail(20).mean())
            vol_ratio = curr_vol / avg_vol

            avg_dollar_vol = avg_vol * price
            market_cap = info.get("marketCap", 0)

            if avg_dollar_vol < MIN_DOLLAR_VOLUME or market_cap < MIN_MARKET_CAP:
                continue

            ma50 = float(df['Close'].rolling(50).mean().iloc[-1])
            ma150 = float(df['Close'].rolling(150).mean().iloc[-1])
            ma200 = float(df['Close'].rolling(200).mean().iloc[-1])

            trend_template = (
                price > ma50 and
                ma50 > ma150 and
                ma150 > ma200
            )

            mom_6m = (price / float(df['Close'].iloc[-126])) - 1

            stock_perf = (price / float(df['Close'].iloc[-63])) - 1
            rel_strength = stock_perf - spy_perf

            df['ADL'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
            adl_trend = "ACCUMULATING" if df['ADL'].iloc[-1] > df['ADL'].iloc[-5] else "DISTRIBUTING"

            up_vol = df[df['Close'] > df['Close'].shift(1)]['Volume'].tail(10).mean()
            dn_vol = df[df['Close'] < df['Close'].shift(1)]['Volume'].tail(10).mean()

            inst_score = up_vol / dn_vol if dn_vol > 0 else 1

            vol_6mo = df['Volume'].tail(126)
            vol_threshold = vol_6mo.quantile(0.95)

            vol_heat = "HIGH (Top 5%)" if curr_vol >= vol_threshold else "Normal"

            rsi = float(ta.rsi(df['Close']).iloc[-1])
            mfi = float(ta.mfi(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1])

            flagpole = (df['Close'].iloc[-5] / df['Close'].iloc[-20]) - 1 > 0.10
            vol_contract = df['Volume'].iloc[-1] < df['Volume'].tail(10).mean()
            higher_lows = df['Low'].iloc[-1] > df['Low'].iloc[-5]
            ma_support = price > ma50

            is_bull_flag = flagpole and vol_contract and higher_lows and ma_support and trend_template

            explosive_breakout = (
                price > df['Close'].tail(20).max() and
                vol_ratio > 1.5 and
                inst_score > 1.2
            )

            action = "HOLD"

            if explosive_breakout:
                action = "EXPLOSIVE BREAKOUT"
            elif is_bull_flag:
                action = "BULL FLAG (RIGID)"

            score = 0
            if trend_template: score += 20
            if inst_score > 1.2: score += 20
            if rel_strength > 0: score += 20
            if mom_6m > 0.20: score += 20
            if vol_heat.startswith("HIGH"): score += 20

            atr = float(ta.atr(df['High'], df['Low'], df['Close']).iloc[-1])

            high_60 = float(df['Close'].tail(60).max())
            low_60 = float(df['Close'].tail(60).min())
            high_52 = float(df['Close'].tail(252).max())
            low_52 = float(df['Close'].tail(252).min())

            company_name = info.get('shortName') or info.get('longName') or "N/A"

            print("\n----------------")
            print("Price:")
            print(f"  {ticker} - {company_name} (${price:.2f}) [{action}]")
            print(f"  Daily Trend: {'BULLISH' if price >= open_price else 'BEARISH'}")
            print(f"  Dist to 60d High/Low: {((high_60/price)-1)*100:+.1f}% / {((price/low_60)-1)*100:+.1f}%")
            print(f"  Dist to 52w High/Low: {((high_52/price)-1)*100:+.1f}% / {((price/low_52)-1)*100:+.1f}%")

            print("Valuations:")
            print(f"  TTL PE: {float(info.get('trailingPE',0) or 0):.1f}")
            print(f"  Fwd PE: {float(info.get('forwardPE',0) or 0):.1f}")
            print(f"  Margin Profile: {float(info.get('profitMargins',0) or 0)*100:.1f}%")
            print(f"  PEG: {info.get('pegRatio',0) or 0}")

            print("Indicators:")
            print(f"  RSI: {rsi:.1f}")
            print(f"  MFI: {mfi:.1f}")
            print(f"  Inst. Accum Score: {inst_score:.2f}x")
            print(f"  A/D Line Trend: {adl_trend}")
            print(f"  Vol Ratio: {vol_ratio:.2f}x")
            print(f"  Vol Heat: {vol_heat}")
            print(f"  Momentum 6M: {mom_6m*100:+.1f}%")
            print(f"  Conviction Score: {score}/100")

            earnings_ts = info.get("earningsTimestamp")
            days_to_earnings = "N/A"

            if earnings_ts:
                earnings_date = datetime.fromtimestamp(earnings_ts, tz=timezone.utc)
                delta = earnings_date - datetime.now(timezone.utc)
                days_to_earnings = delta.days

            print("Results Tracker:")
            print(f"  Days to Earnings: {days_to_earnings}")
            print(f"  Last Gap Reaction: {((df['Open'].iloc[-1]/df['Close'].iloc[-2])-1)*100:+.2f}%")

            if action != "HOLD":
                alerts.append({
                    "ticker": ticker,
                    "action": action,
                    "stop": price - (1.5 * atr)
                })

        except Exception as e:
            print(f"Error on {ticker}: {e}")

    print("\nSTRATEGIC ACTION SUMMARY")

    if alerts:

        for a in alerts:
            print(f"[{a['ticker']}] - {a['action']}")
            print(f"  Calculated Stop: ${a['stop']:.2f}")

    else:
        print("No qualifying setups found.")


if __name__ == "__main__":

    user_tickers = sys.argv[1].split(',') if len(sys.argv) > 1 else ["NVDA","AVGO","TSM","MU"]

    scout_stocks(user_tickers)

