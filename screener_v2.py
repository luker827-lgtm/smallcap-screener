# screener_DIAGNOSTIC.py
import io, os, math, time, datetime as dt
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

MIN_PRICE, MAX_PRICE   = 10, 500       
MIN_CAP, MAX_CAP       = 2e9, 15e12    
MIN_AVG_VOL            = 250_000       
MIN_DOLLAR_VOL         = 1_000_000
BATCH_SIZE             = 50
OUT_DIR                = Path(".")

def _num(x, default=0):
    if x is None: return default
    try:
        f = float(x)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _fetch(url, **kw):
    r = requests.get(url, timeout=30, **kw)
    r.raise_for_status()
    return r.text

def get_us_tickers():
    txt = _fetch("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt")
    df = pd.read_csv(io.StringIO(txt), sep="|")
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
    syms = df["Symbol"].dropna().astype(str)
    return sorted({s for s in syms if s.isalpha() and s.isupper()})

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=20))
def _download(batch):
    return yf.download(batch, period="3mo", group_by="ticker", progress=False, threads=True, auto_adjust=True)

def price_volume_filter(tickers):
    keep = {}
    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        try:
            data = _download(batch)
        except Exception as e:
            continue

        if data is None or data.empty: continue
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_product([[batch[0]], data.columns])

        for t in batch:
            if t not in data.columns.get_level_values(0): continue
            d = data[t].dropna()
            if len(d) < 40 or "Close" not in d or "Volume" not in d: continue
            close  = float(d["Close"].iloc[-1])
            avgvol = float(d["Volume"].tail(20).mean())
            
            if not (MIN_PRICE <= close <= MAX_PRICE): continue
            if avgvol < MIN_AVG_VOL: continue
            if avgvol * close < MIN_DOLLAR_VOL: continue
            keep[t] = d
        time.sleep(1.5)
    return keep

def analyse(t, d):
    # --- GROK'S DIAGNOSTIC VERSION (GATES DISABLED) ---
    try:
        tk   = yf.Ticker(t)
        info = tk.info or {}
        
        # 1. CURRENCY CHECK (Disabled for test)
        # if info.get("financialCurrency") not in (None, "USD"): return None
        
        # 2. NAME FILTER (Disabled for test)
        # name = (info.get("longName") or "").lower()
        # if any(x in name for x in [" etf", " fund", " trust", " municipal", " lp", "partners", "mlp"]): return None
        
        cap = _num(info.get("marketCap"))
        if not (MIN_CAP <= cap <= MAX_CAP): return None

        # 3. QUALITY FILTER (Disabled for test)
        # fcf = _num(info.get("freeCashflow"))
        # if fcf < -200e6: return None 
        
        # 4. INSIDER (Disabled for test)
        # try:
        #     ins = tk.insider_transactions
        #     ...
        # except: pass

        return {
            "ticker": t, 
            "close": round(float(d["Close"].iloc[-1]), 2),
            "market_cap_m": round(cap / 1e6)
        }
    except Exception as e:
        return None

def main():
    print("Fetching symbol list...")
    universe = get_us_tickers()
    print(f"Universe: {len(universe)} tickers")
    
    cands = price_volume_filter(universe)
    print(f"Passed price/volume: {len(cands)}")
    
    if not cands:
        print("STAGE 1 FAILED — no tickers passed price/volume. Check MIN_AVG_VOL, MIN_DOLLAR_VOL, or download errors.")
        return

    rows = []
    print("Testing first 50 tickers from Stage 1...")
    # Only test first 50 to make it fast
    for t, d in list(cands.items())[:50]:
        r = analyse(t, d)
        if r is not None:
            rows.append(r)

    print(f"Passed analyse (first 50): {len(rows)}")
    
    if rows:
        out = pd.DataFrame(rows)
        print("\nSuccessful Tickers:\n", out.to_string(index=False))
    else:
        print("All 50 failed the Stage 2 checks.")

if __name__ == "__main__":
    main()
