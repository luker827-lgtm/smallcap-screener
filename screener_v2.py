# screener_v6.py
# pip install yfinance pandas requests tenacity
import io, os, math, time, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------- Config ----------------
MIN_PRICE, MAX_PRICE   = 10, 500       # Allows high-priced large caps
MIN_CAP, MAX_CAP       = 2e9, 500e9    # $2B to $500 Billion (Mid to Mega Cap)
MIN_AVG_VOL            = 1_000_000
MIN_DOLLAR_VOL         = 1_000_000
BATCH_SIZE             = 100
OUT_DIR                = Path(".")

# ---------------- Helpers ----------------
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
    return yf.download(
        batch, period="3mo", group_by="ticker",
        progress=False, threads=True, auto_adjust=True,
    )

# ---------------- Stage 1: price / volume ----------------
def price_volume_filter(tickers):
    keep = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        try:
            data = _download(batch)
        except Exception as e:
            print(f"[batch {i}] download failed: {e}")
            continue

        if data is None or data.empty:
            continue

        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_product([[batch[0]], data.columns])

        for t in batch:
            if t not in data.columns.get_level_values(0):
                continue
            d = data[t].dropna()
            if len(d) < 40 or "Close" not in d or "Volume" not in d:
                continue
            close  = float(d["Close"].iloc[-1])
            avgvol = float(d["Volume"].tail(20).mean())
            if not (MIN_PRICE <= close <= MAX_PRICE):
                continue
            if avgvol < MIN_AVG_VOL:
                continue
            if avgvol * close < MIN_DOLLAR_VOL:
                continue
            keep[t] = d
        time.sleep(1.0)
    return keep

# ---------------- Stage 2: per-ticker analysis ----------------
def accumulation(d):
    vol_ratio = d["Volume"].tail(10).mean() / max(d["Volume"].tail(50).mean(), 1)
    price_move = d["Close"].iloc[-1] / d["Close"].iloc[-10] - 1
    return float(vol_ratio), float(price_move)

def _insider_signal(tk):
    try:
        ins = tk.insider_transactions
    except Exception:
        return None, None
    if ins is None or ins.empty:
        return 0, 0
    col = "Text" if "Text" in ins.columns else ("Transaction" if "Transaction" in ins.columns else None)
    if col is None:
        return 0, 0
    txt = ins[col].astype(str).str.lower()
    buys  = int(txt.str.contains("purchase|buy", regex=True).sum())
    sells = int(txt.str.contains("sale|sell",       regex=True).sum())
    return buys, sells

def _earnings_days(info, tk):
    ed = None
    try:
        cal = tk.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            ed = cal.loc["Earnings Date"].tolist()
    except Exception:
        pass
    if not ed:
        return None
    try:
        first = pd.Timestamp(ed[0]).date()
        return (first - dt.date.today()).days
    except Exception:
        return None

def analyse(t, d):
    try:
        tk   = yf.Ticker(t)
        info = tk.info or {}
    except Exception:
        return None

    cap = _num(info.get("marketCap"))
    if not (MIN_CAP <= cap <= MAX_CAP):
        return None

    # --- GROK'S FIXES ---
    
    # 1. Reject if financials are in a foreign currency
    if info.get("financialCurrency") != "USD":
        return None

    # 2. Reject funds, BDCs, and trusts
    name = info.get("longName", "").lower()
    if "fund" in name or "trust" in name or "municipal" in name or "income" in name:
        return None
        
    sector = info.get("sector")

    score, flags = 0, []

    # --- QUALITY FILTER ---
    fcf = _num(info.get("freeCashflow"))
    debt_eq = _num(info.get("debtToEquity"), 999)

    if fcf <= 0:
        return None 
    if debt_eq > 250:
        return None
        
    score += 15
    flags.append(f"Strong FCF (${fcf/1e6:.1f}M)")

    # --- FUNDAMENTALS ---
    rev_g   = _num(info.get("revenueGrowth"))
    earn_g  = _num(info.get("earningsGrowth"))
    margin  = _num(info.get("profitMargins"))

    if rev_g > 0.15:   score += 20
    elif rev_g > 0.05: score += 10
    if earn_g > 0:     score += 15
    if margin > 0:     score += 10

    # --- ACCUMULATION ---
    vol_ratio, move = accumulation(d)
    if vol_ratio > 1.3 and -0.03 < move < 0.06:
        score += 20
        flags.append("Quiet accumulation")

    # --- INSIDERS (LOOSENED FILTER) ---
    buys, sells = _insider_signal(tk)
    if buys is not None:
        if buys > sells:   
            score += 15; flags.append(f"Insider buying ({buys})")
        elif sells > (buys * 2): 
            return None # Only reject if selling is more than double the buying
        else:
            flags.append(f"Insider selling ({sells})") # Keep it, but flag it

    try:
        inst = tk.institutional_holders
        if inst is not None and not inst.empty and "pctChange" in inst.columns:
            if _num(inst["pctChange"].mean()) > 0:
                score += 10; flags.append("Funds adding")
            else:
                flags.append("Funds trimming")
    except Exception:
        pass

    # --- VALUATION (GROK'S PEG FIX) ---
    pe_ratio = _num(info.get("trailingPE"))
    forward_pe = _num(info.get("forwardPE"))
    peg_ratio = _num(info.get("pegRatio"))

    if 0.2 < peg_ratio < 1.0:  # Ignore data errors under 0.2
        score += 15
        flags.append(f"Undervalued (PEG {peg_ratio:.2f})")
    elif 1.0 <= peg_ratio < 2.0:
        score += 5
    elif peg_ratio >= 2.0:
        score -= 10
        flags.append(f"Expensive (PEG {peg_ratio:.2f})")

    # --- OTHER FLAGS ---
    short = _num(info.get("shortPercentOfFloat"))
    if short > 0.15:
        flags.append(f"High short interest {short:.0%}")

    rec = _num(info.get("recommendationMean"), default=99)
    if 0 < rec <= 2:
        score += 5; flags.append("Analysts positive")

    days = _earnings_days(info, tk)
    if days is not None and 0 <= days <= 14:
        flags.append(f"Earnings in {days} days")

    return {
        "ticker":       t,
        "score":        score,
        "close":        round(float(d["Close"].iloc[-1]), 2),
        "P/E":          round(pe_ratio, 1) if pe_ratio else None,
        "Fwd P/E":      round(forward_pe, 1) if forward_pe else None,
        "PEG":          round(peg_ratio, 2) if peg_ratio else None,
        "market_cap_m": round(cap / 1e6),
        "rev_growth":   round(rev_g, 3),
        "sector":       sector,
        "flags":        "; ".join(flags),
    }

# ---------------- Main ----------------
def main():
    print("Fetching symbol list...")
    universe = get_us_tickers()
    print(f"Universe: {len(universe)} tickers")

    print("Stage 1: price/volume filter...")
    cands = price_volume_filter(universe)
    print(f"Passed price/volume: {len(cands)}")

    print("Stage 2: fundamentals + signals...")
    rows, skipped = [], 0
    for t, d in cands.items():
        r = analyse(t, d)
        if r is None:
            skipped += 1
        else:
            rows.append(r)
    print(f"Scored: {len(rows)}  (skipped/failed: {skipped})")

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    if not rows:
        print("No candidates passed the filters today.")
        empty_df = pd.DataFrame([{"ticker": "NONE", "score": 0, "flags": "No stocks passed the filters today."}])
        empty_df.to_csv(OUT_DIR / f"report_{stamp}_EMPTY.csv", index=False, sep=",", encoding="utf-8")
        return

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    
    out.to_csv(OUT_DIR / f"report_{stamp}.csv", index=False, sep=",", encoding="utf-8")

    tr = OUT_DIR / "track_record.csv"
    header = not tr.exists() or tr.stat().st_size == 0
    out.assign(run=stamp).to_csv(tr, mode="a", header=header, index=False, sep=",", encoding="utf-8")

    print("\nTop 25:\n")
    print(out.head(25).to_string(index=False))

if __name__ == "__main__":
    main()