# screener_v2.py
# pip install yfinance pandas requests tenacity
import io, os, math, time, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------- Config ----------------
MIN_PRICE, MAX_PRICE   = 5, 10
MIN_CAP, MAX_CAP       = 300e6, 10e9
MIN_AVG_VOL            = 200_000
MIN_DOLLAR_VOL         = 1_000_000      # avg vol * price
BATCH_SIZE             = 100
OUT_DIR                = Path(".")

# ---------------- Helpers ----------------
def _num(x, default=0):
    """Coerce yfinance's NaN/None/np-scalar soup to a real float."""
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

        # Data can be single-level if only one ticker came back
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
        time.sleep(1.0)  # be nice to Yahoo
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

def _dilution_flag(tk):
    try:
        so = tk.get_shares_full(start=dt.date.today() - dt.timedelta(days=365))
        if so is None or len(so) < 2:
            return False
        so = so.sort_index().dropna()
        if len(so) < 2 or so.iloc[0] <= 0:
            return False
        return so.iloc[-1] > so.iloc[0] * 1.10
    except Exception:
        return False

def analyse(t, d):
    try:
        tk   = yf.Ticker(t)
        info = tk.info or {}
    except Exception:
        return None

    cap = _num(info.get("marketCap"))
    if not (MIN_CAP <= cap <= MAX_CAP):
        return None

    score, flags = 0, []

    rev_g   = _num(info.get("revenueGrowth"))
    earn_g  = _num(info.get("earningsGrowth"))
    margin  = _num(info.get("profitMargins"))
    debt_eq = info.get("debtToEquity")

    if rev_g > 0.15:   score += 20
    elif rev_g > 0.05: score += 10
    if earn_g > 0:     score += 15
    if margin > 0:     score += 10
    if debt_eq is not None and _num(debt_eq, 999) < 100:
        score += 10

    vol_ratio, move = accumulation(d)
    # Quiet accumulation: volume up, price flat-to-slightly-up
    if vol_ratio > 1.3 and -0.03 < move < 0.06:
        score += 20
        flags.append("Quiet accumulation")

    buys, sells = _insider_signal(tk)
    if buys is not None:
        if buys > sells:   score += 15; flags.append(f"Insider buying ({buys})")
        elif sells > buys: flags.append(f"Insider selling ({sells})")

    try:
        inst = tk.institutional_holders
        if inst is not None and not inst.empty and "pctChange" in inst.columns:
            if _num(inst["pctChange"].mean()) > 0:
                score += 10; flags.append("Funds adding")
            else:
                flags.append("Funds trimming")
    except Exception:
        pass

    short = _num(info.get("shortPercentOfFloat"))
    if short > 0.15:
        flags.append(f"High short interest {short:.0%}")

    rec = _num(info.get("recommendationMean"), default=99)
    if 0 < rec <= 2:
        score += 5; flags.append("Analysts positive")

    if _dilution_flag(tk):
        score -= 20; flags.append("Dilution >10% in a year")

    days = _earnings_days(info, tk)
    if days is not None and 0 <= days <= 14:
        flags.append(f"Earnings in {days} days")

    return {
        "ticker":       t,
        "score":        score,
        "close":        round(float(d["Close"].iloc[-1]), 2),
        "market_cap_m": round(cap / 1e6),
        "rev_growth":   round(rev_g, 3),
        "vol_ratio":    round(vol_ratio, 2),
        "sector":       info.get("sector"),
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

    if not rows:
        print("No candidates.")
        return

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out.to_csv(OUT_DIR / f"report_{stamp}.csv", index=False)

    # Append to track record safely
    tr = OUT_DIR / "track_record.csv"
    header = not tr.exists() or tr.stat().st_size == 0
    out.assign(run=stamp).to_csv(tr, mode="a", header=header, index=False)

    print("\nTop 25:\n")
    print(out.head(25).to_string(index=False))

if __name__ == "__main__":
    main()
