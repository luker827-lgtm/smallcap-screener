# screener_v13_FINAL_softened.py
# pip install yfinance pandas requests tenacity
import io, os, math, time, datetime as dt
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------- Config ----------------
MIN_PRICE, MAX_PRICE   = 10, 2000
MIN_CAP, MAX_CAP       = 2e9, 15e12
MIN_AVG_VOL            = 250_000
MIN_DOLLAR_VOL         = 1_000_000
BATCH_SIZE             = 50
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
    return yf.download(batch, period="3mo", group_by="ticker", progress=False, threads=True, auto_adjust=True)

# ---------------- Stage 1: price / volume ----------------
def price_volume_filter(tickers):
    keep = {}
    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"Processing batch {batch_num}/{total_batches} ({batch[0]} to {batch[-1]})...")

        try:
            data = _download(batch)
        except Exception as e:
            print(f"  [batch {batch_num}] download failed: {e}")
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
        time.sleep(1.0)
    return keep

# ---------------- Stage 2: per-ticker analysis ----------------
def analyse(t, d):
    try:
        tk   = yf.Ticker(t)
        info = tk.info or {}

        # Soft currency check (only hard-reject explicit non-USD listing currency)
        if info.get("currency") and info.get("currency") != "USD":
            return None

        # Name filter (narrowed — removed broad LP/partners/MLP terms)
        name = (info.get("longName") or "").lower()
        if any(x in name for x in [" etf", " fund", " trust", " municipal"]):
            return None

        cap = _num(info.get("marketCap"))
        if not (MIN_CAP <= cap <= MAX_CAP): return None

        score, flags = 0, []

        # Quality filters (softened FCF)
        fcf = _num(info.get("freeCashflow"))
        raw_de = info.get("debtToEquity")
        if raw_de is not None and _num(raw_de) > 400:
            return None
        # Hard FCF reject removed; negative FCF is now only a flag

        score += 15
        if fcf > 0:
            flags.append(f"Positive FCF (${fcf/1e6:.1f}M)")
        elif fcf < 0:
            flags.append(f"Negative FCF (${fcf/1e6:.1f}M)")

        # Fundamentals
        rev_g   = _num(info.get("revenueGrowth"))
        earn_g  = _num(info.get("earningsGrowth"))
        margin  = _num(info.get("profitMargins"))
        if rev_g > 0.15:   score += 20
        elif rev_g > 0.05: score += 10
        if earn_g > 0:     score += 15
        if margin > 0:     score += 10

        # Accumulation
        vol_ratio = d["Volume"].tail(10).mean() / max(d["Volume"].tail(50).mean(), 1)
        price_move = d["Close"].iloc[-1] / d["Close"].iloc[-10] - 1
        if vol_ratio > 1.3 and -0.03 < price_move < 0.06:
            score += 20
            flags.append("Quiet accumulation")

        # Insiders (soft — no hard reject)
        try:
            ins = tk.insider_transactions
            if ins is not None and not ins.empty:
                col = "Text" if "Text" in ins.columns else ("Transaction" if "Transaction" in ins.columns else None)
                if col:
                    txt = ins[col].astype(str).str.lower()
                    buys  = int(txt.str.contains("purchase|buy", regex=True).sum())
                    sells = int(txt.str.contains("sale|sell", regex=True).sum())
                    if buys > sells:
                        score += 15
                        flags.append(f"Insider buying ({buys})")
                    elif sells > (buys * 2):
                        score -= 10
                        flags.append(f"Heavy insider selling ({sells})")
                    else:
                        flags.append(f"Insider selling ({sells})")
        except Exception:
            pass  # skip on any error, do not reject

        # Valuation
        pe_ratio = _num(info.get("trailingPE"))
        peg_ratio = _num(info.get("pegRatio"))
        if pe_ratio > 0 and 0.2 < peg_ratio < 1.0:
            score += 15
            flags.append(f"Undervalued (PEG {peg_ratio:.2f})")
        elif 1.0 <= peg_ratio < 2.0:
            score += 5
        elif peg_ratio >= 2.0:
            score -= 10
            flags.append(f"Expensive (PEG {peg_ratio:.2f})")

        return {
            "ticker": t,
            "score": score,
            "close": round(float(d["Close"].iloc[-1]), 2),
            "P/E": round(pe_ratio, 1) if pe_ratio else None,
            "PEG": round(peg_ratio, 2) if peg_ratio else None,
            "market_cap_m": round(cap / 1e6),
            "rev_growth": round(rev_g, 3),
            "sector": info.get("sector"),
            "flags": "; ".join(flags),
        }
    except Exception as e:
        print(f"Error processing {t}: {e}")
        return None

# ---------------- Main ----------------
def main():
    universe = get_us_tickers()
    print(f"Universe: {len(universe)} tickers")
    cands = price_volume_filter(universe)
    print(f"Passed price/volume: {len(cands)}")

    if not cands:
        print("STAGE 1 FAILED — no tickers passed price/volume.")
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        pd.DataFrame([{"ticker": "NONE", "score": 0, "flags": "No stocks passed Stage 1."}]).to_csv(
            OUT_DIR / f"report_{stamp}_EMPTY.csv", index=False)
        return

    rows = []
    for t, d in cands.items():
        r = analyse(t, d)
        if r is not None:
            rows.append(r)

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if not rows:
        print("No candidates passed the filters today.")
        pd.DataFrame([{"ticker": "NONE", "score": 0, "flags": "No stocks passed."}]).to_csv(
            OUT_DIR / f"report_{stamp}_EMPTY.csv", index=False)
        return

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    out.to_csv(OUT_DIR / f"report_{stamp}.csv", index=False, sep=",")
    print("\nTop 25:\n", out.head(25).to_string(index=False))

if __name__ == "__main__":
    main()
