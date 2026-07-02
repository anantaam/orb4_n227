#!/usr/bin/env python3
"""
ORB3 - ATR_TRAIL Sensitivity Backtest  v5 (K-Ratio)

Data source : nse.intraday_1min via psycopg2 (db_loader.py)
Cache       : cache/daily_metrics.pkl + cache/bar_cache.pkl
              Use --force-refresh to re-pull from DB when new data arrives.

Run from orb3_package root:
    python scripts/sensitivity_backtest.py
    python scripts/sensitivity_backtest.py --force-refresh

Outputs:
    results/sensitivity_results.csv
    results/sensitivity_plots.png   (5x4 grid: Sharpe | CAGR | MDD | K-Ratio)
"""
import sys, pickle, time as _time, argparse
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from paths import PICKLE_PATH, BAR_CACHE_PATH, RESULTS_DIR
from db_loader import load_universe_from_db, load_all_from_db

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

RESULTS_CSV = RESULTS_DIR / "sensitivity_results.csv"

# ── Constants ──────────────────────────────────────────────────────────────
START_CAP  = 100_000
MARGIN_CAP = 100_000
RISK_PCT   = 0.0125
MAX_POS    = 3
ATR_PER    = 14
VOL_PER    = 14
MIN_ATR    = 4.0

ENTRY_START_SEC = 9  * 3600 + 45 * 60   # 35100
ENTRY_END_SEC   = 11 * 3600 + 30 * 60   # 41400
EXIT_TIME_SEC   = 14 * 3600 + 45 * 60   # 53100

E_BPS  = 5.0
SL_BPS = 3.0
T_BPS  = 2.0

BASE = dict(atr_mult=1.50, rel_vol=1.80, r_factor=0.30, min_turn=25e7, top_k=8)

GRID = [
    ("ATR_Multiplier", "atr_mult",  [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]),
    ("REL_VOL_Thresh", "rel_vol",   [1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.50]),
    ("R_FACTOR",       "r_factor",  [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]),
    ("MinTurnover_Cr", "min_turn",  [v * 1e7 for v in [25, 40, 60, 80, 100, 150]]),
    ("TOP_K",          "top_k",     [4, 5, 6, 7, 8, 10, 12]),
]

# ── Cost helpers ───────────────────────────────────────────────────────────
def slip_price(price, direction, side, bps):
    s = bps / 10_000.0
    if side == "entry":
        return price * (1.0 + s) if direction == 1 else price * (1.0 - s)
    return price * (1.0 - s) if direction == 1 else price * (1.0 + s)

def calc_pnl(entry, exit_, qty, direction):
    buy_p  = entry if direction == 1 else exit_
    sell_p = exit_ if direction == 1 else entry
    bv, sv = buy_p * qty, sell_p * qty
    turn   = bv + sv
    brok   = min(bv * 0.0003, 20.0) + min(sv * 0.0003, 20.0)
    stt    = sv * 0.00025
    exch   = turn * 0.0000297
    sebi   = (turn / 1e7) * 10.0
    gst    = (brok + sebi + exch) * 0.18
    stamp  = max(bv * 0.00003, (bv / 1e7) * 300.0)
    return (exit_ - entry) * qty * direction - (brok + stt + exch + gst + sebi + stamp)

# ── Simulation ─────────────────────────────────────────────────────────────
def simulate_day_np(cand_list, bar_cache, atr_mult_f, r_factor_f):
    trades = []; margin_used = 0.0; n_pos = 0
    for (sym, date, direction, orb_high, orb_low, atr_val) in cand_list:
        if n_pos >= MAX_POS: break
        sym_dict = bar_cache.get(sym)
        if not sym_dict: continue
        bars = sym_dict.get(date)
        if bars is None: continue
        open_arr, high_arr, low_arr, close_arr, tsec_arr = bars
        entry_level = orb_high if direction == 1 else orb_low
        sl_raw      = orb_low  if direction == 1 else orb_high
        r_val = abs(entry_level - sl_raw) * r_factor_f
        if r_val <= 0: continue
        entry_idx = -1
        for i in range(len(tsec_arr)):
            ts = tsec_arr[i]
            if ts < ENTRY_START_SEC: continue
            if ts > ENTRY_END_SEC: break
            if direction == 1 and high_arr[i] >= entry_level: entry_idx = i; break
            elif direction == -1 and low_arr[i] <= entry_level: entry_idx = i; break
        if entry_idx < 0: continue
        ep  = slip_price(entry_level, direction, "entry", E_BPS)
        sl  = (sl_raw - r_val) if direction == 1 else (sl_raw + r_val)
        qty = int((RISK_PCT * MARGIN_CAP) / r_val)
        if qty <= 0: continue
        if margin_used + ep * qty / 5.0 > MARGIN_CAP * 1.05: continue
        margin_used += ep * qty / 5.0; n_pos += 1
        peak = ep; sl_cur = sl; exit_p = 0.0; exited = False
        for i in range(entry_idx, len(tsec_arr)):
            ts = int(tsec_arr[i])
            if ts < ENTRY_START_SEC: continue
            bo = float(open_arr[i]); bh = float(high_arr[i])
            bl = float(low_arr[i]);  bc = float(close_arr[i])
            if direction == 1 and bo <= sl_cur:
                exit_p = slip_price(bo, direction, "exit", SL_BPS); exited = True; break
            if direction == -1 and bo >= sl_cur:
                exit_p = slip_price(bo, direction, "exit", SL_BPS); exited = True; break
            if direction == 1 and bl <= sl_cur:
                exit_p = slip_price(sl_cur, direction, "exit", SL_BPS); exited = True; break
            if direction == -1 and bh >= sl_cur:
                exit_p = slip_price(sl_cur, direction, "exit", SL_BPS); exited = True; break
            if direction == 1:
                if bh > peak: peak = bh
                new_sl = peak - atr_val * atr_mult_f
                if new_sl > sl_cur: sl_cur = new_sl
            else:
                if bl < peak: peak = bl
                new_sl = peak + atr_val * atr_mult_f
                if new_sl < sl_cur: sl_cur = new_sl
            if ts >= EXIT_TIME_SEC:
                exit_p = slip_price(bc, direction, "exit", T_BPS); exited = True; break
        if not exited:
            exit_p = slip_price(float(close_arr[-1]), direction, "exit", T_BPS)
        trades.append(calc_pnl(ep, exit_p, qty, direction))
    return trades

def run_once_np(daily, bar_cache, *, atr_mult, rel_vol, r_factor, min_turn, top_k):
    all_pnl = []; equity = float(START_CAP); eq_list = []
    atr_f = float(atr_mult); rf = float(r_factor)
    for date in sorted(daily["date_only"].unique()):
        dd   = daily[daily["date_only"] == date]
        cand = dd[(dd["atr_orb"]>=MIN_ATR) & (dd["rel_vol"]>=rel_vol) &
                  (dd["turn_B"]>=min_turn) & (dd["direction"]!=0)].nlargest(top_k, "score")
        day_pnl = 0.0
        if not cand.empty:
            cand_list = list(zip(cand["symbol"].values, [date]*len(cand),
                                 cand["direction"].values.astype(int),
                                 cand["orb_high"].values.astype(float),
                                 cand["orb_low"].values.astype(float),
                                 cand["atr_orb"].values.astype(float)))
            for p in simulate_day_np(cand_list, bar_cache, atr_f, rf):
                day_pnl += p; all_pnl.append({"date": date, "net_pnl": p})
        equity += day_pnl; eq_list.append(equity)
    return all_pnl, eq_list

# ── Kestner K-Ratio ────────────────────────────────────────────────────────
def _calc_kfactor(eq_list):
    """Kestner K-Ratio: OLS regression of log(equity) vs time index.
    K = slope / (std_err_of_slope * sqrt(n)). Higher = more linear alpha."""
    n = len(eq_list)
    if n < 10:
        return 0.0, 0.0
    y   = np.log(np.asarray(eq_list, dtype=float))
    x   = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    Sxx = ((x - xm) ** 2).sum()
    Sxy = ((x - xm) * (y - ym)).sum()
    Syy = ((y - ym) ** 2).sum()
    b   = Sxy / Sxx
    res = y - (b * x + (ym - b * xm))
    ss  = (res ** 2).sum()
    r2  = round(1.0 - ss / Syy, 4) if Syy > 0 else 0.0
    se  = np.sqrt(ss / ((n - 2) * Sxx)) if n > 2 else 0.0
    k   = round(b / (se * n ** 0.5), 4) if se > 0 else 0.0
    return k, r2

def calc_stats(all_pnl, eq_list):
    if not all_pnl:
        return {"n_trades":0,"cagr_pct":0.0,"mdd_pct":0.0,"sharpe":0.0,
                "win_rate":0.0,"pf":0.0,"kfactor":0.0,"r2":0.0}
    df   = pd.DataFrame(all_pnl)
    eq   = pd.Series(eq_list, dtype=float)
    peak = eq.cummax()
    mdd  = ((eq - peak) / peak * 100.0).min()
    n_yrs = max(len(eq) / 252.0, 0.1)
    cagr  = (eq.iloc[-1] / START_CAP) ** (1.0/n_yrs) - 1.0
    wins  = df[df["net_pnl"]>0]["net_pnl"]; loss = df[df["net_pnl"]<=0]["net_pnl"]
    pf    = wins.sum()/abs(loss.sum()) if abs(loss.sum())>0 else float("inf")
    dp    = df.groupby("date")["net_pnl"].sum() / START_CAP
    sharpe = dp.mean()/dp.std()*(252**0.5) if dp.std()>0 else 0.0
    k, r2 = _calc_kfactor(eq_list)
    return {"n_trades":len(df),"cagr_pct":round(cagr*100,1),"mdd_pct":round(mdd,1),
            "sharpe":round(sharpe,2),"win_rate":round((df["net_pnl"]>0).mean()*100,1),
            "pf":round(pf,2),"kfactor":k,"r2":r2}

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ORB3 ATR_TRAIL sensitivity backtest")
    ap.add_argument("--force-refresh", action="store_true",
                    help="Re-pull all bar data from DB even if pickles exist")
    args = ap.parse_args()

    t_start = _time.time()
    print("="*70)
    print("ORB3 - ATR_TRAIL SENSITIVITY BACKTEST  v5 (K-Ratio)")
    print("="*70)

    # ── Data loading (DB or cache) ──────────────────────────────────────────
    need_daily = args.force_refresh or not PICKLE_PATH.exists()
    need_bars  = args.force_refresh or not BAR_CACHE_PATH.exists()

    if need_daily or need_bars:
        if args.force_refresh:
            print("\n--force-refresh: re-pulling from DB ...")
        symbols  = load_universe_from_db()
        daily, bar_cache = load_all_from_db(symbols)

        if need_daily:
            print("Saving daily metrics ...")
            with open(PICKLE_PATH, "wb") as f: pickle.dump(daily, f, protocol=4)
            print(f"  -> {PICKLE_PATH}  ({PICKLE_PATH.stat().st_size/1e6:.0f} MB)")

        if need_bars:
            print("Saving bar cache ...")
            with open(BAR_CACHE_PATH, "wb") as f: pickle.dump(bar_cache, f, protocol=4)
            print(f"  -> {BAR_CACHE_PATH}  ({BAR_CACHE_PATH.stat().st_size/1e6:.0f} MB)")

        if not need_daily:
            with open(PICKLE_PATH, "rb") as f: daily = pickle.load(f)
        if not need_bars:
            t0 = _time.time()
            with open(BAR_CACHE_PATH, "rb") as f: bar_cache = pickle.load(f)
            print(f"Loaded bar cache ({_time.time()-t0:.0f}s)")
    else:
        print(f"\nLoading daily metrics from cache ...")
        with open(PICKLE_PATH, "rb") as f: daily = pickle.load(f)
        print(f"  {len(daily):,} rows")
        bc_mb = BAR_CACHE_PATH.stat().st_size / 1e6
        print(f"Loading bar cache ({bc_mb:.0f} MB) ...")
        t0 = _time.time()
        with open(BAR_CACHE_PATH, "rb") as f: bar_cache = pickle.load(f)
        print(f"  {len(bar_cache)} symbols  ({_time.time()-t0:.0f}s)")

    # ── Sensitivity sweep ───────────────────────────────────────────────────
    runs  = [(d, p, v, {**BASE, p: v}) for d, p, vals in GRID for v in vals]
    total = len(runs)
    print(f"\n{total} runs to test.\n")

    if RESULTS_CSV.exists(): RESULTS_CSV.unlink()
    results = []

    for idx, (dim, param, val, params) in enumerate(runs, 1):
        is_bl = abs(float(val) - float(BASE[param])) < 1e-9
        tag   = " [baseline]" if is_bl else ""
        print(f"[{idx:02d}/{total}] {dim}: {param}={val}{tag}", end="  ", flush=True)
        t0 = _time.time()
        pnl, eq = run_once_np(daily, bar_cache, **params)
        s = calc_stats(pnl, eq)
        print(f"Sharpe={s['sharpe']:.2f}  K={s['kfactor']:.3f}  CAGR={s['cagr_pct']}%  "
              f"MDD={s['mdd_pct']}%  ({_time.time()-t0:.0f}s)")
        results.append({"dim": dim, "param": param, "value": val, "is_baseline": is_bl, **s})
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)

    res = pd.DataFrame(results)

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "="*70 + "\nSENSITIVITY SUMMARY\n" + "="*70)
    for dim, param, vals in GRID:
        sub     = res[res["param"] == param].sort_values("value")
        sharpes = sub["sharpe"].astype(float).values
        kfactors = sub["kfactor"].astype(float).values
        bv      = float(BASE[param])
        pk      = float(sub.iloc[int(np.argmax(sharpes))]["value"])
        dist    = abs(pk - bv) / (abs(bv) + 1e-9) * 100
        print(f"\n  {dim}  (baseline={bv})")
        print(f"  {'value':>10}  {'Sharpe':>7}  {'K-Ratio':>8}  {'R2':>6}  "
              f"{'CAGR%':>7}  {'MDD%':>7}  {'WR%':>6}")
        for _, r in sub.iterrows():
            mk = " <<" if abs(float(r["value"]) - bv) < 1e-9 else ""
            print(f"  {float(r['value']):>10.2f}  {float(r['sharpe']):>7.2f}  "
                  f"{float(r['kfactor']):>8.4f}  {float(r['r2']):>6.4f}  "
                  f"{float(r['cagr_pct']):>7.1f}  {float(r['mdd_pct']):>7.1f}  "
                  f"{float(r['win_rate']):>6.1f}{mk}")
        verdict = "ok - near baseline" if dist <= 20 else "CHERRY-PICK RISK"
        print(f"  Peak Sharpe {sharpes.max():.2f} at {param}={pk}  "
              f"dist={dist:.0f}%  [{verdict}]")
        print(f"  Sharpe >= 1.5: {int((sharpes >= 1.5).sum())}/{len(sharpes)}  |  "
              f"K-Ratio > 0: {int((kfactors > 0).sum())}/{len(kfactors)}")

    # ── Dual-criterion: best configs (Sharpe >= 1.5 AND K-Ratio > 0) ───────
    print("\n" + "="*70)
    print("DUAL-CRITERION RANKING  (Sharpe >= 1.5  AND  K-Ratio > 0)")
    print("="*70)
    dual = (res[(res["sharpe"].astype(float) >= 1.5) &
                (res["kfactor"].astype(float) > 0)]
            .sort_values("kfactor", ascending=False)
            .head(10))
    if not dual.empty:
        print(f"\n  {'dim':<22} {'param':<12} {'value':>8}  "
              f"{'Sharpe':>7}  {'K-Ratio':>8}  {'R2':>6}  {'CAGR%':>7}  {'MDD%':>7}")
        for _, r in dual.iterrows():
            bl_tag = " [BL]" if r["is_baseline"] else ""
            print(f"  {r['dim']:<22} {r['param']:<12} {float(r['value']):>8.2f}  "
                  f"{float(r['sharpe']):>7.2f}  {float(r['kfactor']):>8.4f}  "
                  f"{float(r['r2']):>6.4f}  {float(r['cagr_pct']):>7.1f}  "
                  f"{float(r['mdd_pct']):>7.1f}{bl_tag}")
    else:
        print("  (no configs passed both criteria)")

    # ── Sensitivity plots (5x4: Sharpe | CAGR | MDD | K-Ratio) ────────────
    mets = [
        ("sharpe",   "Sharpe",      "steelblue"),
        ("cagr_pct", "CAGR (%)",    "seagreen"),
        ("mdd_pct",  "Max DD (%)",  "firebrick"),
        ("kfactor",  "K-Ratio",     "mediumpurple"),
    ]
    fig, axes = plt.subplots(len(GRID), 4, figsize=(22, 4 * len(GRID)))
    for ri, (dim, param, _) in enumerate(GRID):
        sub = res[res["param"] == param].sort_values("value")
        bv  = float(BASE[param])
        for ci, (met, lbl, col) in enumerate(mets):
            ax = axes[ri][ci]
            ax.plot(sub["value"].astype(float), sub[met].astype(float),
                    "o-", color=col, lw=2, ms=7)
            ax.axvline(bv, color="gray", ls="--", alpha=0.6, label=f"base={bv}")
            if met == "sharpe":
                ax.axhline(1.5, color="orange", ls=":", lw=1.2, alpha=0.9)
            if met == "kfactor":
                ax.axhline(0.0, color="orange", ls=":", lw=1.2, alpha=0.9)
            ax.set_xlabel(dim, fontsize=9); ax.set_ylabel(lbl, fontsize=9)
            ax.set_title(f"{lbl} vs {dim}", fontsize=9, fontweight="bold")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle("ATR_TRAIL - Parameter Sensitivity  (Sharpe | CAGR | MDD | K-Ratio)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "sensitivity_plots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSensitivity plot -> {out}")

    print(f"\nTotal: {(_time.time()-t_start)/60:.1f} min  |  CSV -> {RESULTS_CSV}")
