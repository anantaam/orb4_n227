#!/usr/bin/env python3
"""EOD P&L report to Discord. Fires after hard exit (14:45 IST)."""
import json, os, sys
from datetime import date
from pathlib import Path
import requests

BASE_DIR   = Path(__file__).resolve().parent.parent
EOD_WEBHOOK = os.getenv("EOD_WEBHOOK", "")
CAPITAL    = 100_000
TODAY      = date.today().isoformat()

TRACKS = [
    ("Baseline", BASE_DIR / "state/baseline/trade_journal.jsonl"),
]

EXIT_LABEL = {
    "intraday_sl_or_target": "SL/TGT",
    "hard_exit":             "TIME",
    "software_sl":           "SL",
    "target":                "TGT",
    "sl":                    "SL",
}

def read_journal(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def inr(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}Rs {v:,.0f}"

def summarise(rows):
    today_entries = {r["symbol"]: r for r in rows
                     if r.get("event") == "entry_execution" and r.get("ts_utc","")[:10] == TODAY}
    today_exits   = [r for r in rows
                     if r.get("event") == "exit_execution" and r.get("ts_utc","")[:10] == TODAY]
    all_exits     = [r for r in rows if r.get("event") == "exit_execution"]

    trade_lines = []
    for ex in today_exits:
        sym  = ex["symbol"]
        en   = today_entries.get(sym, {})
        side = ex.get("direction", en.get("direction", "?"))
        qty  = ex.get("quantity",  en.get("quantity",  0))
        epx  = ex.get("entry_price", en.get("entry_price", 0))
        sl   = en.get("sl_price",  0)
        tgt  = en.get("target_price", 0)
        npnl = ex.get("net_pnl", 0)
        etype = EXIT_LABEL.get(ex.get("exit_type",""), ex.get("exit_type","?"))
        elabel = ex.get("exit_label","")
        if elabel and elabel != "UNKNOWN":
            etype = elabel

        arrow = "+" if npnl >= 0 else ""
        trade_lines.append(
            f"  `{sym:<10}` {side:<5}  {qty:>5} @ {epx:.2f}"
            f"  SL={sl:.2f}  TGT={tgt:.2f}"
            f"  [{etype}]  **{arrow}Rs {npnl:,.0f}**"
        )

    today_pnl = sum(r.get("net_pnl", 0) for r in today_exits)
    cum_pnl   = sum(r.get("net_pnl", 0) for r in all_exits)
    equity    = CAPITAL + cum_pnl
    return trade_lines, today_pnl, cum_pnl, equity

sections = [f"**ORB EOD  |  {TODAY}**\n"]

for name, jpath in TRACKS:
    rows = read_journal(jpath)
    trade_lines, today_pnl, cum_pnl, equity = summarise(rows)

    block = [f"**{name}**"]
    if not trade_lines:
        block.append("  Today: idle (no trades)")
    else:
        block.extend(trade_lines)
        block.append(f"  ─────────────────────────────")
        block.append(f"  Today net P&L: **{inr(today_pnl)}**")
    block.append(f"  Cumulative: {inr(cum_pnl)}  |  Equity: Rs {equity:,.0f}")
    sections.append("\n".join(block))

msg = "\n\n".join(sections)
resp = requests.post(EOD_WEBHOOK, json={"content": msg}, timeout=10)
resp.raise_for_status()
print("Sent:\n" + msg)
