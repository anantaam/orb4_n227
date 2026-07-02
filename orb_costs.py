from __future__ import annotations

# Zerodha NSE Equity Intraday (MIS) cost model
BROKERAGE_RATE = 0.0003
BROKERAGE_CAP = 20.0
STT_SELL_RATE = 0.00025
NSE_EXCH_RATE = 0.0000297  # matches backtest calc_pnl (current NSE rate 0.00297%)
SEBI_RATE = 0.000001
STAMP_BUY_RATE = 0.00003
GST_RATE = 0.18


def zerodha_intraday_cost_orb(entry_value: float, exit_value: float, direction: str) -> float:
    """
    direction: "LONG" or "SHORT"
    LONG:  buy at entry, sell at exit
    SHORT: sell at entry, buy at exit
    """
    if direction.upper() == "SHORT":
        sell_value = entry_value
        buy_value = exit_value
    else:  # LONG
        sell_value = exit_value
        buy_value = entry_value

    brok_sell = min(BROKERAGE_RATE * sell_value, BROKERAGE_CAP)
    brok_buy = min(BROKERAGE_RATE * buy_value, BROKERAGE_CAP)
    brok_total = brok_sell + brok_buy
    stt = STT_SELL_RATE * sell_value
    nse = NSE_EXCH_RATE * (sell_value + buy_value)
    sebi = SEBI_RATE * (sell_value + buy_value)
    stamp = STAMP_BUY_RATE * buy_value
    gst = GST_RATE * (brok_total + nse + sebi)
    return brok_total + stt + nse + sebi + stamp + gst
