"""
Trading day utilities backed by live NSE holiday API + persisted API snapshots.

Holidays are stored under the e3 repo `state/`. No hardcoded holiday lists are used.
"""
import json
import os
from datetime import date, datetime, timedelta
import requests

# NSE API
NSE_BASE_URL = "https://www.nseindia.com"
NSE_HOLIDAY_API_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
NSE_DATE_FMT = "%d-%b-%Y"
# Capital Market = equity cash segment; ignore CD/CBM/etc. (e.g. bank closing) for trading-day gates
NSE_SEGMENT_CM = "CM"

def _e3_repo_root() -> str:
    """e3 repository root (parent of this src/ package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _holidays_state_path(year: int) -> str:
    """Path to saved holidays file for a given year (under e3/state/)."""
    return os.path.join(_e3_repo_root(), "state", f"nse_holidays_{year}.json")


def _load_saved_holidays(year: int) -> set[date] | None:
    """Load holiday dates for year from state/nse_holidays_<year>.json. Returns None if missing/invalid."""
    path = _holidays_state_path(year)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None
        out: set[date] = set()
        for item in data:
            if isinstance(item, str):
                try:
                    out.add(datetime.strptime(item, "%Y-%m-%d").date())
                except ValueError:
                    pass
        return out if out else None
    except (json.JSONDecodeError, IOError):
        return None


def _save_holidays(year: int, holidays: set[date]) -> None:
    """Persist holiday set for year to state/nse_holidays_<year>.json (YYYY-MM-DD strings)."""
    path = _holidays_state_path(year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(d.isoformat() for d in holidays), f, indent=0)


def _parse_nse_date(s: str) -> date | None:
    """Parse NSE tradingDate '%d-%b-%Y' to date."""
    try:
        return datetime.strptime(s.strip(), NSE_DATE_FMT).date()
    except (ValueError, TypeError):
        return None


class NSEHolidayService:
    """
    NSE trading holiday service.
    Primary: fetch from NSE API (session to nseindia.com, then GET holiday-master).
    Fallback: persisted snapshots from prior successful API calls.
    """

    _holidays_cache: set[date] | None = None
    _cache_date: date | None = None

    def fetch_live_from_nse(self) -> set[date] | None:
        """
        Fetch CM (equity capital market) trading holidays from NSE API.
        Returns set of holiday dates (possibly empty) on success, or None on HTTP/parse failure.

        Other segments (e.g. CD, CBM) can show holidays on days when CM is open; those must
        not block this cash engine.
        """
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": NSE_BASE_URL + "/",
        })
        try:
            # Required: hit homepage first so NSE sets cookies
            session.get(NSE_BASE_URL + "/", timeout=10)
            r = session.get(NSE_HOLIDAY_API_URL, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return None

        if not isinstance(data, dict):
            return None
        segment_list = data.get(NSE_SEGMENT_CM)
        if not isinstance(segment_list, list):
            return None

        holidays: set[date] = set()
        for item in segment_list:
            if isinstance(item, dict) and "tradingDate" in item:
                d = _parse_nse_date(item["tradingDate"])
                if d is not None:
                    holidays.add(d)
        return holidays

    def _get_last_year_holidays(self, today: date) -> set[date]:
        """Return last year's holidays from saved API snapshot if available."""
        last_year = today.year - 1
        saved = _load_saved_holidays(last_year)
        return saved or set()

    def get_holidays(self) -> set[date]:
        """
        Return set of NSE trading holiday dates.
        API (current year) + last year (saved from API snapshot only).
        """
        today = date.today()
        if self._holidays_cache is not None and self._cache_date == today:
            return self._holidays_cache
        self._holidays_cache = None
        self._cache_date = today

        last_year_holidays = self._get_last_year_holidays(today)

        fetched = self.fetch_live_from_nse()
        if fetched is not None:
            self._holidays_cache = fetched | last_year_holidays
            _save_holidays(today.year, fetched)
            return self._holidays_cache

        saved_current = _load_saved_holidays(today.year) or set()
        self._holidays_cache = saved_current | last_year_holidays
        return self._holidays_cache

    def is_holiday(self, check_date: date) -> bool:
        """Return True if check_date is an NSE trading holiday."""
        return check_date in self.get_holidays()


_holiday_service: NSEHolidayService | None = None


def get_holiday_service() -> NSEHolidayService:
    """Return the shared NSE holiday service instance."""
    global _holiday_service
    if _holiday_service is None:
        _holiday_service = NSEHolidayService()
    return _holiday_service


def is_weekend(check_date: date) -> bool:
    """Return True if check_date is Saturday or Sunday."""
    return check_date.weekday() >= 5


def is_trading_day(check_date: date | datetime | None = None) -> bool:
    """
    Return True if check_date is an NSE trading day (not weekend, not NSE holiday).
    check_date defaults to today (in local date).
    """
    if check_date is None:
        check_date = date.today()
    if isinstance(check_date, datetime):
        check_date = check_date.date()
    if is_weekend(check_date):
        return False
    return not get_holiday_service().is_holiday(check_date)


def next_trading_day(check_date: date) -> date:
    """
    Return the first date >= check_date that is a trading day.
    If check_date is already a trading day, return it; else advance until the next trading day.
    """
    d = check_date
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def add_trading_days(start_date: date, n: int) -> date:
    """
    Return the date that is exactly n trading days after start_date.
    start_date is day 0; the result is the nth trading day after it (e.g. n=1 is the next trading day).
    """
    if n <= 0:
        return next_trading_day(start_date)
    current = next_trading_day(start_date)
    for _ in range(n):
        current += timedelta(days=1)
        current = next_trading_day(current)
    return current
