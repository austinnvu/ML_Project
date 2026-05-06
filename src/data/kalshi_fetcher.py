"""
Utilities for fetching and matching Kalshi NBA market data.

This module supports both the live market endpoints and the historical
endpoints so older settled games and recent/current games can be combined into
one dataset.
"""

from __future__ import annotations

import base64
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)

load_dotenv()


BASE_URL = os.getenv(
    "KALSHI_BASE_URL",
    "https://api.elections.kalshi.com/trade-api/v2",
).rstrip("/")

DEFAULT_TIMEOUT = int(os.getenv("KALSHI_TIMEOUT_SECONDS", "30"))
DEFAULT_SLEEP_SECONDS = float(os.getenv("KALSHI_SLEEP_SECONDS", "0.35"))
DEFAULT_RATE_LIMIT_BACKOFF = float(os.getenv("KALSHI_RATE_LIMIT_BACKOFF_SECONDS", "5.0"))
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "").strip()
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()

MONEYLINE_EXCLUDE_TERMS = {
    "over",
    "under",
    "total",
    "totals",
    "points",
    "combined points",
    "spread",
    "margin",
}

MONEYLINE_TICKER_PREFIXES: Tuple[str, ...] = ("KXNBAGAME",)

TEAM_ALIASES: Dict[str, Sequence[str]] = {
    "ATL": ("atlanta hawks", "hawks", "atlanta"),
    "BOS": ("boston celtics", "celtics", "boston"),
    "BKN": ("brooklyn nets", "nets", "brooklyn"),
    "CHA": ("charlotte hornets", "hornets", "charlotte"),
    "CHI": ("chicago bulls", "bulls", "chicago"),
    "CLE": ("cleveland cavaliers", "cavaliers", "cavs", "cleveland"),
    "DAL": ("dallas mavericks", "mavericks", "mavs", "dallas"),
    "DEN": ("denver nuggets", "nuggets", "denver"),
    "DET": ("detroit pistons", "pistons", "detroit"),
    "GSW": (
        "golden state warriors",
        "warriors",
        "golden state",
    ),
    "HOU": ("houston rockets", "rockets", "houston"),
    "IND": ("indiana pacers", "pacers", "indiana"),
    "LAC": ("la clippers", "los angeles clippers", "clippers"),
    "LAL": ("la lakers", "los angeles lakers", "lakers"),
    "MEM": ("memphis grizzlies", "grizzlies", "memphis"),
    "MIA": ("miami heat", "heat", "miami"),
    "MIL": ("milwaukee bucks", "bucks", "milwaukee"),
    "MIN": ("minnesota timberwolves", "timberwolves", "wolves", "minnesota"),
    "NOP": (
        "new orleans pelicans",
        "pelicans",
        "new orleans",
        "nola",
    ),
    "NYK": ("new york knicks", "knicks", "new york"),
    "OKC": (
        "oklahoma city thunder",
        "thunder",
        "oklahoma city",
        "okc",
    ),
    "ORL": ("orlando magic", "magic", "orlando"),
    "PHI": ("philadelphia 76ers", "76ers", "sixers", "philadelphia"),
    "PHX": ("phoenix suns", "suns", "phoenix"),
    "POR": ("portland trail blazers", "trail blazers", "blazers", "portland"),
    "SAC": ("sacramento kings", "kings", "sacramento"),
    "SAS": ("san antonio spurs", "spurs", "san antonio"),
    "TOR": ("toronto raptors", "raptors", "toronto"),
    "UTA": ("utah jazz", "jazz", "utah"),
    "WAS": ("washington wizards", "wizards", "washington"),
}

ALIAS_TO_TEAM: Dict[str, str] = {}
for team_abbr, aliases in TEAM_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_TEAM[alias] = team_abbr

SORTED_ALIASES = sorted(ALIAS_TO_TEAM.keys(), key=len, reverse=True)


@dataclass
class KalshiMatch:
    game_date: pd.Timestamp
    home_team: str
    away_team: str
    market_ticker: str
    event_ticker: str
    market_title: str
    market_subtitle: str
    market_status: str
    market_source: str
    yes_price: Optional[float]
    home_win_price: Optional[float]
    no_price: Optional[float]
    last_price: Optional[float]
    yes_team: Optional[str]
    volume: Optional[float]
    close_time: Optional[str]
    settlement_time: Optional[str]


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    normalized = str(text).lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_team_mentions(text: str) -> List[Tuple[int, str]]:
    normalized = _normalize_text(text)
    mentions: List[Tuple[int, str]] = []

    for alias in SORTED_ALIASES:
        pattern = rf"\b{re.escape(alias)}\b"
        match = re.search(pattern, normalized)
        if match:
            mentions.append((match.start(), ALIAS_TO_TEAM[alias]))

    mentions.sort(key=lambda item: item[0])

    deduped: List[Tuple[int, str]] = []
    seen = set()
    for position, team_abbr in mentions:
        if team_abbr in seen:
            continue
        deduped.append((position, team_abbr))
        seen.add(team_abbr)
    return deduped


def _extract_two_teams(record: dict) -> List[str]:
    candidate_texts = [
        record.get("title"),
        record.get("subtitle"),
        record.get("yes_sub_title"),
        record.get("no_sub_title"),
        record.get("event_title"),
        record.get("event_sub_title"),
    ]

    ordered: List[str] = []
    for text in candidate_texts:
        for _, team_abbr in _extract_team_mentions(text or ""):
            if team_abbr not in ordered:
                ordered.append(team_abbr)

    return ordered[:2]


def _infer_market_type(record: dict) -> str:
    ticker = str(record.get("ticker") or "")
    if not ticker.startswith(MONEYLINE_TICKER_PREFIXES):
        return "non_moneyline"

    text = " ".join(
        filter(
            None,
            [
                record.get("title"),
                record.get("subtitle"),
                record.get("yes_sub_title"),
                record.get("no_sub_title"),
                record.get("event_title"),
                record.get("event_sub_title"),
            ],
        )
    )
    normalized = _normalize_text(text)

    if any(term in normalized for term in MONEYLINE_EXCLUDE_TERMS):
        return "non_moneyline"

    teams = _extract_two_teams(record)
    if len(teams) == 2:
        return "moneyline"

    return "unknown"


def _infer_yes_team(record: dict, teams: Sequence[str]) -> Optional[str]:
    if len(teams) != 2:
        return None

    yes_sub_title = record.get("yes_sub_title") or ""
    yes_mentions = _extract_team_mentions(yes_sub_title)
    if yes_mentions:
        return yes_mentions[0][1]

    title_mentions = _extract_team_mentions(record.get("title") or "")
    if title_mentions:
        return title_mentions[0][1]

    subtitle_mentions = _extract_team_mentions(record.get("subtitle") or "")
    if subtitle_mentions:
        return subtitle_mentions[0][1]

    return None


def _to_float(value: object) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_game_date(record: dict) -> Optional[pd.Timestamp]:
    for field in ("close_time", "expiration_time", "settlement_ts", "open_time"):
        raw_value = record.get(field)
        if not raw_value:
            continue

        timestamp = pd.to_datetime(raw_value, utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        return timestamp.tz_convert("America/New_York").normalize().tz_localize(None)

    return None


def _choose_yes_price(record: dict) -> Optional[float]:
    ask = _to_float(record.get("yes_ask_dollars"))
    bid = _to_float(record.get("yes_bid_dollars"))
    last_price = _to_float(record.get("last_price_dollars"))
    previous_price = _to_float(record.get("previous_price_dollars"))

    if ask is not None and bid is not None:
        return round((ask + bid) / 2.0, 4)
    if last_price is not None:
        return last_price
    if previous_price is not None:
        return previous_price
    return None


def _choose_no_price(record: dict, yes_price: Optional[float]) -> Optional[float]:
    no_ask = _to_float(record.get("no_ask_dollars"))
    no_bid = _to_float(record.get("no_bid_dollars"))
    if no_ask is not None and no_bid is not None:
        return round((no_ask + no_bid) / 2.0, 4)
    if yes_price is not None:
        return round(1.0 - yes_price, 4)
    return None


class KalshiClient:
    """REST client for Kalshi market data with optional authenticated requests."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        api_key_id: str = KALSHI_API_KEY_ID,
        private_key_path: str = KALSHI_PRIVATE_KEY_PATH,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()
        self.api_key_id = api_key_id.strip()
        self.private_key_path = private_key_path.strip()
        self.private_key = None

        if self.api_key_id and self.private_key_path:
            self.private_key = self._load_private_key(self.private_key_path)
            logger.info("Kalshi client configured with authenticated requests")
        elif self.api_key_id or self.private_key_path:
            logger.warning(
                "Kalshi auth is partially configured. Set both "
                "KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH to enable "
                "authenticated requests."
            )

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key_id and self.private_key)

    def _load_private_key(self, key_path: str):
        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"Kalshi private key file was not found at: {key_path}"
            )

        with open(key_path, "rb") as key_file:
            return serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend(),
            )

    def _build_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        if not self.auth_enabled:
            return {}

        timestamp = str(int(time.time() * 1000))
        sign_path = urlparse(f"{self.base_url}{path}").path
        sign_path = sign_path.split("?")[0]
        message = f"{timestamp}{method.upper()}{sign_path}".encode("utf-8")

        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        # One automatic retry on 429 with exponential backoff. Other status
        # codes (400/404) bubble up so callers can decide to skip.
        backoff = DEFAULT_RATE_LIMIT_BACKOFF
        for attempt in range(2):
            headers = self._build_auth_headers("GET", path)
            response = self.session.get(
                url,
                params=params,
                headers=headers or None,
                timeout=self.timeout,
            )
            if response.status_code == 429 and attempt == 0:
                logger.warning(
                    f"Kalshi rate limit hit on {path}; sleeping {backoff:.1f}s before retry"
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
            return response.json()
        response.raise_for_status()
        return response.json()

    def _paginate(self, path: str, root_key: str, params: Optional[dict] = None) -> List[dict]:
        all_rows: List[dict] = []
        cursor = None

        while True:
            request_params = dict(params or {})
            if cursor:
                request_params["cursor"] = cursor

            payload = self._get(path, request_params)
            all_rows.extend(payload.get(root_key, []))
            cursor = payload.get("cursor")

            if not cursor:
                break

        return all_rows

    def get_historical_cutoff(self) -> dict:
        return self._get("/historical/cutoff")

    def get_series(self, category: Optional[str] = None) -> List[dict]:
        params = {"category": category} if category else None
        payload = self._get("/series", params=params)
        return payload.get("series", [])

    def get_events(
        self,
        series_ticker: str,
        min_close_ts: Optional[int] = None,
    ) -> List[dict]:
        params = {
            "series_ticker": series_ticker,
            "limit": 200,
            "with_nested_markets": "false",
        }
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        return self._paginate("/events", "events", params=params)

    def get_event_markets(self, event_ticker: str, historical: bool = False) -> List[dict]:
        path = "/historical/markets" if historical else "/markets"
        params: Dict[str, object] = {
            "event_ticker": event_ticker,
            "limit": 1000,
        }
        # /markets accepts (and benefits from) mve_filter=exclude to suppress
        # the unused multi-vote-event partitions. /historical/markets rejects
        # it as mutually exclusive with event_ticker -- omit it there.
        if not historical:
            params["mve_filter"] = "exclude"
        return self._paginate(path, "markets", params=params)

    def get_candlesticks(
        self,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 5,
    ) -> List[dict]:
        """Fetch OHLC candlesticks for a single Kalshi market.

        period_interval is in minutes (minimum 1). Returns the list of
        candlestick dicts for the requested ticker, or [] if Kalshi has none.
        """
        payload = self._get(
            "/markets/candlesticks",
            params={
                "market_tickers": market_ticker,
                "start_ts": int(start_ts),
                "end_ts": int(end_ts),
                "period_interval": int(period_interval),
            },
        )
        markets = payload.get("markets") or []
        if not markets:
            return []
        return markets[0].get("candlesticks") or []


NBA_FULL_GAME_SERIES = ("KXNBAGAME",)


def discover_nba_series_tickers(
    client: KalshiClient,
    explicit_series_tickers: Optional[Iterable[str]] = None,
) -> List[str]:
    if explicit_series_tickers:
        return [ticker.strip() for ticker in explicit_series_tickers if ticker.strip()]
    return list(NBA_FULL_GAME_SERIES)


def _event_date_from_row(event_row: dict) -> Optional[pd.Timestamp]:
    for field in ("strike_date", "last_updated_ts"):
        raw_value = event_row.get(field)
        if not raw_value:
            continue
        timestamp = pd.to_datetime(raw_value, utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        return timestamp.tz_convert("America/New_York").normalize().tz_localize(None)
    return None


def _select_pregame_candle(candles: Sequence[dict]) -> Optional[dict]:
    """Pick the latest candle in a window with non-null yes_bid/yes_ask close prices.

    Filters out candles where Kalshi reports no quote (close_dollars is None),
    which happens for periods with no liquidity. Volume is intentionally NOT
    required to be > 0 — quiet pre-game minutes can have non-zero standing
    quotes with zero traded volume, and those are still meaningful prices.
    """
    qualifying: List[dict] = []
    for candle in candles:
        bid = (candle.get("yes_bid") or {}).get("close_dollars")
        ask = (candle.get("yes_ask") or {}).get("close_dollars")
        if bid is None or ask is None:
            continue
        try:
            float(bid)
            float(ask)
        except (TypeError, ValueError):
            continue
        qualifying.append(candle)
    if not qualifying:
        return None
    return max(qualifying, key=lambda c: int(c.get("end_period_ts") or 0))


def fetch_pregame_yes_prices(
    client: KalshiClient,
    matched_markets: pd.DataFrame,
    minutes_before_tipoff: int = 5,
    typical_game_minutes: int = 165,
    sample_window_hours: int = 3,
    period_interval: int = 1,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Pull pre-tipoff YES prices from Kalshi candlesticks for matched markets.

    Kalshi's market `close_time` is when trading stops (≈ end of game), not
    tip-off. We back off by `typical_game_minutes` (NBA games average ~2h45m
    including timeouts) to estimate tipoff, then pull candles in
    [tipoff - sample_window_hours, tipoff - minutes_before_tipoff] and take
    the most recent candle with a valid yes_bid/yes_ask quote. The midpoint
    of that quote becomes the pre-game YES price.

    Returns (price_by_ticker, sampled_ts_by_ticker). Tickers without a
    qualifying pre-game candle are simply absent from both dicts; downstream
    code falls back to the existing settled price.
    """
    pregame_prices: Dict[str, float] = {}
    sampled_timestamps: Dict[str, str] = {}

    relevant = matched_markets.dropna(subset=["kalshi_market_ticker", "kalshi_close_time"])
    total = len(relevant)
    logger.info(f"Fetching pre-game candlesticks for {total} markets")

    for position, row in enumerate(relevant.itertuples(index=False), start=1):
        ticker = getattr(row, "kalshi_market_ticker")
        close_str = getattr(row, "kalshi_close_time")

        close_ts = pd.to_datetime(close_str, utc=True, errors="coerce")
        if pd.isna(close_ts):
            continue

        close_epoch = int(close_ts.timestamp())
        tipoff_epoch = close_epoch - typical_game_minutes * 60
        start_ts = tipoff_epoch - sample_window_hours * 3600
        end_ts = tipoff_epoch - minutes_before_tipoff * 60

        try:
            candles = client.get_candlesticks(
                ticker, start_ts, end_ts, period_interval=period_interval
            )
        except requests.HTTPError as exc:
            logger.warning(f"Kalshi candlestick fetch failed for {ticker}: {exc}")
            continue

        chosen = _select_pregame_candle(candles)
        if chosen is None:
            continue

        bid = float(chosen["yes_bid"]["close_dollars"])
        ask = float(chosen["yes_ask"]["close_dollars"])
        pregame_prices[ticker] = round((bid + ask) / 2.0, 4)
        sampled_timestamps[ticker] = pd.Timestamp(
            int(chosen["end_period_ts"]), unit="s", tz="UTC"
        ).isoformat()

        if position % 50 == 0:
            logger.info(
                f"  pre-game candlesticks: {position}/{total} processed, "
                f"{len(pregame_prices)} priced"
            )

    logger.info(
        f"Pre-game candlestick fetch complete: {len(pregame_prices)}/{total} markets priced"
    )
    return pregame_prices, sampled_timestamps


def collect_nba_candidate_markets(
    nba_df: pd.DataFrame,
    client: Optional[KalshiClient] = None,
    explicit_series_tickers: Optional[Iterable[str]] = None,
) -> List[dict]:
    """
    Fetch candidate Kalshi moneyline markets for the games present in nba_df.

    The Kalshi docs split older settled markets into /historical/markets while
    recent/current markets remain available in /markets, so this function pulls
    from both tiers as needed.
    """

    client = client or KalshiClient()

    min_game_date = pd.to_datetime(nba_df["GAME_DATE"]).min()
    max_game_date = pd.to_datetime(nba_df["GAME_DATE"]).max()
    min_close_ts = int(
        pd.Timestamp(min_game_date).tz_localize("America/New_York").timestamp()
    )

    cutoff_payload = client.get_historical_cutoff()
    cutoff_ts = pd.to_datetime(
        cutoff_payload.get("market_settled_ts"),
        utc=True,
        errors="coerce",
    )
    cutoff_date = None
    if not pd.isna(cutoff_ts):
        cutoff_date = cutoff_ts.tz_convert("America/New_York").normalize().tz_localize(None)

    series_tickers = discover_nba_series_tickers(client, explicit_series_tickers)
    logger.info(f"Using Kalshi series tickers: {series_tickers}")

    candidate_markets: List[dict] = []
    seen_event_tickers = set()

    for series_ticker in series_tickers:
        events = client.get_events(series_ticker=series_ticker, min_close_ts=min_close_ts)
        logger.info(f"Fetched {len(events)} events for Kalshi series {series_ticker}")

        for event in events:
            event_ticker = event.get("event_ticker")
            if not event_ticker or event_ticker in seen_event_tickers:
                continue

            event_date = _event_date_from_row(event)
            if event_date is not None and event_date > max_game_date:
                continue

            teams = _extract_two_teams(
                {
                    "title": event.get("title"),
                    "subtitle": event.get("sub_title"),
                }
            )
            if len(teams) != 2:
                continue

            use_historical = bool(cutoff_date is not None and event_date is not None and event_date < cutoff_date)
            primary_status: Optional[int] = None
            try:
                markets = client.get_event_markets(event_ticker, historical=use_historical)
            except requests.HTTPError as exc:
                primary_status = getattr(exc.response, "status_code", None)
                logger.warning(
                    f"Kalshi market fetch failed for event {event_ticker} "
                    f"(historical={use_historical}): {exc}"
                )
                markets = []
            source = "historical" if use_historical else "live"

            # Only retry on the alternate endpoint when the primary returned
            # an empty list (event genuinely lives in the other archive).
            # 400/404 mean the event isn't there at all -- skip to avoid
            # doubling request load and triggering 429s.
            if not markets and primary_status not in (400, 404):
                try:
                    markets = client.get_event_markets(event_ticker, historical=not use_historical)
                    source = "historical" if source == "live" else "live"
                except requests.HTTPError as exc:
                    logger.warning(
                        f"Kalshi market fetch failed for event {event_ticker} "
                        f"on alternate endpoint: {exc}; skipping"
                    )
                    markets = []

            seen_event_tickers.add(event_ticker)

            for market in markets:
                market_record = dict(market)
                market_record["event_title"] = event.get("title")
                market_record["event_sub_title"] = event.get("sub_title")
                market_record["market_source"] = source
                candidate_markets.append(market_record)

    logger.info(f"Collected {len(candidate_markets)} Kalshi candidate markets")
    return candidate_markets


def build_kalshi_game_prices(
    nba_df: pd.DataFrame,
    candidate_markets: Sequence[dict],
    client: Optional[KalshiClient] = None,
    fetch_pregame: bool = True,
) -> pd.DataFrame:
    """
    Convert raw Kalshi market payloads into one matched moneyline price row per NBA game.

    When `fetch_pregame=True` and a `client` is provided, the matched markets are
    augmented with pre-tipoff YES prices pulled from the candlesticks endpoint,
    and `kalshi_price_home_win` is recomputed using those pre-game prices when
    available (falling back to the settled snapshot price otherwise).
    """

    market_rows = []
    for market in candidate_markets:
        if _infer_market_type(market) != "moneyline":
            continue

        teams = _extract_two_teams(market)
        if len(teams) != 2:
            continue

        market_date = _market_game_date(market)
        if market_date is None:
            continue

        yes_price = _choose_yes_price(market)
        no_price = _choose_no_price(market, yes_price)
        yes_team = _infer_yes_team(market, teams)

        market_rows.append(
            {
                "GAME_DATE": market_date,
                "team_a": teams[0],
                "team_b": teams[1],
                "team_key": "|".join(sorted(teams)),
                "kalshi_market_ticker": market.get("ticker"),
                "kalshi_event_ticker": market.get("event_ticker"),
                "kalshi_market_title": market.get("title"),
                "kalshi_market_subtitle": market.get("subtitle"),
                "kalshi_market_status": market.get("status"),
                "kalshi_market_source": market.get("market_source"),
                "kalshi_yes_price": yes_price,
                "kalshi_no_price": no_price,
                "kalshi_last_price": _to_float(market.get("last_price_dollars")),
                "kalshi_yes_team": yes_team,
                "kalshi_volume": _to_float(market.get("volume_fp")),
                "kalshi_close_time": market.get("close_time"),
                "kalshi_settlement_ts": market.get("settlement_ts"),
            }
        )

    if not market_rows:
        logger.warning("No candidate moneyline markets were parsed from Kalshi data")
        return pd.DataFrame(
            columns=[
                "GAME_DATE",
                "home_TEAM_ABBREVIATION",
                "away_TEAM_ABBREVIATION",
                "kalshi_market_ticker",
                "kalshi_event_ticker",
                "kalshi_market_title",
                "kalshi_market_subtitle",
                "kalshi_market_status",
                "kalshi_market_source",
                "kalshi_yes_price",
                "kalshi_yes_price_pregame",
                "kalshi_pregame_sampled_ts",
                "kalshi_no_price",
                "kalshi_last_price",
                "kalshi_price_home_win",
                "kalshi_yes_team",
                "kalshi_volume",
                "kalshi_close_time",
                "kalshi_settlement_ts",
            ]
        )

    markets_df = pd.DataFrame(market_rows)
    markets_df["GAME_DATE"] = pd.to_datetime(markets_df["GAME_DATE"]).dt.normalize()

    nba_games = nba_df.copy()
    nba_games["GAME_DATE"] = pd.to_datetime(nba_games["GAME_DATE"]).dt.normalize()
    nba_games["team_key"] = nba_games.apply(
        lambda row: "|".join(
            sorted([row["home_TEAM_ABBREVIATION"], row["away_TEAM_ABBREVIATION"]])
        ),
        axis=1,
    )

    merged = nba_games.merge(
        markets_df,
        on=["GAME_DATE", "team_key"],
        how="left",
        suffixes=("", "_kalshi"),
    )

    def compute_home_win_price(row: pd.Series) -> Optional[float]:
        yes_price = row.get("kalshi_yes_price")
        yes_team = row.get("kalshi_yes_team")
        if pd.isna(yes_price) or not yes_team:
            return None
        if yes_team == row["home_TEAM_ABBREVIATION"]:
            return float(yes_price)
        if yes_team == row["away_TEAM_ABBREVIATION"]:
            return round(1.0 - float(yes_price), 4)
        return None

    merged["kalshi_price_home_win"] = merged.apply(compute_home_win_price, axis=1)

    sort_columns = ["kalshi_price_home_win", "kalshi_volume", "kalshi_last_price"]
    merged = merged.sort_values(
        by=sort_columns,
        ascending=[False, False, False],
        na_position="last",
    )

    deduped = merged.drop_duplicates(
        subset=["GAME_DATE", "home_TEAM_ABBREVIATION", "away_TEAM_ABBREVIATION"],
        keep="first",
    ).reset_index(drop=True)

    deduped["kalshi_yes_price_pregame"] = pd.NA
    deduped["kalshi_pregame_sampled_ts"] = pd.NA

    if fetch_pregame and client is not None:
        pregame_prices, sampled_ts = fetch_pregame_yes_prices(client, deduped)
        if pregame_prices:
            deduped["kalshi_yes_price_pregame"] = (
                deduped["kalshi_market_ticker"].map(pregame_prices)
            )
            deduped["kalshi_pregame_sampled_ts"] = (
                deduped["kalshi_market_ticker"].map(sampled_ts)
            )

    def compute_home_win_price_with_pregame(row: pd.Series) -> Optional[float]:
        yes_team = row.get("kalshi_yes_team")
        if not yes_team:
            return None
        pregame_yes = row.get("kalshi_yes_price_pregame")
        settled_yes = row.get("kalshi_yes_price")
        chosen_yes = pregame_yes if pd.notna(pregame_yes) else settled_yes
        if pd.isna(chosen_yes):
            return None
        if yes_team == row["home_TEAM_ABBREVIATION"]:
            return float(chosen_yes)
        if yes_team == row["away_TEAM_ABBREVIATION"]:
            return round(1.0 - float(chosen_yes), 4)
        return None

    deduped["kalshi_price_home_win"] = deduped.apply(
        compute_home_win_price_with_pregame, axis=1
    )

    output_columns = [
        "GAME_DATE",
        "home_TEAM_ABBREVIATION",
        "away_TEAM_ABBREVIATION",
        "kalshi_market_ticker",
        "kalshi_event_ticker",
        "kalshi_market_title",
        "kalshi_market_subtitle",
        "kalshi_market_status",
        "kalshi_market_source",
        "kalshi_yes_price",
        "kalshi_yes_price_pregame",
        "kalshi_pregame_sampled_ts",
        "kalshi_no_price",
        "kalshi_last_price",
        "kalshi_price_home_win",
        "kalshi_yes_team",
        "kalshi_volume",
        "kalshi_close_time",
        "kalshi_settlement_ts",
    ]

    return deduped[output_columns].reset_index(drop=True)
