"""Utility helpers for fetching market-wide valuation averages."""

from __future__ import annotations

import difflib
from functools import lru_cache
from typing import Dict, Optional, Tuple

import pandas as pd
import requests


class MarketAveragesFetcher:
    """Fetch live sector and industry valuation averages from Finviz."""

    INDUSTRY_URL = "https://finviz.com/groups.ashx?g=industry&v=120&o=name"
    SECTOR_URL = "https://finviz.com/groups.ashx?g=sector&v=120&o=name"
    REQUEST_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finviz.com/",
    }
    REQUEST_TIMEOUT = 10

    _SECTOR_FALLBACK_ENTRIES = (
        ("Technology", 25.0, 6.0),
        ("Healthcare", 18.0, 4.0),
        ("Financial Services", 12.0, 2.0),
        ("Financial", 12.0, 2.0),
        ("Consumer Cyclical", 20.0, 1.5),
        ("Consumer Defensive", 15.0, 1.2),
        ("Consumer Staples", 15.0, 1.2),
        ("Communication Services", 22.0, 3.0),
        ("Industrials", 16.0, 1.8),
        ("Energy", 14.0, 1.0),
        ("Utilities", 16.0, 2.0),
        ("Real Estate", 20.0, 8.0),
        ("Basic Materials", 15.0, 1.5),
        ("Materials", 15.0, 1.5),
    )
    SECTOR_FALLBACK: Dict[str, Dict[str, Optional[float]]] = {
        name.lower(): {"name": name, "pe": pe, "ps": ps}
        for name, pe, ps in _SECTOR_FALLBACK_ENTRIES
    }
    INDUSTRY_FALLBACK: Dict[str, Dict[str, Optional[float]]] = {
        key: value.copy() for key, value in SECTOR_FALLBACK.items()
    }

    @classmethod
    def get_sector_metrics(cls, sector_name: str) -> Dict[str, Optional[float]]:
        """Return sector average valuation metrics."""

        data, source = cls._get_sector_data()
        metrics = cls._match_metrics(data, sector_name)
        if metrics:
            result = metrics.copy()
            result.setdefault("name", sector_name or result.get("name", "Sector"))
            result["source"] = source
            return result

        fallback = cls._match_metrics(cls.SECTOR_FALLBACK, sector_name)
        if fallback:
            result = fallback.copy()
            result.setdefault("name", sector_name or result.get("name", "Sector"))
            result["source"] = "Fallback (static)"
            return result

        return {"name": sector_name or "Sector", "pe": None, "ps": None, "source": source}

    @classmethod
    def get_industry_metrics(cls, industry_name: str) -> Dict[str, Optional[float]]:
        """Return industry average valuation metrics."""

        data, source = cls._get_industry_data()
        metrics = cls._match_metrics(data, industry_name)
        if metrics:
            result = metrics.copy()
            result.setdefault("name", industry_name or result.get("name", "Industry"))
            result["source"] = source
            return result

        fallback = cls._match_metrics(cls.INDUSTRY_FALLBACK, industry_name)
        if fallback:
            result = fallback.copy()
            result.setdefault("name", industry_name or result.get("name", "Industry"))
            result["source"] = "Fallback (static)"
            return result

        return {"name": industry_name or "Industry", "pe": None, "ps": None, "source": source}

    @classmethod
    @lru_cache(maxsize=1)
    def _get_sector_data(cls) -> Tuple[Dict[str, Dict[str, Optional[float]]], str]:
        return cls._download_table(cls.SECTOR_URL, cls.SECTOR_FALLBACK)

    @classmethod
    @lru_cache(maxsize=1)
    def _get_industry_data(cls) -> Tuple[Dict[str, Dict[str, Optional[float]]], str]:
        return cls._download_table(cls.INDUSTRY_URL, cls.INDUSTRY_FALLBACK)

    @classmethod
    def _download_table(
        cls,
        url: str,
        fallback: Dict[str, Dict[str, Optional[float]]],
    ) -> Tuple[Dict[str, Dict[str, Optional[float]]], str]:
        try:
            response = requests.get(
                url, headers=cls.REQUEST_HEADERS, timeout=cls.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            tables = pd.read_html(response.text)
        except Exception:
            return {key: value.copy() for key, value in fallback.items()}, "Fallback (static)"

        for table in tables:
            normalized_columns = [
                column[0] if isinstance(column, tuple) else column for column in table.columns
            ]
            table.columns = normalized_columns
            if {"Name", "P/E", "P/S"}.issubset(set(table.columns)):
                subset = table[["Name", "P/E", "P/S"]].copy()
                subset.replace({"-": pd.NA, "N/A": pd.NA}, inplace=True)
                subset["P/E"] = pd.to_numeric(subset["P/E"], errors="coerce")
                subset["P/S"] = pd.to_numeric(subset["P/S"], errors="coerce")

                mapping: Dict[str, Dict[str, Optional[float]]] = {}
                for _, row in subset.iterrows():
                    name = str(row["Name"]).strip()
                    if not name or name.lower() == "name":
                        continue
                    key = name.lower()
                    mapping[key] = {
                        "name": name,
                        "pe": float(row["P/E"]) if pd.notna(row["P/E"]) else None,
                        "ps": float(row["P/S"]) if pd.notna(row["P/S"]) else None,
                    }

                if mapping:
                    return mapping, "Finviz (live)"

        return {key: value.copy() for key, value in fallback.items()}, "Fallback (static)"

    @staticmethod
    def _match_metrics(
        data: Dict[str, Dict[str, Optional[float]]], name: Optional[str]
    ) -> Optional[Dict[str, Optional[float]]]:
        if not name:
            return None

        lookup_key = name.strip().lower()
        if lookup_key in data:
            return data[lookup_key]

        close_matches = difflib.get_close_matches(lookup_key, data.keys(), n=1, cutoff=0.6)
        if close_matches:
            return data[close_matches[0]]

        return None
