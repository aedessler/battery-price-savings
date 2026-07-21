#!/usr/bin/env python
"""Download CAISO public bid data from OASIS.

CAISO publishes masked ("public") bid data 90 days after each trade date.
It is served by the OASIS GroupZip API -- no account or API key required:

    https://oasis.caiso.com/oasisapi/GroupZip
        ?groupid=PUB_DAM_GRP        (day-ahead)  or  PUB_RTM_GRP (real-time)
        &startdatetime=YYYYMMDDT08:00-0000
        &version=3
        &resultformat=6             (6 = CSV, 5 = XML)

Each request returns one full trade day as a zip. Typical sizes per day:
DAM ~0.5 MB zipped (~13 MB CSV), RTM ~1.7 MB zipped (~50 MB CSV).

Note: CAISO's separate bulk S3 bucket (caiso-oasis-s3-prod-groupzips, behind
https://oasis-bulk.caiso.com, requester-pays) only holds LMP/price groups,
not bid data, so this script uses the OASIS API directly.

Examples:
    python download_caiso_bids.py --start 2026-01-01 --end 2026-01-07
    python download_caiso_bids.py --start 2026-01-01 --end 2026-03-31 --market both --extract
"""

import argparse
import datetime as dt
import io
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OASIS_URL = "https://oasis.caiso.com/oasisapi/GroupZip"
GROUP_IDS = {"dam": "PUB_DAM_GRP", "rtm": "PUB_RTM_GRP"}
PUBLICATION_LAG_DAYS = 90
# OASIS rate-limits aggressive clients; ~5 s between requests is the
# commonly safe pace.
DEFAULT_SLEEP = 5.0


def build_url(group_id: str, trade_date: dt.date, result_format: int) -> str:
    # T08:00-0000 = midnight Pacific standard time, i.e. the start of the
    # trade day; GroupZip returns the entire day regardless of the hour.
    params = {
        "groupid": group_id,
        "startdatetime": trade_date.strftime("%Y%m%d") + "T08:00-0000",
        "version": "3",
        "resultformat": str(result_format),
    }
    return f"{OASIS_URL}?{urlencode(params)}"


def download_day(
    market: str,
    trade_date: dt.date,
    out_dir: Path,
    result_format: int,
    extract: bool,
    retries: int = 3,
) -> bool:
    """Download one trade day for one market. Returns True on success."""
    ext = "csv" if result_format == 6 else "xml"
    out_path = out_dir / f"{trade_date:%Y%m%d}_PUB_BID_{market.upper()}_{ext}.zip"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  {out_path.name} already exists, skipping")
        return True

    url = build_url(GROUP_IDS[market], trade_date, result_format)
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "caiso-bid-downloader/1.0"})
            with urlopen(req, timeout=600) as resp:
                data = resp.read()

            # OASIS returns HTTP 200 even for errors; a real result is a zip
            # containing a *_PUB_BID_* file, while errors come back as a tiny
            # zip holding an INVALID_REQUEST message (or as plain text).
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
                names = zf.namelist()
            except zipfile.BadZipFile:
                print(f"  {trade_date}: not a zip (response: {data[:200]!r})")
                return False
            if not any("PUB_BID" in n for n in names):
                print(f"  {trade_date}: OASIS error, zip contains {names}")
                return False

            tmp_path = out_path.with_suffix(".zip.part")
            tmp_path.write_bytes(data)
            tmp_path.rename(out_path)
            print(f"  {out_path.name}  ({len(data) / 1e6:.1f} MB)")

            if extract:
                zf.extractall(out_dir)
                for n in names:
                    print(f"    extracted {n}")
            return True

        except (HTTPError, URLError, TimeoutError) as e:
            wait = 30 * attempt
            print(f"  {trade_date}: {e} (attempt {attempt}/{retries}), "
                  f"retrying in {wait}s")
            if attempt < retries:
                time.sleep(wait)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", required=True, type=dt.date.fromisoformat,
                        help="first trade date, YYYY-MM-DD")
    parser.add_argument("--end", type=dt.date.fromisoformat,
                        help="last trade date, YYYY-MM-DD (default: same as --start)")
    parser.add_argument("--market", choices=["dam", "rtm", "both"], default="dam",
                        help="day-ahead, real-time, or both")
    parser.add_argument("--format", choices=["csv", "xml"], default="csv",
                        help="file format inside the zip")
    parser.add_argument("--out", type=Path, default=Path("caiso_bids"),
                        help="output directory")
    parser.add_argument("--extract", action="store_true",
                        help="also unzip each downloaded file")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help="seconds to wait between requests")
    args = parser.parse_args()

    end = args.end or args.start
    if end < args.start:
        parser.error("--end is before --start")

    latest_available = dt.date.today() - dt.timedelta(days=PUBLICATION_LAG_DAYS)
    if end > latest_available:
        print(f"Public bids are released {PUBLICATION_LAG_DAYS} days after the "
              f"trade date; latest available is {latest_available}. "
              f"Dates after that will be skipped.")
        end = min(end, latest_available)
        if end < args.start:
            return 1

    markets = ["dam", "rtm"] if args.market == "both" else [args.market]
    result_format = 6 if args.format == "csv" else 5
    args.out.mkdir(parents=True, exist_ok=True)

    n_days = (end - args.start).days + 1
    dates = [args.start + dt.timedelta(days=i) for i in range(n_days)]
    print(f"Downloading {len(dates)} day(s) x {markets} to {args.out}/")

    failures = []
    first = True
    for date in dates:
        for market in markets:
            if not first:
                time.sleep(args.sleep)
            first = False
            print(f"{date} {market.upper()}:")
            if not download_day(market, date, args.out, result_format, args.extract):
                failures.append((date, market))

    if failures:
        print(f"\n{len(failures)} download(s) failed:")
        for date, market in failures:
            print(f"  {date} {market.upper()}")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
