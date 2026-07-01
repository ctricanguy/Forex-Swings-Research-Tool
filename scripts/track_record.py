#!/usr/bin/env python3
"""Mark emitted signals to market and print the auditable track record.

    python scripts/track_record.py --db forex.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forex import tracking
from forex.data.storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="forex.db")
    args = parser.parse_args()

    with Storage(args.db) as storage:
        closed = tracking.update_outcomes(storage)
        print(f"marked {closed} closed span(s)\n")
        print(tracking.track_record(storage))


if __name__ == "__main__":
    main()
