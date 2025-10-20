from __future__ import annotations

from .database import init_db
from .simulator import ingest_synthetic


def main() -> None:
    init_db()
    ingest_synthetic(minutes=7*24*60, step_minutes=10)


if __name__ == "__main__":
    main()
