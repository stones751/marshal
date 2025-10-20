from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    select,
    func,
    text,
)
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship, Session

DB_PATH = os.environ.get("PMS_DB_PATH", "/workspace/data/pms.db")

engine = create_engine(f"sqlite+pysqlite:///{DB_PATH}", echo=False, future=True)
Base = declarative_base()


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # generator or transformer
    location: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    telemetry: Mapped[list[Telemetry]] = relationship("Telemetry", back_populates="asset", cascade="all, delete-orphan")
    faults: Mapped[list[Fault]] = relationship("Fault", back_populates="asset", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover
        return f"Asset(id={self.id}, name={self.name}, type={self.type})"


class Telemetry(Base):
    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, default=datetime.utcnow)

    temperature_c: Mapped[float] = mapped_column(Float, nullable=False)
    vibration_mm_s: Mapped[float] = mapped_column(Float, nullable=False)
    voltage_v: Mapped[float] = mapped_column(Float, nullable=False)

    asset: Mapped[Asset] = relationship("Asset", back_populates="telemetry")


class Fault(Base):
    __tablename__ = "faults"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, default=datetime.utcnow)
    code: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # info/warn/critical
    description: Mapped[str] = mapped_column(String(255), nullable=False)

    asset: Mapped[Asset] = relationship("Asset", back_populates="faults")


def init_db(drop_existing: bool = False) -> None:
    if drop_existing and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)


def upsert_default_assets() -> None:
    """Ensure a baseline set of assets exists."""
    defaults = [
        {"name": "Gen-01", "type": "generator", "location": "Plant A"},
        {"name": "Gen-02", "type": "generator", "location": "Plant B"},
        {"name": "TX-01", "type": "transformer", "location": "Substation 1"},
        {"name": "TX-02", "type": "transformer", "location": "Substation 2"},
    ]
    with get_session() as session:
        for d in defaults:
            if not session.scalar(select(func.count()).select_from(Asset).where(Asset.name == d["name"])):
                session.add(Asset(**d))
        session.commit()
