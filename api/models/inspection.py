"""ORM model for inspection records."""
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    part_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # PASS | FAIL | REVIEW
    decision: Mapped[str] = mapped_column(String(32))
    quality_score: Mapped[int] = mapped_column(Integer)
    line: Mapped[str] = mapped_column(String(32), default="demo", index=True)
    defects_json: Mapped[str] = mapped_column(Text, default="[]")
    anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    image_path: Mapped[str] = mapped_column(String(512), default="")
    annotated_path: Mapped[str] = mapped_column(String(512), default="")
    heatmap_path: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
