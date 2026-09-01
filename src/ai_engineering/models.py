from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ai_engineering.data.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    age: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    income: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    city: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True
    )

    bought: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )