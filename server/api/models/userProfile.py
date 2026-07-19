from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

from shared.tickers import is_valid_ticker

Sector = Literal[
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]


class UserProfile(BaseModel):
    user_id: str
    username: str
    password_hash: str  # hashed only; owned by auth, not by generation/retrieval

    sectors: list[Sector] = []
    tickers: list[str] = []
    interests: list[str] = []
    desired_sources: list[str] = []

    created_at: datetime
    updated_at: datetime

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, value: list[str]) -> list[str]:
        invalid = [t for t in value if not is_valid_ticker(t)]
        if invalid:
            raise ValueError(f"Unknown ticker symbol(s): {', '.join(invalid)}")
        return [t.upper() for t in value]
