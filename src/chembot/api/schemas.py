from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class SourceItem(BaseModel):
    excerpts: List[int]
    line: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    cached: bool = False


class UsageResponse(BaseModel):
    date: str
    counts: dict
    limits: dict
