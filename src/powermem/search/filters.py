"""Filter builder utilities for memory search."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


class FilterBuilder:
    """Fluent builder for advanced memory search filters."""

    def __init__(self) -> None:
        self._filters: Dict[str, Any] = {}

    def created_after(self, value: datetime | str) -> "FilterBuilder":
        self._filters["created_after"] = value
        return self

    def created_before(self, value: datetime | str) -> "FilterBuilder":
        self._filters["created_before"] = value
        return self

    def updated_after(self, value: datetime | str) -> "FilterBuilder":
        self._filters["updated_after"] = value
        return self

    def updated_before(self, value: datetime | str) -> "FilterBuilder":
        self._filters["updated_before"] = value
        return self

    def after(self, value: datetime | str) -> "FilterBuilder":
        return self.created_after(value)

    def before(self, value: datetime | str) -> "FilterBuilder":
        return self.created_before(value)

    def importance(self, minimum: Optional[float] = None, maximum: Optional[float] = None) -> "FilterBuilder":
        if minimum is not None:
            self._filters["min_importance"] = minimum
        if maximum is not None:
            self._filters["max_importance"] = maximum
        return self

    def retention(self, minimum: Optional[float] = None, maximum: Optional[float] = None) -> "FilterBuilder":
        if minimum is not None:
            self._filters["min_retention"] = minimum
        if maximum is not None:
            self._filters["max_retention"] = maximum
        return self

    def memory_types(self, memory_types: List[str]) -> "FilterBuilder":
        self._filters["memory_types"] = memory_types
        return self

    def tags(self, tags: List[str], logic: str = "OR") -> "FilterBuilder":
        self._filters["tags"] = tags
        self._filters["tag_logic"] = logic
        return self

    def scopes(self, scopes: List[str]) -> "FilterBuilder":
        self._filters["scopes"] = scopes
        return self

    def user_ids(self, user_ids: List[str]) -> "FilterBuilder":
        self._filters["user_ids"] = user_ids
        return self

    def agent_ids(self, agent_ids: List[str]) -> "FilterBuilder":
        self._filters["agent_ids"] = agent_ids
        return self

    def metadata_contains(self, mapping: Dict[str, Any]) -> "FilterBuilder":
        existing = self._filters.get("metadata_contains", {})
        existing.update(mapping)
        self._filters["metadata_contains"] = existing
        return self

    def metadata_equals(self, mapping: Dict[str, Any]) -> "FilterBuilder":
        existing = self._filters.get("metadata_equals", {})
        existing.update(mapping)
        self._filters["metadata_equals"] = existing
        return self

    def build(self) -> Dict[str, Any]:
        return dict(self._filters)
