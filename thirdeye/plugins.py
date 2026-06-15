from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from thirdeye.adapters.base import AdapterRegistry


ENTRY_POINT_GROUP = "thirdeye.adapters"


def load_adapter_plugins(registry: AdapterRegistry) -> list[str]:
    """Load installed adapter plugins without making them mandatory dependencies."""
    loaded: list[str] = []
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        factory: Any = entry_point.load()
        adapter = factory() if callable(factory) else factory
        registry.register(adapter)
        loaded.append(adapter.adapter_id)
    return loaded

